"""Fused Q/K/V LoRA for self-attention (bf16), training-correct.

In a self-attention projection the *same* activation ``x`` feeds three
independent frozen-Linear + LoRA chains (``q_proj``/``k_proj``/``v_proj``).
Computing them separately reloads ``x`` three times and launches three LoRA
kernels. This module keeps the three cuDNN base GEMMs but fuses all three LoRA
paths into one kernel: ``x`` is read once, the three low-rank hidden states
share SRAM, and the q/k/v LoRA deltas are added to their bases in a single
launch.

Like :mod:`base_lora`, the Triton forward is wrapped in an ``autograd.Function``
so it is training-safe; the backward defaults to bf16 (numerically identical to
the eager LoRA backward) with an opt-in fp32 high-accuracy path. Per-head output
widths, scales and an arbitrary (power-of-two padded) rank are supported.

Clean-room Lulynx implementation; shares no source with any reference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..config import MAX_GRID_BLOCKS


# ---------------------------------------------------------------------------
# PyTorch reference (validation oracle)
# ---------------------------------------------------------------------------

def reference(
    x, bw_q, bw_k, bw_v, bb_q, bb_k, bb_v,
    down_q, down_k, down_v, up_q, up_k, up_v,
    scale_q, scale_k, scale_v,
):
    """Three independent ``base(x) + scale * up(down(x))`` chains."""
    def _path(bw, bb, down, up, scale):
        base = torch.nn.functional.linear(x, bw, bb)
        lora = torch.nn.functional.linear(torch.nn.functional.linear(x, down), up)
        return base + lora * scale

    return (
        _path(bw_q, bb_q, down_q, up_q, scale_q),
        _path(bw_k, bb_k, down_k, up_k, scale_k),
        _path(bw_v, bb_v, down_v, up_v, scale_v),
    )


# ---------------------------------------------------------------------------
# Triton forward kernel
# ---------------------------------------------------------------------------

_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 256, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 256, "BLOCK_N": 64}, num_warps=8, num_stages=1),
]


@triton.jit
def _epilogue(
    h, base_ptr, out_ptr, rows, row_mask, s_base, s_out, up_ptr, s_up, r, r_mask,
    scale, out_features, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """out = base + scale * (h @ up^T) for one head, looped over output cols."""
    base_row = base_ptr + rows[:, None] * s_base
    out_row = out_ptr + rows[:, None] * s_out
    for n0 in range(0, out_features, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        n_mask = n < out_features
        base_blk = tl.load(
            base_row + n[None, :], mask=row_mask[:, None] & n_mask[None, :], other=0.0
        ).to(tl.float32)
        ub = tl.load(
            up_ptr + n[:, None] * s_up + r[None, :],
            mask=n_mask[:, None] & r_mask[None, :], other=0.0,
        ).to(tl.float32)
        y = base_blk + scale * tl.dot(h, tl.trans(ub))
        tl.store(out_row + n[None, :], y.to(tl.bfloat16), mask=row_mask[:, None] & n_mask[None, :])


@triton.autotune(configs=_CONFIGS, key=["n_rows", "in_features", "out_q", "out_k", "out_v"])
@triton.jit
def _fwd_kernel(
    x_ptr, bq_ptr, bk_ptr, bv_ptr,
    dq_ptr, dk_ptr, dv_ptr, uq_ptr, uk_ptr, uv_ptr,
    oq_ptr, ok_ptr, ov_ptr,
    sq, sk, sv,
    n_rows, in_features, out_q, out_k, out_v, rank,
    s_x, s_bq, s_bk, s_bv, s_d, s_u, s_oq, s_ok, s_ov,
    R_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    n_blocks = tl.cdiv(n_rows, BLOCK_M)
    pid = tl.program_id(0)
    r = tl.arange(0, R_PAD)
    r_mask = r < rank

    for blk in range(pid, n_blocks, tl.num_programs(0)):
        rows = blk * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < n_rows
        x_row = x_ptr + rows[:, None] * s_x

        # Phase A: read x once, accumulate h_q/h_k/h_v in SRAM (fp32 acc).
        h_q = tl.zeros([BLOCK_M, R_PAD], dtype=tl.float32)
        h_k = tl.zeros([BLOCK_M, R_PAD], dtype=tl.float32)
        h_v = tl.zeros([BLOCK_M, R_PAD], dtype=tl.float32)
        for k0 in range(0, in_features, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            k_mask = k < in_features
            xb = tl.load(
                x_row + k[None, :], mask=row_mask[:, None] & k_mask[None, :], other=0.0
            ).to(tl.float32)
            dqb = tl.load(dq_ptr + r[:, None] * s_d + k[None, :],
                          mask=r_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
            h_q += tl.dot(xb, tl.trans(dqb))
            dkb = tl.load(dk_ptr + r[:, None] * s_d + k[None, :],
                          mask=r_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
            h_k += tl.dot(xb, tl.trans(dkb))
            dvb = tl.load(dv_ptr + r[:, None] * s_d + k[None, :],
                          mask=r_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
            h_v += tl.dot(xb, tl.trans(dvb))

        # Phase B: per-head epilogue (supports different output widths).
        _epilogue(h_q, bq_ptr, oq_ptr, rows, row_mask, s_bq, s_oq, uq_ptr, s_u, r, r_mask,
                  sq, out_q, BLOCK_M, BLOCK_N)
        _epilogue(h_k, bk_ptr, ok_ptr, rows, row_mask, s_bk, s_ok, uk_ptr, s_u, r, r_mask,
                  sk, out_k, BLOCK_M, BLOCK_N)
        _epilogue(h_v, bv_ptr, ov_ptr, rows, row_mask, s_bv, s_ov, uv_ptr, s_u, r, r_mask,
                  sv, out_v, BLOCK_M, BLOCK_N)


def _pad_rank(rank: int) -> int:
    r = max(16, int(rank))
    return 1 << (r - 1).bit_length()


def _run_forward(x, bw_q, bw_k, bw_v, bb_q, bb_k, bb_v,
                 down_q, down_k, down_v, up_q, up_k, up_v, sq, sk, sv):
    *batch, in_features = x.shape
    n_rows = 1
    for d in batch:
        n_rows *= d
    out_q, out_k, out_v = bw_q.shape[0], bw_k.shape[0], bw_v.shape[0]
    rank = down_q.shape[0]

    x_flat = x.reshape(n_rows, in_features).contiguous()
    base_q = torch.nn.functional.linear(x_flat, bw_q, bb_q)
    base_k = torch.nn.functional.linear(x_flat, bw_k, bb_k)
    base_v = torch.nn.functional.linear(x_flat, bw_v, bb_v)
    oq = torch.empty(n_rows, out_q, dtype=torch.bfloat16, device=x.device)
    ok = torch.empty(n_rows, out_k, dtype=torch.bfloat16, device=x.device)
    ov = torch.empty(n_rows, out_v, dtype=torch.bfloat16, device=x.device)

    grid = lambda meta: (min(triton.cdiv(n_rows, meta["BLOCK_M"]), MAX_GRID_BLOCKS),)
    _fwd_kernel[grid](
        x_flat, base_q, base_k, base_v,
        down_q, down_k, down_v, up_q, up_k, up_v,
        oq, ok, ov,
        float(sq), float(sk), float(sv),
        n_rows, in_features, out_q, out_k, out_v, rank,
        x_flat.stride(0), base_q.stride(0), base_k.stride(0), base_v.stride(0),
        down_q.stride(0), up_q.stride(0), oq.stride(0), ok.stride(0), ov.stride(0),
        R_PAD=_pad_rank(rank),
    )
    return (oq.reshape(*batch, out_q), ok.reshape(*batch, out_k), ov.reshape(*batch, out_v))


# ---------------------------------------------------------------------------
# Autograd wrapper: Triton forward + PyTorch backward
# ---------------------------------------------------------------------------

class _FusedQKVLoRA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bw_q, bw_k, bw_v, bb_q, bb_k, bb_v,
                down_q, down_k, down_v, up_q, up_k, up_v, sq, sk, sv, fp32_backward):
        downs = [t.contiguous() for t in (down_q, down_k, down_v)]
        ups = [t.contiguous() for t in (up_q, up_k, up_v)]
        ctx.save_for_backward(x, bw_q, bw_k, bw_v, *downs, *ups)
        ctx.scales = (float(sq), float(sk), float(sv))
        ctx.fp32_backward = bool(fp32_backward)
        return _run_forward(x, bw_q, bw_k, bw_v, bb_q, bb_k, bb_v, *downs, *ups, sq, sk, sv)

    @staticmethod
    def backward(ctx, gq, gk, gv):
        x, bw_q, bw_k, bw_v, dq, dk, dv, uq, uk, uv = ctx.saved_tensors
        sq, sk, sv = ctx.scales
        in_features = x.shape[-1]
        xf = x.reshape(-1, in_features)
        fp32 = ctx.fp32_backward

        need_x = ctx.needs_input_grad[0]
        # trainable arg positions: down_q=7,down_k=8,down_v=9, up_q=10,up_k=11,up_v=12
        need_d = (ctx.needs_input_grad[7], ctx.needs_input_grad[8], ctx.needs_input_grad[9])
        need_u = (ctx.needs_input_grad[10], ctx.needs_input_grad[11], ctx.needs_input_grad[12])

        xb = xf.float() if fp32 else xf
        grad_x = None
        grads_d = [None, None, None]
        grads_u = [None, None, None]
        heads = (
            (gq, bw_q, dq, uq, sq, 0),
            (gk, bw_k, dk, uk, sk, 1),
            (gv, bw_v, dv, uv, sv, 2),
        )
        for g, bw, d, u, s, i in heads:
            gy = g.reshape(-1, g.shape[-1])
            gyb = gy.float() if fp32 else gy
            db = d.float() if fp32 else d
            ub = u.float() if fp32 else u
            h = xb @ db.t()                # [rows, rank]
            d_h = s * (gyb @ ub)           # [rows, rank]
            if need_d[i]:
                grads_d[i] = (d_h.t() @ xb).to(d.dtype)
            if need_u[i]:
                grads_u[i] = (s * (gyb.t() @ h)).to(u.dtype)
            if need_x:
                contrib = (gy.to(bw.dtype) @ bw)
                contrib = contrib.float() + d_h @ db if fp32 else contrib + (d_h @ db)
                grad_x = contrib if grad_x is None else grad_x + contrib
        if need_x:
            grad_x = grad_x.reshape_as(x).to(x.dtype)

        # align: x, bw_q,bw_k,bw_v, bb_q,bb_k,bb_v, down_q,k,v, up_q,k,v, sq,sk,sv, fp32
        return (grad_x, None, None, None, None, None, None,
                grads_d[0], grads_d[1], grads_d[2], grads_u[0], grads_u[1], grads_u[2],
                None, None, None, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused(
    x, bw_q, bw_k, bw_v, bb_q, bb_k, bb_v,
    down_q, down_k, down_v, up_q, up_k, up_v,
    scale_q, scale_k, scale_v, fp32_backward: bool = False,
):
    """Autograd-aware fused Q/K/V LoRA for a shared input ``x``.

    All three projections take ``x`` ``[*, in_features]``; ``bw_*`` are
    ``[out_*, in_features]`` frozen weights, ``down_*`` ``[rank, in_features]``
    and ``up_*`` ``[out_*, rank]`` trainable factors (shared ``rank``).

    Returns ``(q, k, v)`` bf16 tensors with ``x``'s batch dims.
    """
    trainable = any(t.requires_grad for t in (x, down_q, down_k, down_v, up_q, up_k, up_v))
    if torch.is_grad_enabled() and trainable:
        return _FusedQKVLoRA.apply(
            x, bw_q, bw_k, bw_v, bb_q, bb_k, bb_v,
            down_q, down_k, down_v, up_q, up_k, up_v,
            scale_q, scale_k, scale_v, fp32_backward,
        )
    downs = [t.contiguous() for t in (down_q, down_k, down_v)]
    ups = [t.contiguous() for t in (up_q, up_k, up_v)]
    return _run_forward(x, bw_q, bw_k, bw_v, bb_q, bb_k, bb_v, *downs, *ups,
                        scale_q, scale_k, scale_v)
