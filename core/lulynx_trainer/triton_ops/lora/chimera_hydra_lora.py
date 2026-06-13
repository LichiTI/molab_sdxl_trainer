"""Fused ChimeraHydra dual-pool LoRA delta (bf16), training-correct.

ChimeraHydra adds a mixture-of-experts LoRA delta on top of a frozen Linear:
each pool holds ``E`` low-rank experts and a per-token gate produces ``weights``
(dense softmax or top-k softmax+scatter). The pool delta is

    delta = sum_e weights[..., e] * (up_e @ (down_e @ x)) * scaling

(see ``chimera_hydra._mixed_delta``). The eager path materialises two large
intermediates — ``projected`` ``[*, E, rank]`` and ``deltas`` ``[*, E, out]`` —
and reduces over the expert axis afterwards. This module fuses the whole thing
into one Triton kernel: ``x`` is read once, every expert's hidden state ``h_e``
stays in SRAM, and the gated sum is accumulated per output tile so the
``[*, E, out]`` ``deltas`` tensor is never written to HBM.

The routing itself (gate Linear, softmax, top-k, scatter) stays in PyTorch and
only feeds the dense ``weights`` tensor in — the kernel is purely the dense
batched low-rank projection + gated reduction, so it carries no data-dependent
control flow. Top-k sparsity lives entirely in zero entries of ``weights``.

Like :mod:`base_lora`, the Triton forward is wrapped in an ``autograd.Function``
(bf16 backward by default, opt-in fp32). The expert count is a ``constexpr`` so
the per-expert loop unrolls; any power-of-two-padded rank works.

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
    x: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    weights: torch.Tensor,
    *,
    scaling: float = 1.0,
) -> torch.Tensor:
    """Pure-PyTorch ``sum_e weights_e * up_e(down_e(x)) * scaling``.

    Mirrors ``chimera_hydra._mixed_delta`` (which uses ``scaling=1.0`` and lets
    the caller scale the summed pools). ``down`` is ``[E, rank, in]``, ``up`` is
    ``[E, out, rank]``, ``weights`` is ``[*, E]``.
    """
    projected = torch.einsum("...i,eri->...er", x, down)
    deltas = torch.einsum("...er,eor->...eo", projected, up)
    return (weights.unsqueeze(-1) * deltas).sum(dim=-2) * scaling


# ---------------------------------------------------------------------------
# Triton forward kernel
# ---------------------------------------------------------------------------

_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 256, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_CONFIGS, key=["n_rows", "in_features", "out_features", "num_experts"])
@triton.jit
def _fwd_kernel(
    x_ptr, down_ptr, up_ptr, w_ptr, out_ptr,
    scale,
    n_rows, in_features, out_features, rank, num_experts,
    s_x, s_de, s_dr, s_ue, s_uo, s_w, s_out,
    NUM_EXPERTS: tl.constexpr, R_PAD: tl.constexpr,
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
        out_row = out_ptr + rows[:, None] * s_out

        # One expert at a time (NUM_EXPERTS is constexpr -> the loop unrolls).
        # Triton has no list-of-SRAM-tensors, so instead of caching every h_e we
        # accumulate each expert's gated delta straight into the fp32 ``out``
        # buffer: the [rows, experts, out] deltas are still never materialised —
        # only the [rows, out] output is, read back NUM_EXPERTS-1 times.
        for e in range(NUM_EXPERTS):
            # Phase A: h = x @ down_e^T  ->  [BLOCK_M, R_PAD] in SRAM (fp32 acc)
            h = tl.zeros([BLOCK_M, R_PAD], dtype=tl.float32)
            for k0 in range(0, in_features, BLOCK_K):
                k = k0 + tl.arange(0, BLOCK_K)
                k_mask = k < in_features
                xb = tl.load(
                    x_row + k[None, :],
                    mask=row_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
                db = tl.load(
                    down_ptr + e * s_de + r[:, None] * s_dr + k[None, :],
                    mask=r_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
                h += tl.dot(xb, tl.trans(db))
            w_e = tl.load(w_ptr + rows * s_w + e, mask=row_mask, other=0.0).to(tl.float32)

            # Phase B: add this expert's gated delta to out; scale at the last.
            for n0 in range(0, out_features, BLOCK_N):
                n = n0 + tl.arange(0, BLOCK_N)
                n_mask = n < out_features
                ub = tl.load(
                    up_ptr + e * s_ue + n[:, None] * s_uo + r[None, :],
                    mask=n_mask[:, None] & r_mask[None, :], other=0.0,
                ).to(tl.float32)
                contrib = w_e[:, None] * tl.dot(h, tl.trans(ub))
                ptr = out_row + n[None, :]
                mask = row_mask[:, None] & n_mask[None, :]
                if e == 0:
                    cur = contrib
                else:
                    cur = tl.load(ptr, mask=mask, other=0.0) + contrib
                if e == NUM_EXPERTS - 1:
                    cur = cur * scale
                tl.store(ptr, cur, mask=mask)


def _pad_rank(rank: int) -> int:
    """Smallest power-of-two ``>= max(16, rank)`` (tl.dot / tl.arange need it)."""
    r = max(16, int(rank))
    return 1 << (r - 1).bit_length()


def _run_forward(
    x: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    weights: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Launch the fused pool-delta kernel; returns the delta with ``x``'s batch dims."""
    *batch, in_features = x.shape
    n_rows = 1
    for d in batch:
        n_rows *= d
    num_experts, rank, _in = down.shape
    out_features = up.shape[1]

    x_flat = x.reshape(n_rows, in_features).contiguous()
    w_flat = weights.reshape(n_rows, num_experts).contiguous()
    out = torch.empty(n_rows, out_features, dtype=torch.bfloat16, device=x.device)

    grid = lambda meta: (min(triton.cdiv(n_rows, meta["BLOCK_M"]), MAX_GRID_BLOCKS),)
    _fwd_kernel[grid](
        x_flat, down, up, w_flat, out,
        float(scaling),
        n_rows, in_features, out_features, rank, num_experts,
        x_flat.stride(0), down.stride(0), down.stride(1), up.stride(0), up.stride(1),
        w_flat.stride(0), out.stride(0),
        NUM_EXPERTS=num_experts, R_PAD=_pad_rank(rank),
    )
    return out.reshape(*batch, out_features)


# ---------------------------------------------------------------------------
# Autograd wrapper: Triton forward + PyTorch backward
# ---------------------------------------------------------------------------

class _FusedChimeraDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, down, up, weights, scaling, fp32_backward):
        down = down.contiguous()
        up = up.contiguous()
        weights = weights.contiguous()
        ctx.save_for_backward(x, down, up, weights)
        ctx.scaling = float(scaling)
        ctx.fp32_backward = bool(fp32_backward)
        return _run_forward(x, down, up, weights, scaling)

    @staticmethod
    def backward(ctx, grad_out):
        x, down, up, weights = ctx.saved_tensors
        s = ctx.scaling
        fp32 = ctx.fp32_backward
        n_experts, _rank, in_features = down.shape
        out_features = up.shape[1]

        xf = x.reshape(-1, in_features)
        gy = grad_out.reshape(-1, out_features)
        wf = weights.reshape(-1, n_experts)
        xb = xf.float() if fp32 else xf
        gyb = gy.float() if fp32 else gy

        need_x = ctx.needs_input_grad[0]
        need_down = ctx.needs_input_grad[1]
        need_up = ctx.needs_input_grad[2]
        need_w = ctx.needs_input_grad[3]

        grad_x = None
        grad_down = torch.zeros_like(down) if need_down else None
        grad_up = torch.zeros_like(up) if need_up else None
        grad_w = torch.zeros_like(wf) if need_w else None

        for e in range(n_experts):
            d_e = down[e].float() if fp32 else down[e]      # [rank, in]
            u_e = up[e].float() if fp32 else up[e]          # [out, rank]
            w_e = wf[:, e:e + 1]
            w_e = w_e.float() if fp32 else w_e              # [rows, 1]
            h_e = xb @ d_e.t()                              # [rows, rank]
            # out = sum_e s * w_e * delta_e  ->  d_delta_e = s * w_e * gy
            d_delta = (s * w_e) * gyb                       # [rows, out]
            if need_up:
                grad_up[e] = (d_delta.t() @ h_e).to(up.dtype)
            grad_h = d_delta @ u_e                          # [rows, rank]
            if need_down:
                grad_down[e] = (grad_h.t() @ xb).to(down.dtype)
            if need_x:
                contrib = grad_h @ d_e                      # [rows, in]
                grad_x = contrib if grad_x is None else grad_x + contrib
            if need_w:
                delta_e = h_e @ u_e.t()                     # [rows, out]
                grad_w[:, e] = (s * (delta_e * gyb).sum(dim=-1)).to(weights.dtype)

        if need_x:
            grad_x = grad_x.reshape_as(x).to(x.dtype)
        if need_w:
            grad_w = grad_w.reshape_as(weights)

        # align with forward args: x, down, up, weights, scaling, fp32_backward
        return grad_x, grad_down, grad_up, grad_w, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused(
    x: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    weights: torch.Tensor,
    *,
    scaling: float = 1.0,
    fp32_backward: bool = False,
) -> torch.Tensor:
    """Autograd-aware fused ChimeraHydra pool delta.

    Args:
        x:       ``[*, in_features]`` bf16 activations (already dropout-applied).
        down:    ``[experts, rank, in_features]`` bf16 trainable down-projections.
        up:      ``[experts, out_features, rank]`` bf16 trainable up-projections.
        weights: ``[*, experts]`` gate weights (dense or top-k; zeros allowed).
        scaling: optional post-scale (default ``1.0`` matches ``_mixed_delta``;
            ChimeraHydra applies its ``self.scaling`` on the summed pools).
        fp32_backward: accumulate the LoRA gradients in fp32 (slower, slightly
            more accurate). Default ``False`` matches the eager bf16 backward.

    Returns:
        ``[*, out_features]`` bf16 pool delta.
    """
    if torch.is_grad_enabled() and (
        x.requires_grad or down.requires_grad or up.requires_grad or weights.requires_grad
    ):
        return _FusedChimeraDelta.apply(x, down, up, weights, scaling, fp32_backward)
    return _run_forward(x, down.contiguous(), up.contiguous(), weights.contiguous(), scaling)
