"""Fused FFN first-projection (base+LoRA) + SiLU for Anima DiT blocks (bf16).

An Anima MLP is ``layer2(F.silu(layer1(x)))``. With LoRA, ``layer1`` is a
frozen base Linear plus a low-rank delta. Eagerly the pre-activation
``z = layer1(x)`` is written to HBM, re-read by a standalone SiLU kernel, and
written again before ``layer2`` consumes it. This module folds the activation
into ``layer1``'s fused epilogue: the same split design as :mod:`base_lora`
(cuDNN owns the base GEMM, Triton fuses the LoRA path) but the kernel applies
SiLU before the store, so the separate activation launch and one hidden-tensor
HBM round-trip disappear.

The pre-activation ``z`` is emitted alongside ``silu(z)`` so the backward can
form ``grad_z = grad_out * silu'(z)`` and then reuse the LoRA backward. bf16
backward by default (eager-identical), fp32 opt-in. ``layer2`` is left to the
caller (it stays a normal GEMM, optionally base_lora-fused on its own).

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
    base_weight: torch.Tensor,
    base_bias: torch.Tensor | None,
    down: torch.Tensor,
    up: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Pure-PyTorch ``silu(base(x) + scale * up(down(x)))`` (the layer1 path)."""
    base = torch.nn.functional.linear(x, base_weight, base_bias)
    lora = torch.nn.functional.linear(torch.nn.functional.linear(x, down), up)
    return torch.nn.functional.silu(base + lora * scale)


# ---------------------------------------------------------------------------
# Triton forward kernel
# ---------------------------------------------------------------------------

_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 256, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 256, "BLOCK_N": 64}, num_warps=8, num_stages=1),
]


@triton.autotune(configs=_CONFIGS, key=["n_rows", "in_features", "out_features"])
@triton.jit
def _fwd_kernel(
    x_ptr, base_ptr, down_ptr, up_ptr, out_ptr, z_ptr,
    scale,
    n_rows, in_features, out_features, rank,
    s_x, s_base, s_out, s_z, s_down, s_up,
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

        # Phase A: h = x @ down^T  ->  [BLOCK_M, R_PAD] in SRAM (fp32 acc)
        h = tl.zeros([BLOCK_M, R_PAD], dtype=tl.float32)
        for k0 in range(0, in_features, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            k_mask = k < in_features
            xb = tl.load(x_row + k[None, :], mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
            db = tl.load(down_ptr + r[:, None] * s_down + k[None, :],
                         mask=r_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
            h += tl.dot(xb, tl.trans(db))

        # Phase B: z = base + scale*(h@up^T); store silu(z) and z (for backward).
        base_row = base_ptr + rows[:, None] * s_base
        out_row = out_ptr + rows[:, None] * s_out
        z_row = z_ptr + rows[:, None] * s_z
        for n0 in range(0, out_features, BLOCK_N):
            n = n0 + tl.arange(0, BLOCK_N)
            n_mask = n < out_features
            cell = row_mask[:, None] & n_mask[None, :]
            base_blk = tl.load(base_row + n[None, :], mask=cell, other=0.0).to(tl.float32)
            ub = tl.load(up_ptr + n[:, None] * s_up + r[None, :],
                         mask=n_mask[:, None] & r_mask[None, :], other=0.0).to(tl.float32)
            z = base_blk + scale * tl.dot(h, tl.trans(ub))
            act = z * (1.0 / (1.0 + tl.exp(-z)))  # silu
            tl.store(out_row + n[None, :], act.to(tl.bfloat16), mask=cell)
            tl.store(z_row + n[None, :], z.to(tl.bfloat16), mask=cell)


def _pad_rank(rank: int) -> int:
    r = max(16, int(rank))
    return 1 << (r - 1).bit_length()


def _run_forward(x, base_weight, base_bias, down, up, scale):
    *batch, in_features = x.shape
    n_rows = 1
    for d in batch:
        n_rows *= d
    out_features = base_weight.shape[0]
    rank = down.shape[0]

    x_flat = x.reshape(n_rows, in_features).contiguous()
    base = torch.nn.functional.linear(x_flat, base_weight, base_bias)
    out = torch.empty(n_rows, out_features, dtype=torch.bfloat16, device=x.device)
    z = torch.empty(n_rows, out_features, dtype=torch.bfloat16, device=x.device)

    grid = lambda meta: (min(triton.cdiv(n_rows, meta["BLOCK_M"]), MAX_GRID_BLOCKS),)
    _fwd_kernel[grid](
        x_flat, base, down, up, out, z,
        float(scale),
        n_rows, in_features, out_features, rank,
        x_flat.stride(0), base.stride(0), out.stride(0), z.stride(0), down.stride(0), up.stride(0),
        R_PAD=_pad_rank(rank),
    )
    return out.reshape(*batch, out_features), z.reshape(*batch, out_features)


# ---------------------------------------------------------------------------
# Autograd wrapper: Triton forward + PyTorch backward
# ---------------------------------------------------------------------------

class _FusedFFNSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, base_weight, base_bias, down, up, scale, fp32_backward):
        down = down.contiguous()
        up = up.contiguous()
        out, z = _run_forward(x, base_weight, base_bias, down, up, scale)
        ctx.save_for_backward(x, base_weight, down, up, z)
        ctx.scale = float(scale)
        ctx.fp32_backward = bool(fp32_backward)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, base_weight, down, up, z = ctx.saved_tensors
        scale = ctx.scale
        in_features = x.shape[-1]
        gout = grad_out.reshape(-1, grad_out.shape[-1])
        xf = x.reshape(-1, in_features)

        # grad through SiLU: grad_z = grad_out * silu'(z), computed in fp32.
        zf = z.reshape(-1, z.shape[-1]).float()
        sg = torch.sigmoid(zf)
        silup = sg * (1.0 + zf * (1.0 - sg))
        gz = gout.float() * silup  # [rows, out]

        fp32 = ctx.fp32_backward
        need_x, need_down, need_up = (
            ctx.needs_input_grad[0], ctx.needs_input_grad[3], ctx.needs_input_grad[4],
        )
        grad_x = grad_down = grad_up = None

        if fp32:
            xb = xf.float()
            downf = down.float()
            upf = up.float()
            h = xb @ downf.t()
            d_h = scale * (gz @ upf)
            if need_down:
                grad_down = (d_h.t() @ xb).to(down.dtype)
            if need_up:
                grad_up = (scale * (gz.t() @ h)).to(up.dtype)
            if need_x:
                gx = (gz @ base_weight.float()) + d_h @ downf
                grad_x = gx.reshape_as(x).to(x.dtype)
        else:
            gzb = gz.to(x.dtype)  # bf16 backward (eager-parity), fp32-accum GEMMs
            h = xf @ down.t()
            d_h = (gzb @ up) * scale
            if need_down:
                grad_down = d_h.t() @ xf
            if need_up:
                grad_up = (gzb.t() @ h) * scale
            if need_x:
                grad_x = (gzb @ base_weight + d_h @ down).reshape_as(x)

        # align: x, base_weight, base_bias, down, up, scale, fp32_backward
        return grad_x, None, None, grad_down, grad_up, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: torch.Tensor | None,
    down: torch.Tensor,
    up: torch.Tensor,
    scale: float,
    fp32_backward: bool = False,
) -> torch.Tensor:
    """Autograd-aware fused ``silu(base(x) + scale * up(down(x)))`` (layer1).

    Same argument contract as :func:`base_lora.fused`; returns the activated
    hidden ``[*, hidden]`` bf16 tensor (feed it to ``layer2`` next).
    """
    if torch.is_grad_enabled() and (x.requires_grad or down.requires_grad or up.requires_grad):
        return _FusedFFNSiLU.apply(x, base_weight, base_bias, down, up, scale, fp32_backward)
    out, _ = _run_forward(x, base_weight, base_bias, down.contiguous(), up.contiguous(), scale)
    return out
