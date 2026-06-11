"""Fused AdaLN modulation + LayerNorm for Anima DiT blocks (bf16).

An Anima block modulates its activation before each sub-layer with

    normalized = layer_norm(x.float(), (D,)).to(x.dtype)   # no affine
    out        = normalized * (1 + scale) + shift

where ``scale``/``shift`` are ``[B, D]`` conditioning vectors broadcast over the
token dimension. Eagerly this is two kernels (the LayerNorm, then the affine
FMA) with the intermediate ``normalized`` written to and re-read from HBM. This
module fuses both into one kernel: each row is read once, normalised in SRAM
(fp32 accumulation, matching the eager ``.float()`` cast), modulated, and
written once.

The Triton forward is wrapped in an ``autograd.Function`` with an exact
PyTorch backward (standard affine-free LayerNorm backward; the ``(1 + scale)``
factor is folded into the upstream gradient). bf16 in/out, fp32 math.

Clean-room Lulynx implementation; shares no source with any reference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# F.layer_norm default epsilon; the eager block calls layer_norm without an
# explicit eps, so we must match 1e-5 exactly for numerical parity.
_EPS = 1e-5


# ---------------------------------------------------------------------------
# PyTorch reference (validation oracle)
# ---------------------------------------------------------------------------

def reference(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = _EPS):
    """``layer_norm(x)`` (no affine, fp32) then ``* (1 + scale) + shift``.

    ``x`` is ``[B, N, D]``; ``shift``/``scale`` are ``[B, D]`` broadcast over N.
    """
    normalized = torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), eps=eps).to(x.dtype)
    return normalized * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Triton forward kernel (one program per row)
# ---------------------------------------------------------------------------

def _num_warps(block_d: int) -> int:
    if block_d <= 1024:
        return 4
    if block_d <= 4096:
        return 8
    return 16


@triton.jit
def _adaln_fwd_kernel(
    x_ptr, shift_ptr, scale_ptr, out_ptr, mean_ptr, rstd_ptr,
    n_tokens, D, eps,
    BLOCK_D: tl.constexpr,
):
    """out[m] = norm(x[m]) * (1 + scale[b]) + shift[b], b = m // n_tokens."""
    m = tl.program_id(0)
    b = m // n_tokens
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    x = tl.load(x_ptr + m * D + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    norm = xc * rstd

    sc = tl.load(scale_ptr + b * D + cols, mask=mask, other=0.0).to(tl.float32)
    sh = tl.load(shift_ptr + b * D + cols, mask=mask, other=0.0).to(tl.float32)
    out = norm * (1.0 + sc) + sh
    tl.store(out_ptr + m * D + cols, out.to(tl.bfloat16), mask=mask)
    tl.store(mean_ptr + m, mean)
    tl.store(rstd_ptr + m, rstd)


def _run_forward(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float):
    *batch, D = x.shape
    B = batch[0]
    n_rows = 1
    for d in batch:
        n_rows *= d
    n_tokens = n_rows // B  # tokens per batch row (scale/shift broadcast over these)

    x_flat = x.reshape(n_rows, D).contiguous()
    out = torch.empty_like(x_flat)
    mean = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)

    block_d = triton.next_power_of_2(D)
    _adaln_fwd_kernel[(n_rows,)](
        x_flat, shift.contiguous(), scale.contiguous(), out, mean, rstd,
        n_tokens, D, float(eps),
        BLOCK_D=block_d, num_warps=_num_warps(block_d),
    )
    return out.reshape(*batch, D), mean, rstd


@triton.jit
def _adaln_bwd_dx_kernel(
    x_ptr, g_ptr, scale_ptr, mean_ptr, rstd_ptr, gx_ptr, norm_ptr,
    n_tokens, D,
    BLOCK_D: tl.constexpr,
):
    """grad_x for affine-free LayerNorm with upstream go = g * (1 + scale).

    Also emits ``norm`` so the (tiny) scale/shift gradients can be reduced over
    tokens in one cheap PyTorch pass instead of recomputing the normalisation.
    """
    m = tl.program_id(0)
    b = m // n_tokens
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    x = tl.load(x_ptr + m * D + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + m * D + cols, mask=mask, other=0.0).to(tl.float32)
    sc = tl.load(scale_ptr + b * D + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + m)
    rstd = tl.load(rstd_ptr + m)

    norm = tl.where(mask, (x - mean) * rstd, 0.0)
    go = tl.where(mask, g * (1.0 + sc), 0.0)
    mean_go = tl.sum(go, axis=0) / D
    mean_gonorm = tl.sum(go * norm, axis=0) / D
    gx = rstd * (go - mean_go - norm * mean_gonorm)

    tl.store(gx_ptr + m * D + cols, gx.to(tl.bfloat16), mask=mask)
    tl.store(norm_ptr + m * D + cols, norm.to(tl.bfloat16), mask=mask)


def _run_backward(g, x, scale, mean, rstd, need_x, need_shift, need_scale):
    *batch, D = x.shape
    B = batch[0]
    n_rows = 1
    for d in batch:
        n_rows *= d
    n_tokens = n_rows // B

    grad_x = grad_shift = grad_scale = None
    g_flat = g.reshape(n_rows, D).contiguous()

    if need_x or need_scale:
        x_flat = x.reshape(n_rows, D).contiguous()
        gx = torch.empty_like(g_flat) if need_x else g_flat  # gx unused if not need_x
        norm = torch.empty_like(g_flat)
        block_d = triton.next_power_of_2(D)
        _adaln_bwd_dx_kernel[(n_rows,)](
            x_flat, g_flat, scale.contiguous(), mean, rstd,
            gx if need_x else x_flat, norm,
            n_tokens, D,
            BLOCK_D=block_d, num_warps=_num_warps(block_d),
        )
        if need_x:
            grad_x = gx.reshape(*batch, D).to(x.dtype)
        if need_scale:
            # sum over token dim(s): norm is [n_rows, D] -> [B, D]
            grad_scale = (g_flat * norm).reshape(B, n_tokens, D).sum(dim=1).to(scale.dtype)
    if need_shift:
        grad_shift = g_flat.reshape(B, n_tokens, D).sum(dim=1).to(scale.dtype)

    return grad_x, grad_shift, grad_scale


# ---------------------------------------------------------------------------
# Autograd wrapper: Triton forward + PyTorch backward
# ---------------------------------------------------------------------------

class _FusedAdaLN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift, scale, eps):
        out, mean, rstd = _run_forward(x, shift, scale, eps)
        ctx.save_for_backward(x, scale, mean, rstd)
        ctx.eps = float(eps)
        return out

    @staticmethod
    def backward(ctx, g):
        x, scale, mean, rstd = ctx.saved_tensors
        need_x, need_shift, need_scale = ctx.needs_input_grad[0:3]
        grad_x, grad_shift, grad_scale = _run_backward(
            g, x, scale, mean, rstd, need_x, need_shift, need_scale
        )
        return grad_x, grad_shift, grad_scale, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = _EPS):
    """Autograd-aware fused AdaLN: ``layer_norm(x) * (1 + scale) + shift``.

    ``x`` ``[B, N, D]`` bf16, ``shift``/``scale`` ``[B, D]``. Returns ``[B, N, D]``.
    Falls through to :func:`reference` math via autograd when grad is needed.
    """
    if torch.is_grad_enabled() and (x.requires_grad or scale.requires_grad or shift.requires_grad):
        return _FusedAdaLN.apply(x, shift, scale, eps)
    out, _, _ = _run_forward(x, shift, scale, eps)
    return out
