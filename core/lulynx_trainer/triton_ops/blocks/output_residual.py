"""Fused gated residual for Anima DiT blocks (bf16): ``x + gate * sublayer``.

Each DiT sub-layer closes with ``x = x + gate.unsqueeze(1) * sublayer_out``,
where ``gate`` is a ``[B, D]`` conditioning vector broadcast over tokens. Eager
this is a multiply then an add (two bandwidth-bound elementwise passes with an
intermediate). This module fuses them into one kernel: read ``x``, the
sub-layer output and ``gate`` once, write the result once.

The backward is trivial and cheap: ``grad_x`` is the upstream gradient
unchanged, ``grad_sublayer = grad_out * gate`` (one pass), and
``grad_gate = sum_tokens(grad_out * sublayer)`` (one reduction). Memory-bound,
so fusing helps the forward and the backward. bf16 in/out.

Clean-room Lulynx implementation; shares no source with any reference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# PyTorch reference (validation oracle)
# ---------------------------------------------------------------------------

def reference(x: torch.Tensor, gate: torch.Tensor, sublayer: torch.Tensor) -> torch.Tensor:
    """``x + gate.unsqueeze(1) * sublayer`` (gate ``[B, D]`` over tokens)."""
    return x + gate.unsqueeze(1) * sublayer


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
def _fwd_kernel(x_ptr, s_ptr, gate_ptr, out_ptr, n_tokens, D, BLOCK_D: tl.constexpr):
    m = tl.program_id(0)
    b = m // n_tokens
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    x = tl.load(x_ptr + m * D + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr + m * D + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gate_ptr + b * D + cols, mask=mask, other=0.0).to(tl.float32)
    out = x + g * s
    tl.store(out_ptr + m * D + cols, out.to(tl.bfloat16), mask=mask)


def _run_forward(x: torch.Tensor, gate: torch.Tensor, sublayer: torch.Tensor) -> torch.Tensor:
    *batch, D = x.shape
    B = batch[0]
    n_rows = 1
    for d in batch:
        n_rows *= d
    n_tokens = n_rows // B

    x_flat = x.reshape(n_rows, D).contiguous()
    s_flat = sublayer.reshape(n_rows, D).contiguous()
    out = torch.empty_like(x_flat)
    block_d = triton.next_power_of_2(D)
    _fwd_kernel[(n_rows,)](
        x_flat, s_flat, gate.contiguous(), out, n_tokens, D,
        BLOCK_D=block_d, num_warps=_num_warps(block_d),
    )
    return out.reshape(*batch, D)


# ---------------------------------------------------------------------------
# Autograd wrapper: Triton forward + PyTorch backward
# ---------------------------------------------------------------------------

class _FusedGatedResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, sublayer):
        ctx.save_for_backward(gate, sublayer)
        ctx.gate_shape = gate.shape
        return _run_forward(x, gate, sublayer)

    @staticmethod
    def backward(ctx, grad_out):
        gate, sublayer = ctx.saved_tensors
        need_x, need_gate, need_s = ctx.needs_input_grad[0:3]
        B = gate.shape[0]

        grad_x = grad_out if need_x else None  # d(out)/dx = 1
        grad_gate = grad_s = None
        if need_gate:
            reduce_dims = tuple(range(1, grad_out.dim() - 1))
            gg = grad_out * sublayer
            grad_gate = (gg.sum(dim=reduce_dims) if reduce_dims else gg).to(gate.dtype)
        if need_s:
            grad_s = grad_out * gate.unsqueeze(1)
        return grad_x, grad_gate, grad_s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused(x: torch.Tensor, gate: torch.Tensor, sublayer: torch.Tensor) -> torch.Tensor:
    """Autograd-aware fused ``x + gate.unsqueeze(1) * sublayer``.

    ``x``/``sublayer`` ``[B, N, D]`` bf16, ``gate`` ``[B, D]``. Returns ``[B, N, D]``.
    """
    if torch.is_grad_enabled() and (
        x.requires_grad or gate.requires_grad or sublayer.requires_grad
    ):
        return _FusedGatedResidual.apply(x, gate, sublayer)
    return _run_forward(x, gate, sublayer)
