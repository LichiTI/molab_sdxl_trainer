"""Fused frozen-Linear + LoRA forward (bf16), training-correct.

Computes ``y = base(x) + scale * up(down(x))`` where ``base`` is a frozen
``nn.Linear`` and ``down``/``up`` are the trainable LoRA factors.

Split design: cuDNN computes the frozen base GEMM; a single Triton kernel
then fuses the LoRA down-projection, up-projection, the ``scale`` multiply
and the residual add, keeping the low-rank hidden state ``h`` in SRAM.

Unlike a forward-only kernel, :func:`fused` is wrapped in an
``autograd.Function`` so it is safe inside a training graph: the Triton
kernel drives the forward pass and a compact PyTorch backward produces the
gradients for ``x``, ``down`` and ``up`` (the base weight stays frozen).

The kernel pads the LoRA rank up to a power-of-two ``>= 16`` and masks the
padded lanes, so *any* rank works (8, 24, 48, ...) — not just powers of two.
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
    """Pure-PyTorch ``base(x) + scale * up(down(x))``."""
    base = torch.nn.functional.linear(x, base_weight, base_bias)
    h = torch.nn.functional.linear(x, down)
    lora = torch.nn.functional.linear(h, up)
    return base + lora * scale


# ---------------------------------------------------------------------------
# Triton forward kernel
# ---------------------------------------------------------------------------

_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 256, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 256, "BLOCK_N": 64}, num_warps=8, num_stages=1),
]


@triton.autotune(configs=_CONFIGS, key=["n_rows", "in_features", "out_features"])
@triton.jit
def _fwd_kernel(
    x_ptr, base_ptr, down_ptr, up_ptr, out_ptr,
    scale,
    n_rows, in_features, out_features, rank,
    s_x, s_base, s_out, s_down, s_up,
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

        # Phase A: h = x @ down^T   ->  [BLOCK_M, R_PAD] in SRAM (fp32 acc)
        h = tl.zeros([BLOCK_M, R_PAD], dtype=tl.float32)
        for k0 in range(0, in_features, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            k_mask = k < in_features
            xb = tl.load(
                x_row + k[None, :],
                mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            db = tl.load(
                down_ptr + r[:, None] * s_down + k[None, :],
                mask=r_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            h += tl.dot(xb, tl.trans(db))

        # Phase B: out = base + scale * (h @ up^T)
        base_row = base_ptr + rows[:, None] * s_base
        out_row = out_ptr + rows[:, None] * s_out
        for n0 in range(0, out_features, BLOCK_N):
            n = n0 + tl.arange(0, BLOCK_N)
            n_mask = n < out_features
            base_blk = tl.load(
                base_row + n[None, :],
                mask=row_mask[:, None] & n_mask[None, :], other=0.0,
            ).to(tl.float32)
            ub = tl.load(
                up_ptr + n[:, None] * s_up + r[None, :],
                mask=n_mask[:, None] & r_mask[None, :], other=0.0,
            ).to(tl.float32)
            y = base_blk + scale * tl.dot(h, tl.trans(ub))
            tl.store(
                out_row + n[None, :],
                y.to(tl.bfloat16),
                mask=row_mask[:, None] & n_mask[None, :],
            )


def _pad_rank(rank: int) -> int:
    """Smallest power-of-two ``>= max(16, rank)`` (tl.dot / tl.arange need it)."""
    r = max(16, int(rank))
    return 1 << (r - 1).bit_length()


def _run_forward(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: torch.Tensor | None,
    down: torch.Tensor,
    up: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Launch the fused forward kernel; returns ``y`` with ``x``'s batch dims."""
    *batch, in_features = x.shape
    n_rows = 1
    for d in batch:
        n_rows *= d
    out_features = base_weight.shape[0]
    rank = down.shape[0]

    x_flat = x.reshape(n_rows, in_features).contiguous()
    base = torch.nn.functional.linear(x_flat, base_weight, base_bias)
    out = torch.empty(n_rows, out_features, dtype=torch.bfloat16, device=x.device)

    grid = lambda meta: (min(triton.cdiv(n_rows, meta["BLOCK_M"]), MAX_GRID_BLOCKS),)
    _fwd_kernel[grid](
        x_flat, base, down, up, out,
        float(scale),
        n_rows, in_features, out_features, rank,
        x_flat.stride(0), base.stride(0), out.stride(0), down.stride(0), up.stride(0),
        R_PAD=_pad_rank(rank),
    )
    return out.reshape(*batch, out_features)


# ---------------------------------------------------------------------------
# Autograd wrapper: Triton forward + PyTorch backward
# ---------------------------------------------------------------------------

class _FusedLinearLoRA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, base_weight, base_bias, down, up, scale, fp32_backward):
        down = down.contiguous()
        up = up.contiguous()
        ctx.save_for_backward(x, base_weight, down, up)
        ctx.scale = float(scale)
        ctx.fp32_backward = bool(fp32_backward)
        return _run_forward(x, base_weight, base_bias, down, up, scale)

    @staticmethod
    def backward(ctx, grad_y):
        x, base_weight, down, up = ctx.saved_tensors
        scale = ctx.scale
        in_features = x.shape[-1]

        gy = grad_y.reshape(-1, grad_y.shape[-1])
        xf = x.reshape(-1, in_features)

        if ctx.fp32_backward:
            # High-accuracy path: accumulate the LoRA grads in fp32. Slower
            # (fp32 GEMMs + extra HBM traffic) but a touch more precise than the
            # eager bf16 backward. Opt-in via the ``fp32_backward`` flag.
            gyf = gy.float()
            xff = xf.float()
            downf = down.float()
            upf = up.float()
            h = xff @ downf.t()              # [rows, rank]
            d_h = scale * (gyf @ upf)        # [rows, rank]
            grad_down = grad_up = grad_x = None
            if ctx.needs_input_grad[3]:
                grad_down = (d_h.t() @ xff).to(down.dtype)
            if ctx.needs_input_grad[4]:
                grad_up = (scale * (gyf.t() @ h)).to(up.dtype)
            if ctx.needs_input_grad[0]:
                grad_x_base = (gy.to(base_weight.dtype) @ base_weight).float()
                grad_x = (grad_x_base + d_h @ downf).reshape_as(x).to(x.dtype)
            return grad_x, None, None, grad_down, grad_up, None, None

        # Default path: bf16 GEMMs (fp32 tensor-core accumulation, bf16 output),
        # numerically identical to the eager LoRA backward but launched from the
        # fused graph. This is what makes fused training competitive end-to-end.
        h = xf @ down.t()                    # [rows, rank]
        d_h = (gy @ up) * scale              # [rows, rank]
        grad_down = grad_up = grad_x = None
        if ctx.needs_input_grad[3]:
            grad_down = d_h.t() @ xf
        if ctx.needs_input_grad[4]:
            grad_up = (gy.t() @ h) * scale
        if ctx.needs_input_grad[0]:
            grad_x = (gy @ base_weight + d_h @ down).reshape_as(x)

        # grads align with forward args: x, base_weight, base_bias, down, up, scale, fp32_backward
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
    """Autograd-aware fused base-Linear + LoRA.

    Args:
        x:           ``[*, in_features]`` bf16 activations.
        base_weight: ``[out_features, in_features]`` bf16 frozen weight.
        base_bias:   ``[out_features]`` bf16 frozen bias, or ``None``.
        down:        ``[rank, in_features]`` bf16 trainable down-projection.
        up:          ``[out_features, rank]`` bf16 trainable up-projection.
        scale:       LoRA scaling (``alpha/rank`` or rs-LoRA variant).
        fp32_backward: accumulate the LoRA gradients in fp32 (slightly more
            accurate, slower). Default ``False`` matches the eager bf16 backward.

    Returns:
        ``[*, out_features]`` bf16 tensor.
    """
    if torch.is_grad_enabled() and (
        x.requires_grad or down.requires_grad or up.requires_grad
    ):
        return _FusedLinearLoRA.apply(x, base_weight, base_bias, down, up, scale, fp32_backward)
    return _run_forward(x, base_weight, base_bias, down.contiguous(), up.contiguous(), scale)
