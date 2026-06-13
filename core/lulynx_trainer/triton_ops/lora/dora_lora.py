"""Fused DoRA (weight-decomposed LoRA) forward (bf16), training-correct.

DoRA reparametrises a frozen Linear as a direction + magnitude:

    weight_eff = W0 + scale * (up @ down)         # [out, in]
    norm[o]    = || weight_eff[o, :] ||_2          # per-output-row L2
    row_scale  = magnitude / (norm + eps)          # [out]
    y          = (row_scale[:, None] * weight_eff) @ x^T + bias

(see ``lulynx.dora_layer.DoRALinear._compute_dora_weight``). The per-row
``norm`` is genuinely an ``O(out*in)`` reduction over the full effective weight
— it can't be made low-rank — so we deliberately keep ``row_scale`` (and thus
``norm``) in PyTorch, where autograd carries its gradient back to ``magnitude``,
``down`` and ``up`` through the norm. What the Triton kernel fuses is the *rest*:

    y = row_scale ⊙ (x @ W0^T + scale * up(down(x))) + bias

i.e. the base GEMM, the low-rank LoRA, the per-output-channel ``row_scale``
multiply and the bias add collapse into one launch on top of the cuDNN base
GEMM, with no materialised ``[out, in]`` scaled weight and no extra ``[N, out]``
rescale pass. The win is bandwidth-only (the norm work is unavoidable), so the
honest expectation is a modest speedup at best — but the path is clean-room,
parity-checked and opt-in, which is the bar.

The autograd wrapper returns a gradient for ``row_scale``; because ``row_scale``
is computed in the torch graph from ``magnitude``/``down``/``up``, PyTorch then
folds its norm-path gradient into the LoRA-path gradients the kernel produces
for the same ``down``/``up`` tensors. bf16 backward by default, opt-in fp32.

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
    magnitude: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pure-PyTorch DoRA, mirroring ``DoRALinear._compute_dora_weight`` exactly."""
    weight_eff = torch.addmm(base_weight, up, down, beta=1.0, alpha=float(scale))
    norm = torch.linalg.vector_norm(weight_eff, dim=1, keepdim=True)
    row_scale = magnitude.unsqueeze(1) / (norm + eps)
    return torch.nn.functional.linear(x, weight_eff * row_scale, base_bias)


# ---------------------------------------------------------------------------
# Triton forward kernel
# ---------------------------------------------------------------------------

_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 256, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_CONFIGS, key=["n_rows", "in_features", "out_features"])
@triton.jit
def _fwd_kernel(
    x_ptr, base_ptr, down_ptr, up_ptr, rs_ptr, bias_ptr, out_ptr,
    scale,
    n_rows, in_features, out_features, rank,
    s_x, s_base, s_out, s_down, s_up,
    HAS_BIAS: tl.constexpr, R_PAD: tl.constexpr,
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
            xb = tl.load(
                x_row + k[None, :],
                mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            db = tl.load(
                down_ptr + r[:, None] * s_down + k[None, :],
                mask=r_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            h += tl.dot(xb, tl.trans(db))

        # Phase B: out = row_scale * (base + scale * (h @ up^T)) + bias.
        # base is x @ W0^T WITHOUT bias — row_scale must not scale the bias.
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
            rs = tl.load(rs_ptr + n, mask=n_mask, other=0.0).to(tl.float32)
            y = (base_blk + scale * tl.dot(h, tl.trans(ub))) * rs[None, :]
            if HAS_BIAS:
                b = tl.load(bias_ptr + n, mask=n_mask, other=0.0).to(tl.float32)
                y += b[None, :]
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
    row_scale: torch.Tensor,
) -> torch.Tensor:
    """Launch the fused DoRA forward kernel; returns ``y`` with ``x``'s batch dims."""
    *batch, in_features = x.shape
    n_rows = 1
    for d in batch:
        n_rows *= d
    out_features = base_weight.shape[0]
    rank = down.shape[0]

    x_flat = x.reshape(n_rows, in_features).contiguous()
    base = torch.nn.functional.linear(x_flat, base_weight, None)  # no bias here
    rs = row_scale.contiguous()
    out = torch.empty(n_rows, out_features, dtype=torch.bfloat16, device=x.device)
    has_bias = base_bias is not None
    bias_ptr = base_bias if has_bias else rs  # placeholder when HAS_BIAS is False

    grid = lambda meta: (min(triton.cdiv(n_rows, meta["BLOCK_M"]), MAX_GRID_BLOCKS),)
    _fwd_kernel[grid](
        x_flat, base, down, up, rs, bias_ptr, out,
        float(scale),
        n_rows, in_features, out_features, rank,
        x_flat.stride(0), base.stride(0), out.stride(0), down.stride(0), up.stride(0),
        HAS_BIAS=has_bias, R_PAD=_pad_rank(rank),
    )
    return out.reshape(*batch, out_features)


# ---------------------------------------------------------------------------
# Autograd wrapper: Triton forward + PyTorch backward
# ---------------------------------------------------------------------------

class _FusedDoRALinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, base_weight, base_bias, down, up, scale, row_scale, fp32_backward):
        down = down.contiguous()
        up = up.contiguous()
        row_scale = row_scale.contiguous()
        ctx.save_for_backward(x, base_weight, down, up, row_scale)
        ctx.scale = float(scale)
        ctx.fp32_backward = bool(fp32_backward)
        return _run_forward(x, base_weight, base_bias, down, up, scale, row_scale)

    @staticmethod
    def backward(ctx, grad_y):
        x, base_weight, down, up, row_scale = ctx.saved_tensors
        scale = ctx.scale
        in_features = x.shape[-1]

        gy = grad_y.reshape(-1, grad_y.shape[-1])     # [N, out]
        xf = x.reshape(-1, in_features)               # [N, in]

        if ctx.fp32_backward:
            gyf, xff, downf, upf = gy.float(), xf.float(), down.float(), up.float()
            rsf = row_scale.float()
            h = xff @ downf.t()                       # [N, rank]
            g = (xff @ base_weight.float().t()) + scale * (h @ upf.t())   # [N, out]
            grad_g = gyf * rsf                        # [N, out]
            d_h = scale * (grad_g @ upf)              # [N, rank]
            grad_down = grad_up = grad_x = grad_rs = None
            if ctx.needs_input_grad[3]:
                grad_down = (d_h.t() @ xff).to(down.dtype)
            if ctx.needs_input_grad[4]:
                grad_up = (scale * (grad_g.t() @ h)).to(up.dtype)
            if ctx.needs_input_grad[0]:
                grad_x = (grad_g @ base_weight.float() + d_h @ downf).reshape_as(x).to(x.dtype)
            if ctx.needs_input_grad[6]:
                grad_rs = (gyf * g).sum(dim=0).to(row_scale.dtype)
            return grad_x, None, None, grad_down, grad_up, None, grad_rs, None

        # Default bf16 path (fp32 tensor-core accumulation, bf16 output).
        h = xf @ down.t()                             # [N, rank]
        g = (xf @ base_weight.t()) + scale * (h @ up.t())   # [N, out], un-rescaled
        grad_g = gy * row_scale                       # [N, out]
        d_h = (grad_g @ up) * scale                   # [N, rank]
        grad_down = grad_up = grad_x = grad_rs = None
        if ctx.needs_input_grad[3]:
            grad_down = d_h.t() @ xf
        if ctx.needs_input_grad[4]:
            grad_up = (grad_g.t() @ h) * scale
        if ctx.needs_input_grad[0]:
            grad_x = (grad_g @ base_weight + d_h @ down).reshape_as(x)
        if ctx.needs_input_grad[6]:
            grad_rs = (gy * g).sum(dim=0)

        # align: x, base_weight, base_bias, down, up, scale, row_scale, fp32_backward
        # base_bias is always frozen in DoRA (requires_grad=False) -> None.
        return grad_x, None, None, grad_down, grad_up, None, grad_rs, None


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
    magnitude: torch.Tensor,
    *,
    eps: float = 1e-6,
    fp32_backward: bool = False,
) -> torch.Tensor:
    """Autograd-aware fused DoRA forward.

    Args:
        x:           ``[*, in_features]`` bf16 activations.
        base_weight: ``[out_features, in_features]`` bf16 frozen weight ``W0``.
        base_bias:   ``[out_features]`` bf16 frozen bias, or ``None``.
        down:        ``[rank, in_features]`` bf16 trainable LoRA ``A``.
        up:          ``[out_features, rank]`` bf16 trainable LoRA ``B``.
        scale:       DoRA scaling (``alpha/rank`` or rs-LoRA variant).
        magnitude:   ``[out_features]`` trainable DoRA magnitude ``m``.
        eps:         norm epsilon (DoRALinear uses ``1e-6``).
        fp32_backward: accumulate gradients in fp32 (slower, slightly more accurate).

    Returns:
        ``[*, out_features]`` bf16 tensor.
    """
    # row_scale (and the O(out*in) norm it needs) stays in the torch graph so
    # autograd routes its gradient back to magnitude/down/up through the norm.
    weight_eff = torch.addmm(base_weight, up, down, beta=1.0, alpha=float(scale))
    norm = torch.linalg.vector_norm(weight_eff, dim=1)
    row_scale = magnitude / (norm + eps)
    if torch.is_grad_enabled() and (
        x.requires_grad or down.requires_grad or up.requires_grad or magnitude.requires_grad
    ):
        return _FusedDoRALinear.apply(
            x, base_weight, base_bias, down, up, scale, row_scale, fp32_backward
        )
    return _run_forward(x, base_weight, base_bias, down.contiguous(), up.contiguous(), scale, row_scale)
