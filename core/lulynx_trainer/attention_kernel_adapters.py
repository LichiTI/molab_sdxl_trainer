"""Shared attention kernel adapters for native trainer runtimes.

All functions in this module use ``(batch, heads, sequence, head_dim)`` layout.
Diffusers U-Net processors and DiT attention patchers both adapt to this shape
before calling into the backend-specific kernels.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)

AttentionForward = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

_warned_once: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _warned_once:
        return
    _warned_once.add(key)
    logger.warning(message)


def _validate_bhnd(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 4:
        raise RuntimeError(f"{name} attention tensor must be BHND rank-4, got shape={tuple(tensor.shape)}")


def _scale_kwargs_or_scaled_query(query: torch.Tensor, scale: float | None) -> tuple[torch.Tensor, dict[str, float]]:
    if scale is None:
        return query, {}
    try:
        return query, {"scale": float(scale)}
    except (TypeError, ValueError):
        return query, {}


def sdpa_attention_bhnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """PyTorch SDPA in BHND layout."""

    _validate_bhnd("query", query)
    q_for_sdpa, scale_kwargs = _scale_kwargs_or_scaled_query(query, scale)
    try:
        return F.scaled_dot_product_attention(
            q_for_sdpa,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=float(dropout_p or 0.0),
            is_causal=bool(causal),
            **scale_kwargs,
        )
    except TypeError:
        if scale is None:
            raise
        default_scale = query.shape[-1] ** -0.5
        q_scaled = query * (float(scale) / float(default_scale))
        return F.scaled_dot_product_attention(
            q_scaled,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=float(dropout_p or 0.0),
            is_causal=bool(causal),
        )


def torch_attention_bhnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Manual attention fallback in BHND layout."""

    _validate_bhnd("query", query)
    scale_value = float(scale) if scale is not None else query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_value
    if attention_mask is not None:
        scores = scores + attention_mask
    if causal:
        q_len = query.shape[-2]
        k_len = key.shape[-2]
        causal_mask = torch.ones((q_len, k_len), dtype=torch.bool, device=query.device).tril()
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
    weights = scores.softmax(dim=-1)
    if dropout_p and dropout_p > 0.0:
        weights = F.dropout(weights, p=float(dropout_p))
    return torch.matmul(weights, value)


def ensure_flash2_available() -> Callable:
    if not torch.cuda.is_available():
        raise RuntimeError("flash2 attention requires CUDA")
    if bool(getattr(torch.version, "hip", None)):
        raise RuntimeError("flash2 attention is not available on ROCm/HIP")
    try:
        capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    except Exception:
        capability = None
    if capability is not None and capability < (8, 0):
        raise RuntimeError(f"flash2 attention requires SM80+, got capability={capability}")
    try:
        from flash_attn import flash_attn_func
    except Exception as exc:
        raise RuntimeError(f"flash2 requested but flash_attn is not importable: {exc}") from exc
    if not callable(flash_attn_func):
        raise RuntimeError("flash2 requested but flash_attn_func is not callable")
    return flash_attn_func


def _half_compute_dtype() -> torch.dtype:
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def flash2_attention_bhnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    scale: float | None = None,
    causal: bool = False,
    flash_attn_func: Callable | None = None,
) -> torch.Tensor:
    """FlashAttention-2 in BHND layout."""

    if attention_mask is not None:
        raise RuntimeError("flash2 attention does not support attention_mask in this adapter")
    _validate_bhnd("query", query)
    if query.device.type != "cuda":
        raise RuntimeError("flash2 attention requires CUDA tensors")

    flash_fn = flash_attn_func or ensure_flash2_available()
    original_dtype = query.dtype
    compute_dtype = original_dtype if original_dtype in (torch.float16, torch.bfloat16) else _half_compute_dtype()

    q = query.transpose(1, 2).contiguous().to(compute_dtype)
    k = key.transpose(1, 2).contiguous().to(compute_dtype)
    v = value.transpose(1, 2).contiguous().to(compute_dtype)
    out = flash_fn(
        q,
        k,
        v,
        dropout_p=float(dropout_p or 0.0),
        softmax_scale=scale,
        causal=bool(causal),
    )
    if isinstance(out, tuple):
        out = out[0]
    out = out.transpose(1, 2).contiguous()
    if out.dtype != original_dtype:
        out = out.to(original_dtype)
    return out


def xformers_attention_bhnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """xFormers memory efficient attention in BHND layout."""

    if causal:
        raise RuntimeError("xformers adapter does not support causal=True yet")
    if attention_mask is not None:
        raise RuntimeError("xformers adapter only supports attention_mask=None for training")
    _validate_bhnd("query", query)
    try:
        from xformers.ops import memory_efficient_attention
    except Exception as exc:
        raise RuntimeError(f"xformers requested but memory_efficient_attention is unavailable: {exc}") from exc
    if not callable(memory_efficient_attention):
        raise RuntimeError("xformers requested but memory_efficient_attention is not callable")

    original_dtype = query.dtype
    compute_dtype = original_dtype if original_dtype in (torch.float16, torch.bfloat16) else _half_compute_dtype()
    q = query.transpose(1, 2).contiguous().to(compute_dtype)
    k = key.transpose(1, 2).contiguous().to(compute_dtype)
    v = value.transpose(1, 2).contiguous().to(compute_dtype)
    out = memory_efficient_attention(
        q,
        k,
        v,
        attn_bias=None,
        p=float(dropout_p or 0.0),
        scale=scale,
    )
    if isinstance(out, tuple):
        out = out[0]
    out = out.transpose(1, 2).contiguous()
    if out.dtype != original_dtype:
        out = out.to(original_dtype)
    return out


def _resolve_shim_backward_backend() -> str:
    raw = (
        os.environ.get("LULYNX_ATTENTION_SHIM_BACKWARD")
        or os.environ.get("LULYNX_SAGE_SHIM_BACKWARD")
        or "sdpa"
    )
    normalized = str(raw or "").strip().lower()
    if normalized in {"flash", "flash2", "flashattn", "flashattention", "fa2"}:
        return "flash2"
    return "sdpa"


class _ForwardOnlyAttentionWithRecomputeBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        forward_fn: AttentionForward,
        scale: float | None,
        causal: bool,
        backward_backend: str,
    ) -> torch.Tensor:
        ctx.scale = scale
        ctx.causal = bool(causal)
        ctx.backward_backend = str(backward_backend or "sdpa").strip().lower()
        ctx.save_for_backward(query, key, value)
        return forward_fn(query, key, value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        query, key, value = ctx.saved_tensors
        q = query.detach().requires_grad_(True)
        k = key.detach().requires_grad_(True)
        v = value.detach().requires_grad_(True)
        with torch.enable_grad():
            if ctx.backward_backend == "flash2":
                out = flash2_attention_bhnd(
                    q,
                    k,
                    v,
                    attention_mask=None,
                    dropout_p=0.0,
                    scale=ctx.scale,
                    causal=ctx.causal,
                )
            else:
                out = sdpa_attention_bhnd(
                    q,
                    k,
                    v,
                    attention_mask=None,
                    dropout_p=0.0,
                    scale=ctx.scale,
                    causal=ctx.causal,
                )
            grads = torch.autograd.grad(
                out,
                (q, k, v),
                grad_output,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
        return grads[0], grads[1], grads[2], None, None, None, None


def forward_only_attention_bhnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    forward_fn: AttentionForward,
    scale: float | None = None,
    causal: bool = False,
    backward_backend: str | None = None,
) -> torch.Tensor:
    """Wrap a forward-only attention kernel with recompute-based gradients."""

    if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad):
        return _ForwardOnlyAttentionWithRecomputeBackward.apply(
            query,
            key,
            value,
            forward_fn,
            scale,
            bool(causal),
            backward_backend or _resolve_shim_backward_backend(),
        )
    return forward_fn(query, key, value)


def sage_attention_bhnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """SageAttention forward with recompute-based training gradients."""

    if attention_mask is not None:
        raise RuntimeError("sageattention adapter does not support attention_mask")
    if dropout_p and dropout_p > 0.0:
        raise RuntimeError("sageattention adapter does not support dropout_p > 0")
    _validate_bhnd("query", query)
    if query.device.type != "cuda":
        raise RuntimeError("sageattention requires CUDA tensors")
    if query.shape[-1] not in (64, 96, 128):
        raise RuntimeError(f"sageattention supports head_dim 64/96/128, got {query.shape[-1]}")
    try:
        from sageattention import sageattn
    except Exception as exc:
        raise RuntimeError(f"sageattention requested but package is unavailable: {exc}") from exc
    if not callable(sageattn):
        raise RuntimeError("sageattention requested but sageattn is not callable")

    def _forward(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
        original_dtype = q_in.dtype
        compute_dtype = original_dtype if original_dtype in (torch.float16, torch.bfloat16) else _half_compute_dtype()
        q_run = q_in.contiguous().to(compute_dtype)
        k_run = k_in.contiguous().to(compute_dtype)
        v_run = v_in.contiguous().to(compute_dtype)
        out = sageattn(q_run, k_run, v_run, tensor_layout="HND", is_causal=bool(causal), sm_scale=scale)
        if isinstance(out, tuple):
            out = out[0]
        if out.dtype != original_dtype:
            out = out.to(original_dtype)
        return out

    return forward_only_attention_bhnd(
        query,
        key,
        value,
        forward_fn=_forward,
        scale=scale,
        causal=causal,
    )


def _select_sparge2_forward() -> Callable:
    try:
        import spas_sage_attn
    except Exception as exc:
        raise RuntimeError(f"spargeattn2 requested but spas_sage_attn is unavailable: {exc}") from exc
    for name in (
        "spas_sage2_attn_meansim_cuda",
        "spas_sage_attn_meansim_cuda",
        "block_sparse_sage2_attn_cuda",
    ):
        fn = getattr(spas_sage_attn, name, None)
        if callable(fn):
            return fn
    raise RuntimeError("spargeattn2 requested but no callable Sparge/Sage2 attention kernel was found")


def sparge2_attention_bhnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """SpargeAttention2 forward with recompute-based training gradients."""

    if attention_mask is not None:
        raise RuntimeError("spargeattn2 adapter does not support attention_mask")
    if dropout_p and dropout_p > 0.0:
        raise RuntimeError("spargeattn2 adapter does not support dropout_p > 0")
    _validate_bhnd("query", query)
    if query.device.type != "cuda":
        raise RuntimeError("spargeattn2 requires CUDA tensors")
    if query.shape[-1] not in (64, 128):
        raise RuntimeError(f"spargeattn2 supports head_dim 64/128, got {query.shape[-1]}")
    if query.shape[-2] < 128 or key.shape[-2] < 128:
        raise RuntimeError(
            "spargeattn2 requires query/key sequence length >= 128 "
            f"(got q={query.shape[-2]}, k={key.shape[-2]})"
        )
    sparge_fn = _select_sparge2_forward()

    def _forward(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
        original_dtype = q_in.dtype
        compute_dtype = original_dtype if original_dtype in (torch.float16, torch.bfloat16) else _half_compute_dtype()
        output_dtype = compute_dtype
        q_run = q_in.contiguous().to(compute_dtype)
        k_run = k_in.contiguous().to(compute_dtype)
        v_run = v_in.contiguous().to(compute_dtype)
        out = sparge_fn(
            q_run,
            k_run,
            v_run,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=bool(causal),
            scale=scale,
            tensor_layout="HND",
            output_dtype=output_dtype,
        )
        if isinstance(out, tuple):
            out = out[0]
        if out.dtype != original_dtype:
            out = out.to(original_dtype)
        return out

    return forward_only_attention_bhnd(
        query,
        key,
        value,
        forward_fn=_forward,
        scale=scale,
        causal=causal,
    )
