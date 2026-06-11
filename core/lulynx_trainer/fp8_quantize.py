"""FP8 base-model quantization helper.

When ``fp8_base`` is enabled, the UNet/transformer's frozen base weights are
cast to ``torch.float8_e4m3fn`` to slash VRAM.  Adapter (LoRA/LyCORIS) weights
are left in their original dtype so gradient computation stays numerically sound.

The public entry-point is :func:`quantize_base_weights_fp8`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_FP8_DTYPE = torch.float8_e4m3fn


def _collect_lora_param_ids(lora_injector: Any) -> Set[int]:
    """Return the ``id()`` set of all parameters managed by *lora_injector*.

    This lets the quantizer skip adapter weights without depending on the
    ``_lora_leaf`` attribute convention.
    """
    ids: Set[int] = set()
    if lora_injector is None:
        return ids

    # Standard LoRAInjector exposes get_trainable_params().
    if hasattr(lora_injector, "get_trainable_params"):
        for p in lora_injector.get_trainable_params():
            ids.add(id(p))

    # Also collect all parameters from injected_layers (covers non-trainable
    # adapter state like frozen VeRA shared buffers).
    injected = getattr(lora_injector, "injected_layers", None)
    if isinstance(injected, dict):
        for layer in injected.values():
            for p in layer.parameters():
                ids.add(id(p))

    return ids


def quantize_base_weights_fp8(
    model: nn.Module,
    lora_injector: Any = None,
) -> float:
    """Quantize frozen base-model weights to ``torch.float8_e4m3fn``.

    Walks ``model.parameters()`` and casts every parameter that:

    1. does **not** require gradients (frozen base weight), **and**
    2. is **not** managed by *lora_injector* (adapter weights),

    to ``float8_e4m3fn``.  This roughly halves the VRAM footprint of the
    base model while keeping adapter weights in bf16/fp16 for accurate
    gradient computation.

    Args:
        model: The UNet / DiT model whose base weights to quantize.
        lora_injector: Optional :class:`LoRAInjector` (or compatible object)
            whose parameters must be skipped.

    Returns:
        Estimated VRAM savings in megabytes (based on element count *
        bytes-saved-per-element).
    """
    lora_ids = _collect_lora_param_ids(lora_injector)

    _scaled_mm = getattr(torch._C, "_scaled_mm", None)

    linear_count = 0
    other_count = 0
    bytes_saved = 0

    for name, param in model.named_parameters():
        # Skip adapter-managed parameters
        if id(param) in lora_ids:
            continue
        # Skip _lora_leaf-marked parameters (redundant safety net)
        if getattr(param, "_lora_leaf", False):
            continue
        # Only quantize frozen (non-trainable) weights
        if param.requires_grad:
            continue

        orig_dtype = param.dtype
        orig_bytes_per_elem = orig_dtype.itemsize if hasattr(orig_dtype, "itemsize") else 2

        try:
            param.data = param.data.to(_FP8_DTYPE)
            new_bytes_per_elem = _FP8_DTYPE.itemsize if hasattr(_FP8_DTYPE, "itemsize") else 1
            bytes_saved += param.numel() * max(orig_bytes_per_elem - new_bytes_per_elem, 0)

            # If this param is the weight of a Linear module, try _scaled_mm
            # and store a per-tensor scale for hardware-accelerated matmul.
            if _scaled_mm is not None and param.dim() == 2 and param.is_cuda:
                # Walk modules to find the Linear that owns this parameter
                for mod in model.modules():
                    if isinstance(mod, nn.Linear) and param is getattr(mod, "weight", None):
                        amax = param.data.abs().amax().clamp(min=1e-12)
                        scale = amax.to(torch.float32) / torch.finfo(_FP8_DTYPE).max
                        mod._fp8_scale = scale.item()
                        linear_count += 1
                        break
                else:
                    other_count += 1
            else:
                # Check if it's a linear weight (for counting)
                for mod in model.modules():
                    if isinstance(mod, nn.Linear) and param is getattr(mod, "weight", None):
                        mod._fp8_scale = None
                        linear_count += 1
                        break
                else:
                    other_count += 1
        except RuntimeError:
            # Some shapes / dtypes may not support FP8 — skip silently.
            pass

    vram_saved_mb = bytes_saved / (1024 * 1024)

    logger.info(
        "FP8 base quantization: %d linear weights, %d other params quantized "
        "(dtype=%s, estimated VRAM savings=%.1f MB)",
        linear_count, other_count, _FP8_DTYPE, vram_saved_mb,
    )

    return vram_saved_mb


def _resolve_scaled_mm():
    """Return ``torch._scaled_mm`` (public or private), or None if unavailable."""
    fn = getattr(torch, "_scaled_mm", None)
    if fn is None:
        fn = getattr(torch._C, "_scaled_mm", None)
    return fn


@torch.no_grad()
def _fp8_compute_supported(linear: nn.Module, x: torch.Tensor) -> bool:
    """Whether the FP8 tensor-core base GEMM can run for this layer/input.

    Requires CUDA, an fp8 weight, ``torch._scaled_mm``, and 16-aligned GEMM
    dims (the tensor-core contract).  Anything else falls back to bf16.
    """
    weight = getattr(linear, "weight", None)
    if weight is None or weight.dtype != _FP8_DTYPE or not x.is_cuda:
        return False
    if weight.dim() != 2:
        return False
    out_features, in_features = weight.shape
    if x.shape[-1] != in_features:
        return False
    if (in_features % 16) or (out_features % 16):
        return False
    return _resolve_scaled_mm() is not None


def fp8_base_linear_forward(linear: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a frozen base ``nn.Linear`` GEMM on Ada FP8 tensor cores.

    The base weight is already ``float8_e4m3fn`` (a *direct* cast done by
    :func:`quantize_base_weights_fp8`), so the fp8 values approximate the real
    weights and the weight dequant scale is ``1.0``.  We dynamically per-tensor
    quantize the activation to e4m3 and call ``torch._scaled_mm`` so the matmul
    runs on the FP8 tensor cores; the LoRA path is untouched and added by the
    caller.  Any unsupported shape / dtype / runtime error falls back to the
    bf16 dequantized ``F.linear`` so correctness is never at risk.
    """
    weight = linear.weight
    bias = getattr(linear, "bias", None)

    def _fallback() -> torch.Tensor:
        b = bias.to(x.dtype) if bias is not None else None
        return F.linear(x, weight.to(x.dtype), b)

    if not _fp8_compute_supported(linear, x):
        return _fallback()

    scaled_mm = _resolve_scaled_mm()
    out_features, in_features = weight.shape
    fp8_max = torch.finfo(_FP8_DTYPE).max
    try:
        x2d = x.reshape(-1, in_features)
        x_amax = x2d.abs().amax().clamp(min=1e-12).float()
        x_scale = (x_amax / fp8_max).reshape(())
        x_fp8 = (x2d.float() / x_scale).clamp(-fp8_max, fp8_max).to(_FP8_DTYPE)
        scale_a = x_scale.to(torch.float32)
        scale_b = torch.ones((), dtype=torch.float32, device=x.device)  # direct-cast weight → scale 1
        b = bias.to(x.dtype) if bias is not None else None
        out = scaled_mm(x_fp8, weight.t(), scale_a, scale_b, bias=b, out_dtype=x.dtype)
        return out.reshape(*x.shape[:-1], out_features)
    except (RuntimeError, TypeError):
        return _fallback()

