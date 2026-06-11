"""Anima train-norm compatibility helpers."""

from __future__ import annotations

from typing import Dict, Optional

import torch


def _normalized_export_base(base_key: str) -> str:
    if base_key.startswith("unet_"):
        return f"lora_{base_key}"
    return base_key


def _tensor_or_default(value: Optional[torch.Tensor], default: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().float()
    return default.detach().float()


def export_anima_train_norm_state_dict(
    state_dict: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, str]],
    *,
    injector,
) -> tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]:
    exported = dict(state_dict)
    metadata_out = {} if metadata is None else dict(metadata)
    exported_count = 0

    for full_name, layer in getattr(injector, "injected_layers", {}).items():
        if type(layer).__name__ != "_NormAdapter":
            continue

        base_name = full_name.replace(".", "_")
        scale_key = f"{base_name}.norm_scale"
        bias_key = f"{base_name}.norm_bias"
        if scale_key not in exported and bias_key not in exported:
            continue

        export_base = _normalized_export_base(base_name)
        alpha = float(getattr(layer, "alpha", 1.0) or 1.0)
        scale = exported.get(scale_key, getattr(layer, "scale", None))
        bias = exported.get(bias_key, getattr(layer, "bias", None))

        base_weight = _tensor_or_default(
            getattr(layer, "base_weight", None),
            torch.ones_like(scale if isinstance(scale, torch.Tensor) else getattr(layer, "scale")),
        )
        scale_tensor = _tensor_or_default(scale, getattr(layer, "scale"))
        diff = base_weight * scale_tensor * alpha
        exported[f"{export_base}.diff"] = diff.to(dtype=diff.dtype).contiguous()
        exported_count += 1

        base_bias = getattr(layer, "base_bias", None)
        bias_tensor = _tensor_or_default(bias, getattr(layer, "bias"))
        bias_default = torch.zeros_like(bias_tensor)
        merged_bias_diff = bias_tensor * alpha
        if isinstance(base_bias, torch.Tensor):
            merged_bias_diff = (_tensor_or_default(base_bias, bias_default) + bias_tensor * alpha) - _tensor_or_default(base_bias, bias_default)
        exported[f"{export_base}.diff_b"] = merged_bias_diff.to(dtype=merged_bias_diff.dtype).contiguous()
        exported_count += 1

        exported.pop(scale_key, None)
        exported.pop(bias_key, None)

    if exported_count:
        metadata_out["ss_train_norm_export_format"] = "comfyui_diff"
        metadata_out["ss_train_norm_exported_count"] = str(exported_count)

    return exported, metadata_out


def resolve_anima_train_norm_state_for_layer(
    state_dict: Dict[str, torch.Tensor],
    *,
    layer: torch.nn.Module,
    layer_base_name: str,
) -> Optional[Dict[str, torch.Tensor]]:
    scale_key = f"{layer_base_name}.norm_scale"
    bias_key = f"{layer_base_name}.norm_bias"
    assignments: Dict[str, torch.Tensor] = {}

    direct_scale = state_dict.get(scale_key)
    direct_bias = state_dict.get(bias_key)
    if isinstance(direct_scale, torch.Tensor):
        assignments["norm_scale"] = direct_scale.detach()
    if isinstance(direct_bias, torch.Tensor):
        assignments["norm_bias"] = direct_bias.detach()
    if assignments:
        return assignments

    export_base = _normalized_export_base(layer_base_name)
    diff_key_candidates = (
        f"{export_base}.diff",
        f"{layer_base_name}.diff",
    )
    diff_bias_candidates = (
        f"{export_base}.diff_b",
        f"{layer_base_name}.diff_b",
    )
    weight_key_candidates = (
        f"{export_base}.weight",
        f"{layer_base_name}.weight",
    )
    bias_key_candidates = (
        f"{export_base}.bias",
        f"{layer_base_name}.bias",
    )

    alpha = float(getattr(layer, "alpha", 1.0) or 1.0)
    if abs(alpha) < 1e-12:
        alpha = 1.0

    base_weight = _tensor_or_default(getattr(layer, "base_weight", None), getattr(layer, "scale").new_ones(getattr(layer, "scale").shape))
    base_bias = _tensor_or_default(getattr(layer, "base_bias", None), getattr(layer, "bias").new_zeros(getattr(layer, "bias").shape))

    def _first_tensor(candidates: tuple[str, ...]) -> Optional[torch.Tensor]:
        for key in candidates:
            value = state_dict.get(key)
            if isinstance(value, torch.Tensor):
                return value.detach().float()
        return None

    diff = _first_tensor(diff_key_candidates)
    diff_b = _first_tensor(diff_bias_candidates)
    if diff is not None or diff_b is not None:
        if diff is not None:
            denom = base_weight * alpha
            safe = torch.where(denom.abs() > 1e-12, diff / denom, torch.zeros_like(diff))
            assignments["norm_scale"] = safe
        if diff_b is not None:
            assignments["norm_bias"] = diff_b / alpha
        return assignments or None

    weight_abs = _first_tensor(weight_key_candidates)
    bias_abs = _first_tensor(bias_key_candidates)
    if weight_abs is not None or bias_abs is not None:
        if weight_abs is not None:
            denom = base_weight * alpha
            safe = torch.where(denom.abs() > 1e-12, (weight_abs - base_weight) / denom, torch.zeros_like(weight_abs))
            assignments["norm_scale"] = safe
        if bias_abs is not None:
            assignments["norm_bias"] = (bias_abs - base_bias) / alpha
        return assignments or None

    return None
