"""Warehouse LoKr export rules."""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

import torch

from .lokr_load_resolver import materialize_lokr_matrices
from .lokr_weight_format import collect_lokr_weight_layouts


class LoKrExportMode(str, Enum):
    NATIVE = "native"
    LORA_COMPATIBLE = "lora_compatible"


def _scalar_tensor(value: float) -> torch.Tensor:
    return torch.tensor(float(value), dtype=torch.float32)


def _normalized_export_base(base_key: str) -> str:
    if base_key.startswith("unet_"):
        return f"lora_{base_key}"
    return base_key


def _scrub_lokr_only_metadata(metadata: Dict[str, str]) -> None:
    for key in (
        "lycoris_lokr_factor",
        "lycoris_factor",
        "lokr_rank_dropout",
        "lokr_module_dropout",
        "lokr_full_matrix",
        "lokr_decompose_both",
        "lokr_unbalanced_factorization",
        "lokr_export_mode",
    ):
        metadata.pop(key, None)


def export_lokr_state_dict(
    state_dict: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, str]],
    *,
    export_mode: str = "native",
) -> tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]:
    mode = LoKrExportMode(str(export_mode or "native").strip().lower())
    layouts = collect_lokr_weight_layouts(state_dict)
    if not layouts:
        return state_dict, metadata

    exported: Dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    for base_key, layout in layouts.items():
        w1, w2 = materialize_lokr_matrices(state_dict, layout)
        rank = max(1, int(layout.rank or 1))
        alpha = float(layout.alpha if layout.alpha is not None else rank)
        scale = alpha / float(rank)
        export_base = _normalized_export_base(base_key)

        if mode is LoKrExportMode.NATIVE:
            exported[f"{export_base}.lokr_w1"] = w1.to(dtype=w1.dtype).contiguous()
            exported[f"{export_base}.lokr_w2"] = (w2 * scale).to(dtype=w2.dtype).contiguous()
            exported[f"{export_base}.alpha"] = _scalar_tensor(rank)
        else:
            in_features = int(w1.shape[1] * w2.shape[1])
            delta = torch.kron(w1, w2).reshape(int(w1.shape[0] * w2.shape[0]), in_features) * scale
            eye = torch.eye(in_features, dtype=delta.dtype, device=delta.device)
            exported[f"{export_base}.lora_down.weight"] = eye.contiguous()
            exported[f"{export_base}.lora_up.weight"] = delta.contiguous()
            exported[f"{export_base}.alpha"] = _scalar_tensor(in_features)

        consumed.update(layout.tensor_keys.values())

    for key, value in state_dict.items():
        if key in consumed:
            continue
        base, sep, suffix = key.rpartition(".")
        if sep and base in layouts and (suffix.startswith("lokr_") or suffix == "alpha"):
            continue
        exported[key] = value

    metadata_out = {} if metadata is None else dict(metadata)
    _scrub_lokr_only_metadata(metadata_out)
    metadata_out["ss_lokr_export_mode"] = mode.value
    metadata_out["ss_lokr_rank_exported"] = "false"
    if mode is LoKrExportMode.NATIVE:
        metadata_out["ss_network_module"] = "networks.lora_anima"
        metadata_out["ss_anima_adapter_type"] = "lokr"
        metadata_out["ss_adapter_variant"] = "lokr"
        metadata_out["ss_lokr_native_export"] = "true"
        metadata_out.pop("ss_lokr_compatible_export", None)
        metadata_out["ss_lokr_scale_export_format"] = "comfyui_baked_single_scale"
    else:
        metadata_out["ss_network_module"] = "networks.lora"
        metadata_out["ss_anima_adapter_type"] = "lora"
        metadata_out["ss_adapter_variant"] = "lora"
        metadata_out["ss_lokr_native_export"] = "false"
        metadata_out["ss_lokr_compatible_export"] = "true"
        metadata_out["ss_lokr_scale_export_format"] = "lora_identity_down"

    return exported, metadata_out

