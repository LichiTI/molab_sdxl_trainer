# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""GLoRA export rules.

Two export modes mirror the LoKr setup:

- ``native``: keep the raw ``a1/a2/b1/b2`` tensors plus alpha.  Downstream
  tooling that understands the GLoRA contract (ΔW = W·A + B) can re-apply the
  delta against the same base model.
- ``lora_compatible``: bake the full delta against the *base* state-dict
  (the caller passes ``base_weights`` for any layer key to materialize),
  emitting a standard ``lora_down.weight = I`` + ``lora_up.weight = ΔW`` pair
  so the export loads as a plain LoRA in inference pipelines that have no
  GLoRA support.  Layers whose base weight is unavailable fall back to the
  native tensors so we never silently drop them.

The exporter is intentionally narrow: it does not touch other adapter
families' keys and leaves the metadata alone for them.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Mapping, Optional, Tuple

import torch


class GLoRAExportMode(str, Enum):
    NATIVE = "native"
    LORA_COMPATIBLE = "lora_compatible"


_GLORA_SUFFIXES: Tuple[str, ...] = ("a1.weight", "a2.weight", "b1.weight", "b2.weight")


def _scalar(value: float) -> torch.Tensor:
    return torch.tensor(float(value), dtype=torch.float32)


def _normalized_export_base(base_key: str) -> str:
    return f"lora_{base_key}" if base_key.startswith("unet_") else base_key


def collect_glora_layer_bases(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Group the GLoRA tensors per layer base name."""
    grouped: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        for suffix in _GLORA_SUFFIXES:
            if key.endswith("." + suffix):
                base = key[: -(len(suffix) + 1)]
                slot = grouped.setdefault(base, {})
                slot[suffix] = value
                break
        else:
            if key.endswith(".alpha"):
                base = key[: -len(".alpha")]
                slot = grouped.setdefault(base, {})
                slot.setdefault("alpha", value)
    return {base: slot for base, slot in grouped.items() if any(s in slot for s in _GLORA_SUFFIXES)}


def _layer_alpha_rank(slot: Dict[str, torch.Tensor]) -> Tuple[float, int]:
    a1 = slot.get("a1.weight")
    if a1 is None:
        rank = 1
    else:
        rank = int(a1.shape[1])
    alpha_tensor = slot.get("alpha")
    if alpha_tensor is None:
        return float(rank), max(1, rank)
    return float(alpha_tensor.detach().float().item()), max(1, rank)


def export_glora_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    metadata: Optional[Dict[str, str]],
    *,
    export_mode: str = "native",
    base_weights: Optional[Mapping[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]:
    """Rewrite a GLoRA state-dict for the requested export mode.

    ``base_weights`` maps the GLoRA layer base name (e.g.
    ``unet_net_blocks_0_self_attn_q_proj``) to the corresponding frozen base
    weight, used only for the ``lora_compatible`` mode.
    """
    mode = GLoRAExportMode(str(export_mode or "native").strip().lower())
    layers = collect_glora_layer_bases(state_dict)

    if not layers:
        return dict(state_dict), metadata

    exported: Dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    base_weights = base_weights or {}

    for base, slot in layers.items():
        alpha_value, rank = _layer_alpha_rank(slot)
        scaling = alpha_value / float(rank) if rank > 0 else 1.0
        export_base = _normalized_export_base(base)

        if mode is GLoRAExportMode.NATIVE:
            for suffix in _GLORA_SUFFIXES:
                tensor = slot.get(suffix)
                if tensor is None:
                    continue
                exported[f"{export_base}.{suffix}"] = tensor.contiguous()
            exported[f"{export_base}.alpha"] = _scalar(alpha_value)
        else:
            base_weight = base_weights.get(base)
            a1, a2 = slot.get("a1.weight"), slot.get("a2.weight")
            b1, b2 = slot.get("b1.weight"), slot.get("b2.weight")
            if base_weight is None or a1 is None or a2 is None or b1 is None or b2 is None:
                # Fall back to native for any layer we can't fully materialize.
                for suffix in _GLORA_SUFFIXES:
                    tensor = slot.get(suffix)
                    if tensor is None:
                        continue
                    exported[f"{export_base}.{suffix}"] = tensor.contiguous()
                exported[f"{export_base}.alpha"] = _scalar(alpha_value)
            else:
                w = base_weight.detach().to(dtype=a1.dtype).reshape(b1.shape[0], -1)
                delta = (w @ (a1 @ a2) + (b1 @ b2)) * scaling
                eye = torch.eye(delta.shape[1], dtype=delta.dtype, device=delta.device)
                exported[f"{export_base}.lora_down.weight"] = eye.contiguous()
                exported[f"{export_base}.lora_up.weight"] = delta.contiguous()
                exported[f"{export_base}.alpha"] = _scalar(float(delta.shape[1]))

        for suffix in _GLORA_SUFFIXES + ("alpha",):
            consumed.add(f"{base}.{suffix}")

    for key, value in state_dict.items():
        if key in consumed:
            continue
        exported[key] = value

    metadata_out = {} if metadata is None else dict(metadata)
    metadata_out["ss_glora_export_mode"] = mode.value
    if mode is GLoRAExportMode.NATIVE:
        metadata_out["ss_glora_native_export"] = "true"
        metadata_out.pop("ss_glora_compatible_export", None)
        metadata_out.setdefault("ss_network_module", "networks.lora_anima")
        metadata_out.setdefault("ss_anima_adapter_type", "glora")
        metadata_out.setdefault("ss_adapter_variant", "glora")
    else:
        metadata_out["ss_glora_native_export"] = "false"
        metadata_out["ss_glora_compatible_export"] = "true"
        metadata_out["ss_network_module"] = "networks.lora"
        metadata_out["ss_anima_adapter_type"] = "lora"
        metadata_out["ss_adapter_variant"] = "lora"

    return exported, metadata_out


__all__ = [
    "GLoRAExportMode",
    "export_glora_state_dict",
    "collect_glora_layer_bases",
]
