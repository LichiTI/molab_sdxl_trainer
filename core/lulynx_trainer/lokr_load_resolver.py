"""Warehouse LoKr load resolution and reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .lokr_weight_format import (
    LoKrWeightLayout,
    collect_lokr_weight_layouts,
    scalar_from_tensor,
)


@dataclass
class LoKrResolvedWeights:
    direct_assignments: Dict[str, torch.Tensor]
    effective_rank: int
    effective_alpha: float
    export_scale_mode: str


def _state_tensor(
    state_dict: Dict[str, torch.Tensor],
    layout: LoKrWeightLayout,
    suffix: str,
) -> Optional[torch.Tensor]:
    key = layout.tensor_keys.get(suffix)
    value = state_dict.get(key) if key is not None else None
    return value if isinstance(value, torch.Tensor) else None


def materialize_lokr_matrices(
    state_dict: Dict[str, torch.Tensor],
    layout: LoKrWeightLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    w1 = _state_tensor(state_dict, layout, "lokr_w1")
    if w1 is None:
        w1_a = _state_tensor(state_dict, layout, "lokr_w1_a")
        w1_b = _state_tensor(state_dict, layout, "lokr_w1_b")
        if w1_a is None or w1_b is None:
            raise RuntimeError(f"Incomplete LoKr w1 branch for {layout.base_key}")
        w1 = w1_a.detach().float() @ w1_b.detach().float()
    else:
        w1 = w1.detach().float()

    w2 = _state_tensor(state_dict, layout, "lokr_w2")
    if w2 is None:
        w2_a = _state_tensor(state_dict, layout, "lokr_w2_a")
        w2_b = _state_tensor(state_dict, layout, "lokr_w2_b")
        if w2_a is None or w2_b is None:
            raise RuntimeError(f"Incomplete LoKr w2 branch for {layout.base_key}")
        w2 = w2_a.detach().float() @ w2_b.detach().float()
    else:
        w2 = w2.detach().float()

    return w1, w2


def _metadata_flag(metadata: Optional[Dict[str, str]], key: str) -> bool:
    if not metadata:
        return False
    return str(metadata.get(key, "")).strip().lower() == "true"


def _resolve_source_scale(
    layout: LoKrWeightLayout,
    *,
    metadata: Optional[Dict[str, str]],
) -> tuple[float, int, float, str]:
    rank = max(1, int(layout.rank or 1))
    alpha = float(layout.alpha if layout.alpha is not None else rank)
    if _metadata_flag(metadata, "ss_lokr_native_export") or str((metadata or {}).get("ss_lokr_export_mode", "")).strip().lower() == "native":
        return 1.0, rank, float(rank), "comfyui_baked_single_scale"
    return alpha / float(rank), rank, alpha, "alpha_div_rank"


def _copy_if_exact_match(
    layer: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    layout: LoKrWeightLayout,
) -> Dict[str, torch.Tensor]:
    assignments: Dict[str, torch.Tensor] = {}
    for attr in ("lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b"):
        if not hasattr(layer, attr):
            continue
        key = layout.tensor_keys.get(attr)
        source = state_dict.get(key) if key is not None else None
        target = getattr(layer, attr)
        if not isinstance(source, torch.Tensor):
            return {}
        if tuple(source.shape) != tuple(target.shape):
            return {}
        assignments[attr] = source.detach()
    return assignments


def _factorize_matrix_for_target(
    matrix: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
    rank = int(target_a.shape[1])
    if tuple(target_b.shape) != (rank, cols):
        raise RuntimeError(
            f"Incompatible LoKr target factor shapes: {tuple(target_a.shape)} and {tuple(target_b.shape)} for {tuple(matrix.shape)}"
        )
    working = matrix.detach().float()
    u, s, vh = torch.linalg.svd(working, full_matrices=False)
    k = min(rank, int(s.shape[0]))
    a = torch.zeros((rows, rank), dtype=working.dtype, device=working.device)
    b = torch.zeros((rank, cols), dtype=working.dtype, device=working.device)
    if k > 0:
        sqrt_s = torch.sqrt(s[:k])
        a[:, :k] = u[:, :k] * sqrt_s.unsqueeze(0)
        b[:k, :] = sqrt_s.unsqueeze(1) * vh[:k, :]
    return a, b


def resolve_lokr_state_for_layer(
    state_dict: Dict[str, torch.Tensor],
    *,
    layer: torch.nn.Module,
    layer_base_name: str,
    metadata: Optional[Dict[str, str]] = None,
) -> Optional[LoKrResolvedWeights]:
    layouts = collect_lokr_weight_layouts(state_dict)
    layout = layouts.get(layer_base_name)
    if layout is None:
        return None

    direct_assignments = _copy_if_exact_match(layer, state_dict, layout)
    source_scale, effective_rank, effective_alpha, scale_mode = _resolve_source_scale(layout, metadata=metadata)
    target_scale = float(getattr(layer, "scaling", 1.0) or 1.0)
    if direct_assignments:
        return LoKrResolvedWeights(
            direct_assignments=direct_assignments,
            effective_rank=effective_rank,
            effective_alpha=effective_alpha,
            export_scale_mode=scale_mode,
        )

    w1, w2 = materialize_lokr_matrices(state_dict, layout)
    w2 = w2 * (source_scale / max(target_scale, 1e-12))

    assignments: Dict[str, torch.Tensor] = {}
    if hasattr(layer, "lokr_w1"):
        target = getattr(layer, "lokr_w1")
        if tuple(target.shape) != tuple(w1.shape):
            raise RuntimeError(f"Shape mismatch for {layer_base_name}.lokr_w1: {tuple(w1.shape)} != {tuple(target.shape)}")
        assignments["lokr_w1"] = w1
    elif hasattr(layer, "lokr_w1_a") and hasattr(layer, "lokr_w1_b"):
        w1_a, w1_b = _factorize_matrix_for_target(w1, getattr(layer, "lokr_w1_a"), getattr(layer, "lokr_w1_b"))
        assignments["lokr_w1_a"] = w1_a
        assignments["lokr_w1_b"] = w1_b
    else:
        raise RuntimeError(f"Unsupported LoKr target w1 layout for {layer_base_name}")

    if hasattr(layer, "lokr_w2"):
        target = getattr(layer, "lokr_w2")
        if tuple(target.shape) != tuple(w2.shape):
            raise RuntimeError(f"Shape mismatch for {layer_base_name}.lokr_w2: {tuple(w2.shape)} != {tuple(target.shape)}")
        assignments["lokr_w2"] = w2
    elif hasattr(layer, "lokr_w2_a") and hasattr(layer, "lokr_w2_b"):
        w2_a, w2_b = _factorize_matrix_for_target(w2, getattr(layer, "lokr_w2_a"), getattr(layer, "lokr_w2_b"))
        assignments["lokr_w2_a"] = w2_a
        assignments["lokr_w2_b"] = w2_b
    else:
        raise RuntimeError(f"Unsupported LoKr target w2 layout for {layer_base_name}")

    return LoKrResolvedWeights(
        direct_assignments=assignments,
        effective_rank=effective_rank,
        effective_alpha=effective_alpha,
        export_scale_mode=scale_mode,
    )

