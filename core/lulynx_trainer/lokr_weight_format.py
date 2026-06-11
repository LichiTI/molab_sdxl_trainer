"""Warehouse LoKr weight layout detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


LOKR_LAYOUT_DIRECT = "direct"
LOKR_LAYOUT_DECOMPOSED = "decomposed"

_LOKR_SUFFIXES = (
    "lokr_w1",
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_w2",
    "lokr_w2_a",
    "lokr_w2_b",
    "lokr_rank",
    "alpha",
)


@dataclass
class LoKrWeightLayout:
    base_key: str
    layout_kind: str
    has_rank_key: bool
    rank: Optional[int]
    alpha: Optional[float]
    factor: Optional[int]
    shapes: Dict[str, tuple[int, ...]]
    tensor_keys: Dict[str, str]


def _tensor_shape(value: object) -> Optional[tuple[int, ...]]:
    if isinstance(value, torch.Tensor):
        return tuple(int(dim) for dim in value.shape)
    return None


def scalar_from_tensor(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().reshape(-1)[0].cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_lokr_base_name(base_key: str) -> str:
    if base_key.startswith("lora_unet_"):
        return base_key[len("lora_"):]
    return base_key


def infer_lokr_rank(
    tensor_shapes: Dict[str, tuple[int, ...]],
    *,
    alpha: Optional[float] = None,
    explicit_rank: Optional[float] = None,
) -> Optional[int]:
    if explicit_rank is not None:
        return max(1, int(round(float(explicit_rank))))
    for suffix in ("lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b"):
        shape = tensor_shapes.get(suffix)
        if shape is None or len(shape) != 2:
            continue
        if suffix.endswith("_a"):
            return max(1, int(shape[1]))
        return max(1, int(shape[0]))
    if alpha is not None:
        rounded = int(round(float(alpha)))
        if rounded > 0 and abs(float(alpha) - float(rounded)) <= 1e-6:
            return rounded
    return None


def infer_lokr_factor(tensor_shapes: Dict[str, tuple[int, ...]]) -> Optional[int]:
    direct_w2 = tensor_shapes.get("lokr_w2")
    if direct_w2 is not None and len(direct_w2) == 2 and direct_w2[0] == direct_w2[1] and direct_w2[0] > 0:
        return int(direct_w2[0])

    w2_a = tensor_shapes.get("lokr_w2_a")
    w2_b = tensor_shapes.get("lokr_w2_b")
    if w2_a is not None and w2_b is not None and len(w2_a) == 2 and len(w2_b) == 2:
        if w2_a[0] == w2_b[1] and w2_a[0] > 0:
            return int(w2_a[0])

    w1_a = tensor_shapes.get("lokr_w1_a")
    w1_b = tensor_shapes.get("lokr_w1_b")
    if w1_a is not None and w1_b is not None and len(w1_a) == 2 and len(w1_b) == 2:
        if w1_a[0] == w1_b[1] and w1_a[0] > 0:
            return int(w1_a[0])
    return None


def collect_lokr_weight_layouts(state_dict: Dict[str, torch.Tensor]) -> Dict[str, LoKrWeightLayout]:
    grouped: Dict[str, Dict[str, object]] = {}
    for key, value in state_dict.items():
        base, sep, suffix = key.rpartition(".")
        if not sep or suffix not in _LOKR_SUFFIXES:
            continue
        normalized = normalize_lokr_base_name(base)
        entry = grouped.setdefault(
            normalized,
            {
                "tensor_keys": {},
                "shapes": {},
                "alpha": None,
                "rank_value": None,
            },
        )
        entry["tensor_keys"][suffix] = key
        shape = _tensor_shape(value)
        if shape is not None:
            entry["shapes"][suffix] = shape
        if suffix == "alpha":
            entry["alpha"] = scalar_from_tensor(value)
        elif suffix == "lokr_rank":
            entry["rank_value"] = scalar_from_tensor(value)

    layouts: Dict[str, LoKrWeightLayout] = {}
    for base_key, payload in grouped.items():
        tensor_keys = dict(payload["tensor_keys"])
        shapes = dict(payload["shapes"])
        has_decomposed = any(name in tensor_keys for name in ("lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b"))
        layout_kind = LOKR_LAYOUT_DECOMPOSED if has_decomposed else LOKR_LAYOUT_DIRECT
        alpha = payload["alpha"]
        explicit_rank = payload["rank_value"]
        layouts[base_key] = LoKrWeightLayout(
            base_key=base_key,
            layout_kind=layout_kind,
            has_rank_key="lokr_rank" in tensor_keys,
            rank=infer_lokr_rank(shapes, alpha=alpha, explicit_rank=explicit_rank),
            alpha=alpha,
            factor=infer_lokr_factor(shapes),
            shapes=shapes,
            tensor_keys=tensor_keys,
        )
    return layouts

