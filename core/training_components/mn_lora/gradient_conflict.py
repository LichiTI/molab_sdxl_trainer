"""Gradient conflict surgery helpers for MN-LoRA.

The existing trainer-level PCGrad resolves conflicts between accumulated
microbatches. This module is optimizer-local and works on named gradient maps,
so MN-LoRA regularizer gradients such as Fisher/EWC can be resolved against the
main training gradient before they are written back to parameters.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch


def _flatten_map(grad_map: Mapping[str, torch.Tensor], names: Iterable[str]) -> Optional[torch.Tensor]:
    chunks: List[torch.Tensor] = []
    for name in names:
        grad = grad_map.get(name)
        if isinstance(grad, torch.Tensor):
            chunks.append(grad.detach().float().reshape(-1))
    if not chunks:
        return None
    return torch.cat(chunks)


def _cosine(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor], names: Iterable[str]) -> float:
    flat_a = _flatten_map(a, names)
    flat_b = _flatten_map(b, names)
    if flat_a is None or flat_b is None:
        return 1.0
    norm_a = torch.linalg.vector_norm(flat_a)
    norm_b = torch.linalg.vector_norm(flat_b)
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 1.0
    return float(torch.dot(flat_a, flat_b).div(norm_a * norm_b).clamp(-1.0, 1.0).detach().cpu())


def _project_against(target: Dict[str, torch.Tensor], reference: Mapping[str, torch.Tensor], names: Iterable[str]) -> int:
    projections = 0
    for name in names:
        g_t = target.get(name)
        g_r = reference.get(name)
        if g_t is None or g_r is None:
            continue
        t = g_t.float()
        r = g_r.float()
        dot = torch.dot(t.reshape(-1), r.reshape(-1))
        ref_norm_sq = torch.dot(r.reshape(-1), r.reshape(-1))
        if ref_norm_sq <= 1e-12 or dot >= 0:
            continue
        target[name] = (t - (dot / ref_norm_sq) * r).to(dtype=g_t.dtype)
        projections += 1
    return projections


class MNLoRAGradientConflictController:
    """Resolve conflicts between named objective gradients."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        conflict_threshold: float = 0.0,
        protect_main_gradient: bool = True,
        reduction: str = "sum",
    ) -> None:
        self.enabled = bool(enabled)
        self.conflict_threshold = float(conflict_threshold)
        self.protect_main_gradient = bool(protect_main_gradient)
        self.reduction = str(reduction or "sum").strip().lower()
        if self.reduction not in {"sum", "mean"}:
            self.reduction = "sum"

        self._calls = 0
        self._maps_seen = 0
        self._total_pairs = 0
        self._conflict_pairs = 0
        self._projections = 0
        self._last_stats: Dict[str, Any] = {}

    def resolve(self, gradient_maps: List[Mapping[str, torch.Tensor]]) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        self._calls += 1
        maps = [dict(item) for item in gradient_maps if item]
        self._maps_seen += len(maps)
        names: List[str] = []
        seen = set()
        for grad_map in maps:
            for name in grad_map.keys():
                if name not in seen:
                    seen.add(name)
                    names.append(name)

        stats: Dict[str, Any] = {
            "enabled": bool(self.enabled),
            "input_maps": len(maps),
            "param_count": len(names),
            "total_pairs": 0,
            "conflict_pairs": 0,
            "projections": 0,
            "conflict_rate": 0.0,
            "avg_conflict_angle": 0.0,
            "conflict_threshold": float(self.conflict_threshold),
            "protect_main_gradient": bool(self.protect_main_gradient),
            "reduction": self.reduction,
        }
        if not self.enabled or len(maps) <= 1 or not names:
            reduced = self._reduce(maps, names)
            self._last_stats = stats
            return reduced, stats

        projected = [
            {name: grad.detach().clone() for name, grad in grad_map.items()}
            for grad_map in maps
        ]
        conflict_angles: List[float] = []
        start_idx = 1 if self.protect_main_gradient else 0
        for i in range(len(maps)):
            for j in range(i + 1, len(maps)):
                stats["total_pairs"] += 1
                cosine = _cosine(maps[i], maps[j], names)
                if cosine >= self.conflict_threshold:
                    continue
                stats["conflict_pairs"] += 1
                conflict_angles.append(math.degrees(math.acos(max(min(cosine, 1.0), -1.0))))

        for i in range(start_idx, len(projected)):
            for j in range(len(maps)):
                if i == j:
                    continue
                cosine = _cosine(projected[i], maps[j], names)
                if cosine >= self.conflict_threshold:
                    continue
                stats["projections"] += _project_against(projected[i], maps[j], names)

        if stats["total_pairs"]:
            stats["conflict_rate"] = float(stats["conflict_pairs"] / max(1, stats["total_pairs"]))
        if conflict_angles:
            stats["avg_conflict_angle"] = float(sum(conflict_angles) / len(conflict_angles))

        self._total_pairs += int(stats["total_pairs"])
        self._conflict_pairs += int(stats["conflict_pairs"])
        self._projections += int(stats["projections"])
        self._last_stats = stats
        return self._reduce(projected, names), stats

    def _reduce(self, maps: List[Mapping[str, torch.Tensor]], names: Iterable[str]) -> Dict[str, torch.Tensor]:
        reduced: Dict[str, torch.Tensor] = {}
        divisor = max(1, len(maps)) if self.reduction == "mean" else 1
        for name in names:
            grads = [grad_map[name].float() for grad_map in maps if name in grad_map]
            if not grads:
                continue
            reduced[name] = torch.stack(grads).sum(dim=0) / divisor
        return reduced

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "calls": int(self._calls),
            "maps_seen": int(self._maps_seen),
            "total_pairs": int(self._total_pairs),
            "conflict_pairs": int(self._conflict_pairs),
            "conflict_rate": float(self._conflict_pairs / max(1, self._total_pairs)),
            "projections": int(self._projections),
            "conflict_threshold": float(self.conflict_threshold),
            "protect_main_gradient": bool(self.protect_main_gradient),
            "reduction": str(self.reduction),
            "last": dict(self._last_stats),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "conflict_threshold": self.conflict_threshold,
            "protect_main_gradient": self.protect_main_gradient,
            "reduction": self.reduction,
            "telemetry": self.get_telemetry_snapshot(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.enabled = bool(state_dict.get("enabled", self.enabled))
        self.conflict_threshold = float(state_dict.get("conflict_threshold", self.conflict_threshold))
        self.protect_main_gradient = bool(state_dict.get("protect_main_gradient", self.protect_main_gradient))
        self.reduction = str(state_dict.get("reduction", self.reduction) or self.reduction)
