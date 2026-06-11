"""Offline adapter target layer/rank selection policies.

This primitive turns profiler metrics into a deterministic layer selection plan.
It is intended for FG-LoRA / gradient / CKA experiments and does not change the
existing LoRA injector or checkpoint format.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


SUPPORTED_ADAPTER_TARGET_POLICIES = {"all", "profiled", "gradient_selected", "cka_selected"}


@dataclass(frozen=True)
class AdapterLayerMetric:
    name: str
    parameter_count: int = 0
    gradient_norm: float = 0.0
    cka_dissimilarity: float = 0.0
    sensitivity: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AdapterLayerMetric":
        return cls(
            name=str(value.get("name") or value.get("module") or value.get("layer") or ""),
            parameter_count=max(int(value.get("parameter_count", value.get("params", 0)) or 0), 0),
            gradient_norm=max(float(value.get("gradient_norm", value.get("grad_norm", 0.0)) or 0.0), 0.0),
            cka_dissimilarity=max(float(value.get("cka_dissimilarity", value.get("cka", 0.0)) or 0.0), 0.0),
            sensitivity=max(float(value.get("sensitivity", value.get("loss_delta", 0.0)) or 0.0), 0.0),
        )


@dataclass(frozen=True)
class AdapterTargetPolicyConfig:
    policy: str = "all"
    base_rank: int = 16
    min_rank: int = 1
    max_rank: int = 64
    target_fraction: float = 1.0
    top_k: int = 0
    min_score: float = 0.0

    def normalized(self) -> "AdapterTargetPolicyConfig":
        policy = str(self.policy or "all").strip().lower().replace("-", "_")
        if policy not in SUPPORTED_ADAPTER_TARGET_POLICIES:
            policy = "all"
        base_rank = max(int(self.base_rank or 1), 1)
        min_rank = max(int(self.min_rank or 1), 1)
        max_rank = max(int(self.max_rank or base_rank), min_rank)
        base_rank = min(max(base_rank, min_rank), max_rank)
        fraction = 1.0 if self.target_fraction is None else float(self.target_fraction)
        return AdapterTargetPolicyConfig(
            policy=policy,
            base_rank=base_rank,
            min_rank=min_rank,
            max_rank=max_rank,
            target_fraction=min(max(fraction, 0.0), 1.0),
            top_k=max(int(self.top_k or 0), 0),
            min_score=max(float(self.min_score or 0.0), 0.0),
        )


@dataclass(frozen=True)
class AdapterTargetRow:
    name: str
    selected: bool
    score: float
    rank: int
    reason: str
    parameter_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "selected": bool(self.selected),
            "score": float(self.score),
            "rank": int(self.rank),
            "reason": self.reason,
            "parameter_count": int(self.parameter_count),
        }


@dataclass(frozen=True)
class AdapterTargetPolicyPlan:
    policy: str
    selected_count: int
    total_count: int
    rows: tuple[AdapterTargetRow, ...]
    default_behavior_changed: bool = False

    @property
    def selected_names(self) -> tuple[str, ...]:
        return tuple(row.name for row in self.rows if row.selected)

    @property
    def rank_by_name(self) -> dict[str, int]:
        return {row.name: row.rank for row in self.rows if row.selected}

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan": "adapter_target_policy_plan_v0",
            "policy": self.policy,
            "selected_count": int(self.selected_count),
            "total_count": int(self.total_count),
            "selected_names": list(self.selected_names),
            "rank_by_name": self.rank_by_name,
            "default_behavior_changed": bool(self.default_behavior_changed),
            "rows": [row.as_dict() for row in self.rows],
        }


def build_adapter_target_policy_plan(
    metrics: Sequence[AdapterLayerMetric | Mapping[str, Any]],
    config: AdapterTargetPolicyConfig | Mapping[str, Any] | None = None,
) -> AdapterTargetPolicyPlan:
    cfg = _config(config)
    parsed_layers = [_metric(item) for item in metrics]
    layers = tuple(layer for layer in parsed_layers if layer.name)
    if not layers:
        return AdapterTargetPolicyPlan(policy=cfg.policy, selected_count=0, total_count=0, rows=())

    scores = _scores(layers, cfg.policy)
    select_count = _select_count(len(layers), cfg)
    ranked_indices = sorted(range(len(layers)), key=lambda idx: (-scores[idx], layers[idx].name))
    selected_indices = set(ranked_indices[:select_count])
    if cfg.min_score > 0.0 and cfg.policy != "all":
        selected_indices = {idx for idx in selected_indices if scores[idx] >= cfg.min_score}

    rows = []
    for idx, layer in enumerate(layers):
        selected = idx in selected_indices
        score = scores[idx]
        rows.append(
            AdapterTargetRow(
                name=layer.name,
                selected=selected,
                score=score,
                rank=_rank_for_score(score, cfg) if selected else 0,
                reason=_reason(selected, cfg.policy, score, cfg.min_score),
                parameter_count=layer.parameter_count,
            )
        )
    rows = tuple(sorted(rows, key=lambda row: row.name))
    return AdapterTargetPolicyPlan(
        policy=cfg.policy,
        selected_count=sum(1 for row in rows if row.selected),
        total_count=len(rows),
        rows=rows,
        default_behavior_changed=False,
    )


def _config(config: AdapterTargetPolicyConfig | Mapping[str, Any] | None) -> AdapterTargetPolicyConfig:
    if isinstance(config, Mapping):
        return AdapterTargetPolicyConfig(**config).normalized()
    return (config or AdapterTargetPolicyConfig()).normalized()


def _metric(value: AdapterLayerMetric | Mapping[str, Any]) -> AdapterLayerMetric:
    if isinstance(value, AdapterLayerMetric):
        return value
    if isinstance(value, Mapping):
        return AdapterLayerMetric.from_mapping(value)
    raise TypeError("adapter layer metric must be AdapterLayerMetric or Mapping")


def _scores(layers: Sequence[AdapterLayerMetric], policy: str) -> list[float]:
    if policy == "all":
        return [1.0 for _ in layers]
    gradients = _normalize([layer.gradient_norm for layer in layers])
    cka = _normalize([layer.cka_dissimilarity for layer in layers])
    sensitivity = _normalize([layer.sensitivity for layer in layers])
    if policy == "gradient_selected":
        return gradients
    if policy == "cka_selected":
        return cka
    return [0.45 * g + 0.35 * c + 0.20 * s for g, c, s in zip(gradients, cka, sensitivity)]


def _normalize(values: Sequence[float]) -> list[float]:
    maximum = max((float(value) for value in values), default=0.0)
    if maximum <= 0.0:
        return [0.0 for _ in values]
    return [float(value) / maximum for value in values]


def _select_count(total: int, cfg: AdapterTargetPolicyConfig) -> int:
    if cfg.policy == "all":
        return total
    if cfg.top_k > 0:
        return min(cfg.top_k, total)
    return min(max(int(math.ceil(total * cfg.target_fraction)), 1), total)


def _rank_for_score(score: float, cfg: AdapterTargetPolicyConfig) -> int:
    if cfg.policy == "all":
        return cfg.base_rank
    span = cfg.max_rank - cfg.min_rank
    return min(max(int(math.ceil(cfg.min_rank + span * max(min(score, 1.0), 0.0))), cfg.min_rank), cfg.max_rank)


def _reason(selected: bool, policy: str, score: float, min_score: float) -> str:
    if selected:
        return "selected_all" if policy == "all" else "selected_by_score"
    if min_score > 0.0 and score < min_score:
        return "below_min_score"
    return "not_in_top_budget"


__all__ = [
    "AdapterLayerMetric",
    "AdapterTargetPolicyConfig",
    "AdapterTargetPolicyPlan",
    "AdapterTargetRow",
    "build_adapter_target_policy_plan",
]
