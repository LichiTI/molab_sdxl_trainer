"""Experimental sparse swap planner for Streaming Offload.

The planner is deliberately side-effect free: it classifies candidate units
into resident / prefetch / on-demand buckets, but does not mutate modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


SPARSE_SWAP_DECISIONS = {"resident", "warm_prefetch", "cold_on_demand", "skip"}


@dataclass(frozen=True)
class SparseSwapUnit:
    block_index: int
    module_name: str
    parameter_count: int
    reason: str = ""
    decision: str = ""

    @classmethod
    def from_any(cls, unit: Any) -> "SparseSwapUnit":
        return cls(
            block_index=int(getattr(unit, "block_index", 0)),
            module_name=str(getattr(unit, "module_name", "")),
            parameter_count=int(getattr(unit, "parameter_count", 0)),
            reason=str(getattr(unit, "reason", "")),
            decision=str(getattr(unit, "decision", "")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_index": int(self.block_index),
            "module_name": self.module_name,
            "parameter_count": int(self.parameter_count),
            "reason": self.reason,
            "source_decision": self.decision,
        }


@dataclass(frozen=True)
class SparseSwapAssignment:
    unit: SparseSwapUnit
    decision: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        payload = self.unit.as_dict()
        payload.update({"sparse_decision": self.decision, "sparse_reason": self.reason})
        return payload


@dataclass
class SparseSwapPlan:
    assignments: list[SparseSwapAssignment]
    budget_mb: float | None
    hot_tokens: tuple[str, ...]
    edge_blocks: int
    min_param_count: int

    @property
    def planned_parameter_count(self) -> int:
        return sum(item.unit.parameter_count for item in self.assignments if item.decision in {"warm_prefetch", "cold_on_demand"})

    @property
    def planned_transfer_mb_fp16(self) -> float:
        return self.planned_parameter_count * 2.0 / (1024.0 * 1024.0)

    def count(self, decision: str) -> int:
        return sum(1 for item in self.assignments if item.decision == decision)

    def as_dict(self, *, sample_limit: int = 24) -> dict[str, Any]:
        return {
            "budget_mb": self.budget_mb,
            "hot_tokens": list(self.hot_tokens),
            "edge_blocks": int(self.edge_blocks),
            "min_param_count": int(self.min_param_count),
            "resident_count": self.count("resident"),
            "warm_prefetch_count": self.count("warm_prefetch"),
            "cold_on_demand_count": self.count("cold_on_demand"),
            "skip_count": self.count("skip"),
            "planned_transfer_mb_fp16": round(float(self.planned_transfer_mb_fp16), 3),
            "sample": [item.as_dict() for item in self.assignments[:sample_limit]],
        }


def _matches_hot_token(name: str, hot_tokens: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(token and token.lower() in lowered for token in hot_tokens)


def build_sparse_swap_plan(
    units: Iterable[Any],
    *,
    budget_mb: float | None = None,
    hot_tokens: Iterable[str] = ("attn", "attention", "mod", "embed", "q_proj", "k_proj", "v_proj"),
    edge_blocks: int = 1,
    min_param_count: int = 262_144,
    warm_prefetch_fraction: float = 0.35,
) -> SparseSwapPlan:
    normalized = [SparseSwapUnit.from_any(unit) for unit in units]
    hot = tuple(str(token).strip().lower() for token in hot_tokens if str(token).strip())
    max_block = max((unit.block_index for unit in normalized), default=-1)
    edge_blocks = max(int(edge_blocks or 0), 0)
    min_param_count = max(int(min_param_count or 0), 0)
    warm_prefetch_fraction = min(max(float(warm_prefetch_fraction), 0.0), 1.0)

    candidates: list[SparseSwapUnit] = []
    assignments: list[SparseSwapAssignment] = []
    for unit in normalized:
        if unit.parameter_count < min_param_count:
            assignments.append(SparseSwapAssignment(unit, "skip", "small"))
        elif unit.block_index < edge_blocks or unit.block_index > max_block - edge_blocks:
            assignments.append(SparseSwapAssignment(unit, "resident", "edge"))
        elif _matches_hot_token(unit.module_name, hot):
            assignments.append(SparseSwapAssignment(unit, "resident", "hot"))
        else:
            candidates.append(unit)

    candidates.sort(key=lambda item: (item.block_index, -item.parameter_count, item.module_name))
    budget_params = None if budget_mb is None else int(max(float(budget_mb), 0.0) * 1024.0 * 1024.0 / 2.0)
    candidate_params = sum(unit.parameter_count for unit in candidates)
    warm_budget = int(candidate_params * warm_prefetch_fraction)
    if budget_params is not None:
        warm_budget = min(warm_budget, budget_params)

    used = 0
    for unit in candidates:
        if used < warm_budget:
            assignments.append(SparseSwapAssignment(unit, "warm_prefetch", "budgeted_prefetch"))
            used += unit.parameter_count
        else:
            assignments.append(SparseSwapAssignment(unit, "cold_on_demand", "cold"))

    assignments.sort(key=lambda item: (item.unit.block_index, item.unit.module_name))
    return SparseSwapPlan(
        assignments=assignments,
        budget_mb=None if budget_mb is None else float(budget_mb),
        hot_tokens=hot,
        edge_blocks=edge_blocks,
        min_param_count=min_param_count,
    )
