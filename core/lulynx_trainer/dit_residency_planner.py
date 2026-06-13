"""Shared native DiT residency planning.

This module owns the hot/cold layer selection logic used by Anima and Newbie
Streaming Offload.  The model-specific wrappers still decide how to find their
block list, but the offload policy is shared and explainable here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
import torch.nn as nn

from .lora_injector import LoRALinear
from .native_unet.weight_residency import LulynxManagedLinear
from .sparse_swap_policy import build_sparse_swap_plan


VALID_DIT_BLOCK_RESIDENCY_MODES = {"resident", "streaming_offload", "block_cpu_pinned"}
AUTO_STREAMING_MIN_PARAMETER_COUNT = 262_144
AUTO_STREAMING_TARGET_COLD_PARAM_FRACTION = 0.60
STREAMING_RESIDENT_EDGE_BLOCKS = 1


def normalize_dit_block_residency(value: Any) -> str:
    mode = str(value or "resident").strip().lower().replace("-", "_")
    aliases = {
        "off": "resident",
        "gpu": "resident",
        "none": "resident",
        "cpu_pinned": "block_cpu_pinned",
        "block_pinned": "block_cpu_pinned",
        "blocks_cpu_pinned": "block_cpu_pinned",
        "linear_cpu_pinned": "block_cpu_pinned",
        "balanced": "streaming_offload",
        "hot": "streaming_offload",
        "hotaware": "streaming_offload",
        "hot_aware": "streaming_offload",
        "hot_aware_cpu_pinned": "streaming_offload",
        "streaming": "streaming_offload",
        "streaming_cpu_offload": "streaming_offload",
        "streaming_pinned": "streaming_offload",
        "steaming": "streaming_offload",
        "steaming_offload": "streaming_offload",
    }
    mode = aliases.get(mode, mode)
    return mode if mode in VALID_DIT_BLOCK_RESIDENCY_MODES else "resident"


def default_hot_tokens(family: str) -> tuple[str, ...]:
    normalized = str(family or "").strip().lower()
    common = (
        "attention",
        "attn",
        "adaln",
        "mod",
        "embed",
        "q_proj",
        "k_proj",
        "v_proj",
        "output_proj",
        "to_q",
        "to_k",
        "to_v",
        "to_out",
    )
    if normalized == "anima":
        return (
            "self_attn",
            "cross_attn",
            *common,
            "llm_adapter",
        )
    if normalized == "newbie":
        return (
            *common,
            "modulation",
            "time",
            "qkv",
            "out",
        )
    return common


@dataclass
class DitResidencyUnit:
    block_index: int
    module_name: str
    parameter_count: int
    from_lora: bool
    decision: str
    reason: str
    module: LulynxManagedLinear = field(repr=False, compare=False)
    sparse_decision: str = ""

    @property
    def cpu_pinned(self) -> bool:
        return self.decision == "cpu_pinned"

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_index": int(self.block_index),
            "module_name": self.module_name,
            "parameter_count": int(self.parameter_count),
            "from_lora": bool(self.from_lora),
            "decision": self.decision,
            "reason": self.reason,
            "sparse_decision": self.sparse_decision,
        }


@dataclass
class DitResidencyPlan:
    family: str
    mode: str
    block_count: int
    requested_min_parameter_count: int
    min_parameter_count: int
    auto_min_parameter_count: bool
    strategy: str
    units: list[DitResidencyUnit]
    auto_threshold_candidate_count: int = 0
    auto_threshold_total_parameter_count: int = 0

    @property
    def managed_linear_count(self) -> int:
        return len(self.units)

    @property
    def lora_wrapped_linear_count(self) -> int:
        return sum(1 for unit in self.units if unit.from_lora)

    @property
    def planned_linear_count(self) -> int:
        return sum(1 for unit in self.units if unit.cpu_pinned)

    @property
    def skipped_small_count(self) -> int:
        return sum(1 for unit in self.units if unit.reason == "small")

    @property
    def hot_resident_count(self) -> int:
        return sum(1 for unit in self.units if unit.reason == "hot")

    @property
    def edge_resident_count(self) -> int:
        return sum(1 for unit in self.units if unit.reason == "edge")

    @property
    def cold_candidate_count(self) -> int:
        return sum(1 for unit in self.units if unit.reason in {"cold", "cold_emergency"})

    @property
    def sparse_warm_prefetch_count(self) -> int:
        return sum(1 for unit in self.units if unit.sparse_decision == "warm_prefetch")

    @property
    def sparse_cold_on_demand_count(self) -> int:
        return sum(1 for unit in self.units if unit.sparse_decision == "cold_on_demand")

    @property
    def planned_parameter_count(self) -> int:
        return sum(int(unit.parameter_count) for unit in self.units if unit.cpu_pinned)

    @property
    def planned_cpu_parameter_mb(self) -> float:
        # Planner estimate assumes bf16/fp16-ish frozen weights. The apply step
        # records actual CPU tensor bytes after enabling residency.
        return self.planned_parameter_count * 2.0 / (1024 * 1024)

    def as_dict(self, *, sample_limit: int = 16) -> dict[str, Any]:
        return {
            "family": self.family,
            "mode": self.mode,
            "block_count": int(self.block_count),
            "strategy": self.strategy,
            "managed_linear_count": int(self.managed_linear_count),
            "planned_linear_count": int(self.planned_linear_count),
            "lora_wrapped_linear_count": int(self.lora_wrapped_linear_count),
            "skipped_small_count": int(self.skipped_small_count),
            "hot_resident_count": int(self.hot_resident_count),
            "edge_resident_count": int(self.edge_resident_count),
            "cold_candidate_count": int(self.cold_candidate_count),
            "sparse_warm_prefetch_count": int(self.sparse_warm_prefetch_count),
            "sparse_cold_on_demand_count": int(self.sparse_cold_on_demand_count),
            "requested_min_parameter_count": int(self.requested_min_parameter_count),
            "min_parameter_count": int(self.min_parameter_count),
            "auto_min_parameter_count": bool(self.auto_min_parameter_count),
            "auto_threshold_candidate_count": int(self.auto_threshold_candidate_count),
            "auto_threshold_total_parameter_count": int(self.auto_threshold_total_parameter_count),
            "planned_cpu_parameter_mb": round(float(self.planned_cpu_parameter_mb), 3),
            "unit_sample": [unit.as_dict() for unit in self.units[:sample_limit]],
        }


def module_parameter_count(module: nn.Module) -> int:
    return sum(int(param.numel()) for param in module.parameters(recurse=False))


def managed_base_linear(module: nn.Module) -> tuple[LulynxManagedLinear | None, bool]:
    if isinstance(module, LulynxManagedLinear):
        return module, False
    if isinstance(module, LoRALinear):
        original = getattr(module, "original", None)
        if isinstance(original, LulynxManagedLinear):
            return original, True
    return None, False


def _is_hot_resident_linear(name: str, *, hot_tokens: Iterable[str]) -> bool:
    lowered = str(name or "").lower()
    return any(token in lowered for token in hot_tokens)


def _is_edge_block(block_index: int, block_count: int, *, edge_blocks: int) -> bool:
    if edge_blocks <= 0 or block_count <= 0:
        return False
    return block_index < edge_blocks or block_index >= block_count - edge_blocks


def _auto_streaming_threshold(cold_parameter_counts: list[int]) -> tuple[int, int, int]:
    if not cold_parameter_counts:
        return AUTO_STREAMING_MIN_PARAMETER_COUNT, 0, 0
    sorted_counts = sorted((max(int(value), 0) for value in cold_parameter_counts), reverse=True)
    total = sum(sorted_counts)
    target = max(int(total * AUTO_STREAMING_TARGET_COLD_PARAM_FRACTION), 1)
    selected_total = 0
    threshold = sorted_counts[-1]
    selected = 0
    for count in sorted_counts:
        selected += 1
        selected_total += count
        threshold = count
        if selected_total >= target:
            break
    return max(threshold, AUTO_STREAMING_MIN_PARAMETER_COUNT), len(sorted_counts), total


def build_dit_residency_plan(
    blocks: list[nn.Module],
    *,
    family: str,
    mode: str,
    requested_min_parameter_count: int = 0,
    edge_blocks: int = STREAMING_RESIDENT_EDGE_BLOCKS,
    hot_tokens: Iterable[str] | None = None,
    sparse_swap_enabled: bool = False,
    sparse_swap_budget_mb: float | None = None,
    sparse_swap_warm_fraction: float = 0.35,
) -> DitResidencyPlan:
    normalized = normalize_dit_block_residency(mode)
    requested_min = max(int(requested_min_parameter_count or 0), 0)
    block_count = len(blocks)
    tokens = tuple(hot_tokens or default_hot_tokens(family))

    raw_units: list[tuple[int, str, LulynxManagedLinear, bool, int, bool, bool]] = []
    seen: set[int] = set()
    cold_parameter_counts: list[int] = []
    for block_index, block in enumerate(blocks):
        for module_name, module in block.named_modules():
            base, from_lora = managed_base_linear(module)
            if base is None:
                continue
            ident = id(base)
            if ident in seen:
                continue
            seen.add(ident)
            parameter_count = module_parameter_count(base)
            is_edge = _is_edge_block(block_index, block_count, edge_blocks=edge_blocks)
            is_hot = _is_hot_resident_linear(module_name, hot_tokens=tokens)
            raw_units.append((block_index, module_name, base, from_lora, parameter_count, is_edge, is_hot))
            if normalized == "streaming_offload" and not is_edge and not is_hot:
                cold_parameter_counts.append(parameter_count)

    effective_min = requested_min
    auto_min = False
    auto_candidate_count = 0
    auto_total_params = 0
    strategy = normalized
    if normalized == "streaming_offload":
        strategy = "hot_aware_streaming"
        if requested_min == 0:
            effective_min, auto_candidate_count, auto_total_params = _auto_streaming_threshold(cold_parameter_counts)
            auto_min = True
            strategy = "hot_aware_streaming_auto_threshold"
        if sparse_swap_enabled:
            strategy = f"{strategy}_sparse_swap"
    elif normalized == "block_cpu_pinned":
        strategy = "block_cpu_pinned_all_frozen"

    sparse_by_key: dict[tuple[int, str], str] = {}
    sparse_reason_by_key: dict[tuple[int, str], str] = {}
    if normalized == "streaming_offload" and sparse_swap_enabled:
        sparse_units = [
            type(
                "SparseCandidate",
                (),
                {
                    "block_index": block_index,
                    "module_name": module_name,
                    "parameter_count": parameter_count,
                    "reason": "",
                    "decision": "",
                },
            )()
            for block_index, module_name, _base, _from_lora, parameter_count, _is_edge, _is_hot in raw_units
        ]
        sparse_plan = build_sparse_swap_plan(
            sparse_units,
            budget_mb=sparse_swap_budget_mb,
            hot_tokens=tokens,
            edge_blocks=edge_blocks,
            min_param_count=effective_min,
            warm_prefetch_fraction=sparse_swap_warm_fraction,
        )
        for assignment in sparse_plan.assignments:
            key = (int(assignment.unit.block_index), str(assignment.unit.module_name))
            sparse_by_key[key] = str(assignment.decision)
            sparse_reason_by_key[key] = str(assignment.reason)

    units: list[DitResidencyUnit] = []
    for block_index, module_name, base, from_lora, parameter_count, is_edge, is_hot in raw_units:
        decision = "resident"
        reason = "resident"
        sparse_decision = ""
        if normalized == "block_cpu_pinned":
            if parameter_count >= effective_min:
                decision = "cpu_pinned"
                reason = "cold_emergency"
            else:
                reason = "small"
        elif normalized == "streaming_offload":
            if sparse_swap_enabled:
                sparse_decision = sparse_by_key.get((block_index, module_name), "")
                sparse_reason = sparse_reason_by_key.get((block_index, module_name), "")
                if sparse_decision in {"warm_prefetch", "cold_on_demand"}:
                    decision = "cpu_pinned"
                    reason = sparse_decision
                elif sparse_decision == "skip":
                    reason = "small"
                elif sparse_decision == "resident":
                    reason = sparse_reason or "resident"
                elif is_edge:
                    reason = "edge"
                elif is_hot:
                    reason = "hot"
                elif parameter_count < effective_min:
                    reason = "small"
                else:
                    decision = "cpu_pinned"
                    reason = "cold"
            else:
                if is_edge:
                    reason = "edge"
                elif is_hot:
                    reason = "hot"
                elif parameter_count < effective_min:
                    reason = "small"
                else:
                    decision = "cpu_pinned"
                    reason = "cold"
        units.append(
            DitResidencyUnit(
                block_index=block_index,
                module_name=module_name,
                module=base,
                parameter_count=parameter_count,
                from_lora=from_lora,
                decision=decision,
                reason=reason,
                sparse_decision=sparse_decision,
            )
        )

    return DitResidencyPlan(
        family=str(family or "").strip().lower(),
        mode=normalized,
        block_count=block_count,
        requested_min_parameter_count=requested_min,
        min_parameter_count=effective_min,
        auto_min_parameter_count=auto_min,
        strategy=strategy,
        units=units,
        auto_threshold_candidate_count=auto_candidate_count,
        auto_threshold_total_parameter_count=auto_total_params,
    )
