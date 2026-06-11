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


class DitBlockPrefetchController:
    """Forward-order async prefetch for CPU-pinned frozen DiT Linear weights."""

    def __init__(
        self,
        blocks: list[nn.Module],
        plan: DitResidencyPlan,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None = None,
        depth: int = 1,
        install_hooks: bool = True,
    ) -> None:
        self.blocks = list(blocks)
        self.plan = plan
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype
        self.depth = max(int(depth or 0), 0)
        self.enabled = False
        self.reason = "disabled"
        self._handles: list[Any] = []
        self._stream: torch.cuda.Stream | None = None
        self._module_to_index: dict[int, int] = {id(block): index for index, block in enumerate(self.blocks)}
        self._units_by_block: dict[int, list[DitResidencyUnit]] = {}
        self._before_calls = 0
        self._prefetch_calls = 0
        self._submitted = 0
        self._skipped = 0
        self._errors = 0
        self._last_error = ""

        for unit in plan.units:
            if unit.cpu_pinned and unit.sparse_decision != "cold_on_demand":
                self._units_by_block.setdefault(int(unit.block_index), []).append(unit)

        if self.depth <= 0:
            self.reason = "prefetch depth is 0"
            return
        if self.device is None or self.device.type != "cuda":
            self.reason = "prefetch requires a CUDA training device"
            return
        if not torch.cuda.is_available():
            self.reason = "CUDA is not available"
            return
        if not self._units_by_block:
            self.reason = "no CPU-pinned Linear units are planned"
            return
        try:
            self._stream = torch.cuda.Stream(device=self.device)
        except Exception as exc:
            self.reason = f"failed to create CUDA prefetch stream: {type(exc).__name__}: {exc}"
            self._errors += 1
            self._last_error = self.reason
            return
        self.enabled = True
        self.reason = "active"
        if install_hooks:
            self.install_hooks()

    @property
    def planned_block_count(self) -> int:
        return len(self._units_by_block)

    @property
    def planned_linear_count(self) -> int:
        return sum(len(units) for units in self._units_by_block.values())

    def install_hooks(self) -> None:
        if not self.enabled or self._handles:
            return

        def _make_hook(block_index: int) -> Any:
            def _hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
                self.before_block(block_index, *args)

            return _hook

        for index, block in enumerate(self.blocks):
            self._handles.append(block.register_forward_pre_hook(_make_hook(index)))

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()
        for units in self._units_by_block.values():
            for unit in units:
                unit.module.clear_cpu_pinned_prefetch()

    def before_block_module(self, block: nn.Module, *inputs: Any) -> None:
        block_index = self._module_to_index.get(id(block))
        if block_index is not None:
            self.before_block(block_index, *inputs)

    def before_block(self, block_index: int, *inputs: Any) -> None:
        if not self.enabled:
            return
        self._before_calls += 1
        device, dtype = self._resolve_target(inputs)
        if device is None or device.type != "cuda":
            self._skipped += 1
            return
        for offset in range(0, self.depth + 1):
            self._prefetch_block(int(block_index) + offset, device=device, dtype=dtype)

    def _resolve_target(self, inputs: Any) -> tuple[torch.device | None, torch.dtype | None]:
        tensor = self._first_tensor(inputs)
        if tensor is not None and tensor.device.type == "cuda":
            return tensor.device, tensor.dtype if tensor.is_floating_point() else self.dtype
        return self.device, self.dtype

    def _first_tensor(self, value: Any) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, dict):
            for item in value.values():
                found = self._first_tensor(item)
                if found is not None:
                    return found
        if isinstance(value, (list, tuple)):
            for item in value:
                found = self._first_tensor(item)
                if found is not None:
                    return found
        return None

    def _prefetch_block(self, block_index: int, *, device: torch.device, dtype: torch.dtype | None) -> None:
        units = self._units_by_block.get(int(block_index))
        if not units:
            return
        for unit in units:
            self._prefetch_calls += 1
            try:
                submitted = unit.module.prefetch_cpu_pinned_residency(
                    device=device,
                    dtype=dtype or self.dtype,
                    stream=self._stream,
                )
                if submitted:
                    self._submitted += 1
                else:
                    self._skipped += 1
            except Exception as exc:
                self._errors += 1
                self._last_error = f"{type(exc).__name__}: {exc}"

    def as_dict(self) -> dict[str, Any]:
        submitted = consumed = missed = errors = pending = 0
        for units in self._units_by_block.values():
            for unit in units:
                stats = unit.module.get_cpu_pinned_prefetch_stats()
                submitted += int(stats.get("submitted", 0))
                consumed += int(stats.get("consumed", 0))
                missed += int(stats.get("missed", 0))
                errors += int(stats.get("errors", 0))
                pending += int(stats.get("pending", 0))
        return {
            "enabled": bool(self.enabled),
            "reason": self.reason,
            "family": self.plan.family,
            "mode": self.plan.mode,
            "depth": int(self.depth),
            "device": str(self.device or ""),
            "planned_block_count": int(self.planned_block_count),
            "planned_linear_count": int(self.planned_linear_count),
            "before_block_calls": int(self._before_calls),
            "prefetch_calls": int(self._prefetch_calls),
            "submitted": int(submitted),
            "submitted_controller": int(self._submitted),
            "consumed": int(consumed),
            "missed": int(missed),
            "skipped": int(self._skipped),
            "errors": int(errors + self._errors),
            "pending": int(pending),
            "last_error": self._last_error,
        }


def clear_dit_block_prefetch_controller(model: nn.Module) -> None:
    controller = getattr(model, "_lulynx_dit_prefetch_controller", None)
    if controller is not None and hasattr(controller, "close"):
        try:
            controller.close()
        except Exception:
            pass
    try:
        setattr(model, "_lulynx_dit_prefetch_controller", None)
    except Exception:
        pass


def install_dit_block_prefetch_controller(
    model: nn.Module,
    blocks: list[nn.Module],
    plan: DitResidencyPlan,
    *,
    enabled: bool,
    depth: int,
    device: torch.device | str | None,
    dtype: torch.dtype | None = None,
    install_hooks: bool = True,
) -> DitBlockPrefetchController | None:
    clear_dit_block_prefetch_controller(model)
    if not enabled:
        return None
    controller = DitBlockPrefetchController(
        blocks,
        plan,
        device=device,
        dtype=dtype,
        depth=depth,
        install_hooks=install_hooks,
    )
    try:
        setattr(model, "_lulynx_dit_prefetch_controller", controller)
    except Exception:
        controller.close()
        return None
    return controller


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
