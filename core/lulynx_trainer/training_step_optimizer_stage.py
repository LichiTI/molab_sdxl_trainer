"""Optimizer-stage planning for the Lulynx staged training pipeline.

The planner describes the optimizer step route after gradient accumulation. It
does not call optimizer APIs, clip gradients, or synchronize CUDA.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


LULYNX_TRAINING_STEP_OPTIMIZER_STAGE_PLAN = "lulynx_training_step_optimizer_stage_plan_v0"


@dataclass(frozen=True)
class LulynxTrainingStepOptimizerStagePlan:
    optimizer_type: str
    route: str
    gradient_accumulation_steps: int
    optimizer_step_executed: bool
    scheduler_step_executed: bool
    zero_grad_called: bool
    uses_step_closure: bool
    uses_fused_backward: bool
    uses_turbocore_native_update: bool
    optimizer_update_profile_contract: Mapping[str, Any]
    compile_caution_reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return bool(self.optimizer_type)

    @property
    def compile_static_graph_risk(self) -> bool:
        return bool(self.compile_caution_reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan": LULYNX_TRAINING_STEP_OPTIMIZER_STAGE_PLAN,
            "ok": self.ok,
            "optimizer_type": self.optimizer_type,
            "route": self.route,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer_step_executed": self.optimizer_step_executed,
            "scheduler_step_executed": self.scheduler_step_executed,
            "zero_grad_called": self.zero_grad_called,
            "uses_step_closure": self.uses_step_closure,
            "uses_fused_backward": self.uses_fused_backward,
            "uses_turbocore_native_update": self.uses_turbocore_native_update,
            "optimizer_update_profile_contract": dict(self.optimizer_update_profile_contract),
            "compile_static_graph_risk": self.compile_static_graph_risk,
            "compile_caution_reasons": list(self.compile_caution_reasons),
        }


def build_lulynx_training_step_optimizer_stage_plan(
    *,
    optimizer: Any,
    gradient_accumulation_steps: int,
    optimizer_step_executed: bool,
    scheduler_step_executed: bool,
    zero_grad_called: bool,
    uses_step_closure: bool = False,
    uses_fused_backward: bool = False,
    native_update_runtime: Mapping[str, Any] | None = None,
) -> LulynxTrainingStepOptimizerStagePlan:
    native_runtime = native_update_runtime if isinstance(native_update_runtime, Mapping) else {}
    uses_native_update = _uses_native_update_runtime(native_runtime)
    cautions: list[str] = []
    if bool(uses_step_closure):
        cautions.append("optimizer_step_closure_replays_accumulated_microbatches")
    if bool(uses_fused_backward):
        cautions.append("optimizer_uses_fused_backward_route")
    if uses_native_update:
        cautions.append("turbocore_native_update_changes_optimizer_route")
    return LulynxTrainingStepOptimizerStagePlan(
        optimizer_type=type(optimizer).__name__ if optimizer is not None else "",
        route=_resolve_route(
            optimizer_step_executed=optimizer_step_executed,
            uses_step_closure=uses_step_closure,
            uses_fused_backward=uses_fused_backward,
            uses_native_update=uses_native_update,
        ),
        gradient_accumulation_steps=max(int(gradient_accumulation_steps or 1), 1),
        optimizer_step_executed=bool(optimizer_step_executed),
        scheduler_step_executed=bool(scheduler_step_executed),
        zero_grad_called=bool(zero_grad_called),
        uses_step_closure=bool(uses_step_closure),
        uses_fused_backward=bool(uses_fused_backward),
        uses_turbocore_native_update=uses_native_update,
        optimizer_update_profile_contract=_build_optimizer_update_profile_contract(
            optimizer_step_executed=optimizer_step_executed,
            scheduler_step_executed=scheduler_step_executed,
            zero_grad_called=zero_grad_called,
        ),
        compile_caution_reasons=tuple(_dedupe(cautions)),
    )


def _resolve_route(
    *,
    optimizer_step_executed: bool,
    uses_step_closure: bool,
    uses_fused_backward: bool,
    uses_native_update: bool,
) -> str:
    if uses_fused_backward:
        return "fused_backward_optimizer_route"
    if uses_native_update:
        return "turbocore_native_update_route"
    if uses_step_closure:
        return "closure_optimizer_step"
    if optimizer_step_executed:
        return "standard_optimizer_step"
    return "optimizer_step_skipped"


def _uses_native_update_runtime(native_runtime: Mapping[str, Any]) -> bool:
    return bool(
        native_runtime.get("native_step_executed")
        or native_runtime.get("native_update_training_dispatch_enabled")
        or native_runtime.get("training_dispatch")
        or native_runtime.get("training_path_enabled")
    )


def _build_optimizer_update_profile_contract(
    *,
    optimizer_step_executed: bool,
    scheduler_step_executed: bool,
    zero_grad_called: bool,
) -> dict[str, Any]:
    subphase_labels: list[str] = []
    if optimizer_step_executed:
        subphase_labels.append("optimizer_step")
    if scheduler_step_executed:
        subphase_labels.append("scheduler_step")
    if zero_grad_called:
        subphase_labels.append("zero_grad")
    return {
        "schema_version": 1,
        "contract": "lulynx_optimizer_update_profile_contract_v0",
        "contract_only": True,
        "outer_phase_label": "optimizer_update_total",
        "subphase_labels": subphase_labels,
        "runtime_default_change": False,
        "guarded_tail_validation_ready": bool(subphase_labels),
    }


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


__all__ = [
    "LULYNX_TRAINING_STEP_OPTIMIZER_STAGE_PLAN",
    "LulynxTrainingStepOptimizerStagePlan",
    "build_lulynx_training_step_optimizer_stage_plan",
]
