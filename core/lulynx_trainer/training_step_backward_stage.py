"""Backward-stage planning for the Lulynx train-step pipeline.

The planner describes the backward route and gradient side effects. It does not
call ``loss.backward()``, inspect gradients, or modify optimizer state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LULYNX_TRAINING_STEP_BACKWARD_STAGE_PLAN = "lulynx_training_step_backward_stage_plan_v0"


@dataclass(frozen=True)
class LulynxTrainingStepBackwardStagePlan:
    route: str
    do_backward: bool
    sync_gradients: bool
    gradient_accumulation_steps: int
    uses_step_closure: bool
    uses_fused_backward: bool
    uses_gradient_release: bool
    gradient_release_mode: str
    create_graph_backward: bool
    compile_caution_reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return bool(self.route)

    @property
    def compile_static_graph_risk(self) -> bool:
        return bool(self.compile_caution_reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan": LULYNX_TRAINING_STEP_BACKWARD_STAGE_PLAN,
            "ok": self.ok,
            "route": self.route,
            "do_backward": self.do_backward,
            "sync_gradients": self.sync_gradients,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "uses_step_closure": self.uses_step_closure,
            "uses_fused_backward": self.uses_fused_backward,
            "uses_gradient_release": self.uses_gradient_release,
            "gradient_release_mode": self.gradient_release_mode,
            "create_graph_backward": self.create_graph_backward,
            "compile_static_graph_risk": self.compile_static_graph_risk,
            "compile_caution_reasons": list(self.compile_caution_reasons),
        }


def build_lulynx_training_step_backward_stage_plan(
    *,
    do_backward: bool,
    sync_gradients: bool,
    gradient_accumulation_steps: int,
    uses_step_closure: bool = False,
    uses_fused_backward: bool = False,
    gradient_release_mode: str = "",
    create_graph_backward: bool = False,
) -> LulynxTrainingStepBackwardStagePlan:
    release_mode = str(gradient_release_mode or "")
    uses_release = bool(release_mode)
    cautions: list[str] = []
    if uses_step_closure:
        cautions.append("optimizer_step_closure_defers_or_replays_backward")
    if uses_fused_backward:
        cautions.append("optimizer_uses_fused_backward_route")
    if uses_release:
        cautions.append(f"gradient_release_mode:{release_mode}")
    if create_graph_backward:
        cautions.append("create_graph_backward_requested")
    return LulynxTrainingStepBackwardStagePlan(
        route=_resolve_route(
            do_backward=bool(do_backward),
            uses_step_closure=bool(uses_step_closure),
            uses_fused_backward=bool(uses_fused_backward),
        ),
        do_backward=bool(do_backward),
        sync_gradients=bool(sync_gradients),
        gradient_accumulation_steps=max(int(gradient_accumulation_steps or 1), 1),
        uses_step_closure=bool(uses_step_closure),
        uses_fused_backward=bool(uses_fused_backward),
        uses_gradient_release=uses_release,
        gradient_release_mode=release_mode,
        create_graph_backward=bool(create_graph_backward),
        compile_caution_reasons=tuple(_dedupe(cautions)),
    )


def _resolve_route(*, do_backward: bool, uses_step_closure: bool, uses_fused_backward: bool) -> str:
    if not do_backward:
        return "no_backward_validation_or_probe"
    if uses_fused_backward:
        return "fused_backward"
    if uses_step_closure:
        return "closure_backward"
    return "standard_loss_backward"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


__all__ = [
    "LULYNX_TRAINING_STEP_BACKWARD_STAGE_PLAN",
    "LulynxTrainingStepBackwardStagePlan",
    "build_lulynx_training_step_backward_stage_plan",
]
