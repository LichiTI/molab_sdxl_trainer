"""Telemetry-stage planning for the Lulynx staged training pipeline.

The planner describes which telemetry surfaces were attached to a completed
training step. It does not collect metrics itself or call user callbacks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


LULYNX_TRAINING_STEP_TELEMETRY_STAGE_PLAN = "lulynx_training_step_telemetry_stage_plan_v0"


@dataclass(frozen=True)
class LulynxTrainingStepTelemetryStagePlan:
    step_wall_seconds: float
    has_step_phase_profile: bool
    has_training_loop_runtime: bool
    has_transfer_profile: bool
    has_vram_smart_sensing: bool
    has_peak_vram_diagnostics: bool
    has_bubble_profile: bool

    @property
    def ok(self) -> bool:
        return self.step_wall_seconds >= 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan": LULYNX_TRAINING_STEP_TELEMETRY_STAGE_PLAN,
            "ok": self.ok,
            "step_wall_seconds": round(float(self.step_wall_seconds), 6),
            "has_step_phase_profile": self.has_step_phase_profile,
            "has_training_loop_runtime": self.has_training_loop_runtime,
            "has_transfer_profile": self.has_transfer_profile,
            "has_vram_smart_sensing": self.has_vram_smart_sensing,
            "has_peak_vram_diagnostics": self.has_peak_vram_diagnostics,
            "has_bubble_profile": self.has_bubble_profile,
        }


def build_lulynx_training_step_telemetry_stage_plan(
    *,
    step_info: Mapping[str, Any],
    step_wall_seconds: float,
) -> LulynxTrainingStepTelemetryStagePlan:
    info = step_info if isinstance(step_info, Mapping) else {}
    training_loop_runtime = info.get("training_loop_runtime")
    return LulynxTrainingStepTelemetryStagePlan(
        step_wall_seconds=max(float(step_wall_seconds or 0.0), 0.0),
        has_step_phase_profile=isinstance(info.get("step_phase_profile"), Mapping),
        has_training_loop_runtime=isinstance(training_loop_runtime, Mapping),
        has_transfer_profile=isinstance(info.get("data_transfer_profile"), Mapping),
        has_vram_smart_sensing=isinstance(info.get("vram_smart_sensing_runtime"), Mapping),
        has_peak_vram_diagnostics=isinstance(info.get("peak_vram_diagnostics"), Mapping),
        has_bubble_profile=_has_bubble_profile(training_loop_runtime),
    )


def _has_bubble_profile(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    bubble = value.get("bubble_runtime_controller")
    if isinstance(bubble, Mapping):
        return True
    phase_profile = value.get("step_phase_bubble_profile")
    return isinstance(phase_profile, Mapping)


__all__ = [
    "LULYNX_TRAINING_STEP_TELEMETRY_STAGE_PLAN",
    "LulynxTrainingStepTelemetryStagePlan",
    "build_lulynx_training_step_telemetry_stage_plan",
]
