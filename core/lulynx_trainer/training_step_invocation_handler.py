"""Train-step invocation handler for Lulynx train epochs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .training_step_orchestrator_runtime import build_lulynx_stage_orchestrator_runtime


@dataclass(frozen=True)
class LulynxTrainStepInvocationStageExecution:
    loss_result: Any
    legacy_signature_fallback_used: bool
    orchestrator_runtime: dict[str, Any]


def run_lulynx_train_step_invocation_stage_handler(
    *,
    train_step: Callable[..., Any],
    batch: dict[str, Any],
    accumulation_steps: int,
    return_loss_tensor: bool,
    step_phase_profiler: Any,
) -> LulynxTrainStepInvocationStageExecution:
    """Invoke the current train_step function and record train_step_total."""

    phase_start = _start_phase(step_phase_profiler)
    fallback_used = False
    try:
        loss_result = train_step(
            batch,
            accumulation_steps=int(accumulation_steps or 1),
            return_loss_tensor=bool(return_loss_tensor),
        )
    except TypeError as exc:
        if "return_loss_tensor" not in str(exc):
            raise
        fallback_used = True
        loss_result = train_step(batch, accumulation_steps=int(accumulation_steps or 1))
    _record_phase(step_phase_profiler, "train_step_total", phase_start)
    return LulynxTrainStepInvocationStageExecution(
        loss_result=loss_result,
        legacy_signature_fallback_used=fallback_used,
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=(
                "batch_contract",
                "host_to_device",
                "conditioning",
                "noise_timestep",
                "forward",
                "loss",
                "backward",
            ),
            status="train_step_invocation_stage_handler_executed",
            handler_source="existing_training_loop_train_step_invocation_path",
            extra={
                "accumulation_steps": int(accumulation_steps or 1),
                "return_loss_tensor": bool(return_loss_tensor),
                "legacy_signature_fallback_used": fallback_used,
            },
        ),
    )


def _start_phase(step_phase_profiler: Any) -> Any:
    start = getattr(step_phase_profiler, "start", None)
    return start() if callable(start) else None


def _record_phase(step_phase_profiler: Any, label: str, started: Any) -> None:
    record = getattr(step_phase_profiler, "record", None)
    if callable(record):
        record(label, started)


__all__ = [
    "LulynxTrainStepInvocationStageExecution",
    "run_lulynx_train_step_invocation_stage_handler",
]
