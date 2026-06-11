"""Epoch lifecycle handlers for Lulynx training loops."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .training_step_housekeeping_handler import (
    run_lulynx_initial_step_skip_housekeeping_stage_handler,
)
from .training_step_orchestrator_runtime import build_lulynx_stage_orchestrator_runtime


@dataclass(frozen=True)
class LulynxEpochIterationGuardStageExecution:
    global_step: int
    completed_by_step_limit: bool
    should_stop: bool
    should_break_epoch: bool
    should_continue_epoch: bool
    data_wait_reset: bool
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxEpochFinalizationStageExecution:
    result: dict[str, Any]
    epoch_callback_called: bool
    native_update_executor_closed: bool
    orchestrator_runtime: dict[str, Any]


def run_lulynx_epoch_iteration_guard_stage_handler(
    *,
    should_stop: bool,
    total_steps: int,
    global_step: int,
    skip_until_initial_step: bool,
    initial_step_target: int,
    lora_injector: Any,
    progress_bar: Any,
    callback: Any,
    epoch: int,
) -> LulynxEpochIterationGuardStageExecution:
    """Run top-of-iteration stop/step-limit/initial-skip guards."""

    step = int(global_step)
    if bool(should_stop):
        return _iteration_guard_execution(
            global_step=step,
            completed_by_step_limit=False,
            should_stop=True,
            should_break_epoch=True,
            should_continue_epoch=False,
            data_wait_reset=False,
            status="epoch_iteration_guard_stop_requested",
        )
    if int(total_steps or 0) > 0 and step >= int(total_steps):
        return _iteration_guard_execution(
            global_step=step,
            completed_by_step_limit=True,
            should_stop=True,
            should_break_epoch=True,
            should_continue_epoch=False,
            data_wait_reset=False,
            status="epoch_iteration_guard_step_limit_reached",
        )
    if bool(skip_until_initial_step) and step < int(initial_step_target):
        skip_execution = run_lulynx_initial_step_skip_housekeeping_stage_handler(
            global_step=step,
            initial_step_target=int(initial_step_target),
            lora_injector=lora_injector,
            progress_bar=progress_bar,
            callback=callback,
            epoch=epoch,
        )
        return LulynxEpochIterationGuardStageExecution(
            global_step=skip_execution.global_step,
            completed_by_step_limit=False,
            should_stop=False,
            should_break_epoch=False,
            should_continue_epoch=True,
            data_wait_reset=True,
            orchestrator_runtime=skip_execution.orchestrator_runtime,
        )
    return _iteration_guard_execution(
        global_step=step,
        completed_by_step_limit=False,
        should_stop=False,
        should_break_epoch=False,
        should_continue_epoch=False,
        data_wait_reset=False,
        status="epoch_iteration_guard_passed",
    )


def run_lulynx_epoch_finalization_stage_handler(
    *,
    total_loss: float,
    num_steps: int,
    epoch: int,
    on_epoch_end: Callable[[int, dict[str, Any]], Any] | None,
    turbocore_native_update_defer_state_sync: bool,
    close_turbocore_native_update_training_executor: Callable[[], Any] | None,
) -> LulynxEpochFinalizationStageExecution:
    """Finalize one train epoch and preserve callback/close ordering."""

    result = {"avg_loss": float(total_loss or 0.0) / max(int(num_steps or 0), 1), "steps": int(num_steps or 0)}
    callback_called = False
    if callable(on_epoch_end):
        on_epoch_end(int(epoch), {"avg_loss": result["avg_loss"]})
        callback_called = True
    executor_closed = False
    if not bool(turbocore_native_update_defer_state_sync) and callable(
        close_turbocore_native_update_training_executor
    ):
        close_turbocore_native_update_training_executor()
        executor_closed = True
    return LulynxEpochFinalizationStageExecution(
        result=result,
        epoch_callback_called=callback_called,
        native_update_executor_closed=executor_closed,
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=("telemetry",),
            status="epoch_finalization_stage_handler_executed",
            handler_source="existing_training_loop_epoch_finalization_path",
            extra={
                "steps": result["steps"],
                "epoch_callback_called": callback_called,
                "native_update_executor_closed": executor_closed,
            },
        ),
    )


def _iteration_guard_execution(
    *,
    global_step: int,
    completed_by_step_limit: bool,
    should_stop: bool,
    should_break_epoch: bool,
    should_continue_epoch: bool,
    data_wait_reset: bool,
    status: str,
) -> LulynxEpochIterationGuardStageExecution:
    return LulynxEpochIterationGuardStageExecution(
        global_step=int(global_step),
        completed_by_step_limit=bool(completed_by_step_limit),
        should_stop=bool(should_stop),
        should_break_epoch=bool(should_break_epoch),
        should_continue_epoch=bool(should_continue_epoch),
        data_wait_reset=bool(data_wait_reset),
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=("batch_contract",) if should_continue_epoch else (),
            status=status,
            handler_source="existing_training_loop_epoch_iteration_guard_path",
            extra={
                "global_step": int(global_step),
                "completed_by_step_limit": bool(completed_by_step_limit),
                "should_stop": bool(should_stop),
                "should_break_epoch": bool(should_break_epoch),
                "should_continue_epoch": bool(should_continue_epoch),
                "data_wait_reset": bool(data_wait_reset),
            },
        ),
    )


__all__ = [
    "LulynxEpochFinalizationStageExecution",
    "LulynxEpochIterationGuardStageExecution",
    "run_lulynx_epoch_finalization_stage_handler",
    "run_lulynx_epoch_iteration_guard_stage_handler",
]
