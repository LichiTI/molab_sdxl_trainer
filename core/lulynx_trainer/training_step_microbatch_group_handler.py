"""Microbatch group handler for Lulynx train epochs."""

from __future__ import annotations

import time
from collections.abc import Callable, MutableSequence
from dataclasses import dataclass
from typing import Any

from .training_step_orchestrator_runtime import build_lulynx_stage_orchestrator_runtime


@dataclass(frozen=True)
class LulynxMicrobatchGroupStageExecution:
    current_group_target: int
    accumulation_wall_start: float | None
    closure_microbatches: list[dict[str, Any]]
    data_wait_started: Any
    micro_batch_index: int
    micro_batch_count: int
    sync_gradients: bool
    accumulation_group_start: bool
    before_train_step_callback_called: bool
    before_train_step_callback_error: str
    block_offloader_prepared: bool
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxAccumulationGroupTailStageExecution:
    current_group_target: int
    microbatches_in_group: int
    closure_microbatches: list[dict[str, Any]]
    accumulation_wall_start: float | None
    completed_by_step_limit: bool
    should_stop: bool
    should_break_epoch: bool
    audit_ran: bool
    orchestrator_runtime: dict[str, Any]


def run_lulynx_microbatch_group_stage_handler(
    *,
    batch: dict[str, Any],
    step: int,
    total_microbatches: int,
    current_group_target: int,
    microbatches_in_group: int,
    gradient_accumulation_steps: int,
    dynamic_batch_scheduler: Any,
    step_phase_profiler: Any,
    data_wait_started: Any,
    accumulation_wall_start: float | None,
    closure_microbatches: MutableSequence[dict[str, Any]] | None,
    step_requires_closure: bool,
    capture_optimizer_step_rng_state: Callable[[], Any] | None,
    on_before_train_step: Callable[[int], Any] | None,
    global_step: int,
    block_offloader: Any,
    logger: Any,
) -> LulynxMicrobatchGroupStageExecution:
    """Prepare per-microbatch group state without running the train step."""

    group_target = max(int(current_group_target or 1), 1)
    closure_items = list(closure_microbatches or [])
    group_start = int(microbatches_in_group or 0) == 0
    wall_start = accumulation_wall_start
    if group_start:
        closure_items = []
        wall_start = time.perf_counter()
        _call_noargs(getattr(step_phase_profiler, "reset_group", None))
        if dynamic_batch_scheduler is not None:
            batch_res = dynamic_batch_scheduler.get_batch_resolution(batch)
            group_target = dynamic_batch_scheduler.compute_accumulation_steps(
                batch_res,
                total_microbatches_remaining=max(int(total_microbatches) - int(step), 0),
            )
    record_cpu = getattr(step_phase_profiler, "record_cpu", None)
    next_data_wait_started = (
        record_cpu("data_wait", data_wait_started) if callable(record_cpu) else data_wait_started
    )
    is_boundary = int(microbatches_in_group or 0) + 1 >= group_target or (
        int(step) + 1
    ) == int(total_microbatches)
    micro_index = int(microbatches_in_group or 0) + 1
    if bool(step_requires_closure):
        closure_items.append(
            {
                "batch": batch,
                "rng_state": (
                    capture_optimizer_step_rng_state()
                    if callable(capture_optimizer_step_rng_state)
                    else None
                ),
                "micro_batch_index": micro_index,
                "micro_batch_count": group_target,
                "sync_gradients": bool(is_boundary),
                "accumulation_group_start": bool(group_start),
            }
        )
    callback_called = False
    callback_error = ""
    if callable(on_before_train_step):
        callback_called = True
        try:
            on_before_train_step(int(global_step))
        except Exception as exc:
            callback_error = f"{type(exc).__name__}: {exc}"
            log_debug = getattr(logger, "debug", None)
            if callable(log_debug):
                log_debug("on_before_train_step callback skipped: %s", exc)
    block_prepared = _maybe_prepare_block_offloader(block_offloader)
    return LulynxMicrobatchGroupStageExecution(
        current_group_target=group_target,
        accumulation_wall_start=wall_start,
        closure_microbatches=closure_items,
        data_wait_started=next_data_wait_started,
        micro_batch_index=micro_index,
        micro_batch_count=group_target,
        sync_gradients=bool(is_boundary),
        accumulation_group_start=bool(group_start),
        before_train_step_callback_called=callback_called,
        before_train_step_callback_error=callback_error,
        block_offloader_prepared=block_prepared,
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=("batch_contract",),
            status="microbatch_group_stage_handler_executed",
            handler_source="existing_training_loop_microbatch_group_path",
            extra={
                "micro_batch_index": micro_index,
                "micro_batch_count": group_target,
                "sync_gradients": bool(is_boundary),
                "accumulation_group_start": bool(group_start),
                "closure_microbatch_count": len(closure_items),
                "before_train_step_callback_called": callback_called,
                "before_train_step_callback_error": callback_error,
                "block_offloader_prepared": block_prepared,
            },
        ),
    )


def run_lulynx_accumulation_group_tail_stage_handler(
    *,
    auditor: Any,
    auditor_interval: int,
    run_audit: Callable[[], Any] | None,
    global_step: int,
    total_steps: int,
    step: int,
    total_microbatches: int,
    gradient_accumulation_steps: int,
    current_group_target: int,
    closure_microbatches: MutableSequence[dict[str, Any]] | None,
    accumulation_wall_start: float | None,
) -> LulynxAccumulationGroupTailStageExecution:
    """Run post-telemetry epoch-tail checks and reset accumulation group state."""

    audit_ran = False
    if auditor and int(global_step) % int(auditor_interval) == 0:
        if callable(run_audit):
            run_audit()
            audit_ran = True
    completed_by_step_limit = False
    should_stop = False
    should_break = False
    if int(total_steps or 0) > 0 and int(global_step) >= int(total_steps):
        completed_by_step_limit = True
        should_stop = True
        should_break = True
    if should_break:
        next_group_target = max(int(current_group_target or 1), 1)
        next_microbatches = 0
        next_closure = list(closure_microbatches or [])
        next_wall_start = accumulation_wall_start
    else:
        next_group_target = max(int(current_group_target or 1), 1)
        remaining_after_step = int(total_microbatches) - (int(step) + 1)
        if remaining_after_step > 0:
            target = max(int(gradient_accumulation_steps or 1), 1)
            next_group_target = min(target, remaining_after_step)
        next_microbatches = 0
        next_closure = []
        next_wall_start = None
    return LulynxAccumulationGroupTailStageExecution(
        current_group_target=next_group_target,
        microbatches_in_group=next_microbatches,
        closure_microbatches=next_closure,
        accumulation_wall_start=next_wall_start,
        completed_by_step_limit=completed_by_step_limit,
        should_stop=should_stop,
        should_break_epoch=should_break,
        audit_ran=audit_ran,
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=(
                "batch_contract",
                "host_to_device",
                "conditioning",
                "noise_timestep",
                "forward",
                "loss",
                "backward",
                "optimizer_step",
                "telemetry",
            ),
            status="accumulation_group_tail_stage_handler_executed",
            handler_source="existing_training_loop_accumulation_tail_path",
            extra={
                "audit_ran": audit_ran,
                "completed_by_step_limit": completed_by_step_limit,
                "should_stop": should_stop,
                "should_break_epoch": should_break,
                "next_group_target": next_group_target,
            },
        ),
    )


def _maybe_prepare_block_offloader(block_offloader: Any) -> bool:
    if block_offloader is None or not bool(getattr(block_offloader, "needs_step_prepare", False)):
        return False
    prepare = getattr(block_offloader, "prepare_before_forward", None)
    if not callable(prepare):
        return False
    prepare()
    return True


def _call_noargs(callback: Callable[[], Any] | None) -> Any:
    return callback() if callable(callback) else None


__all__ = [
    "LulynxAccumulationGroupTailStageExecution",
    "LulynxMicrobatchGroupStageExecution",
    "run_lulynx_accumulation_group_tail_stage_handler",
    "run_lulynx_microbatch_group_stage_handler",
]
