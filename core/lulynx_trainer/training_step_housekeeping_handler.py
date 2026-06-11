"""Post-optimizer housekeeping handler for Lulynx train steps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .training_step_orchestrator_runtime import build_lulynx_stage_orchestrator_runtime


@dataclass(frozen=True)
class LulynxPostOptimizerHousekeepingStageExecution:
    global_step: int
    validation_info: dict[str, Any] | None
    text_encoder_removed: bool
    lora_step_synced: bool
    drift_checked: bool
    safe_state_saved: bool
    progress_updated: bool
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxInitialStepSkipHousekeepingStageExecution:
    global_step: int
    lora_step_synced: bool
    progress_updated: bool
    callback_called: bool
    orchestrator_runtime: dict[str, Any]


def run_lulynx_initial_step_skip_housekeeping_stage_handler(
    *,
    global_step: int,
    initial_step_target: int,
    lora_injector: Any,
    progress_bar: Any,
    callback: Any,
    epoch: int,
) -> LulynxInitialStepSkipHousekeepingStageExecution:
    """Run housekeeping for skipped initial steps."""

    next_step = int(global_step) + 1
    lora_synced = _sync_lora_step(lora_injector, next_step)
    progress_updated = _update_skip_progress(progress_bar, initial_step_target=initial_step_target)
    callback_called = False
    if callable(callback):
        callback(next_step, 0.0, {"lr": 0.0, "epoch": int(epoch), "skipped": True})
        callback_called = True
    return LulynxInitialStepSkipHousekeepingStageExecution(
        global_step=next_step,
        lora_step_synced=lora_synced,
        progress_updated=progress_updated,
        callback_called=callback_called,
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=("batch_contract",),
            status="initial_step_skip_housekeeping_stage_handler_executed",
            handler_source="existing_training_loop_initial_step_skip_path",
            extra={
                "global_step": next_step,
                "initial_step_target": int(initial_step_target),
                "lora_step_synced": lora_synced,
                "progress_updated": progress_updated,
                "callback_called": callback_called,
            },
        ),
    )


def run_lulynx_post_optimizer_housekeeping_stage_handler(
    *,
    te_manager: Any,
    text_encoder_1: Any,
    text_encoder_2: Any,
    optimizer: Any,
    train_text_encoder_any: bool,
    step_phase_profiler: Any,
    update_phase_start: Any,
    global_step: int,
    lora_injector: Any,
    drift_monitor: Any,
    drift_check_interval: int,
    unet: Any,
    maybe_save_safe_state: Callable[[], Any] | None,
    progress_bar: Any,
    loss: Any,
    lr_scheduler: Any,
    validation_dataloader: Any,
    eval_every_n_steps: int,
    validate_epoch: Callable[[Any, int], Mapping[str, Any]] | None,
    epoch: int,
    logger: Any,
) -> LulynxPostOptimizerHousekeepingStageExecution:
    """Run post-optimizer housekeeping while preserving TrainingLoop ordering."""

    pre_record_start = _start_cpu(step_phase_profiler)
    removed = _maybe_remove_text_encoders(
        te_manager=te_manager,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        optimizer=optimizer,
        train_text_encoder_any=bool(train_text_encoder_any),
        global_step=int(global_step),
        logger=logger,
    )
    _record_optimizer_update_substage(
        step_phase_profiler,
        "optimizer_update_housekeeping_pre_record",
        pre_record_start,
    )
    _record_phase(step_phase_profiler, "optimizer_update_total", update_phase_start)

    next_step = int(global_step) + 1
    lora_synced = _sync_lora_step(lora_injector, next_step)
    drift_checked = _maybe_check_drift(drift_monitor, drift_check_interval, next_step, unet)
    safe_saved = False
    if callable(maybe_save_safe_state):
        maybe_save_safe_state()
        safe_saved = True
    progress_updated = _update_progress(progress_bar, loss=loss, lr_scheduler=lr_scheduler)
    validation_info = _maybe_validate_step(
        validation_dataloader=validation_dataloader,
        eval_every_n_steps=int(eval_every_n_steps or 0),
        global_step=next_step,
        validate_epoch=validate_epoch,
        epoch=int(epoch),
        logger=logger,
    )
    return LulynxPostOptimizerHousekeepingStageExecution(
        global_step=next_step,
        validation_info=validation_info,
        text_encoder_removed=removed,
        lora_step_synced=lora_synced,
        drift_checked=drift_checked,
        safe_state_saved=safe_saved,
        progress_updated=progress_updated,
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
            ),
            status="post_optimizer_housekeeping_stage_handler_executed",
            handler_source="existing_training_loop_post_optimizer_housekeeping_path",
            extra={
                "global_step": next_step,
                "text_encoder_removed": removed,
                "lora_step_synced": lora_synced,
                "drift_checked": drift_checked,
                "safe_state_saved": safe_saved,
                "progress_updated": progress_updated,
                "validation_present": validation_info is not None,
            },
        ),
    )


def _maybe_remove_text_encoders(
    *,
    te_manager: Any,
    text_encoder_1: Any,
    text_encoder_2: Any,
    optimizer: Any,
    train_text_encoder_any: bool,
    global_step: int,
    logger: Any,
) -> bool:
    remove = getattr(te_manager, "maybe_remove_text_encoders", None)
    if not (te_manager and text_encoder_1 is not None and callable(remove) and train_text_encoder_any):
        return False
    removed = bool(remove(text_encoder_1, text_encoder_2, optimizer))
    if removed:
        info = getattr(logger, "info", None)
        if callable(info):
            info("[TrainingLoop] Text encoders offloaded to CPU at step %d", global_step)
    return removed


def _record_phase(step_phase_profiler: Any, label: str, started: Any) -> None:
    record = getattr(step_phase_profiler, "record", None)
    if callable(record):
        record(label, started)


def _start_cpu(step_phase_profiler: Any) -> Any:
    start_cpu = getattr(step_phase_profiler, "start_cpu", None)
    if callable(start_cpu):
        return start_cpu()
    start = getattr(step_phase_profiler, "start", None)
    return start() if callable(start) else None


def _record_optimizer_update_substage(
    step_phase_profiler: Any,
    label: str,
    started: Any,
) -> None:
    record = getattr(step_phase_profiler, "record_optimizer_update_substage", None)
    if callable(record):
        record(label, started)


def _sync_lora_step(lora_injector: Any, global_step: int) -> bool:
    set_global_step = getattr(lora_injector, "set_global_step", None)
    if not callable(set_global_step):
        return False
    set_global_step(global_step)
    return True


def _maybe_check_drift(drift_monitor: Any, interval: int, global_step: int, unet: Any) -> bool:
    if drift_monitor is None:
        return False
    denominator = max(int(interval or 1), 1)
    if int(global_step) % denominator != 0:
        return False
    check_drift = getattr(drift_monitor, "check_drift", None)
    if not callable(check_drift):
        return False
    check_drift(unet)
    return True


def _update_progress(progress_bar: Any, *, loss: Any, lr_scheduler: Any) -> bool:
    set_postfix = getattr(progress_bar, "set_postfix", None)
    if not callable(set_postfix):
        return False
    set_postfix(
        {
            "loss": f"{loss:.4f}",
            "lr": f"{_last_scheduler_lr(lr_scheduler):.2e}" if lr_scheduler else "0.00e+00",
        }
    )
    return True


def _update_skip_progress(progress_bar: Any, *, initial_step_target: int) -> bool:
    set_postfix = getattr(progress_bar, "set_postfix", None)
    if not callable(set_postfix):
        return False
    set_postfix({"skip": f"until step {int(initial_step_target)}"})
    return True


def _last_scheduler_lr(lr_scheduler: Any) -> float:
    get_last_lr = getattr(lr_scheduler, "get_last_lr", None)
    if not callable(get_last_lr):
        return 0.0
    values = get_last_lr()
    if not values:
        return 0.0
    return float(values[0])


def _maybe_validate_step(
    *,
    validation_dataloader: Any,
    eval_every_n_steps: int,
    global_step: int,
    validate_epoch: Callable[[Any, int], Mapping[str, Any]] | None,
    epoch: int,
    logger: Any,
) -> dict[str, Any] | None:
    if not (
        validation_dataloader is not None
        and eval_every_n_steps > 0
        and global_step > 0
        and global_step % eval_every_n_steps == 0
        and callable(validate_epoch)
    ):
        return None
    try:
        val_result = validate_epoch(validation_dataloader, epoch)
        info = {
            "avg_loss": float(val_result.get("avg_loss", 0.0) or 0.0),
            "steps": int(val_result.get("steps", 0) or 0),
            "trigger": "step",
        }
        log_info = getattr(logger, "info", None)
        if callable(log_info):
            log_info(
                "Validation step %d: avg_loss=%.4f (%d steps)",
                global_step,
                info["avg_loss"],
                info["steps"],
            )
        return info
    except Exception as exc:
        log_warning = getattr(logger, "warning", None)
        if callable(log_warning):
            log_warning("Validation failed at step %d: %s", global_step, exc)
        return {"error": str(exc), "trigger": "step"}


__all__ = [
    "LulynxInitialStepSkipHousekeepingStageExecution",
    "LulynxPostOptimizerHousekeepingStageExecution",
    "run_lulynx_initial_step_skip_housekeeping_stage_handler",
    "run_lulynx_post_optimizer_housekeeping_stage_handler",
]
