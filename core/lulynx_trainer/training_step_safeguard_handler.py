"""SafeGuard stage handler for Lulynx train epochs."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, MutableSequence, Sequence
from dataclasses import dataclass
from typing import Any

from .safe_guard import SafeGuardAction
from .training_step_orchestrator_runtime import build_lulynx_stage_orchestrator_runtime


@dataclass(frozen=True)
class LulynxSafeguardStageExecution:
    action: str
    severity: str
    pending_filenames: list[str]
    closure_microbatches: list[dict[str, Any]]
    microbatches_in_group: int
    pending_loss_total: Any
    pending_loss_count: int
    data_wait_started: Any
    should_stop: bool
    should_break_epoch: bool
    should_continue_epoch: bool
    pcgrad_cleared: bool
    optimizer_zero_grad_called: bool
    lr_reduced: bool
    restored: bool
    event_emitted: bool
    orchestrator_runtime: dict[str, Any]


def run_lulynx_safeguard_stage_handler(
    *,
    safeguard: Any,
    is_accumulation_boundary: bool,
    fast_accumulation: bool,
    global_step: int,
    epoch: int,
    loss: Any,
    lr_scheduler: Any,
    batch_filenames: Any,
    pending_filenames: Sequence[str] | None,
    collect_gradients_for_safeguard: Callable[[str], Any] | None,
    normalize_safeguard_gradient_scan_mode: Callable[[Any], str] | None,
    emit_runtime_event: Callable[[Mapping[str, Any]], Any] | None,
    optimizer: Any,
    clear_pcgrad_pending_grads: Callable[[], Any] | None,
    restore_safe_state: Callable[[Any], bool] | None,
    closure_microbatches: MutableSequence[dict[str, Any]] | None,
    microbatches_in_group: int,
    pending_loss_total: Any,
    pending_loss_count: int,
    logger: Any,
) -> LulynxSafeguardStageExecution:
    """Run SafeGuard checks at the accumulation boundary."""

    filenames = [str(item) for item in (pending_filenames or [])]
    closure_items = list(closure_microbatches or [])
    if not ((bool(is_accumulation_boundary) or not bool(fast_accumulation)) and safeguard):
        if bool(is_accumulation_boundary) and bool(fast_accumulation):
            filenames = []
        return _execution(
            action="not_run",
            severity="info",
            pending_filenames=filenames,
            closure_microbatches=closure_items,
            microbatches_in_group=int(microbatches_in_group or 0),
            pending_loss_total=pending_loss_total,
            pending_loss_count=int(pending_loss_count or 0),
            data_wait_started=None,
            should_stop=False,
            should_break_epoch=False,
            should_continue_epoch=False,
            pcgrad_cleared=False,
            optimizer_zero_grad_called=False,
            lr_reduced=False,
            restored=False,
            event_emitted=False,
        )
    try:
        safeguard_filenames = filenames if bool(fast_accumulation) else batch_filenames
        config = getattr(safeguard, "config", None)
        gradient_check_interval = max(
            1,
            int(
                getattr(
                    config,
                    "gradient_check_interval",
                    getattr(config, "nan_check_interval", 1),
                )
                or 1
            ),
        )
        nan_check_interval = max(1, int(getattr(config, "nan_check_interval", 1) or 1))
        gradient_scan_mode = _normalize_gradient_scan_mode(
            normalize_safeguard_gradient_scan_mode,
            getattr(config, "gradient_scan_mode", "batched"),
        )
        gradient_scan_due = (
            bool(getattr(config, "enable_nan_detection", True))
            and int(global_step) % nan_check_interval == 0
            and (not bool(fast_accumulation) or int(global_step) % gradient_check_interval == 0)
        )
        gradients = (
            collect_gradients_for_safeguard(gradient_scan_mode)
            if gradient_scan_mode != "off" and gradient_scan_due and callable(collect_gradients_for_safeguard)
            else None
        )
        action = safeguard.check(
            step=int(global_step),
            loss=loss,
            lr=_last_scheduler_lr(lr_scheduler),
            gradients=gradients,
            filenames=safeguard_filenames,
        )
        event_emitted = _emit_safeguard_event(
            emit_runtime_event=emit_runtime_event,
            safeguard=safeguard,
            action=action,
            global_step=global_step,
            epoch=epoch,
            loss=loss,
            lr=_last_scheduler_lr(lr_scheduler),
            filenames=safeguard_filenames,
        )
        if bool(fast_accumulation):
            filenames = []
        if action == SafeGuardAction.STOP:
            _log(getattr(logger, "error", None), "[SafeGuard] Requested training stop")
            pcgrad_cleared = _call_noargs(clear_pcgrad_pending_grads)
            zero_grad_called = _zero_grad(optimizer)
            return _execution(
                action=action.value,
                severity="warning",
                pending_filenames=filenames,
                closure_microbatches=closure_items,
                microbatches_in_group=int(microbatches_in_group or 0),
                pending_loss_total=pending_loss_total,
                pending_loss_count=int(pending_loss_count or 0),
                data_wait_started=None,
                should_stop=True,
                should_break_epoch=True,
                should_continue_epoch=False,
                pcgrad_cleared=pcgrad_cleared,
                optimizer_zero_grad_called=zero_grad_called,
                lr_reduced=False,
                restored=False,
                event_emitted=event_emitted,
            )
        if action == SafeGuardAction.ROLLBACK:
            rollback_state = safeguard.get_rollback_state()
            rollback_step = rollback_state.get("global_step", global_step) if rollback_state else global_step
            restored = bool(restore_safe_state(rollback_state)) if callable(restore_safe_state) else False
            reduction_factor = _lr_reduction_factor(config)
            _scale_optimizer_lr(optimizer, reduction_factor)
            _emit_rollback_callback_and_log(
                safeguard=safeguard,
                restored=restored,
                rollback_step=rollback_step,
                reduction_factor=reduction_factor,
                logger=logger,
            )
            zero_grad_called = _zero_grad(optimizer)
            pcgrad_cleared = _call_noargs(clear_pcgrad_pending_grads)
            return _execution(
                action=action.value,
                severity="warning",
                pending_filenames=[],
                closure_microbatches=[],
                microbatches_in_group=0,
                pending_loss_total=None,
                pending_loss_count=0,
                data_wait_started=time.perf_counter(),
                should_stop=False,
                should_break_epoch=False,
                should_continue_epoch=True,
                pcgrad_cleared=pcgrad_cleared,
                optimizer_zero_grad_called=zero_grad_called,
                lr_reduced=True,
                restored=restored,
                event_emitted=event_emitted,
            )
        if action == SafeGuardAction.REDUCE_LR:
            reduction_factor = _lr_reduction_factor(config)
            _scale_optimizer_lr(optimizer, reduction_factor)
            _log(
                getattr(logger, "warning", None),
                "[SafeGuard] Reduced LR by factor %.3f due to action %s",
                reduction_factor,
                action.value,
            )
            return _execution(
                action=action.value,
                severity="warning",
                pending_filenames=filenames,
                closure_microbatches=closure_items,
                microbatches_in_group=int(microbatches_in_group or 0),
                pending_loss_total=pending_loss_total,
                pending_loss_count=int(pending_loss_count or 0),
                data_wait_started=None,
                should_stop=False,
                should_break_epoch=False,
                should_continue_epoch=False,
                pcgrad_cleared=False,
                optimizer_zero_grad_called=False,
                lr_reduced=True,
                restored=False,
                event_emitted=event_emitted,
            )
        return _execution(
            action=getattr(action, "value", str(action)),
            severity="info",
            pending_filenames=filenames,
            closure_microbatches=closure_items,
            microbatches_in_group=int(microbatches_in_group or 0),
            pending_loss_total=pending_loss_total,
            pending_loss_count=int(pending_loss_count or 0),
            data_wait_started=None,
            should_stop=False,
            should_break_epoch=False,
            should_continue_epoch=False,
            pcgrad_cleared=False,
            optimizer_zero_grad_called=False,
            lr_reduced=False,
            restored=False,
            event_emitted=event_emitted,
        )
    except (ValueError, TypeError) as exc:
        _log(getattr(logger, "warning", None), f"SafeGuard check skipped (param error): {exc}")
        return _execution(
            action="skipped_param_error",
            severity="warning",
            pending_filenames=filenames,
            closure_microbatches=closure_items,
            microbatches_in_group=int(microbatches_in_group or 0),
            pending_loss_total=pending_loss_total,
            pending_loss_count=int(pending_loss_count or 0),
            data_wait_started=None,
            should_stop=False,
            should_break_epoch=False,
            should_continue_epoch=False,
            pcgrad_cleared=False,
            optimizer_zero_grad_called=False,
            lr_reduced=False,
            restored=False,
            event_emitted=False,
        )
    except Exception as exc:
        log_exception = getattr(logger, "exception", None)
        if callable(log_exception):
            log_exception(f"SafeGuard CRITICAL FAILURE: {exc}")
        raise


def _execution(**values: Any) -> LulynxSafeguardStageExecution:
    return LulynxSafeguardStageExecution(
        **values,
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=("batch_contract", "forward", "loss", "backward"),
            status="safeguard_stage_handler_executed",
            handler_source="existing_training_loop_safeguard_path",
            extra={
                "action": values["action"],
                "severity": values["severity"],
                "should_stop": values["should_stop"],
                "should_continue_epoch": values["should_continue_epoch"],
                "lr_reduced": values["lr_reduced"],
                "event_emitted": values["event_emitted"],
            },
        ),
    )


def _normalize_gradient_scan_mode(callback: Callable[[Any], str] | None, mode: Any) -> str:
    return callback(mode) if callable(callback) else str(mode or "batched")


def _last_scheduler_lr(lr_scheduler: Any) -> float:
    get_last_lr = getattr(lr_scheduler, "get_last_lr", None)
    if not callable(get_last_lr):
        return 0.0
    values = get_last_lr()
    return float(values[0]) if values else 0.0


def _emit_safeguard_event(
    *,
    emit_runtime_event: Callable[[Mapping[str, Any]], Any] | None,
    safeguard: Any,
    action: SafeGuardAction,
    global_step: int,
    epoch: int,
    loss: Any,
    lr: float,
    filenames: Any,
) -> bool:
    if not callable(emit_runtime_event):
        return False
    try:
        stats = dict(getattr(safeguard, "get_stats", lambda: {})() or {})
    except Exception:
        stats = {}
    emit_runtime_event(
        {
            "event_type": "safeguard",
            "step": int(global_step),
            "epoch": int(epoch),
            "severity": "warning" if action != SafeGuardAction.CONTINUE else "info",
            "summary": f"safeguard action={action.value}",
            "data": {
                "action": action.value,
                "loss": float(loss),
                "lr": float(lr),
                "filenames": list(filenames or []),
                "stats": stats,
            },
        }
    )
    return True


def _lr_reduction_factor(config: Any) -> float:
    return max(min(float(getattr(config, "lr_reduction_factor", 0.5)), 1.0), 0.0)


def _scale_optimizer_lr(optimizer: Any, factor: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] *= factor


def _emit_rollback_callback_and_log(
    *,
    safeguard: Any,
    restored: bool,
    rollback_step: Any,
    reduction_factor: float,
    logger: Any,
) -> None:
    config = getattr(safeguard, "config", None)
    if restored:
        on_rollback = getattr(config, "on_rollback", None)
        if callable(on_rollback):
            on_rollback(rollback_step, "auto_recovery")
        _log(
            getattr(logger, "warning", None),
            "[SafeGuard] Rolled back to step %s and reduced LR by factor %.3f",
            rollback_step,
            reduction_factor,
        )
        return
    _log(
        getattr(logger, "warning", None),
        "[SafeGuard] Rollback requested but no safe state was restorable; reduced LR by factor %.3f",
        reduction_factor,
    )


def _zero_grad(optimizer: Any) -> bool:
    if optimizer is None:
        return False
    optimizer.zero_grad(set_to_none=True)
    return True


def _call_noargs(callback: Callable[[], Any] | None) -> bool:
    if not callable(callback):
        return False
    callback()
    return True


def _log(callback: Callable[..., Any] | None, message: str, *args: Any) -> None:
    if callable(callback):
        callback(message, *args)


__all__ = [
    "LulynxSafeguardStageExecution",
    "run_lulynx_safeguard_stage_handler",
]
