"""Optimizer execution handler for Lulynx train steps."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .optimizer_step_contracts import bind_loss_value_closure, bind_step_closure, optimizer_uses_fused_backward
from .training_step_optimizer_stage import (
    LulynxTrainingStepOptimizerStagePlan,
    build_lulynx_training_step_optimizer_stage_plan,
)
from .training_step_orchestrator_runtime import build_lulynx_stage_orchestrator_runtime
from .turbocore_native_update_runtime_profile import build_turbocore_native_update_runtime_profile


@dataclass(frozen=True)
class LulynxOptimizerExecutionStageExecution:
    optimizer_stage_plan: LulynxTrainingStepOptimizerStagePlan
    completed_trace: Any
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxOptimizerFinalizeStageExecution:
    scheduler_step_executed: bool
    zero_grad_called: bool
    gradient_release_sync_called: bool
    gradient_release_after_step_called: bool
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxOptimizerStepRouteStageExecution:
    optimizer_step_executed: bool
    route: str
    grad_snapshot: Any
    native_update_skipped_pytorch_grad_clip: bool
    fallback_grad_clip_ran: bool
    step_closure_used: bool
    pytorch_optimizer_step_called: bool
    optimizer_step_route_report: dict[str, Any]
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxBeforeOptimizerHookStageExecution:
    emit_after_optimizer_step_event: Callable[..., Any] | None
    before_optimizer_event_emitted: bool
    before_optimizer_callback_called: bool
    before_optimizer_callback_error: str
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxAfterOptimizerHookStageExecution:
    after_optimizer_event_emitted: bool
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxTurboCoreNativeUpdateRuntimeProfileStageExecution:
    profile: dict[str, Any]
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxPostOptimizerMaintenanceStageExecution:
    vram_optimizer_mb: float
    peak_vram_diagnostics: dict[str, Any] | None
    precision_swap_offload_report: dict[str, Any]
    cache_release_report: dict[str, Any]
    pcgrad_cleared: bool
    optimizer_vram_captured: bool
    precision_swap_offload_called: bool
    orchestrator_runtime: dict[str, Any]


def run_lulynx_post_optimizer_maintenance_stage_handler(
    *,
    clear_pcgrad_pending_grads: Callable[[], Any] | None,
    vram_diag_step: bool,
    cuda_memory_snapshot: Callable[[], Mapping[str, Any]] | None,
    last_peak_vram_diagnostics: Mapping[str, Any] | None,
    build_peak_vram_diagnostics: Callable[[dict[str, dict[str, float]]], Mapping[str, Any]] | None,
    block_offloader: Any,
    maybe_release_cuda_cache: Callable[..., Mapping[str, Any]] | None,
    global_step: int,
) -> LulynxPostOptimizerMaintenanceStageExecution:
    """Run post-optimizer maintenance that feeds telemetry reports."""

    pcgrad_cleared = False
    if callable(clear_pcgrad_pending_grads):
        clear_pcgrad_pending_grads()
        pcgrad_cleared = True

    optimizer_vram_mb = 0.0
    peak_vram_diagnostics = dict(last_peak_vram_diagnostics) if isinstance(last_peak_vram_diagnostics, Mapping) else None
    optimizer_vram_captured = False
    if bool(vram_diag_step) and callable(cuda_memory_snapshot):
        optimizer_diag = dict(cuda_memory_snapshot() or {})
        optimizer_vram_mb = float(optimizer_diag.get("peak_reserved_mb", 0.0) or 0.0)
        stage_details = dict((peak_vram_diagnostics or {}).get("stages", {}))
        stage_details["optimizer"] = optimizer_diag
        if callable(build_peak_vram_diagnostics):
            peak_vram_diagnostics = dict(build_peak_vram_diagnostics(stage_details) or {})
        optimizer_vram_captured = True

    precision_swap_offload_report: dict[str, Any] = {}
    precision_swap_offload_called = False
    if block_offloader is not None:
        offload_selected = getattr(block_offloader, "offload_selected_after_step", None)
        if callable(offload_selected):
            precision_swap_offload_report = dict(offload_selected() or {})
            precision_swap_offload_called = True

    cache_release_report: dict[str, Any] = {}
    if callable(maybe_release_cuda_cache):
        cache_release_report = dict(maybe_release_cuda_cache("after_optimizer", int(global_step)) or {})

    return LulynxPostOptimizerMaintenanceStageExecution(
        vram_optimizer_mb=optimizer_vram_mb,
        peak_vram_diagnostics=peak_vram_diagnostics,
        precision_swap_offload_report=precision_swap_offload_report,
        cache_release_report=cache_release_report,
        pcgrad_cleared=pcgrad_cleared,
        optimizer_vram_captured=optimizer_vram_captured,
        precision_swap_offload_called=precision_swap_offload_called,
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
            status="post_optimizer_maintenance_stage_handler_executed",
            handler_source="existing_training_loop_post_optimizer_maintenance_path",
            extra={
                "pcgrad_cleared": pcgrad_cleared,
                "optimizer_vram_captured": optimizer_vram_captured,
                "precision_swap_offload_called": precision_swap_offload_called,
                "has_cache_release_report": bool(cache_release_report),
            },
        ),
    )


def run_lulynx_turbocore_native_update_runtime_profile_stage_handler(
    *,
    shadow: Any,
    gate: Any,
    readiness: Mapping[str, Any] | None,
    runtime_context: Mapping[str, Any],
    dispatch_runtime: Any,
    dispatch_armer: Any,
    shadow_report: Mapping[str, Any] | None = None,
    gate_report: Mapping[str, Any] | None = None,
    dispatch_arming: Mapping[str, Any] | None = None,
    dispatch_runtime_report: Mapping[str, Any] | None = None,
    dispatch_recovery: Mapping[str, Any] | None = None,
    diagnostic_replay: Mapping[str, Any] | None = None,
    step: int | None = None,
    memory_optimization_state: MutableMapping[str, Any] | None = None,
    profile_builder: Callable[..., Mapping[str, Any]] = build_turbocore_native_update_runtime_profile,
) -> LulynxTurboCoreNativeUpdateRuntimeProfileStageExecution:
    """Refresh TurboCore native-update runtime profile at the optimizer boundary."""

    try:
        profile = dict(
            profile_builder(
                shadow=shadow,
                gate=gate,
                readiness=readiness,
                runtime_context=runtime_context,
                dispatch_runtime=dispatch_runtime,
                dispatch_armer=dispatch_armer,
                shadow_report=shadow_report,
                gate_report=gate_report,
                dispatch_arming=dispatch_arming,
                dispatch_runtime_report=dispatch_runtime_report,
                dispatch_recovery=dispatch_recovery,
                diagnostic_replay=diagnostic_replay,
                step=step,
            )
            or {}
        )
    except Exception as exc:
        gate_config = getattr(getattr(gate, "config", None), "mode", "off")
        shadow_config = getattr(getattr(shadow, "config", None), "mode", "off")
        profile = {
            "schema_version": 1,
            "profile": "turbocore_native_update_runtime_profile_v0",
            "requested": bool(getattr(gate, "requested", False)),
            "shadow_enabled": bool(getattr(shadow, "enabled", False)),
            "mode": str(gate_config or "off"),
            "shadow_mode": str(shadow_config or "off"),
            "resolved": "profile_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    if profile and isinstance(memory_optimization_state, MutableMapping):
        memory_optimization_state["turbocore_native_update"] = dict(profile)
    return LulynxTurboCoreNativeUpdateRuntimeProfileStageExecution(
        profile=profile,
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
            status="turbocore_native_update_runtime_profile_stage_handler_executed",
            handler_source="existing_training_loop_turbocore_native_update_runtime_profile_path",
            extra={
                "profile_present": bool(profile),
                "requested": bool(profile.get("requested", False)),
                "resolved": str(profile.get("resolved", "") or ""),
            },
        ),
    )


def run_lulynx_before_optimizer_hook_stage_handler(
    *,
    hook_context: Mapping[str, Any],
    global_step: int,
    on_before_optimizer_step: Callable[[int], Any] | None,
    logger: Any,
    hook_api: Mapping[str, Any] | None = None,
) -> LulynxBeforeOptimizerHookStageExecution:
    """Run optimizer-stage before hooks and return the after hook callable."""

    hooks = dict(hook_api) if isinstance(hook_api, Mapping) else _load_optimizer_hook_api()
    emit_before_optimizer_step_event = hooks.get("emit_before_optimizer_step_event")
    emit_after_optimizer_step_event = hooks.get("emit_after_optimizer_step_event")
    before_event_emitted = False
    callback_called = False
    callback_error = ""

    if callable(emit_before_optimizer_step_event):
        emit_before_optimizer_step_event(**dict(hook_context))
        before_event_emitted = True
    if on_before_optimizer_step is not None:
        callback_called = True
        try:
            on_before_optimizer_step(int(global_step))
        except Exception as exc:
            callback_error = f"{type(exc).__name__}: {exc}"
            debug = getattr(logger, "debug", None)
            if callable(debug):
                debug("on_before_optimizer_step callback skipped: %s", exc)

    return LulynxBeforeOptimizerHookStageExecution(
        emit_after_optimizer_step_event=(
            emit_after_optimizer_step_event if callable(emit_after_optimizer_step_event) else None
        ),
        before_optimizer_event_emitted=before_event_emitted,
        before_optimizer_callback_called=callback_called,
        before_optimizer_callback_error=callback_error,
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
            status="before_optimizer_hook_stage_handler_executed",
            handler_source="existing_training_loop_before_optimizer_hook_path",
            extra={
                "before_optimizer_event_emitted": before_event_emitted,
                "before_optimizer_callback_called": callback_called,
                "before_optimizer_callback_error": callback_error,
            },
        ),
    )


def run_lulynx_after_optimizer_hook_stage_handler(
    *,
    hook_context: Mapping[str, Any],
    optimizer_step_executed: bool,
    scheduler_step_executed: bool,
    zero_grad_called: bool,
    emit_after_optimizer_step_event: Callable[..., Any] | None,
) -> LulynxAfterOptimizerHookStageExecution:
    """Run optimizer-stage after hooks after optimizer finalization."""

    after_event_emitted = False
    if callable(emit_after_optimizer_step_event):
        emit_after_optimizer_step_event(
            **dict(hook_context),
            optimizer_step_executed=bool(optimizer_step_executed),
            scheduler_step_executed=bool(scheduler_step_executed),
            zero_grad_called=bool(zero_grad_called),
        )
        after_event_emitted = True
    return LulynxAfterOptimizerHookStageExecution(
        after_optimizer_event_emitted=after_event_emitted,
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
            status="after_optimizer_hook_stage_handler_executed",
            handler_source="existing_training_loop_after_optimizer_hook_path",
            extra={"after_optimizer_event_emitted": after_event_emitted},
        ),
    )


def run_lulynx_optimizer_step_route_stage_handler(
    *,
    optimizer: Any,
    loss: Any,
    gradient_release_manager: Any,
    step_requires_closure: bool,
    closure_microbatches: Sequence[Mapping[str, Any]],
    accumulation_steps: int,
    make_step_closure: Callable[[list[Mapping[str, Any]], int], Callable[[], Any]],
    native_update_runtime: Mapping[str, Any] | None,
    native_update_skipped_pytorch_grad_clip: bool,
    trainable_params: Sequence[Any],
    max_grad_norm: float,
    grad_tracker: Any,
    native_update_loop_timing: MutableMapping[str, Any],
    sync_native_update_training_executor_to_pytorch: Callable[[str], Any],
    step_phase_profiler: Any,
) -> LulynxOptimizerStepRouteStageExecution:
    """Execute or skip the optimizer step according to the current route."""

    native_runtime = native_update_runtime if isinstance(native_update_runtime, Mapping) else {}
    optimizer_step_start = _profiler_start(step_phase_profiler)
    optimizer_step_micro_start = _profiler_start_cpu(step_phase_profiler)
    grad_snapshot = None
    fallback_grad_clip_ran = False
    step_closure_used = False
    pytorch_optimizer_step_called = False
    optimizer_step_executed = False
    route = "standard_pytorch_optimizer_step"

    if optimizer_uses_fused_backward(optimizer):
        optimizer_step_executed = True
        route = "fused_backward_optimizer_route"
        _profiler_record_optimizer_step_micro(
            step_phase_profiler,
            "fused_backward_route",
            optimizer_step_micro_start,
        )
    elif (
        gradient_release_manager is not None
        and not gradient_release_manager.needs_external_optimizer_step
        and not step_requires_closure
    ):
        optimizer_step_executed = True
        route = "gradient_release_internal_optimizer_step"
        _profiler_record_optimizer_step_micro(
            step_phase_profiler,
            "gradient_release_internal_route",
            optimizer_step_micro_start,
        )
    else:
        bind_loss_value_closure(optimizer, loss)
        route_start = _profiler_record_optimizer_step_micro(
            step_phase_profiler,
            "loss_closure_bind",
            optimizer_step_micro_start,
        )
        if step_requires_closure:
            step_closure = make_step_closure(
                list(closure_microbatches),
                max(int(accumulation_steps or 1), 1),
            )
            bind_step_closure(optimizer, step_closure)
            call_start = _profiler_record_optimizer_step_micro(
                step_phase_profiler,
                "closure_build_and_bind",
                route_start,
            )
            optimizer.step(step_closure)
            _profiler_record_optimizer_step_micro(
                step_phase_profiler,
                "pytorch_closure_call",
                call_start,
            )
            step_closure_used = True
            pytorch_optimizer_step_called = True
            route = "closure_optimizer_step"
        elif bool(native_runtime.get("native_step_executed", False)):
            route = "turbocore_native_update_route"
            _profiler_record_optimizer_step_micro(
                step_phase_profiler,
                "native_route_already_executed",
                route_start,
            )
        else:
            if native_update_skipped_pytorch_grad_clip and trainable_params:
                grad_clip_start = _profiler_start_cpu(step_phase_profiler)
                total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                if grad_tracker:
                    grad_snapshot = grad_tracker.update(float(total_norm), trainable_params)
                native_update_skipped_pytorch_grad_clip = False
                fallback_grad_clip_ran = True
                native_update_loop_timing["pytorch_grad_clip_fallback_ms"] = _elapsed_ms(optimizer_step_start)
                route_start = _profiler_record_optimizer_step_micro(
                    step_phase_profiler,
                    "grad_clip_fallback",
                    grad_clip_start,
                )
            sync_start = _profiler_start_cpu(step_phase_profiler)
            sync_native_update_training_executor_to_pytorch("before_pytorch_optimizer_fallback")
            call_start = _profiler_record_optimizer_step_micro(
                step_phase_profiler,
                "sync_before_pytorch_fallback",
                sync_start,
            )
            optimizer.step()
            _profiler_record_optimizer_step_micro(
                step_phase_profiler,
                "pytorch_call",
                call_start,
            )
            pytorch_optimizer_step_called = True
            route = "standard_pytorch_optimizer_step"
        optimizer_step_executed = True

    _profiler_record(step_phase_profiler, "optimizer_step", optimizer_step_start)
    optimizer_step_route_report = {
        "schema_version": 1,
        "report": "lulynx_optimizer_step_route_report_v0",
        "report_only": True,
        "subphase_label": "optimizer_step",
        "route": route,
        "optimizer_step_executed": optimizer_step_executed,
        "pytorch_optimizer_step_called": pytorch_optimizer_step_called,
        "step_closure_used": step_closure_used,
        "fallback_grad_clip_ran": fallback_grad_clip_ran,
        "native_step_executed": bool(native_runtime.get("native_step_executed", False)),
        "gradient_release_internal_step": route == "gradient_release_internal_optimizer_step",
        "timing_source": "step_phase_profile.phases_ms.optimizer_step",
        "runtime_default_change": False,
    }
    return LulynxOptimizerStepRouteStageExecution(
        optimizer_step_executed=optimizer_step_executed,
        route=route,
        grad_snapshot=grad_snapshot,
        native_update_skipped_pytorch_grad_clip=bool(native_update_skipped_pytorch_grad_clip),
        fallback_grad_clip_ran=fallback_grad_clip_ran,
        step_closure_used=step_closure_used,
        pytorch_optimizer_step_called=pytorch_optimizer_step_called,
        optimizer_step_route_report=optimizer_step_route_report,
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
            status="optimizer_step_route_stage_handler_executed",
            handler_source="existing_training_loop_optimizer_step_route_path",
            extra={
                "optimizer_step_executed": optimizer_step_executed,
                "route": route,
                "fallback_grad_clip_ran": fallback_grad_clip_ran,
                "step_closure_used": step_closure_used,
                "pytorch_optimizer_step_called": pytorch_optimizer_step_called,
                "optimizer_step_route_report": optimizer_step_route_report,
            },
        ),
    )


def run_lulynx_optimizer_finalize_stage_handler(
    *,
    optimizer: Any,
    lr_scheduler: Any,
    loss: Any,
    gradient_release_manager: Any,
    step_phase_profiler: Any,
) -> LulynxOptimizerFinalizeStageExecution:
    """Execute scheduler/gradient-release/zero-grad finalization for optimizer stage."""

    scheduler_step_executed = False
    zero_grad_called = False
    gradient_release_sync_called = False
    gradient_release_after_step_called = False

    if lr_scheduler:
        scheduler_step_start = _profiler_start(step_phase_profiler)
        if hasattr(lr_scheduler, "get_loss_aware_state"):
            lr_scheduler.step(loss)
        else:
            lr_scheduler.step()
        scheduler_step_executed = True
        if gradient_release_manager is not None:
            gradient_release_manager.sync_learning_rate()
            gradient_release_sync_called = True
        _profiler_record(step_phase_profiler, "scheduler_step", scheduler_step_start)

    zero_grad_start = _profiler_start(step_phase_profiler)
    if gradient_release_manager is not None and gradient_release_manager.mode == "post_step":
        gradient_release_manager.release_gradients_after_step()
        zero_grad_called = True
        gradient_release_after_step_called = True
    elif gradient_release_manager is not None and gradient_release_manager.mode == "during_backward":
        zero_grad_called = True
    else:
        optimizer.zero_grad(set_to_none=True)
        zero_grad_called = True
    _profiler_record(step_phase_profiler, "zero_grad", zero_grad_start)

    return LulynxOptimizerFinalizeStageExecution(
        scheduler_step_executed=scheduler_step_executed,
        zero_grad_called=zero_grad_called,
        gradient_release_sync_called=gradient_release_sync_called,
        gradient_release_after_step_called=gradient_release_after_step_called,
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
            status="optimizer_finalize_stage_handler_executed",
            handler_source="existing_training_loop_optimizer_finalize_path",
            extra={
                "scheduler_step_executed": scheduler_step_executed,
                "zero_grad_called": zero_grad_called,
                "gradient_release_sync_called": gradient_release_sync_called,
                "gradient_release_after_step_called": gradient_release_after_step_called,
            },
        ),
    )


def run_lulynx_optimizer_execution_stage_handler(
    *,
    optimizer: Any,
    trace: Any,
    gradient_accumulation_steps: int,
    optimizer_step_executed: bool,
    scheduler_step_executed: bool,
    zero_grad_called: bool,
    uses_step_closure: bool,
    uses_fused_backward: bool,
    native_update_runtime: Mapping[str, Any] | None = None,
) -> LulynxOptimizerExecutionStageExecution:
    """Record the already-executed optimizer stage without changing behavior."""

    optimizer_stage_plan = build_lulynx_training_step_optimizer_stage_plan(
        optimizer=optimizer,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimizer_step_executed=optimizer_step_executed,
        scheduler_step_executed=scheduler_step_executed,
        zero_grad_called=zero_grad_called,
        uses_step_closure=uses_step_closure,
        uses_fused_backward=uses_fused_backward,
        native_update_runtime=native_update_runtime,
    )
    completed_trace = trace.mark_completed_stage(
        "optimizer_step",
        optimizer_step_executed=bool(optimizer_step_executed),
        scheduler_step_executed=bool(scheduler_step_executed),
        zero_grad_called=bool(zero_grad_called),
        optimizer_stage_plan=optimizer_stage_plan.as_dict(),
    )
    return LulynxOptimizerExecutionStageExecution(
        optimizer_stage_plan=optimizer_stage_plan,
        completed_trace=completed_trace,
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
            status="optimizer_execution_stage_handler_executed",
            handler_source="existing_training_loop_optimizer_execution_path",
            stage_plans={"optimizer_stage_plan": optimizer_stage_plan.as_dict()},
            extra={
                "optimizer_step_executed": bool(optimizer_step_executed),
                "scheduler_step_executed": bool(scheduler_step_executed),
                "zero_grad_called": bool(zero_grad_called),
            },
        ),
    )


def _profiler_start(step_phase_profiler: Any) -> Any:
    start = getattr(step_phase_profiler, "start", None)
    return start() if callable(start) else None


def _profiler_start_cpu(step_phase_profiler: Any) -> Any:
    start = getattr(step_phase_profiler, "start_cpu", None)
    if callable(start):
        return start()
    return _profiler_start(step_phase_profiler)


def _profiler_record(step_phase_profiler: Any, label: str, started: Any) -> None:
    record = getattr(step_phase_profiler, "record", None)
    if callable(record):
        record(label, started)


def _profiler_record_optimizer_step_micro(
    step_phase_profiler: Any,
    label: str,
    started: Any,
) -> Any:
    record = getattr(step_phase_profiler, "record_optimizer_step_micro_substage", None)
    if callable(record):
        return record(label, started)
    return _profiler_start_cpu(step_phase_profiler)


def _elapsed_ms(started: Any) -> float:
    try:
        import time

        cpu_started = started.cpu_started if hasattr(started, "cpu_started") else started
        return round((time.perf_counter() - float(cpu_started)) * 1000.0, 4)
    except Exception:
        return 0.0


def _load_optimizer_hook_api() -> dict[str, Any]:
    try:
        from core.services.training_hooks import (
            emit_after_optimizer_step_event,
            emit_before_optimizer_step_event,
        )
    except Exception:  # pragma: no cover - optional launcher/plugin surface
        return {}
    return {
        "emit_after_optimizer_step_event": emit_after_optimizer_step_event,
        "emit_before_optimizer_step_event": emit_before_optimizer_step_event,
    }


__all__ = [
    "LulynxAfterOptimizerHookStageExecution",
    "LulynxBeforeOptimizerHookStageExecution",
    "LulynxOptimizerExecutionStageExecution",
    "LulynxOptimizerFinalizeStageExecution",
    "LulynxOptimizerStepRouteStageExecution",
    "LulynxPostOptimizerMaintenanceStageExecution",
    "LulynxTurboCoreNativeUpdateRuntimeProfileStageExecution",
    "run_lulynx_after_optimizer_hook_stage_handler",
    "run_lulynx_before_optimizer_hook_stage_handler",
    "run_lulynx_optimizer_execution_stage_handler",
    "run_lulynx_optimizer_finalize_stage_handler",
    "run_lulynx_optimizer_step_route_stage_handler",
    "run_lulynx_post_optimizer_maintenance_stage_handler",
    "run_lulynx_turbocore_native_update_runtime_profile_stage_handler",
]
