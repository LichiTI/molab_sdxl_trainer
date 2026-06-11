"""Telemetry execution handler for Lulynx train steps."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .training_step_orchestrator_runtime import build_lulynx_stage_orchestrator_runtime
from .training_step_telemetry_stage import (
    LulynxTrainingStepTelemetryStagePlan,
    build_lulynx_training_step_telemetry_stage_plan,
)


@dataclass(frozen=True)
class LulynxTelemetryExecutionStageExecution:
    telemetry_stage_plan: LulynxTrainingStepTelemetryStagePlan
    completed_trace: Any
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxTelemetryCallbackStageExecution:
    callback_called: bool
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxTelemetryStepInfoStageExecution:
    step_info: dict[str, Any]
    transfer_profile: dict[str, Any]
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxTelemetrySideEffectsStageExecution:
    step_info: dict[str, Any]
    training_loop_runtime: dict[str, Any]
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxTelemetryNoCallbackMaintenanceStageExecution:
    transfer_profile: dict[str, Any]
    orchestrator_runtime: dict[str, Any]


def run_lulynx_telemetry_step_info_stage_handler(
    *,
    lr_scheduler: Any,
    epoch: int,
    step_wall_seconds: float,
    pcgrad_runtime_state: Any,
    b_tier_last_state: Mapping[str, Any] | None,
    step_phase_profiler: Any,
    accumulation_steps: int,
    transfer_profile: Mapping[str, Any] | None,
    validation_info: Mapping[str, Any] | None = None,
    vram_diag_step: bool = False,
    peak_vram_stages: Mapping[str, Any] | None = None,
    vram_forward_mb: float = 0.0,
    vram_backward_mb: float = 0.0,
    vram_optimizer_mb: float = 0.0,
    peak_vram_diagnostics: Mapping[str, Any] | None = None,
    report_fields: Mapping[str, Any] | None = None,
) -> LulynxTelemetryStepInfoStageExecution:
    """Build the step-end telemetry payload without invoking user callbacks."""

    transfer_snapshot = dict(transfer_profile) if isinstance(transfer_profile, Mapping) else {}
    profiler_transfer_profile = transfer_profile if isinstance(transfer_profile, Mapping) else None
    step_info: dict[str, Any] = {
        "lr": _last_scheduler_lr(lr_scheduler),
        "epoch": int(epoch),
        "step_wall_seconds": float(step_wall_seconds or 0.0),
        "pcgrad": pcgrad_runtime_state() if callable(pcgrad_runtime_state) else {},
        "b_tier": dict(b_tier_last_state) if isinstance(b_tier_last_state, Mapping) and b_tier_last_state else None,
    }
    if bool(getattr(step_phase_profiler, "enabled", False)):
        step_info["step_phase_profile"] = step_phase_profiler.snapshot(
            step_wall_seconds=step_wall_seconds,
            accumulation_steps=accumulation_steps,
            transfer_profile=profiler_transfer_profile,
        )
    loss_scheduler_state = getattr(lr_scheduler, "get_loss_aware_state", None)
    if callable(loss_scheduler_state):
        step_info["loss_aware_scheduler"] = loss_scheduler_state()
    if validation_info is not None:
        step_info["validation"] = dict(validation_info) if isinstance(validation_info, Mapping) else validation_info
    if vram_diag_step:
        stage_peaks = dict(peak_vram_stages) if isinstance(peak_vram_stages, Mapping) else {}
        step_info["peak_vram_stages"] = {
            "forward_mb": round(float(stage_peaks.get("forward_mb", vram_forward_mb) or 0.0), 1),
            "backward_mb": round(float(stage_peaks.get("backward_mb", vram_backward_mb) or 0.0), 1),
            "optimizer_mb": round(float(vram_optimizer_mb or 0.0), 1),
        }
        if isinstance(peak_vram_diagnostics, Mapping) and peak_vram_diagnostics:
            step_info["peak_vram_diagnostics"] = dict(peak_vram_diagnostics)
    if transfer_snapshot:
        step_info["data_transfer_profile"] = transfer_snapshot
    for key, value in (report_fields or {}).items():
        if value:
            step_info[str(key)] = dict(value) if isinstance(value, Mapping) else value
    return LulynxTelemetryStepInfoStageExecution(
        step_info=step_info,
        transfer_profile=transfer_snapshot,
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
            status="telemetry_step_info_stage_handler_executed",
            handler_source="existing_training_loop_step_info_packaging_path",
            extra={
                "has_step_phase_profile": "step_phase_profile" in step_info,
                "has_data_transfer_profile": "data_transfer_profile" in step_info,
                "has_peak_vram_stages": "peak_vram_stages" in step_info,
            },
        ),
    )


def run_lulynx_telemetry_no_callback_maintenance_stage_handler(
    *,
    step_wall_seconds: float,
    record_transfer_profile_step: Callable[[float], Mapping[str, Any] | None] | None = None,
    refresh_module_offload_stats: Callable[[], Any] | None = None,
    update_block_swap_profile: Callable[[], Any] | None = None,
    update_precision_swap_observations: Callable[..., Any] | None = None,
    update_vram_smart_sensing_runtime: Callable[..., Mapping[str, Any] | None] | None = None,
) -> LulynxTelemetryNoCallbackMaintenanceStageExecution:
    """Run telemetry maintenance for steps without a user callback."""

    seconds = float(step_wall_seconds or 0.0)
    transfer_profile = {}
    if callable(record_transfer_profile_step):
        value = record_transfer_profile_step(seconds)
        transfer_profile = dict(value) if isinstance(value, Mapping) else {}
    _call_noargs(refresh_module_offload_stats)
    _call_noargs(update_block_swap_profile)
    if callable(update_precision_swap_observations):
        update_precision_swap_observations(seconds)
    if callable(update_vram_smart_sensing_runtime):
        update_vram_smart_sensing_runtime(seconds)
    return LulynxTelemetryNoCallbackMaintenanceStageExecution(
        transfer_profile=transfer_profile,
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
            status="telemetry_no_callback_maintenance_stage_handler_executed",
            handler_source="existing_training_loop_no_callback_telemetry_maintenance_path",
            extra={"has_data_transfer_profile": bool(transfer_profile)},
        ),
    )


def run_lulynx_telemetry_side_effects_stage_handler(
    *,
    step_info: Mapping[str, Any],
    step_wall_seconds: float,
    global_step: int,
    loss: Any,
    advanced_monitoring: bool = False,
    peak_vram_diag_interval: int = 1,
    auditor: Any = None,
    entropy_probe_step: bool = False,
    loss_tracker: Any = None,
    act_drift_step: bool = False,
    act_drift_tracker: Any = None,
    grad_snapshot: Any = None,
    hessian_trace: Any = None,
    hessian_layers: Any = None,
    layer_monitor_info: Mapping[str, Any] | None = None,
    forgetting_probe: Any = None,
    forgetting_probe_interval: int = 1,
    validation_step: Callable[..., Any] | None = None,
    manifold_tracker: Any = None,
    manifold_snapshot_interval: int = 1,
    get_trainable_params: Callable[[], Sequence[Any]] | None = None,
    aggressive_component_residency: bool = False,
    ensure_cpu_resident_components: Callable[[str], Any] | None = None,
    verify_phase_module_states: Callable[[str], Any] | None = None,
    refresh_module_offload_stats: Callable[[], Any] | None = None,
    update_block_swap_profile: Callable[[], Any] | None = None,
    update_precision_swap_observations: Callable[..., Any] | None = None,
    update_vram_smart_sensing_runtime: Callable[..., Mapping[str, Any] | None] | None = None,
    refresh_training_loop_runtime_profile: Callable[[], Mapping[str, Any] | None] | None = None,
    memory_optimization_state: Mapping[str, Any] | None = None,
) -> LulynxTelemetrySideEffectsStageExecution:
    """Run step-end telemetry side effects while preserving loop ordering."""

    info = dict(step_info) if isinstance(step_info, Mapping) else {}

    if bool(advanced_monitoring) and _modulo(global_step, peak_vram_diag_interval):
        from .anima_attention import snapshot_attention_stats

        info["attention_stats"] = snapshot_attention_stats()
    audit_mode = _auditor_mode(auditor)
    if audit_mode is not None:
        info["audit_mode"] = audit_mode
    if bool(entropy_probe_step):
        from .attn_entropy import collect_probe, snapshot_entropy_stats

        entropy_value = collect_probe()
        if entropy_value is not None:
            info["attn_entropy"] = round(float(entropy_value), 4)
            info["attn_entropy_stats"] = snapshot_entropy_stats()
    loss_snapshot = _maybe_snapshot(loss_tracker)
    if loss_snapshot:
        info["loss_modifiers"] = loss_snapshot
    if bool(act_drift_step) and act_drift_tracker is not None:
        if not bool(getattr(act_drift_tracker, "has_baseline", False)):
            _call_noargs(getattr(act_drift_tracker, "capture_baseline", None))
        else:
            compute_drift = getattr(act_drift_tracker, "compute_drift", None)
            if callable(compute_drift):
                info["act_drift"] = compute_drift()
        _call_noargs(getattr(act_drift_tracker, "clear_features", None))
    if grad_snapshot is not None:
        as_dict = getattr(grad_snapshot, "as_dict", None)
        info["grad_stats"] = as_dict() if callable(as_dict) else grad_snapshot
    if hessian_trace is not None:
        info["hessian_trace"] = hessian_trace
    if hessian_layers is not None:
        info["hessian_layers"] = hessian_layers
    if layer_monitor_info:
        info["layer_monitor"] = dict(layer_monitor_info)
    _maybe_run_forgetting_probe(
        info=info,
        forgetting_probe=forgetting_probe,
        interval=forgetting_probe_interval,
        validation_step=validation_step,
        global_step=global_step,
    )
    _maybe_run_manifold_snapshot(
        manifold_tracker=manifold_tracker,
        interval=manifold_snapshot_interval,
        get_trainable_params=get_trainable_params,
        global_step=global_step,
        loss=loss,
    )
    if bool(aggressive_component_residency):
        _call_with_label(ensure_cpu_resident_components, "after_optimizer_step")
        _call_with_label(verify_phase_module_states, "after_optimizer_step")
    _call_noargs(refresh_module_offload_stats)
    _call_noargs(update_block_swap_profile)
    if callable(update_precision_swap_observations):
        update_precision_swap_observations(float(step_wall_seconds or 0.0), info)
    smart_sensing_report = {}
    if callable(update_vram_smart_sensing_runtime):
        smart_sensing_report = dict(update_vram_smart_sensing_runtime(float(step_wall_seconds or 0.0), info) or {})
    if smart_sensing_report:
        info["vram_smart_sensing_runtime"] = smart_sensing_report
    training_loop_runtime = _refresh_runtime(refresh_training_loop_runtime_profile)
    if training_loop_runtime:
        info["training_loop_runtime"] = dict(training_loop_runtime)
    info["memory_optimization"] = dict(memory_optimization_state) if isinstance(memory_optimization_state, Mapping) else {}
    return LulynxTelemetrySideEffectsStageExecution(
        step_info=info,
        training_loop_runtime=training_loop_runtime,
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
            status="telemetry_side_effects_stage_handler_executed",
            handler_source="existing_training_loop_telemetry_side_effects_path",
            extra={
                "has_training_loop_runtime": bool(training_loop_runtime),
                "has_memory_optimization": "memory_optimization" in info,
                "has_vram_smart_sensing_runtime": "vram_smart_sensing_runtime" in info,
            },
        ),
    )


def _last_scheduler_lr(lr_scheduler: Any) -> float:
    if lr_scheduler is None:
        return 0.0
    get_last_lr = getattr(lr_scheduler, "get_last_lr", None)
    if not callable(get_last_lr):
        return 0.0
    last_lr = get_last_lr()
    if not last_lr:
        return 0.0
    return float(last_lr[0])


def run_lulynx_telemetry_execution_stage_handler(
    *,
    trace: Any,
    step_info: Mapping[str, Any],
    step_wall_seconds: float,
) -> LulynxTelemetryExecutionStageExecution:
    """Record telemetry surfaces after the current step collected them."""

    telemetry_stage_plan = build_lulynx_training_step_telemetry_stage_plan(
        step_info=step_info,
        step_wall_seconds=step_wall_seconds,
    )
    completed_trace = trace.mark_completed_stage(
        "telemetry",
        telemetry_stage_plan=telemetry_stage_plan.as_dict(),
    )
    return LulynxTelemetryExecutionStageExecution(
        telemetry_stage_plan=telemetry_stage_plan,
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
                "telemetry",
            ),
            status="telemetry_execution_stage_handler_executed",
            handler_source="existing_training_loop_telemetry_packaging_path",
            stage_plans={"telemetry_stage_plan": telemetry_stage_plan.as_dict()},
            extra={
                "has_training_loop_runtime": telemetry_stage_plan.has_training_loop_runtime,
                "has_step_phase_profile": telemetry_stage_plan.has_step_phase_profile,
                "has_bubble_profile": telemetry_stage_plan.has_bubble_profile,
            },
        ),
    )


def run_lulynx_telemetry_callback_stage_handler(
    *,
    callback: Any,
    global_step: int,
    loss: Any,
    step_info: Mapping[str, Any],
) -> LulynxTelemetryCallbackStageExecution:
    """Run the user-facing step-end callback at the telemetry boundary."""

    callback_called = False
    if callable(callback):
        callback(int(global_step), loss, dict(step_info))
        callback_called = True
    return LulynxTelemetryCallbackStageExecution(
        callback_called=callback_called,
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
            status="telemetry_callback_stage_handler_executed",
            handler_source="existing_training_loop_step_end_callback_path",
            extra={"callback_called": callback_called},
        ),
    )


def _modulo(value: int, interval: int) -> bool:
    denominator = max(int(interval or 1), 1)
    return int(value or 0) % denominator == 0


def _auditor_mode(auditor: Any) -> Any:
    mode = getattr(auditor, "_current_mode", None) if auditor is not None else None
    if mode is None:
        return None
    return getattr(mode, "value", mode)


def _maybe_snapshot(owner: Any) -> Any:
    snapshot = getattr(owner, "snapshot", None) if owner is not None else None
    return snapshot() if callable(snapshot) else None


def _maybe_run_forgetting_probe(
    *,
    info: dict[str, Any],
    forgetting_probe: Any,
    interval: int,
    validation_step: Callable[..., Any] | None,
    global_step: int,
) -> None:
    if forgetting_probe is None or not bool(getattr(forgetting_probe, "has_anchors", False)):
        return
    if not _modulo(global_step, interval):
        return
    probe = getattr(forgetting_probe, "probe", None)
    if not callable(probe):
        return
    snapshot = probe(validation_step, global_step)
    if snapshot is None:
        return
    as_dict = getattr(snapshot, "as_dict", None)
    info["forgetting"] = as_dict() if callable(as_dict) else snapshot


def _maybe_run_manifold_snapshot(
    *,
    manifold_tracker: Any,
    interval: int,
    get_trainable_params: Callable[[], Sequence[Any]] | None,
    global_step: int,
    loss: Any,
) -> None:
    if manifold_tracker is None or not _modulo(global_step, interval) or not callable(get_trainable_params):
        return
    trainable_params = list(get_trainable_params() or [])
    if not trainable_params:
        return
    snapshot = getattr(manifold_tracker, "snapshot", None)
    if callable(snapshot):
        snapshot(global_step, trainable_params, float(loss))


def _call_noargs(callback: Callable[[], Any] | None) -> Any:
    return callback() if callable(callback) else None


def _call_with_label(callback: Callable[[str], Any] | None, label: str) -> Any:
    return callback(label) if callable(callback) else None


def _refresh_runtime(callback: Callable[[], Mapping[str, Any] | None] | None) -> dict[str, Any]:
    value = callback() if callable(callback) else {}
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = [
    "LulynxTelemetryCallbackStageExecution",
    "LulynxTelemetryExecutionStageExecution",
    "LulynxTelemetryNoCallbackMaintenanceStageExecution",
    "LulynxTelemetrySideEffectsStageExecution",
    "LulynxTelemetryStepInfoStageExecution",
    "run_lulynx_telemetry_callback_stage_handler",
    "run_lulynx_telemetry_execution_stage_handler",
    "run_lulynx_telemetry_no_callback_maintenance_stage_handler",
    "run_lulynx_telemetry_side_effects_stage_handler",
    "run_lulynx_telemetry_step_info_stage_handler",
]
