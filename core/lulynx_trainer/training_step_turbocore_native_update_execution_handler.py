"""TurboCore native-update execution handlers for Lulynx optimizer stages."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.turbocore_native_update_dispatch_diagnostic_executor import build_shadow_owner_native_diagnostic_executor
from core.turbocore_native_update_dispatch_runtime import TurboCoreNativeUpdateDispatchRuntime
from core.turbocore_native_update_probe_cache import retain_native_update_probe_evidence

from .training_step_orchestrator_runtime import build_lulynx_stage_orchestrator_runtime
from .turbocore_native_update_readiness_adapter import build_native_update_runtime_context_with_shadow_evidence


@dataclass(frozen=True)
class LulynxTurboCoreNativeUpdatePreOptimizerStageExecution:
    previous_gate: dict[str, Any]
    dispatch_arming: dict[str, Any]
    dispatch_runtime_report: dict[str, Any]
    runtime_recovery_observation: dict[str, Any]
    runtime_context: dict[str, Any]
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxTurboCoreNativeUpdatePostOptimizerStageExecution:
    gate_report: dict[str, Any]
    arming_observation: dict[str, Any]
    diagnostic_replay_report: dict[str, Any]
    runtime_context: dict[str, Any]
    orchestrator_runtime: dict[str, Any]


def run_lulynx_turbocore_native_update_pre_optimizer_stage_handler(
    *,
    gate: Any,
    dispatch_armer: Any,
    dispatch_runtime: Any,
    runtime_context: Mapping[str, Any],
    trainable_params: Sequence[Any],
    step: int,
    get_training_executor: Callable[[Sequence[Any]], Any],
    native_update_loop_timing: MutableMapping[str, Any],
    logger: Any,
) -> LulynxTurboCoreNativeUpdatePreOptimizerStageExecution:
    """Prepare native-update arming/runtime reports before optimizer.step."""

    context = dict(runtime_context) if isinstance(runtime_context, Mapping) else {}
    previous_gate = _call_mapping(dispatch_armer, "last_gate_report")
    dispatch_arming: dict[str, Any] = {}
    recovery_observation: dict[str, Any] = {}
    dispatch_runtime_report: dict[str, Any] = {}

    if not bool(getattr(gate, "requested", False)):
        return LulynxTurboCoreNativeUpdatePreOptimizerStageExecution(
            previous_gate=previous_gate,
            dispatch_arming=dispatch_arming,
            dispatch_runtime_report=dispatch_runtime_report,
            runtime_recovery_observation=recovery_observation,
            runtime_context=context,
            orchestrator_runtime=_runtime(
                "turbocore_native_update_pre_optimizer_stage_handler_noop",
                requested=False,
                dispatch_arming_present=False,
                dispatch_runtime_present=False,
            ),
        )

    previous_fallback = dict(previous_gate.get("fallback_policy", {}) or {})
    started = time.perf_counter()
    observe_recovery = getattr(dispatch_runtime, "observe_recovery_policy", None)
    if callable(observe_recovery):
        recovery_observation = dict(observe_recovery(previous_fallback.get("runtime_recovery")) or {})
    native_update_loop_timing["recovery_observe_ms"] = _elapsed_ms(started)

    started = time.perf_counter()
    prepare_before_optimizer = getattr(dispatch_armer, "prepare_before_optimizer", None)
    if callable(prepare_before_optimizer):
        dispatch_arming = dict(
            prepare_before_optimizer(
                step=step,
                runtime_state=_runtime_snapshot(dispatch_runtime),
                runtime_context=context,
            )
            or {}
        )
    native_update_loop_timing["dispatch_arming_ms"] = _elapsed_ms(started)

    native_executor = None
    if bool(context.get("native_update_training_dispatch_enabled", False)):
        started = time.perf_counter()
        try:
            native_executor = get_training_executor(list(trainable_params))
        except Exception as exc:
            context = dict(context)
            context["native_update_executor_present"] = False
            context["native_update_training_executor_error"] = f"{type(exc).__name__}: {exc}"
            _debug(logger, "TurboCore native update training executor unavailable: %s", exc)
        native_update_loop_timing["executor_get_ms"] = _elapsed_ms(started)

    started = time.perf_counter()
    prepare_step = getattr(dispatch_runtime, "prepare_step", None)
    if callable(prepare_step):
        dispatch_runtime_report = dict(
            prepare_step(
                step=step,
                arming_report=dispatch_arming,
                kernel_launch_plan=previous_gate.get("kernel_launch_plan"),
                runtime_context=context,
                native_executor=native_executor,
            )
            or {}
        )
    native_update_loop_timing["dispatch_runtime_prepare_ms"] = _elapsed_ms(started)

    return LulynxTurboCoreNativeUpdatePreOptimizerStageExecution(
        previous_gate=previous_gate,
        dispatch_arming=dispatch_arming,
        dispatch_runtime_report=dispatch_runtime_report,
        runtime_recovery_observation=recovery_observation,
        runtime_context=context,
        orchestrator_runtime=_runtime(
            "turbocore_native_update_pre_optimizer_stage_handler_executed",
            requested=True,
            dispatch_arming_present=bool(dispatch_arming),
            dispatch_runtime_present=bool(dispatch_runtime_report),
        ),
    )


def run_lulynx_turbocore_native_update_post_optimizer_stage_handler(
    *,
    gate: Any,
    dispatch_armer: Any,
    previous_gate: Mapping[str, Any],
    shadow_report: Mapping[str, Any],
    dispatch_runtime_report: Mapping[str, Any],
    runtime_context: Mapping[str, Any],
    optimizer: Any,
    trainable_param_count: int,
    step: int,
    can_retain_gate: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], bool],
    refresh_readiness: Callable[[dict[str, Any]], Mapping[str, Any]],
    diagnostic_executor_replay: bool,
    native_update_loop_timing: MutableMapping[str, Any],
    logger: Any,
) -> LulynxTurboCoreNativeUpdatePostOptimizerStageExecution:
    """Update native-update gate and optional diagnostics after optimizer.step."""

    context = dict(runtime_context) if isinstance(runtime_context, Mapping) else {}
    previous = dict(previous_gate) if isinstance(previous_gate, Mapping) else {}
    shadow = dict(shadow_report) if isinstance(shadow_report, Mapping) else {}
    dispatch_runtime = dict(dispatch_runtime_report) if isinstance(dispatch_runtime_report, Mapping) else {}
    gate_report: dict[str, Any] = {}
    arming_observation: dict[str, Any] = {}
    diagnostic_replay_report: dict[str, Any] = {}

    if not bool(getattr(gate, "requested", False)):
        return LulynxTurboCoreNativeUpdatePostOptimizerStageExecution(
            gate_report=gate_report,
            arming_observation=arming_observation,
            diagnostic_replay_report=diagnostic_replay_report,
            runtime_context=context,
            orchestrator_runtime=_runtime(
                "turbocore_native_update_post_optimizer_stage_handler_noop",
                requested=False,
                gate_report_present=False,
                arming_observation_present=False,
                diagnostic_replay_present=False,
            ),
        )

    started = time.perf_counter()
    if can_retain_gate(previous, shadow, dispatch_runtime):
        gate_report = retain_native_update_probe_evidence(previous, step=int(step))
    else:
        readiness_report = refresh_readiness(shadow)
        context = build_native_update_runtime_context_with_shadow_evidence(context, shadow)
        try:
            update = getattr(gate, "update")
            gate_report = dict(
                update(
                    shadow_report=shadow,
                    optimizer=optimizer,
                    trainable_param_count=int(trainable_param_count),
                    runtime_context=context,
                    readiness_report=dict(readiness_report) if isinstance(readiness_report, Mapping) else {},
                )
                or {}
            )
        except Exception as exc:
            gate_report = _gate_error_report(gate, exc)
            _debug(logger, "TurboCore native update gate skipped: %s", exc)
    native_update_loop_timing["gate_update_ms"] = _elapsed_ms(started)

    started = time.perf_counter()
    observe_after_optimizer = getattr(dispatch_armer, "observe_after_optimizer", None)
    if callable(observe_after_optimizer):
        arming_observation = dict(observe_after_optimizer(gate_report) or {})
    native_update_loop_timing["arming_observe_ms"] = _elapsed_ms(started)

    if diagnostic_executor_replay:
        diagnostic_replay_report = _run_diagnostic_replay(
            shadow_report=shadow,
            runtime_context=context,
            step=int(step),
        )

    return LulynxTurboCoreNativeUpdatePostOptimizerStageExecution(
        gate_report=gate_report,
        arming_observation=arming_observation,
        diagnostic_replay_report=diagnostic_replay_report,
        runtime_context=context,
        orchestrator_runtime=_runtime(
            "turbocore_native_update_post_optimizer_stage_handler_executed",
            requested=True,
            gate_report_present=bool(gate_report),
            arming_observation_present=bool(arming_observation),
            diagnostic_replay_present=bool(diagnostic_replay_report),
        ),
    )


def _runtime(status: str, **extra: Any) -> dict[str, Any]:
    return build_lulynx_stage_orchestrator_runtime(
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
        status=status,
        handler_source="existing_training_loop_turbocore_native_update_pre_optimizer_path",
        extra=extra,
    )


def _call_mapping(owner: Any, method_name: str) -> dict[str, Any]:
    method = getattr(owner, method_name, None)
    if not callable(method):
        return {}
    value = method()
    return dict(value) if isinstance(value, Mapping) else {}


def _runtime_snapshot(dispatch_runtime: Any) -> dict[str, Any]:
    snapshot = getattr(dispatch_runtime, "snapshot", None)
    if not callable(snapshot):
        return {}
    value = snapshot()
    return dict(value) if isinstance(value, Mapping) else {}


def _debug(logger: Any, message: str, *args: Any) -> None:
    debug = getattr(logger, "debug", None)
    if callable(debug):
        debug(message, *args)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 4)


def _gate_error_report(gate: Any, exc: Exception) -> dict[str, Any]:
    config = getattr(gate, "config", None)
    return {
        "schema_version": 1,
        "gate": "turbocore_native_update_gate_v0",
        "mode": getattr(config, "mode", None),
        "requested": True,
        "would_enable_native_update": False,
        "training_path_enabled": False,
        "native_kernel_present": False,
        "performance_test_ready": False,
        "stream_lifetime_bound": False,
        "fallback_policy": {
            "schema_version": 1,
            "policy": "turbocore_native_update_fallback_v0",
            "training_path_enabled": False,
            "actions": ["disable_native_update_due_to_gate_error"],
        },
        "error": f"{type(exc).__name__}: {exc}",
        "blocked_reasons": ["gate_error"],
    }


def _run_diagnostic_replay(
    *,
    shadow_report: Mapping[str, Any],
    runtime_context: Mapping[str, Any],
    step: int,
) -> dict[str, Any]:
    diagnostic_replay_runtime = TurboCoreNativeUpdateDispatchRuntime()
    report = dict(
        diagnostic_replay_runtime.prepare_step(
            step=step,
            arming_report={
                "previous_request_requested": True,
                "armed_for_native_dispatch": True,
                "execute_native_step": True,
            },
            kernel_launch_plan={"launch_allowed": True, "launch_attempted": False},
            runtime_context={
                **dict(runtime_context),
                "training_path_enabled": False,
                "native_update_runtime_execution_guard_enabled": True,
                "native_update_diagnostic_executor_call_enabled": True,
                "native_update_diagnostic_clone_context_enabled": True,
            },
            native_executor=build_shadow_owner_native_diagnostic_executor(dict(shadow_report)),
        )
        or {}
    )
    report["diagnostic_replay"] = True
    report["source"] = "current_step_shadow_owner_native_probe"
    return report


__all__ = [
    "LulynxTurboCoreNativeUpdatePreOptimizerStageExecution",
    "LulynxTurboCoreNativeUpdatePostOptimizerStageExecution",
    "run_lulynx_turbocore_native_update_pre_optimizer_stage_handler",
    "run_lulynx_turbocore_native_update_post_optimizer_stage_handler",
]
