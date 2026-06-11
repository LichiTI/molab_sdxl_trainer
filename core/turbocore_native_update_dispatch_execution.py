"""Report-only execution plan for TurboCore native update dispatch."""

from __future__ import annotations

from typing import Any, Mapping


def build_native_update_dispatch_execution_plan(
    *,
    arming_report: Mapping[str, Any] | None = None,
    kernel_launch_plan: Mapping[str, Any] | None = None,
    runtime_context: Mapping[str, Any] | None = None,
    runtime_state: Mapping[str, Any] | None = None,
    native_executor: Any | None = None,
) -> dict[str, Any]:
    """Describe the future native step call boundary without executing it."""

    arming = _as_dict(arming_report)
    launch = _as_dict(kernel_launch_plan)
    context = _as_dict(runtime_context)
    state = _as_dict(runtime_state)
    disabled_for_run = bool(state.get("disabled_for_run", False))
    executor_present = callable(native_executor) or bool(context.get("native_update_executor_present", False))
    execution_guard = bool(context.get("native_update_runtime_execution_guard_enabled", False))
    mutation_guard = bool(context.get("native_update_training_mutation_guard_enabled", False))
    requested_training_path = bool(context.get("training_path_enabled", False))
    training_dispatch_enabled = bool(context.get("native_update_training_dispatch_enabled", False))
    runtime_dispatch_available = bool(context.get("native_update_runtime_dispatch_available", training_dispatch_enabled))
    diagnostic_executor_call = bool(context.get("native_update_diagnostic_executor_call_enabled", False))
    diagnostic_clone_context = bool(context.get("native_update_diagnostic_clone_context_enabled", False))
    armed = bool(arming.get("armed_for_native_dispatch", False) and arming.get("execute_native_step", False))
    launch_allowed = bool(launch.get("launch_allowed", False))
    explicit_training_context = bool(training_dispatch_enabled and requested_training_path)
    training_executor_preconditions_ready = bool(
        armed
        and launch_allowed
        and executor_present
        and execution_guard
        and mutation_guard
        and requested_training_path
        and not disabled_for_run
    )
    execution_allowed = bool(training_executor_preconditions_ready and training_dispatch_enabled and runtime_dispatch_available)
    diagnostic_executor_preconditions_ready = bool(
        armed
        and launch_allowed
        and executor_present
        and execution_guard
        and diagnostic_clone_context
        and not disabled_for_run
    )
    training_blocked = _training_blocked_reasons(
        arming=arming,
        launch=launch,
        disabled_for_run=disabled_for_run,
        explicit_training_context=explicit_training_context,
        executor_present=executor_present,
        execution_guard=execution_guard,
        mutation_guard=mutation_guard,
        requested_training_path=requested_training_path,
        training_dispatch_enabled=training_dispatch_enabled,
        runtime_dispatch_available=runtime_dispatch_available,
    )
    diagnostic_blocked = _diagnostic_blocked_reasons(
        arming=arming,
        launch=launch,
        disabled_for_run=disabled_for_run,
        explicit_training_context=explicit_training_context,
        executor_present=executor_present,
        execution_guard=execution_guard,
        diagnostic_clone_context=diagnostic_clone_context,
    )
    return {
        "schema_version": 1,
        "plan": "turbocore_native_update_dispatch_execution_plan_v0",
        "training_dispatch": training_dispatch_enabled,
        "training_path_enabled": bool(requested_training_path and training_dispatch_enabled),
        "runtime_dispatch_available": runtime_dispatch_available,
        "native_executor_present": executor_present,
        "training_dispatch_enabled": training_dispatch_enabled,
        "explicit_training_context_requested": explicit_training_context,
        "diagnostic_executor_call_enabled": diagnostic_executor_call,
        "diagnostic_clone_context_enabled": diagnostic_clone_context,
        "runtime_execution_guard_enabled": execution_guard,
        "training_mutation_guard_enabled": mutation_guard,
        "requested_training_path_enabled": requested_training_path,
        "armed_for_native_dispatch": armed,
        "kernel_launch_allowed": launch_allowed,
        "executor_preconditions_ready": training_executor_preconditions_ready,
        "training_executor_preconditions_ready": training_executor_preconditions_ready,
        "diagnostic_executor_preconditions_ready": diagnostic_executor_preconditions_ready,
        "execution_allowed": execution_allowed,
        "diagnostic_executor_probe_allowed": bool(diagnostic_executor_preconditions_ready and diagnostic_executor_call),
        "would_call_native_executor": execution_allowed,
        "would_mutate_training_parameters": execution_allowed,
        "should_call_pytorch_optimizer_step": not execution_allowed,
        "training_blocked_reasons": _dedupe(training_blocked),
        "diagnostic_executor_blocked_reasons": _dedupe(diagnostic_blocked),
        "blocked_reasons": _dedupe(training_blocked),
    }


def _base_blocked_reasons(
    *,
    arming: Mapping[str, Any],
    launch: Mapping[str, Any],
    disabled_for_run: bool,
    explicit_training_context: bool,
    executor_present: bool,
    execution_guard: bool,
) -> list[str]:
    blocked: list[str] = []
    if disabled_for_run:
        blocked.append("native_dispatch_disabled_for_run")
    if not arming:
        blocked.append("dispatch_arming_report_missing")
    elif not bool(arming.get("armed_for_native_dispatch", False)):
        blocked.append("dispatch_not_armed")
    if arming and not bool(arming.get("execute_native_step", False)):
        blocked.append("native_step_execution_disabled")
    if not launch:
        blocked.append("kernel_launch_plan_missing")
    elif not bool(launch.get("launch_allowed", False)):
        blocked.append("kernel_launch_not_allowed")
    if not explicit_training_context:
        blocked.append("native_dispatch_training_runtime_executor_default_off")
    elif not executor_present:
        blocked.append("native_dispatch_runtime_executor_missing")
    if explicit_training_context and not execution_guard:
        blocked.append("native_dispatch_runtime_execution_guard_disabled")
    return blocked


def _training_blocked_reasons(
    *,
    arming: Mapping[str, Any],
    launch: Mapping[str, Any],
    disabled_for_run: bool,
    explicit_training_context: bool,
    executor_present: bool,
    execution_guard: bool,
    mutation_guard: bool,
    requested_training_path: bool,
    training_dispatch_enabled: bool,
    runtime_dispatch_available: bool,
) -> list[str]:
    blocked = _base_blocked_reasons(
        arming=arming,
        launch=launch,
        disabled_for_run=disabled_for_run,
        explicit_training_context=explicit_training_context,
        executor_present=executor_present,
        execution_guard=execution_guard,
    )
    if explicit_training_context and not mutation_guard:
        blocked.append("native_dispatch_training_mutation_guard_disabled")
    if not requested_training_path:
        blocked.append("native_dispatch_training_path_default_off")
    if not training_dispatch_enabled:
        blocked.append("native_dispatch_training_path_disabled")
        blocked.append("native_dispatch_runtime_default_off")
    if not runtime_dispatch_available:
        blocked.append("native_dispatch_runtime_not_implemented")
    return blocked


def _diagnostic_blocked_reasons(
    *,
    arming: Mapping[str, Any],
    launch: Mapping[str, Any],
    disabled_for_run: bool,
    explicit_training_context: bool,
    executor_present: bool,
    execution_guard: bool,
    diagnostic_clone_context: bool,
) -> list[str]:
    blocked = _base_blocked_reasons(
        arming=arming,
        launch=launch,
        disabled_for_run=disabled_for_run,
        explicit_training_context=explicit_training_context,
        executor_present=executor_present,
        execution_guard=execution_guard,
    )
    if not diagnostic_clone_context:
        blocked.append("native_dispatch_diagnostic_clone_context_disabled")
    return blocked


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = ["build_native_update_dispatch_execution_plan"]
