"""Default-off runtime boundary for TurboCore native update dispatch."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_native_update_dispatch_executor import (
    run_native_update_dispatch_executor_probe,
    run_native_update_training_executor,
)
from core.turbocore_native_update_dispatch_execution import build_native_update_dispatch_execution_plan
from core.turbocore_native_update_recovery_integration import build_native_update_recovery_integration_report


class TurboCoreNativeUpdateDispatchRuntime:
    """Default-off runtime facade for guarded native update dispatch."""

    def __init__(self) -> None:
        self._attempts = 0
        self._native_steps = 0
        self._disabled_for_run = False
        self._disable_reason = ""

    def prepare_step(
        self,
        *,
        step: int,
        arming_report: Mapping[str, Any] | None = None,
        kernel_launch_plan: Mapping[str, Any] | None = None,
        runtime_context: Mapping[str, Any] | None = None,
        native_executor: Any | None = None,
    ) -> dict[str, Any]:
        arming = _as_dict(arming_report)
        launch = _as_dict(kernel_launch_plan)
        context = _as_dict(runtime_context)
        requested = bool(arming.get("previous_request_requested", False))
        if requested:
            self._attempts += 1
        state = self.snapshot()
        execution_plan = build_native_update_dispatch_execution_plan(
            arming_report=arming,
            kernel_launch_plan=launch,
            runtime_context=context,
            runtime_state=state,
            native_executor=native_executor,
        )
        execute_native = bool(execution_plan.get("execution_allowed", False))
        executor_probe = _training_dispatch_skips_diagnostic_probe() if execute_native else run_native_update_dispatch_executor_probe(
            execution_plan=execution_plan,
            runtime_context=context,
            native_executor=native_executor,
        )
        training_executor = run_native_update_training_executor(
            execution_plan=execution_plan,
            runtime_context=context,
            native_executor=native_executor,
        ) if execute_native else {}
        native_step_executed = bool(training_executor.get("native_step_executed", False))
        if native_step_executed:
            self._native_steps += 1
        elif execute_native and training_executor:
            self.disable_for_run(_training_executor_failure_reason(training_executor))
        final_state = self.snapshot()
        blocked = _blocked_reasons(
            arming=arming,
            launch=launch,
            execution_plan=execution_plan,
            executor_probe=executor_probe,
            training_executor=training_executor,
            disabled_for_run=self._disabled_for_run,
        )
        return {
            "schema_version": 1,
            "runtime": "turbocore_native_update_dispatch_runtime_v0",
            "step": int(step),
            "requested": requested,
            "training_dispatch": bool(context.get("native_update_training_dispatch_enabled", False)),
            "training_path_enabled": bool(context.get("training_path_enabled", False) and context.get("native_update_training_dispatch_enabled", False)),
            "runtime_dispatch_available": bool(context.get("native_update_runtime_dispatch_available", context.get("native_update_training_dispatch_enabled", False))),
            "native_step_executed": native_step_executed,
            "native_kernel_launched": bool(training_executor.get("native_kernel_launched", False)) if training_executor else False,
            "should_call_pytorch_optimizer_step": not native_step_executed,
            "should_call_python_scheduler": True,
            "should_zero_grad_with_pytorch": True,
            "fallback_to_pytorch_required": not native_step_executed,
            "state": final_state,
            "evidence": {
                "arming_present": bool(arming),
                "arming_execute_native_step": bool(arming.get("execute_native_step", False)),
                "kernel_launch_plan_present": bool(launch),
                "kernel_launch_allowed": bool(launch.get("launch_allowed", False)),
                "kernel_launch_attempted": bool(launch.get("launch_attempted", False)),
            },
            "execution_plan": execution_plan,
            "executor_probe": executor_probe,
            "training_executor": training_executor,
            "blocked_reasons": _dedupe(blocked),
        }

    def disable_for_run(self, reason: str) -> dict[str, Any]:
        self._disabled_for_run = True
        self._disable_reason = str(reason or "native_dispatch_disabled_for_run")
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "state": "turbocore_native_update_dispatch_runtime_state_v0",
            "training_path_enabled": False,
            "disabled_for_run": bool(self._disabled_for_run),
            "disable_reason": self._disable_reason,
            "native_dispatch_attempts": int(self._attempts),
            "native_steps_executed": int(self._native_steps),
        }

    def observe_recovery_policy(self, recovery_policy: Mapping[str, Any] | None) -> dict[str, Any]:
        recovery = _as_dict(recovery_policy)
        self._observe_recovery(recovery)
        runtime_state = self.snapshot()
        recovery_runtime = _as_dict(recovery.get("runtime"))
        recovery_state = _as_dict(recovery.get("state_safety"))
        integration = build_native_update_recovery_integration_report(
            mode=str(recovery.get("mode", "off") or "off"),
            policy_defined=bool(recovery.get("policy_defined", False)),
            disable_native_update_for_run=bool(recovery.get("disable_native_update_for_run", False)),
            runtime_error_observed=bool(recovery_runtime.get("runtime_error_observed", False)),
            state_mismatch_observed=bool(recovery_state.get("state_mismatch_observed", False)),
            runtime_state=runtime_state,
        )
        if recovery:
            integration["runtime_observation_present"] = True
        return {
            "schema_version": 1,
            "state": "turbocore_native_update_dispatch_runtime_recovery_observation_v0",
            "training_path_enabled": False,
            "recovery_policy_present": bool(recovery),
            "recovery_disable_native_update_for_run": bool(recovery.get("disable_native_update_for_run", False)),
            "disabled_for_run": bool(self._disabled_for_run),
            "disable_reason": self._disable_reason,
            "integration": integration,
        }

    def _observe_recovery(self, recovery: Mapping[str, Any]) -> None:
        if bool(recovery.get("disable_native_update_for_run", False)):
            reason = "recovery_policy_disable_native_update_for_run"
            blocked = recovery.get("blocked_reasons") if isinstance(recovery.get("blocked_reasons"), list) else []
            if blocked:
                reason = str(blocked[0])
            self.disable_for_run(reason)


def _blocked_reasons(
    *,
    arming: Mapping[str, Any],
    launch: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    executor_probe: Mapping[str, Any],
    training_executor: Mapping[str, Any],
    disabled_for_run: bool,
) -> list[str]:
    blocked: list[str] = []
    if disabled_for_run:
        blocked.append("native_dispatch_disabled_for_run")
    if not arming:
        blocked.append("dispatch_arming_report_missing")
    elif not bool(arming.get("armed_for_native_dispatch", False)):
        blocked.append("dispatch_not_armed")
    if not launch:
        blocked.append("kernel_launch_plan_missing")
    elif not bool(launch.get("launch_allowed", False)):
        blocked.append("kernel_launch_not_allowed")
    if not bool(execution_plan.get("execution_allowed", False)):
        blocked.append("native_step_execution_disabled")
    blocked.extend(_strings(execution_plan.get("blocked_reasons")))
    blocked.extend(_strings(executor_probe.get("blocked_reasons")))
    blocked.extend(_strings(training_executor.get("blocked_reasons")))
    if not bool(execution_plan.get("runtime_dispatch_available", False)):
        blocked.append("native_dispatch_runtime_not_implemented")
    if not bool(execution_plan.get("training_dispatch_enabled", False)):
        blocked.append("native_dispatch_training_path_disabled")
    return blocked


def _training_dispatch_skips_diagnostic_probe() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_dispatch_executor_probe_v0",
        "attempted": False,
        "called": False,
        "ok": True,
        "reason": "training_dispatch_uses_training_executor",
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_step_executed": False,
        "native_kernel_launched": False,
        "training_parameters_mutated": False,
        "should_call_pytorch_optimizer_step": True,
        "result": {},
        "blocked_reasons": [],
    }


def _training_executor_failure_reason(report: Mapping[str, Any]) -> str:
    reasons = _strings(report.get("blocked_reasons"))
    if reasons:
        return reasons[0]
    return str(report.get("reason", "") or "native_update_training_executor_failed")


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result



__all__ = ["TurboCoreNativeUpdateDispatchRuntime"]
