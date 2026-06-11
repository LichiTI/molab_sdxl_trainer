"""Default-off native update executor probe for TurboCore dispatch runtime."""

from __future__ import annotations

from typing import Any, Mapping


def run_native_update_dispatch_executor_probe(
    *,
    execution_plan: Mapping[str, Any] | None = None,
    runtime_context: Mapping[str, Any] | None = None,
    native_executor: Any | None = None,
) -> dict[str, Any]:
    """Call an explicitly supplied diagnostic executor without training mutation.

    This is the first callable slot for a future native optimizer executor.  It
    is deliberately diagnostic-only: even a successful call must not claim to
    replace the PyTorch optimizer step or mutate training parameters.
    """

    plan = _as_dict(execution_plan)
    context = _as_dict(runtime_context)
    diagnostic_enabled = bool(context.get("native_update_diagnostic_executor_call_enabled", False))
    executor_present = callable(native_executor)
    explicit_training_context = bool(
        context.get("native_update_training_dispatch_enabled", False)
        and context.get("training_path_enabled", False)
    )
    preconditions_ready = bool(
        plan.get("diagnostic_executor_preconditions_ready", plan.get("executor_preconditions_ready", False))
    )
    diagnostic_blockers = _strings(plan.get("diagnostic_executor_blocked_reasons"))
    blocked = _preflight_blockers(
        diagnostic_enabled=diagnostic_enabled,
        executor_present=executor_present,
        explicit_training_context=explicit_training_context,
        preconditions_ready=preconditions_ready,
        diagnostic_blockers=diagnostic_blockers,
    )
    if blocked:
        return _payload(
            attempted=False,
            called=False,
            ok=False,
            reason=blocked[0],
            result={},
            blocked_reasons=blocked,
        )

    try:
        result = native_executor(  # type: ignore[misc]
            {
                "schema_version": 1,
                "request": "turbocore_native_update_dispatch_executor_probe_request_v0",
                "execution_plan": dict(plan),
                "runtime_context": dict(context),
                "training_dispatch": False,
                "training_path_enabled": False,
            }
        )
        result_payload = _as_dict(result)
        post_blockers = _post_call_blockers(result_payload)
        return _payload(
            attempted=True,
            called=True,
            ok=bool(result_payload.get("ok", False)) and not post_blockers,
            reason="called" if not post_blockers else post_blockers[0],
            result=result_payload,
            blocked_reasons=post_blockers,
        )
    except Exception as exc:  # pragma: no cover - defensive boundary
        return _payload(
            attempted=True,
            called=True,
            ok=False,
            reason="native_dispatch_executor_probe_error",
            result={"error": f"{type(exc).__name__}: {exc}"},
            blocked_reasons=["native_dispatch_executor_probe_error"],
        )


def _payload(
    *,
    attempted: bool,
    called: bool,
    ok: bool,
    reason: str,
    result: Mapping[str, Any],
    blocked_reasons: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_dispatch_executor_probe_v0",
        "attempted": bool(attempted),
        "called": bool(called),
        "ok": bool(ok),
        "reason": str(reason or ""),
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_step_executed": False,
        "native_kernel_launched": bool(_as_dict(result).get("native_kernel_launched", False)) if called else False,
        "training_parameters_mutated": False,
        "should_call_pytorch_optimizer_step": True,
        "result": dict(result),
        "blocked_reasons": _dedupe(blocked_reasons),
    }


def _preflight_blockers(
    *,
    diagnostic_enabled: bool,
    executor_present: bool,
    explicit_training_context: bool,
    preconditions_ready: bool,
    diagnostic_blockers: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not diagnostic_enabled:
        blocked.append("native_dispatch_diagnostic_executor_call_disabled")
    if not executor_present and explicit_training_context:
        blocked.append("native_dispatch_runtime_executor_missing")
    if diagnostic_enabled and not preconditions_ready:
        blocked.extend(diagnostic_blockers or ["native_dispatch_diagnostic_executor_preconditions_not_ready"])
    return blocked


def _post_call_blockers(result: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    if not bool(result.get("ok", False)):
        blocked.append(str(result.get("reason", "native_dispatch_executor_probe_not_ok") or "native_dispatch_executor_probe_not_ok"))
    if bool(result.get("training_dispatch", False)) or bool(result.get("training_path_enabled", False)):
        blocked.append("native_dispatch_executor_reported_training_path_enabled")
    if bool(result.get("native_step_executed", False)) or bool(result.get("training_parameters_mutated", False)):
        blocked.append("native_dispatch_executor_reported_training_mutation")
    if bool(result.get("should_call_pytorch_optimizer_step", True)) is False:
        blocked.append("native_dispatch_executor_attempted_to_skip_pytorch_optimizer")
    return blocked


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


def run_native_update_training_executor(
    *,
    execution_plan: Mapping[str, Any] | None = None,
    runtime_context: Mapping[str, Any] | None = None,
    native_executor: Any | None = None,
) -> dict[str, Any]:
    """Call an explicitly supplied training executor for native update dispatch."""

    plan = _as_dict(execution_plan)
    context = _as_dict(runtime_context)
    blocked = _training_preflight_blockers(
        execution_allowed=bool(plan.get("execution_allowed", False)),
        executor_present=callable(native_executor),
        training_dispatch_enabled=bool(context.get("native_update_training_dispatch_enabled", False)),
    )
    if blocked:
        return _training_payload(
            attempted=False,
            called=False,
            ok=False,
            reason=blocked[0],
            result={},
            blocked_reasons=blocked,
        )
    try:
        result = native_executor(  # type: ignore[misc]
            {
                "schema_version": 1,
                "request": "turbocore_native_update_training_executor_request_v0",
                "execution_plan": dict(plan),
                "runtime_context": dict(context),
                "training_dispatch": True,
                "training_path_enabled": True,
            }
        )
        result_payload = _as_dict(result)
        post_blockers = _training_post_call_blockers(result_payload)
        return _training_payload(
            attempted=True,
            called=True,
            ok=bool(result_payload.get("ok", False)) and not post_blockers,
            reason="called" if not post_blockers else post_blockers[0],
            result=result_payload,
            blocked_reasons=post_blockers,
        )
    except Exception as exc:  # pragma: no cover - defensive boundary
        return _training_payload(
            attempted=True,
            called=True,
            ok=False,
            reason="native_update_training_executor_error",
            result=_exception_payload(exc),
            blocked_reasons=["native_update_training_executor_error"],
        )


def _training_payload(
    *,
    attempted: bool,
    called: bool,
    ok: bool,
    reason: str,
    result: Mapping[str, Any],
    blocked_reasons: list[str],
) -> dict[str, Any]:
    payload = _as_dict(result)
    executed = bool(payload.get("native_step_executed", False)) if called else False
    return {
        "schema_version": 1,
        "executor": "turbocore_native_update_training_executor_call_v0",
        "attempted": bool(attempted),
        "called": bool(called),
        "ok": bool(ok),
        "reason": str(reason or ""),
        "training_dispatch": True,
        "training_path_enabled": True,
        "native_step_executed": executed,
        "native_kernel_launched": bool(payload.get("native_kernel_launched", False)) if called else False,
        "training_parameters_mutated": bool(payload.get("training_parameters_mutated", False)) if called else False,
        "should_call_pytorch_optimizer_step": bool(payload.get("should_call_pytorch_optimizer_step", True)) if called else True,
        "result": payload,
        "blocked_reasons": _dedupe(blocked_reasons),
    }


def _exception_payload(exc: Exception) -> dict[str, Any]:
    payload: dict[str, Any] = {"error": f"{type(exc).__name__}: {exc}"}
    native_report = _as_dict(getattr(exc, "native_report", {}))
    if native_report:
        payload["native_report"] = native_report
        payload["reason"] = str(native_report.get("reason", "") or "")
        payload["blocked_reasons"] = _strings(native_report.get("blocked_reasons"))
        borrowed = _as_dict(native_report.get("borrowed_stream_policy"))
        launch_evidence = _as_dict(native_report.get("borrowed_stream_launch_evidence"))
        if borrowed:
            payload["borrowed_stream_policy"] = borrowed
        if launch_evidence:
            payload["borrowed_stream_launch_evidence"] = launch_evidence
    return payload


def _training_preflight_blockers(
    *,
    execution_allowed: bool,
    executor_present: bool,
    training_dispatch_enabled: bool,
) -> list[str]:
    blocked: list[str] = []
    if not training_dispatch_enabled:
        blocked.append("native_dispatch_training_path_disabled")
    if not executor_present:
        blocked.append("native_dispatch_runtime_executor_missing")
    if not execution_allowed:
        blocked.append("native_step_execution_disabled")
    return blocked


def _training_post_call_blockers(result: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    if not bool(result.get("ok", False)):
        blocked.append(str(result.get("reason", "native_update_training_executor_not_ok") or "native_update_training_executor_not_ok"))
    if not bool(result.get("training_dispatch", False)) or not bool(result.get("training_path_enabled", False)):
        blocked.append("native_update_training_executor_did_not_report_training_dispatch")
    if not bool(result.get("native_step_executed", False)):
        blocked.append("native_update_training_executor_did_not_execute_step")
    if bool(result.get("training_parameters_mutated", False)) is False:
        blocked.append("native_update_training_executor_did_not_mutate_parameters")
    if bool(result.get("should_call_pytorch_optimizer_step", True)):
        blocked.append("native_update_training_executor_requires_pytorch_optimizer_step")
    return blocked


__all__ = ["run_native_update_dispatch_executor_probe", "run_native_update_training_executor"]
