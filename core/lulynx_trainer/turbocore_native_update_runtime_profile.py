"""Compact TurboCore native-update runtime profile for loop state and manifests."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def build_turbocore_native_update_runtime_profile(
    *,
    shadow: Any,
    gate: Any,
    readiness: Mapping[str, Any] | None = None,
    runtime_context: Mapping[str, Any] | None = None,
    dispatch_runtime: Any | None = None,
    dispatch_armer: Any | None = None,
    shadow_report: Mapping[str, Any] | None = None,
    gate_report: Mapping[str, Any] | None = None,
    dispatch_arming: Mapping[str, Any] | None = None,
    dispatch_runtime_report: Mapping[str, Any] | None = None,
    dispatch_recovery: Mapping[str, Any] | None = None,
    diagnostic_replay: Mapping[str, Any] | None = None,
    step: int | None = None,
) -> Dict[str, Any]:
    gate_config = _config_dict(getattr(gate, "config", None))
    shadow_config = _config_dict(getattr(shadow, "config", None))
    requested = bool(getattr(gate, "requested", False))
    shadow_enabled = bool(getattr(shadow, "enabled", False))
    if not requested and not shadow_enabled:
        return {}

    context = _as_dict(runtime_context)
    readiness_summary = _readiness_summary(readiness)
    gate_summary = _gate_summary(gate_report)
    shadow_summary = _shadow_summary(shadow_report, shadow_enabled=shadow_enabled, shadow_config=shadow_config)
    arming_summary = _arming_summary(dispatch_arming)
    dispatch_summary = _dispatch_runtime_summary(dispatch_runtime_report)
    recovery_summary = _recovery_summary(dispatch_recovery)
    diagnostic_summary = _diagnostic_replay_summary(diagnostic_replay)
    runtime_state = _snapshot(dispatch_runtime)
    arming_state = _snapshot(dispatch_armer)

    native_step_executed = bool(dispatch_summary.get("native_step_executed", False))
    would_enable = bool(gate_summary.get("would_enable_native_update", False))
    if native_step_executed:
        resolved = "native_training_step"
    elif would_enable:
        resolved = "gate_ready_profile"
    elif requested:
        resolved = "profile_only"
    elif shadow_enabled:
        resolved = "shadow_only"
    else:
        resolved = "off"

    blocked_reasons = _dedupe(
        _strings(readiness_summary.get("blocked_reasons"))
        + _strings(gate_summary.get("blocked_reasons"))
        + _strings(shadow_summary.get("blocked_reasons"))
        + _strings(arming_summary.get("blocked_reasons"))
        + _strings(dispatch_summary.get("blocked_reasons"))
    )

    profile: Dict[str, Any] = {
        "schema_version": 1,
        "profile": "turbocore_native_update_runtime_profile_v0",
        "requested": requested,
        "resolved": resolved,
        "mode": str(gate_config.get("mode", "off") or "off"),
        "shadow_enabled": shadow_enabled,
        "shadow_mode": str(shadow_config.get("mode", "off") or "off"),
        "dispatch_enabled": bool(gate_config.get("dispatch_enabled", False)),
        "training_path_requested": bool(context.get("training_path_enabled", False)),
        "training_dispatch_enabled": bool(context.get("native_update_training_dispatch_enabled", False)),
        "runtime_dispatch_available": bool(context.get("native_update_runtime_dispatch_available", False)),
        "native_step_executed": native_step_executed,
        "native_kernel_launched": bool(dispatch_summary.get("native_kernel_launched", False)),
        "fallback_to_pytorch_required": bool(dispatch_summary.get("fallback_to_pytorch_required", not native_step_executed)),
        "blocked_reasons": blocked_reasons,
        "readiness": readiness_summary,
        "shadow": shadow_summary,
        "gate": gate_summary,
        "dispatch_arming": arming_summary,
        "dispatch_runtime": dispatch_summary,
        "dispatch_recovery": recovery_summary,
        "runtime_state": runtime_state,
        "arming_state": arming_state,
    }
    if diagnostic_summary:
        profile["diagnostic_replay"] = diagnostic_summary
    if step is not None:
        profile["step"] = int(step)
    executor_error = str(context.get("native_update_training_executor_error", "") or "")
    if executor_error:
        profile["training_executor_error"] = executor_error
    return profile


def _readiness_summary(value: Mapping[str, Any] | None) -> Dict[str, Any]:
    report = _as_dict(value)
    if not report:
        return {"present": False, "ok": False, "blocked_reasons": []}
    return {
        "present": True,
        "ok": bool(report.get("ok", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_kernel_present": bool(report.get("native_kernel_present", False)),
        "performance_test_ready": bool(report.get("performance_test_ready", False)),
        "stream_lifetime_bound": bool(report.get("stream_lifetime_bound", False)),
        "stream_ordering_verified": bool(report.get("stream_ordering_verified", False)),
        "event_chain_verified": bool(report.get("event_chain_verified", False)),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _gate_summary(value: Mapping[str, Any] | None) -> Dict[str, Any]:
    report = _as_dict(value)
    if not report:
        return {"present": False, "would_enable_native_update": False, "blocked_reasons": []}
    request = _as_dict(report.get("dispatch_request"))
    launch = _as_dict(report.get("kernel_launch_plan"))
    fallback = _as_dict(report.get("fallback_policy"))
    probe_cache = _as_dict(report.get("probe_cache_retention"))
    return {
        "present": True,
        "mode": str(report.get("mode", "") or ""),
        "requested": bool(report.get("requested", False)),
        "would_enable_native_update": bool(report.get("would_enable_native_update", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "required_shadow_passes": _int(report.get("required_shadow_passes")),
        "consecutive_shadow_passes": _int(report.get("consecutive_shadow_passes")),
        "native_kernel_present": bool(report.get("native_kernel_present", False)),
        "performance_test_ready": bool(report.get("performance_test_ready", False)),
        "stream_lifetime_bound": bool(report.get("stream_lifetime_bound", False)),
        "dispatch_allowed": bool(request.get("dispatch_allowed", False)),
        "kernel_launch_allowed": bool(launch.get("launch_allowed", False)),
        "retained_after_shadow_autostop": bool(report.get("retained_after_shadow_autostop", False)),
        "retained_probe_evidence": bool(report.get("retained_probe_evidence", False) or probe_cache.get("retained", False)),
        "probe_cache_source": str(report.get("probe_cache_source", probe_cache.get("source", "")) or ""),
        "probe_cache_reused_steps": _int(report.get("probe_cache_reused_steps") or probe_cache.get("reused_steps")),
        "fallback_actions": _strings(fallback.get("actions")),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _shadow_summary(
    value: Mapping[str, Any] | None,
    *,
    shadow_enabled: bool,
    shadow_config: Mapping[str, Any],
) -> Dict[str, Any]:
    report = _as_dict(value)
    summary: Dict[str, Any] = {
        "present": bool(report),
        "enabled": bool(shadow_enabled),
        "mode": str(shadow_config.get("mode", "off") or "off"),
        "direct_grad": bool(shadow_config.get("direct_grad", False)),
        "copyback_probe": bool(shadow_config.get("copyback_probe", False)),
        "copyback_dispatch_experimental": bool(shadow_config.get("copyback_dispatch_experimental", False)),
        "native_binding_probe": bool(shadow_config.get("native_binding_probe", False)),
        "owner_native_launch_probe": bool(shadow_config.get("owner_native_launch_probe", False)),
        "blocked_reasons": [],
    }
    if not report:
        return summary
    after = _as_dict(report.get("after_optimizer"))
    copyback = _as_dict(report.get("copyback_probe"))
    dispatch = _as_dict(report.get("copyback_dispatch_probe"))
    native_binding = _as_dict(report.get("native_binding_probe"))
    owner_native = _as_dict(report.get("owner_native_launch_probe"))
    direct_grad = _as_dict(report.get("direct_grad_audit"))
    summary.update(
        {
            "stage": str(report.get("stage", "") or ""),
            "step": _int(report.get("step")),
            "parameter_tensors": _int(report.get("parameter_tensors")),
            "parameter_numel": _int(report.get("parameter_numel")),
            "training_path_enabled": bool(report.get("training_path_enabled", False)),
            "error": str(report.get("error", "") or ""),
            "skip_reason": str(report.get("skip_reason", report.get("reason", "")) or ""),
            "after_optimizer_compared": bool(after.get("compared", False)),
            "parity_ok_loose": bool(after.get("parity_ok_loose", False)),
            "max_abs_param_diff": _optional_float(after.get("max_abs_param_diff")),
            "mean_abs_param_diff": _optional_float(after.get("mean_abs_param_diff")),
            "direct_grad_parity_ok": bool(direct_grad.get("parity_ok", False)) if direct_grad else None,
            "copyback_scratch_validated": bool(copyback.get("scratch_copyback_validated", False)) if copyback else None,
            "copyback_real_parameters_mutated": bool(copyback.get("real_parameters_mutated", False)) if copyback else None,
            "copyback_dispatch_present": bool(dispatch),
            "copyback_dispatch_validated": bool(dispatch.get("copyback_dispatch_validated", False)) if dispatch else None,
            "copyback_dispatch_real_parameters_mutated": bool(dispatch.get("real_parameters_mutated", False)) if dispatch else None,
            "copyback_dispatch_real_parameters_restored": bool(dispatch.get("real_parameters_restored", False)) if dispatch else None,
            "native_binding_present": bool(native_binding),
            "native_binding_request_shape_ready": bool(native_binding.get("request_shape_ready", False)) if native_binding else None,
            "native_binding_stream_lifetime_bound": bool(native_binding.get("stream_lifetime_bound", False)) if native_binding else None,
            "native_binding_event_chain_verified": bool(native_binding.get("event_chain_verified", False)) if native_binding else None,
            "owner_native_launch_present": bool(owner_native),
            "owner_native_launch_attempted": bool(owner_native.get("attempted", False)) if owner_native else None,
            "owner_native_launch_kernel_executed": bool(owner_native.get("kernel_executed", False)) if owner_native else None,
            "owner_native_launch_ok": bool(owner_native.get("ok", False)) if owner_native else None,
            "owner_native_launch_parity_ok": bool(owner_native.get("parity_ok", False)) if owner_native else None,
        }
    )
    blocked: list[str] = []
    if summary.get("error"):
        blocked.append("shadow_error")
    if summary.get("skip_reason"):
        blocked.append(str(summary["skip_reason"]))
    if after and not bool(after.get("parity_ok_loose", False)):
        blocked.append("shadow_parity_not_ok")
    if dispatch and not bool(dispatch.get("copyback_dispatch_validated", False)):
        blocked.append("copyback_dispatch_not_validated")
    summary["blocked_reasons"] = _dedupe(blocked)
    return summary


def _arming_summary(value: Mapping[str, Any] | None) -> Dict[str, Any]:
    report = _as_dict(value)
    if not report:
        return {"present": False, "armed_for_native_dispatch": False, "blocked_reasons": []}
    return {
        "present": True,
        "step": _int(report.get("step")),
        "armed_for_native_dispatch": bool(report.get("armed_for_native_dispatch", False)),
        "execute_native_step": bool(report.get("execute_native_step", False)),
        "call_pytorch_optimizer_step": bool(report.get("call_pytorch_optimizer_step", True)),
        "previous_gate_present": bool(report.get("previous_gate_present", False)),
        "previous_request_requested": bool(report.get("previous_request_requested", False)),
        "previous_request_allowed": bool(report.get("previous_request_allowed", False)),
        "retained_probe_evidence": bool(report.get("retained_probe_evidence", False)),
        "probe_cache_source": str(report.get("probe_cache_source", "") or ""),
        "probe_cache_reused_steps": _int(report.get("probe_cache_reused_steps")),
        "runtime_disabled_for_run": bool(report.get("runtime_disabled_for_run", False)),
        "runtime_disable_reason": str(report.get("runtime_disable_reason", "") or ""),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _dispatch_runtime_summary(value: Mapping[str, Any] | None) -> Dict[str, Any]:
    report = _as_dict(value)
    if not report:
        return {"present": False, "native_step_executed": False, "fallback_to_pytorch_required": True, "blocked_reasons": []}
    return {
        "present": True,
        "step": _int(report.get("step")),
        "requested": bool(report.get("requested", False)),
        "training_dispatch": bool(report.get("training_dispatch", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "runtime_dispatch_available": bool(report.get("runtime_dispatch_available", False)),
        "native_step_executed": bool(report.get("native_step_executed", False)),
        "native_kernel_launched": bool(report.get("native_kernel_launched", False)),
        "fallback_to_pytorch_required": bool(report.get("fallback_to_pytorch_required", True)),
        "should_call_pytorch_optimizer_step": bool(report.get("should_call_pytorch_optimizer_step", True)),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _recovery_summary(value: Mapping[str, Any] | None) -> Dict[str, Any]:
    report = _as_dict(value)
    if not report:
        return {"present": False, "disabled_for_run": False}
    return {
        "present": True,
        "recovery_policy_present": bool(report.get("recovery_policy_present", False)),
        "recovery_disable_native_update_for_run": bool(report.get("recovery_disable_native_update_for_run", False)),
        "disabled_for_run": bool(report.get("disabled_for_run", False)),
        "disable_reason": str(report.get("disable_reason", "") or ""),
    }


def _diagnostic_replay_summary(value: Mapping[str, Any] | None) -> Dict[str, Any]:
    report = _as_dict(value)
    if not report:
        return {}
    return {
        "present": True,
        "diagnostic_replay": bool(report.get("diagnostic_replay", False)),
        "native_step_executed": bool(report.get("native_step_executed", False)),
        "native_kernel_launched": bool(report.get("native_kernel_launched", False)),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _snapshot(value: Any) -> Dict[str, Any]:
    snapshot = getattr(value, "snapshot", None)
    if callable(snapshot):
        try:
            return _as_dict(snapshot())
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
    return {}


def _config_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "as_dict"):
        return _as_dict(value.as_dict())
    return _as_dict(value)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["build_turbocore_native_update_runtime_profile"]
