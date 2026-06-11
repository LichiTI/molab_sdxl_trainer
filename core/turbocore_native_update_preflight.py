"""Dispatch preflight for future TurboCore native optimizer updates.

This module turns scattered readiness/shadow evidence into one explicit hard
gate.  It is intentionally stricter than the legacy report-only gate: passing
this preflight should mean a future native update dispatch has the minimum
evidence needed to be considered, not that dispatch is currently enabled.
"""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_native_update_performance import build_native_update_performance_gate


REQUIRED_DISPATCH_BLOCKERS = (
    "native_dispatch_runtime_not_implemented",
    "native_dispatch_training_path_disabled",
)


def build_native_update_dispatch_preflight(
    *,
    mode: str,
    requested: bool,
    readiness_report: Mapping[str, Any] | None,
    shadow_report: Mapping[str, Any] | None,
    gate_blocked_reasons: list[str] | None = None,
    consecutive_shadow_passes: int = 0,
    required_shadow_passes: int = 1,
    allow_missing_native_kernel: bool = False,
    runtime_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a report-only hard gate for future native update dispatch."""

    readiness = _as_dict(readiness_report)
    shadow = _as_dict(shadow_report)
    context = _as_dict(runtime_context)
    runtime_recovery = _runtime_recovery_status(readiness, shadow)
    native_binding = _native_binding_status(shadow, context)
    performance = build_native_update_performance_gate(
        readiness_report=readiness,
        shadow_report=shadow,
        performance_report=_as_dict(shadow.get("performance_report")),
    )
    short_training_performance_ready = bool(
        context.get("native_update_allow_short_training_dispatch_evidence", False)
        and context.get("native_update_short_training_dispatch_performance_ready", False)
    )
    performance_for_blockers = dict(performance)
    if short_training_performance_ready:
        performance_for_blockers["blocked_reasons"] = []
        performance_for_blockers["performance_test_ready"] = True
        performance_for_blockers["short_training_dispatch_performance_ready"] = True
        performance_for_blockers["training_dispatch_performance_gate_ready"] = True
    evidence = {
        "request": _request_status(mode=mode, requested=requested),
        "readiness": _readiness_status(readiness),
        "shadow": _shadow_status(shadow, consecutive_shadow_passes, required_shadow_passes),
        "copyback": _copyback_status(shadow),
        "owner_native_kernel": _owner_native_status(shadow),
        "native_binding": native_binding,
        "runtime_recovery": runtime_recovery,
        "performance": performance_for_blockers,
        "runtime_safety": _runtime_safety_status(readiness, shadow, allow_missing_native_kernel, runtime_recovery, performance_for_blockers),
    }
    blockers: list[str] = []
    for status in evidence.values():
        blockers.extend(_strings(status.get("blocked_reasons")))
    blockers.extend(str(item) for item in (gate_blocked_reasons or []) if str(item))
    explicit_training_context = bool(
        requested
        and context.get("native_update_training_dispatch_enabled", False)
        and context.get("training_path_enabled", False)
        and context.get("native_update_runtime_dispatch_available", False)
    )
    if not explicit_training_context:
        blockers.extend(REQUIRED_DISPATCH_BLOCKERS)
    unique_blockers = _dedupe(blockers)
    dispatch_allowed = bool(explicit_training_context and not unique_blockers)
    return {
        "schema_version": 1,
        "preflight": "turbocore_native_update_dispatch_preflight_v0",
        "mode": _normalize_mode(mode),
        "requested": bool(requested),
        "dispatch_preflight_passed": dispatch_allowed,
        "would_allow_native_dispatch": dispatch_allowed,
        "training_dispatch": dispatch_allowed,
        "training_path_enabled": dispatch_allowed,
        "native_kernel_present": bool(evidence["runtime_safety"].get("native_kernel_present", False)),
        "performance_test_ready": bool(performance_for_blockers.get("performance_test_ready", False)),
        "training_dispatch_performance_gate_ready": bool(
            performance_for_blockers.get("training_dispatch_performance_gate_ready", False)
            or performance.get("representative_performance_gate_ready", False)
        ),
        "stream_lifetime_bound": bool(
            native_binding.get("stream_lifetime_bound", False)
            or native_binding.get("stream_ordering_verified", False)
        ),
        "stream_lifetime_ownership_bound": bool(native_binding.get("stream_lifetime_ownership_bound", False)),
        "stream_ordering_verified": bool(native_binding.get("stream_ordering_verified", False)),
        "event_chain_verified": bool(native_binding.get("event_chain_verified", False)),
        "evidence": evidence,
        "blocked_reasons": unique_blockers,
    }


def _request_status(*, mode: str, requested: bool) -> dict[str, Any]:
    blocked: list[str] = []
    if not requested:
        blocked.append("native_update_not_requested")
    return {"ok": not blocked, "mode": _normalize_mode(mode), "requested": bool(requested), "blocked_reasons": blocked}


def _readiness_status(readiness: Mapping[str, Any]) -> dict[str, Any]:
    blocked = _strings(readiness.get("blocked_reasons"))
    if not readiness:
        blocked.append("readiness_report_missing")
    elif not bool(readiness.get("ok", False)) and not blocked:
        blocked.append("readiness_not_ok")
    return {
        "ok": bool(readiness and not blocked),
        "present": bool(readiness),
        "native_kernel_present": bool(readiness.get("native_kernel_present", False)),
        "performance_test_ready": bool(readiness.get("performance_test_ready", False)),
        "stream_lifetime_bound": bool(readiness.get("stream_lifetime_bound", False)),
        "stream_lifetime_ownership_bound": bool(
            readiness.get("stream_lifetime_ownership_bound", readiness.get("stream_lifetime_bound", False))
        ),
        "stream_ordering_verified": bool(readiness.get("stream_ordering_verified", False)),
        "event_chain_verified": bool(readiness.get("event_chain_verified", False)),
        "blocked_reasons": _dedupe(blocked),
    }


def _shadow_status(shadow: Mapping[str, Any], consecutive: int, required: int) -> dict[str, Any]:
    after = _as_dict(shadow.get("after_optimizer"))
    blocked: list[str] = []
    compared = bool(after.get("compared", False))
    parity_ok = bool(after.get("parity_ok_loose", False))
    warmup_ok = int(consecutive or 0) >= max(int(required or 1), 1)
    if not shadow:
        blocked.append("shadow_report_missing")
    if not compared:
        blocked.append("shadow_not_compared")
    if not parity_ok:
        blocked.append("shadow_parity_not_ok")
    if not warmup_ok:
        blocked.append("shadow_warmup_not_satisfied")
    return {
        "ok": not blocked,
        "compared": compared,
        "parity_ok": parity_ok,
        "consecutive_shadow_passes": int(consecutive or 0),
        "required_shadow_passes": max(int(required or 1), 1),
        "warmup_satisfied": warmup_ok,
        "max_abs_param_diff": _float_or_none(after.get("max_abs_param_diff")),
        "mean_abs_param_diff": _float_or_none(after.get("mean_abs_param_diff")),
        "blocked_reasons": blocked,
    }


def _copyback_status(shadow: Mapping[str, Any]) -> dict[str, Any]:
    copyback = _as_dict(shadow.get("copyback_probe"))
    dispatch = _as_dict(shadow.get("copyback_dispatch_probe"))
    scratch_ok = bool(copyback.get("scratch_copyback_validated", False))
    dispatch_validated = bool(dispatch.get("copyback_dispatch_validated", False))
    restored = bool(dispatch.get("real_parameters_restored", False)) if dispatch else False
    blocked: list[str] = []
    if not copyback:
        blocked.append("copyback_probe_missing")
    elif not scratch_ok:
        blocked.append("copyback_scratch_validation_failed")
    if bool(copyback.get("real_parameters_mutated", False)):
        blocked.append("copyback_probe_mutated_training_parameters")
    if not dispatch:
        blocked.append("copyback_dispatch_probe_missing")
    elif not dispatch_validated:
        blocked.append("copyback_dispatch_validation_failed")
    if dispatch and bool(dispatch.get("real_parameters_mutated", False)) and not restored:
        blocked.append("copyback_dispatch_left_training_parameters_mutated")
    return {
        "ok": not blocked,
        "scratch_copyback_validated": scratch_ok,
        "dispatch_probe_present": bool(dispatch),
        "dispatch_validated": dispatch_validated,
        "real_parameters_restored": restored if dispatch else None,
        "blocked_reasons": blocked,
    }


def _owner_native_status(shadow: Mapping[str, Any]) -> dict[str, Any]:
    owner = _as_dict(shadow.get("owner_native_launch_probe"))
    blocked: list[str] = []
    if not owner:
        blocked.append("owner_native_launch_probe_missing")
    elif bool(owner.get("skipped", False)):
        blocked.append(str(owner.get("reason", "owner_native_launch_probe_skipped") or "owner_native_launch_probe_skipped"))
    elif not bool(owner.get("ok", False)):
        blocked.append("owner_native_launch_probe_failed")
    if owner and not bool(owner.get("kernel_executed", False)):
        blocked.append("owner_native_kernel_not_executed")
    if owner and not bool(owner.get("parity_ok", False)):
        blocked.append("owner_native_launch_parity_failed")
    if owner and bool(owner.get("persistent_owner_mutated", False)):
        blocked.append("owner_native_probe_mutated_persistent_owner")
    event_requested = bool(owner.get("event_chain_probe_requested", False)) if owner else False
    event_verified = bool(owner.get("event_chain_verified", False)) if owner else False
    if event_requested and not event_verified:
        blocked.append("owner_native_event_chain_not_verified")
    return {
        "ok": not blocked,
        "present": bool(owner),
        "attempted": bool(owner.get("attempted", False)) if owner else False,
        "kernel_executed": bool(owner.get("kernel_executed", False)) if owner else False,
        "parity_ok": bool(owner.get("parity_ok", False)) if owner else False,
        "probe_owner_reused": bool(owner.get("probe_owner_reused", False)) if owner else False,
        "binding_session_reused": bool(owner.get("binding_session_reused", False)) if owner else False,
        "runtime_session_reused": bool(owner.get("runtime_session_reused", False)) if owner else False,
        "event_chain_probe_requested": event_requested,
        "event_chain_probe_attempted": bool(owner.get("event_chain_probe_attempted", False)) if owner else False,
        "event_chain_verified": event_verified,
        "pre_launch_ordering_verified": bool(owner.get("pre_launch_ordering_verified", False)) if owner else False,
        "post_launch_ordering_verified": bool(owner.get("post_launch_ordering_verified", False)) if owner else False,
        "stream_wait_event_verified": bool(owner.get("stream_wait_event_verified", False)) if owner else False,
        "max_abs_diff": _float_or_none(owner.get("max_abs_diff")) if owner else None,
        "elapsed_ms": _float_or_none(owner.get("elapsed_ms")) if owner else None,
        "blocked_reasons": _dedupe(blocked),
    }


def _native_binding_status(shadow: Mapping[str, Any], context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    runtime_context = _as_dict(context)
    binding = _as_dict(shadow.get("native_binding_probe"))
    owner = _as_dict(shadow.get("owner_native_launch_probe"))
    native_event_verified = bool(binding.get("event_chain_verified", False)) if binding else False
    owner_event_verified = bool(owner.get("event_chain_verified", False)) if owner else False
    event_verified = bool(native_event_verified or owner_event_verified)
    native_ordering_verified = bool(
        native_event_verified
        or binding.get("pre_launch_ordering_verified", False)
        or binding.get("post_launch_ordering_verified", False)
        or binding.get("stream_wait_event_verified", False)
    ) if binding else False
    owner_ordering_verified = bool(
        owner_event_verified
        or owner.get("pre_launch_ordering_verified", False)
        or owner.get("post_launch_ordering_verified", False)
        or owner.get("stream_wait_event_verified", False)
    ) if owner else False
    stream_ordering_verified = bool(native_ordering_verified or owner_ordering_verified)
    runtime_stream_ownership_bound = bool(
        runtime_context.get("native_update_stream_lifetime_ownership_runtime_bound", False)
        or runtime_context.get("native_update_stream_lifetime_ownership_bound", False)
    )
    runtime_stream_guard_enabled = bool(
        runtime_context.get("native_update_stream_lifetime_ownership_guard_enabled", False)
    )
    stream_lifetime_ownership_bound = bool(
        (binding.get("stream_lifetime_bound", False) if binding else False)
        or runtime_stream_ownership_bound
    )
    stream_guard_ready = bool(
        (binding.get("stream_guard_ready", False) if binding else False)
        or (runtime_stream_ownership_bound and runtime_stream_guard_enabled)
    )
    blocked: list[str] = []
    if not binding:
        blocked.append("native_binding_probe_missing")
    if binding and not bool(binding.get("request_shape_ready", False)):
        blocked.append("native_binding_request_shape_not_ready")
    if binding and not bool(binding.get("tensor_object_binding_ready", False)):
        blocked.append("native_binding_tensor_object_not_ready")
    if binding and not bool(binding.get("launch_plan_ready", False)):
        blocked.append("native_binding_launch_plan_not_ready")
    if not stream_lifetime_ownership_bound:
        blocked.append("stream_lifetime_ownership_not_promoted" if stream_ordering_verified else "stream_lifetime_unbound")
    if binding and not stream_guard_ready:
        blocked.append("stream_guard_not_ready")
    if binding and not event_verified:
        blocked.append("event_chain_not_verified")
    return {
        "ok": not blocked,
        "present": bool(binding),
        "request_shape_ready": bool(binding.get("request_shape_ready", False)) if binding else False,
        "tensor_object_binding_ready": bool(binding.get("tensor_object_binding_ready", False)) if binding else False,
        "launch_plan_ready": bool(binding.get("launch_plan_ready", False)) if binding else False,
        "stream_lifetime_bound": bool(stream_lifetime_ownership_bound or stream_ordering_verified),
        "stream_lifetime_ownership_bound": stream_lifetime_ownership_bound,
        "runtime_stream_lifetime_ownership_bound": runtime_stream_ownership_bound,
        "stream_ordering_verified": stream_ordering_verified,
        "stream_guard_ready": stream_guard_ready,
        "event_chain_verified": event_verified,
        "native_binding_event_chain_verified": native_event_verified,
        "owner_native_event_chain_verified": owner_event_verified,
        "native_binding_ordering_verified": native_ordering_verified,
        "owner_native_ordering_verified": owner_ordering_verified,
        "blocked_reasons": _dedupe(blocked),
    }


def _runtime_recovery_status(readiness: Mapping[str, Any], shadow: Mapping[str, Any]) -> dict[str, Any]:
    policy = _as_dict(shadow.get("runtime_recovery_policy"))
    fallback_policy = _as_dict(shadow.get("fallback_policy"))
    if not policy:
        policy = _as_dict(fallback_policy.get("runtime_recovery"))
    owner = _as_dict(shadow.get("owner_native_launch_probe"))
    readiness_blockers = _strings(readiness.get("blocked_reasons"))
    runtime_blockers = [item for item in readiness_blockers if "runtime" in item]
    runtime_error = bool(policy.get("runtime", {}).get("runtime_error_observed", False)) if isinstance(policy.get("runtime"), Mapping) else False
    runtime_error = bool(runtime_error or owner.get("error") or runtime_blockers)
    blocked: list[str] = []
    if not policy:
        blocked.append("native_runtime_recovery_policy_missing")
    elif not bool(policy.get("policy_defined", False)):
        blocked.append("native_runtime_recovery_policy_not_defined")
    if not bool(policy.get("dispatch_integration_ready", False)):
        blocked.append("native_runtime_recovery_dispatch_not_integrated")
    if runtime_error:
        blocked.append("native_runtime_error_observed")
    integration = _as_dict(policy.get("integration"))
    return {
        "ok": not blocked,
        "present": bool(policy),
        "policy_defined": bool(policy.get("policy_defined", False)),
        "runtime_recovery_ready": bool(policy.get("runtime_recovery_ready", False)),
        "policy_observation_integrated": bool(policy.get("policy_observation_integrated", False)),
        "run_disable_latch_integrated": bool(policy.get("run_disable_latch_integrated", False)),
        "pre_step_arming_observes_latch": bool(policy.get("pre_step_arming_observes_latch", False)),
        "default_off_recovery_bridge_ready": bool(policy.get("default_off_recovery_bridge_ready", False)),
        "recovery_observation_bridge_ready": bool(policy.get("recovery_observation_bridge_ready", False)),
        "training_dispatch_recovery_ready": bool(policy.get("training_dispatch_recovery_ready", False)),
        "training_dispatch_recovery_blocked": bool(policy.get("training_dispatch_recovery_blocked", False)),
        "training_dispatch_recovery_blocker": str(policy.get("training_dispatch_recovery_blocker", "") or ""),
        "dispatch_integration_ready": bool(policy.get("dispatch_integration_ready", False)),
        "runtime_error_observed": runtime_error,
        "fallback_to_pytorch_enabled": bool(policy.get("fallback_to_pytorch_enabled", False)),
        "integration": integration,
        "actions": _strings(policy.get("actions")),
        "blocked_reasons": _dedupe(blocked + _strings(policy.get("blocked_reasons"))),
    }


def _runtime_safety_status(
    readiness: Mapping[str, Any],
    shadow: Mapping[str, Any],
    allow_missing_kernel: bool,
    runtime_recovery: Mapping[str, Any],
    performance: Mapping[str, Any],
) -> dict[str, Any]:
    owner = _as_dict(shadow.get("owner_native_launch_probe"))
    native_kernel_present = bool(readiness.get("native_kernel_present", False)) or bool(owner.get("kernel_executed", False))
    blocked: list[str] = []
    if not bool(runtime_recovery.get("dispatch_integration_ready", False)):
        blocked.append("native_runtime_error_recovery_missing")
    performance_ready = bool(
        performance.get("representative_performance_gate_ready", False)
        or performance.get("training_dispatch_performance_gate_ready", False)
    )
    if not performance_ready:
        blocked.append("representative_performance_gate_missing")
    if not native_kernel_present and not bool(allow_missing_kernel):
        blocked.append("native_kernel_missing")
    return {
        "ok": False,
        "native_kernel_present": native_kernel_present,
        "allow_missing_native_kernel": bool(allow_missing_kernel),
        "error_recovery_ready": bool(runtime_recovery.get("dispatch_integration_ready", False)),
        "representative_performance_gate_ready": bool(performance.get("representative_performance_gate_ready", False)),
        "training_dispatch_performance_gate_ready": bool(
            performance.get("training_dispatch_performance_gate_ready", False)
        ),
        "blocked_reasons": blocked,
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_mode(value: str) -> str:
    normalized = str(value or "off").strip().lower().replace("-", "_")
    return normalized if normalized in {"off", "profile", "native_experimental"} else "off"


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = ["build_native_update_dispatch_preflight"]
