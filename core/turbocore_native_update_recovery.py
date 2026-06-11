"""Runtime recovery contract for future TurboCore native update dispatch.

The real training dispatcher is still disabled.  This module only describes the
actions that a future native optimizer path must take after a native runtime,
stream, state, or copyback failure before PyTorch can safely remain the
authoritative optimizer.
"""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_native_update_recovery_integration import build_native_update_recovery_integration_report


def build_native_update_runtime_recovery_policy(
    *,
    mode: str,
    strict: bool = False,
    readiness_report: Mapping[str, Any] | None = None,
    shadow_report: Mapping[str, Any] | None = None,
    runtime_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a report-only recovery policy for native update failures."""

    readiness = _as_dict(readiness_report)
    shadow = _as_dict(shadow_report)
    owner_native = _as_dict(shadow.get("owner_native_launch_probe"))
    copyback_dispatch = _as_dict(shadow.get("copyback_dispatch_probe"))
    runtime = _runtime_error_status(readiness, owner_native)
    state = _state_safety_status(shadow, owner_native, copyback_dispatch)
    integration = build_native_update_recovery_integration_report(
        mode=mode,
        policy_defined=True,
        disable_native_update_for_run=bool(runtime["runtime_error_observed"] or state["state_mismatch_observed"]),
        runtime_error_observed=bool(runtime["runtime_error_observed"]),
        state_mismatch_observed=bool(state["state_mismatch_observed"]),
        runtime_context=runtime_context,
    )
    recovery_ready = bool(integration["training_dispatch_recovery_ready"])
    actions = _recovery_actions(
        strict=bool(strict),
        runtime=runtime,
        state=state,
        training_dispatch_recovery_ready=recovery_ready,
    )
    actions.extend(_strings(integration.get("actions")))
    blocked = list(integration["blocked_reasons"])
    if bool(runtime["runtime_error_observed"]):
        blocked.append("native_runtime_error_observed")
    if bool(state["state_mismatch_observed"]):
        blocked.append("native_state_mismatch_observed")
    if bool(state["requires_parameter_restore_before_fallback"]):
        blocked.append("training_parameter_restore_required_before_fallback")
    return {
        "schema_version": 1,
        "policy": "turbocore_native_update_runtime_recovery_v0",
        "mode": _normalize_mode(mode),
        "strict": bool(strict),
        "training_dispatch": recovery_ready,
        "training_path_enabled": recovery_ready,
        "policy_defined": True,
        "runtime_recovery_ready": recovery_ready,
        "policy_observation_integrated": bool(integration["recovery_policy_observation_integrated"]),
        "run_disable_latch_integrated": bool(integration["run_disable_latch_integrated"]),
        "pre_step_arming_observes_latch": bool(integration["pre_step_arming_observes_latch"]),
        "default_off_recovery_bridge_ready": bool(integration["default_off_recovery_bridge_ready"]),
        "recovery_observation_bridge_ready": bool(integration["recovery_observation_bridge_ready"]),
        "training_dispatch_recovery_ready": bool(integration["training_dispatch_recovery_ready"]),
        "training_dispatch_recovery_blocked": bool(integration["training_dispatch_recovery_blocked"]),
        "training_dispatch_recovery_blocker": str(integration["training_dispatch_recovery_blocker"]),
        "dispatch_integration_ready": bool(integration["dispatch_integration_ready"]),
        "fallback_to_pytorch_enabled": not bool(strict),
        "fail_fast_if_strict": bool(strict),
        "safe_to_retry_native": recovery_ready,
        "disable_native_update_for_run": bool(runtime["runtime_error_observed"] or state["state_mismatch_observed"]),
        "runtime": runtime,
        "state_safety": state,
        "integration": integration,
        "actions": _dedupe(actions),
        "blocked_reasons": _dedupe(blocked),
    }


def _runtime_error_status(readiness: Mapping[str, Any], owner_native: Mapping[str, Any]) -> dict[str, Any]:
    runtime_session = _as_dict(owner_native.get("runtime_session"))
    binding_session = _as_dict(owner_native.get("binding_session"))
    launch = _as_dict(owner_native.get("launch"))
    readiness_errors = [item for item in _strings(readiness.get("blocked_reasons")) if "runtime" in item]
    owner_attempted = bool(owner_native.get("attempted", False))
    owner_skipped = bool(owner_native.get("skipped", False))
    native_launch_attempted = bool(owner_native.get("native_launch_attempted", False) or launch)
    launch_ok = bool(owner_native.get("native_launch_ok", False) or launch.get("ok", False)) if native_launch_attempted else None
    error_text = str(owner_native.get("error", "") or launch.get("error", "") or "")
    failed_attempt = bool(owner_native and owner_attempted and not owner_skipped and not bool(owner_native.get("ok", False)))
    runtime_error = bool(
        readiness_errors
        or error_text
        or failed_attempt
        or _has_error(runtime_session)
        or _has_error(binding_session)
        or _has_error(launch)
    )
    return {
        "runtime_error_observed": runtime_error,
        "readiness_runtime_blockers": readiness_errors,
        "owner_native_present": bool(owner_native),
        "owner_native_attempted": owner_attempted,
        "owner_native_skipped": owner_skipped,
        "native_launch_attempted": native_launch_attempted,
        "native_launch_ok": launch_ok,
        "kernel_executed": bool(owner_native.get("kernel_executed", False) or launch.get("kernel_executed", False)),
        "reason": str(owner_native.get("reason", "") or launch.get("reason", "") or ""),
        "error": error_text,
        "runtime_session_id": int(owner_native.get("runtime_session_id", 0) or runtime_session.get("runtime_session_id", 0) or 0),
        "binding_session_id": int(owner_native.get("binding_session_id", 0) or binding_session.get("session_id", 0) or 0),
    }


def _state_safety_status(
    shadow: Mapping[str, Any],
    owner_native: Mapping[str, Any],
    copyback_dispatch: Mapping[str, Any],
) -> dict[str, Any]:
    after = _as_dict(shadow.get("after_optimizer"))
    shadow_autostopped = _shadow_autostop_skip(shadow, after)
    owner_attempted = bool(owner_native.get("attempted", False))
    owner_kernel_executed = bool(owner_native.get("kernel_executed", False))
    owner_parity_ok = bool(owner_native.get("parity_ok", False)) if owner_native else True
    owner_mutated = bool(owner_native.get("persistent_owner_mutated", False))
    dispatch_mutated = bool(copyback_dispatch.get("real_parameters_mutated", False))
    dispatch_restored = bool(copyback_dispatch.get("real_parameters_restored", False))
    copyback_left_mutated = bool(dispatch_mutated and not dispatch_restored)
    shadow_parity_known = bool(after and not shadow_autostopped)
    shadow_parity_ok = bool(after.get("parity_ok_loose", False)) if shadow_parity_known else True
    state_mismatch = bool(
        owner_mutated
        or copyback_left_mutated
        or (owner_attempted and owner_kernel_executed and not owner_parity_ok)
        or (shadow_parity_known and not shadow_parity_ok)
    )
    return {
        "state_mismatch_observed": state_mismatch,
        "shadow_parity_known": shadow_parity_known,
        "shadow_parity_ok": shadow_parity_ok,
        "owner_native_parity_ok": owner_parity_ok if owner_native else None,
        "owner_native_persistent_owner_mutated": owner_mutated if owner_native else None,
        "copyback_dispatch_probe_present": bool(copyback_dispatch),
        "copyback_dispatch_real_parameters_mutated": dispatch_mutated if copyback_dispatch else None,
        "copyback_dispatch_real_parameters_restored": dispatch_restored if copyback_dispatch else None,
        "requires_parameter_restore_before_fallback": copyback_left_mutated,
        "shadow_auto_stopped_after_consecutive_passes": shadow_autostopped,
        "shadow_revalidation_required": True,
    }


def _recovery_actions(
    *,
    strict: bool,
    runtime: Mapping[str, Any],
    state: Mapping[str, Any],
    training_dispatch_recovery_ready: bool,
) -> list[str]:
    actions = [
        "record_recovery_evidence_in_gate_report",
        "require_shadow_parity_revalidation_after_recovery",
        "on_non_finite_gradient_match_pytorch_skip_policy",
    ]
    if training_dispatch_recovery_ready:
        actions.append("allow_native_training_dispatch_recovery_bridge")
    else:
        actions.extend(
            [
                "keep_pytorch_optimizer_authoritative",
                "keep_training_dispatch_disabled_until_recovery_integrated",
            ]
        )
    if bool(runtime.get("runtime_error_observed", False)):
        actions.extend(
            [
                "disable_native_update_for_run_on_runtime_error",
                "destroy_native_runtime_session_on_error",
                "drop_native_tensor_binding_session_on_error",
            ]
        )
    if bool(state.get("state_mismatch_observed", False)):
        actions.append("disable_native_update_for_run_on_state_mismatch")
    if bool(state.get("requires_parameter_restore_before_fallback", False)):
        actions.append("restore_training_parameters_before_fallback")
    else:
        actions.append("verify_training_parameters_unchanged_before_fallback")
    actions.append("fail_fast_if_strict_after_native_error" if strict else "fallback_to_pytorch_optimizer_if_non_strict")
    return actions


def _has_error(value: Mapping[str, Any]) -> bool:
    if not value:
        return False
    if value.get("error"):
        return True
    if value.get("reason") and value.get("ok") is False:
        return True
    return False


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _shadow_autostop_skip(shadow: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    reason = str(after.get("reason", "") or shadow.get("reason", "") or "")
    return bool(
        reason == "auto_stopped_after_consecutive_passes"
        or (
            bool(after.get("skipped", False))
            and reason == "auto_stopped_after_consecutive_passes"
        )
    )


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _normalize_mode(value: str) -> str:
    normalized = str(value or "off").strip().lower().replace("-", "_")
    return normalized if normalized in {"off", "profile", "native_experimental"} else "off"


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = ["build_native_update_runtime_recovery_policy"]
