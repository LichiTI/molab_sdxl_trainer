"""Stream lifetime lease evidence for V5 borrowed-stream AdamW canaries."""

from __future__ import annotations

from typing import Any, Mapping


CONTRACT = "turbocore_v5_stream_lifetime_lease_evidence_v0"
VALID_LEASE_SCOPES = {
    "borrowed_stream_single_step",
    "native_adamw_runtime_step",
    "native_update_training_step",
}


def build_stream_lifetime_lease_evidence(
    *,
    stream_guard: Mapping[str, Any] | None = None,
    lease_evidence: Mapping[str, Any] | None = None,
    runtime_context: Mapping[str, Any] | None = None,
    requested_policy: str = "context_synchronize",
) -> dict[str, Any]:
    """Return conservative proof that a borrowed stream may skip ctx sync."""

    guard = _as_dict(stream_guard)
    source = _as_dict(lease_evidence)
    context = _as_dict(runtime_context)
    requested = str(requested_policy or "context_synchronize") == "borrowed_stream_event_chain"
    explicit_training = bool(
        source.get("explicit_training_context_requested", False)
        or (
            context.get("native_update_training_dispatch_enabled", False)
            and context.get("training_path_enabled", False)
            and context.get("native_update_runtime_dispatch_available", False)
        )
    )
    guard_enabled = bool(
        source.get("ownership_guard_enabled", False)
        or context.get("native_update_stream_lifetime_ownership_guard_enabled", False)
    )
    binding_enabled = bool(
        source.get("ownership_binding_enabled", False)
        or context.get("native_update_stream_lifetime_ownership_bound", False)
        or context.get("native_update_stream_lifetime_ownership_runtime_bound", False)
    )
    recovery_ready = bool(
        source.get("runtime_recovery_ready", False)
        or source.get("training_dispatch_recovery_ready", False)
        or context.get("native_update_training_dispatch_recovery_ready", False)
        or context.get("native_update_runtime_recovery_ready", False)
    )
    native_error_recovery_verified = bool(source.get("native_error_recovery_verified", recovery_ready))
    lease_scope = str(source.get("lease_scope", source.get("scope", "")) or "")
    lease_active = bool(source.get("lease_active_for_current_step", source.get("lease_active", False)))
    ordering_verified = bool(
        guard.get("event_chain_verified", False)
        and guard.get("pre_launch_ordering_verified", False)
        and guard.get("post_launch_ordering_verified", False)
        and guard.get("stream_wait_event_verified", False)
    )
    external_stream = bool(
        guard.get("stream_handle_nonzero", False)
        and str(guard.get("stream_handle_kind", "") or "") == "external_cuda_stream_handle"
    )
    checks = {
        "v5_p11_borrowed_stream_policy_missing": requested,
        "v5_p11_external_stream_missing": external_stream,
        "v5_p11_event_chain_ordering_missing": ordering_verified,
        "v5_p11_explicit_training_context_missing": explicit_training,
        "v5_p11_ownership_guard_missing": guard_enabled,
        "v5_p11_ownership_binding_missing": binding_enabled,
        "v5_p11_recovery_bridge_missing": recovery_ready,
        "v5_p11_native_error_recovery_missing": native_error_recovery_verified,
        "v5_p11_lease_scope_missing": lease_scope in VALID_LEASE_SCOPES,
        "v5_p11_lease_not_active_for_current_step": lease_active,
    }
    blocked = [name for name, ok in checks.items() if not ok]
    ready = not blocked
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "ok": ready,
        "ready_for_runtime_stream_guard": ready,
        "stream_lifetime_bound": ready,
        "native_error_recovery_verified": ready and native_error_recovery_verified,
        "requested_policy": str(requested_policy or "context_synchronize"),
        "lease_scope": lease_scope,
        "lease_active_for_current_step": lease_active,
        "explicit_training_context_requested": explicit_training,
        "ownership_guard_enabled": guard_enabled,
        "ownership_binding_enabled": binding_enabled,
        "runtime_recovery_ready": recovery_ready,
        "ordering_verified": ordering_verified,
        "external_stream_verified": external_stream,
        "default_behavior_changed": False,
        "requires_explicit_opt_in": True,
        "training_path_enabled": False,
        "default_training_path_enabled": False,
        "auto_rollout_allowed": False,
        "blocked_reasons": _dedupe(blocked + _strings(source.get("blocked_reasons"))),
    }


def build_single_step_lifetime_lease_request(
    *,
    explicit_training_context: bool,
    recovery_ready: bool,
    lease_scope: str = "native_update_training_step",
) -> dict[str, Any]:
    """Build the internal request-side lease evidence shape."""

    active = bool(explicit_training_context)
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "lease_scope": str(lease_scope or "native_update_training_step"),
        "lease_active_for_current_step": active,
        "explicit_training_context_requested": active,
        "ownership_guard_enabled": active,
        "ownership_binding_enabled": active,
        "runtime_recovery_ready": bool(recovery_ready and active),
        "training_dispatch_recovery_ready": bool(recovery_ready and active),
        "native_error_recovery_verified": bool(recovery_ready and active),
        "default_behavior_changed": False,
        "requires_explicit_opt_in": True,
        "training_path_enabled": False,
        "default_training_path_enabled": False,
        "auto_rollout_allowed": False,
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


__all__ = [
    "build_single_step_lifetime_lease_request",
    "build_stream_lifetime_lease_evidence",
]
