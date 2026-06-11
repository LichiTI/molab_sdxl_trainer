"""Fallback policy report for future TurboCore native update dispatch."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_native_update_recovery import build_native_update_runtime_recovery_policy


def build_native_update_fallback_policy(
    *,
    mode: str,
    strict: bool = False,
    readiness_report: Mapping[str, Any] | None = None,
    shadow_report: Mapping[str, Any] | None = None,
    runtime_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = str(mode or "off").strip().lower().replace("-", "_")
    if normalized not in {"off", "profile", "native_experimental"}:
        normalized = "off"
    readiness = dict(readiness_report or {})
    shadow = dict(shadow_report or {})
    readiness_blockers = [str(item) for item in readiness.get("blocked_reasons", [])] if isinstance(readiness.get("blocked_reasons"), list) else []
    after = shadow.get("after_optimizer") if isinstance(shadow.get("after_optimizer"), Mapping) else {}
    copyback = shadow.get("copyback_probe") if isinstance(shadow.get("copyback_probe"), Mapping) else {}
    dispatch = shadow.get("copyback_dispatch_probe") if isinstance(shadow.get("copyback_dispatch_probe"), Mapping) else {}
    native_binding = shadow.get("native_binding_probe") if isinstance(shadow.get("native_binding_probe"), Mapping) else {}
    owner_native = shadow.get("owner_native_launch_probe") if isinstance(shadow.get("owner_native_launch_probe"), Mapping) else {}
    native_event_verified = bool(native_binding.get("event_chain_verified", False)) if native_binding else False
    owner_event_verified = bool(owner_native.get("event_chain_verified", False)) if owner_native else False
    stream_ordering_verified = bool(
        native_event_verified
        or owner_event_verified
        or native_binding.get("pre_launch_ordering_verified", False)
        or native_binding.get("post_launch_ordering_verified", False)
        or native_binding.get("stream_wait_event_verified", False)
        or owner_native.get("pre_launch_ordering_verified", False)
        or owner_native.get("post_launch_ordering_verified", False)
        or owner_native.get("stream_wait_event_verified", False)
    )
    runtime_recovery = build_native_update_runtime_recovery_policy(
        mode=normalized,
        strict=bool(strict),
        readiness_report=readiness_report,
        shadow_report=shadow_report,
        runtime_context=runtime_context,
    )
    shadow_ok = bool(after.get("parity_ok_loose", False)) if after else False
    copyback_ok = bool(copyback.get("scratch_copyback_validated", False)) if copyback else True
    copyback_mutated_real_params = bool(copyback.get("real_parameters_mutated", False)) if copyback else False
    dispatch_enabled = bool(dispatch.get("copyback_dispatch_enabled", False)) if dispatch else False
    dispatch_ok = bool(dispatch.get("copyback_dispatch_validated", False)) if dispatch else None
    dispatch_mutated = bool(dispatch.get("real_parameters_mutated", False)) if dispatch else None
    dispatch_restored = bool(dispatch.get("real_parameters_restored", False)) if dispatch else None
    actions: list[str] = []
    if normalized == "off":
        actions.append("native_update_not_requested")
    if readiness_blockers:
        actions.append("keep_pytorch_optimizer_due_to_readiness_blockers")
    if shadow and not shadow_ok:
        actions.append("keep_pytorch_optimizer_due_to_shadow_parity")
    if copyback and not copyback_ok:
        actions.append("keep_pytorch_optimizer_due_to_copyback_validation")
    if copyback_mutated_real_params:
        actions.append("disable_native_update_due_to_copyback_mutation")
    if copyback and not dispatch:
        actions.append("keep_pytorch_optimizer_due_to_copyback_dispatch_disabled")
    if dispatch and not dispatch_ok:
        actions.append("keep_pytorch_optimizer_due_to_copyback_dispatch_validation")
    if dispatch_mutated and not dispatch_restored:
        actions.append("disable_native_update_due_to_unrestored_copyback_dispatch_mutation")
    if native_binding and not bool(native_binding.get("stream_lifetime_bound", False)):
        actions.append(
            "keep_pytorch_optimizer_due_to_stream_lifetime_ownership_not_promoted"
            if stream_ordering_verified
            else "keep_pytorch_optimizer_due_to_unbound_stream_lifetime"
        )
    if native_binding and not bool(native_binding.get("stream_guard_ready", False)):
        actions.append("keep_pytorch_optimizer_due_to_stream_guard_not_ready")
    if native_binding and not bool(native_binding.get("stream_identity_ready", False)):
        actions.append("keep_pytorch_optimizer_due_to_stream_identity_not_ready")
    if native_binding and not bool(native_event_verified or owner_event_verified):
        actions.append("keep_pytorch_optimizer_due_to_event_chain_not_verified")
    stream_contract = native_binding.get("stream_contract") if isinstance(native_binding.get("stream_contract"), Mapping) else {}
    if native_binding and not bool(native_binding.get("launch_plan_ready", False)):
        actions.append("keep_pytorch_optimizer_due_to_native_binding_plan")
    if owner_native and bool(owner_native.get("attempted", False)) and not bool(owner_native.get("ok", False)):
        actions.append("keep_pytorch_optimizer_due_to_owner_native_launch_probe")
    actions.extend(
        [
            "on_native_error_disable_native_update_for_run",
            "on_non_finite_gradient_match_pytorch_skip_policy",
            "on_state_mismatch_fallback_to_pytorch_optimizer",
        ]
    )
    recovery_ready = bool(runtime_recovery.get("training_dispatch_recovery_ready", False))
    return {
        "schema_version": 1,
        "policy": "turbocore_native_update_fallback_v0",
        "mode": normalized,
        "strict": bool(strict),
        "training_path_enabled": recovery_ready,
        "training_dispatch": recovery_ready,
        "fallback_to_pytorch_enabled": not bool(strict),
        "fail_fast_if_strict": bool(strict),
        "native_attempted": False,
        "readiness_blockers": readiness_blockers,
        "shadow_parity_ok": shadow_ok if shadow else None,
        "copyback_scratch_validated": copyback_ok if copyback else None,
        "copyback_real_parameters_mutated": copyback_mutated_real_params if copyback else None,
        "copyback_dispatch_probe_present": bool(dispatch),
        "copyback_dispatch_enabled": dispatch_enabled if dispatch else None,
        "copyback_dispatch_validated": dispatch_ok,
        "copyback_dispatch_real_parameters_mutated": dispatch_mutated,
        "copyback_dispatch_real_parameters_restored": dispatch_restored,
        "native_binding_probe_present": bool(native_binding),
        "native_binding_launch_plan_ready": bool(native_binding.get("launch_plan_ready", False)) if native_binding else None,
        "native_binding_stream_lifetime_bound": bool(native_binding.get("stream_lifetime_bound", False)) if native_binding else None,
        "native_binding_stream_lifetime_ownership_bound": bool(native_binding.get("stream_lifetime_bound", False)) if native_binding else None,
        "stream_ordering_verified": stream_ordering_verified if (native_binding or owner_native) else None,
        "native_binding_stream_contract_present": bool(stream_contract) if native_binding else None,
        "native_binding_stream_kind": str(stream_contract.get("stream_kind", "") or "") if stream_contract else None,
        "native_binding_stream_lease_id": int(native_binding.get("stream_lease_id", 0) or 0) if native_binding else None,
        "native_binding_stream_guard_present": bool(native_binding.get("stream_guard_present", False)) if native_binding else None,
        "native_binding_stream_guard_ready": bool(native_binding.get("stream_guard_ready", False)) if native_binding else None,
        "native_binding_stream_identity_ready": bool(native_binding.get("stream_identity_ready", False)) if native_binding else None,
        "native_binding_stream_guard_level": str(native_binding.get("stream_guard_level", "") or "") if native_binding else None,
        "native_binding_stream_handle_kind": str(native_binding.get("stream_handle_kind", "") or "") if native_binding else None,
        "native_binding_stream_handle_reported": bool(native_binding.get("stream_handle_reported", False)) if native_binding else None,
        "native_binding_stream_handle_nonzero": bool(native_binding.get("stream_handle_nonzero", False)) if native_binding else None,
        "native_binding_synchronization_guard_ready": bool(native_binding.get("synchronization_guard_ready", False)) if native_binding else None,
        "native_binding_synchronization_strategy": str(native_binding.get("synchronization_strategy", "") or "") if native_binding else None,
        "native_binding_event_chain_contract": str(native_binding.get("event_chain_contract", "") or "") if native_binding else None,
        "native_binding_event_chain_state": str(native_binding.get("event_chain_state", "") or "") if native_binding else None,
        "native_binding_event_chain_probe_requested": bool(native_binding.get("event_chain_probe_requested", False)) if native_binding else None,
        "native_binding_event_chain_probe_attempted": bool(native_binding.get("event_chain_probe_attempted", False)) if native_binding else None,
        "native_binding_event_chain_verified": bool(native_binding.get("event_chain_verified", False)) if native_binding else None,
        "native_binding_pre_launch_ordering_verified": bool(native_binding.get("pre_launch_ordering_verified", False)) if native_binding else None,
        "native_binding_post_launch_ordering_verified": bool(native_binding.get("post_launch_ordering_verified", False)) if native_binding else None,
        "native_binding_stream_wait_event_verified": bool(native_binding.get("stream_wait_event_verified", False)) if native_binding else None,
        "native_binding_native_launch_candidate": bool(native_binding.get("native_launch_candidate", False)) if native_binding else None,
        "native_binding_borrowed_external_stream": bool(native_binding.get("borrowed_external_stream", False)) if native_binding else None,
        "native_binding_stream_device_match": bool(native_binding.get("stream_device_match", False)) if native_binding else None,
        "owner_native_launch_probe_present": bool(owner_native),
        "owner_native_launch_attempted": bool(owner_native.get("attempted", False)) if owner_native else None,
        "owner_native_launch_ok": bool(owner_native.get("ok", False)) if owner_native else None,
        "owner_native_launch_kernel_executed": bool(owner_native.get("kernel_executed", False)) if owner_native else None,
        "owner_native_launch_parity_ok": bool(owner_native.get("parity_ok", False)) if owner_native else None,
        "owner_native_launch_persistent_owner_mutated": bool(owner_native.get("persistent_owner_mutated", False)) if owner_native else None,
        "runtime_recovery": runtime_recovery,
        "actions": _dedupe(actions),
    }


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = ["build_native_update_fallback_policy"]
