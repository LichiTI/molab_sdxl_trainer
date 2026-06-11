"""Retention helpers for TurboCore native-update probe evidence."""

from __future__ import annotations

from typing import Any, Mapping


PROBE_CACHE_SCHEMA = "turbocore_native_update_probe_evidence_retention_v0"


def can_retain_native_update_probe_evidence(
    *,
    previous_gate: Mapping[str, Any] | None,
    shadow_report: Mapping[str, Any] | None,
    dispatch_runtime_report: Mapping[str, Any] | None,
    defer_state_sync: bool,
) -> bool:
    """Return true when the previous successful gate can be reused after shadow autostop."""

    gate = _as_dict(previous_gate)
    shadow = _as_dict(shadow_report)
    runtime = _as_dict(dispatch_runtime_report)
    if not bool(defer_state_sync):
        return False
    if not gate or not shadow or not runtime:
        return False
    if not bool(runtime.get("native_step_executed", False)):
        return False
    runtime_state = _as_dict(runtime.get("state"))
    if bool(runtime_state.get("disabled_for_run", False)):
        return False
    if "native_dispatch_disabled_for_run" in _strings(runtime.get("blocked_reasons")):
        return False
    if not _is_shadow_autostop_skip(shadow):
        return False
    if not _previous_gate_allows_training_dispatch(gate):
        return False
    if _previous_gate_requires_revalidation(gate):
        return False
    return _previous_gate_has_probe_evidence(gate)


def retain_native_update_probe_evidence(
    previous_gate: Mapping[str, Any],
    *,
    step: int,
    source: str = "previous_successful_gate_report",
) -> dict[str, Any]:
    """Annotate a retained gate report so benchmark/profile surfaces can audit reuse."""

    gate = dict(previous_gate or {})
    previous_cache = _as_dict(gate.get("probe_cache_retention"))
    reused_steps = _int(gate.get("probe_cache_reused_steps") or previous_cache.get("reused_steps")) + 1
    evidence = _probe_evidence_summary(gate)
    retention = {
        "schema_version": 1,
        "cache": PROBE_CACHE_SCHEMA,
        "retained": True,
        "source": str(source or "previous_successful_gate_report"),
        "reused_steps": reused_steps,
        "retained_at_step": int(step),
        "evidence": evidence,
    }
    gate.update(
        {
            "retained_after_shadow_autostop": True,
            "retained_probe_evidence": True,
            "probe_cache_source": retention["source"],
            "probe_cache_reused_steps": reused_steps,
            "probe_cache_retained_at_step": int(step),
            "probe_cache_retained_evidence": evidence,
            "probe_cache_retention": retention,
        }
    )
    return gate


def probe_cache_summary(gate_report: Mapping[str, Any] | None) -> dict[str, Any]:
    gate = _as_dict(gate_report)
    cache = _as_dict(gate.get("probe_cache_retention"))
    evidence = _as_dict(gate.get("probe_cache_retained_evidence")) or _as_dict(cache.get("evidence"))
    return {
        "retained": bool(gate.get("retained_probe_evidence", False) or cache.get("retained", False)),
        "source": str(gate.get("probe_cache_source", cache.get("source", "")) or ""),
        "reused_steps": _int(gate.get("probe_cache_reused_steps") or cache.get("reused_steps")),
        "retained_at_step": _int(gate.get("probe_cache_retained_at_step") or cache.get("retained_at_step")),
        "evidence": evidence,
    }


def _previous_gate_allows_training_dispatch(gate: Mapping[str, Any]) -> bool:
    request = _as_dict(gate.get("dispatch_request"))
    contract = _as_dict(gate.get("dispatch_contract"))
    kernel = _as_dict(gate.get("kernel_launch_plan"))
    return bool(
        request.get("requested", False)
        and request.get("dispatch_allowed", False)
        and contract.get("dispatch_rehearsal_ready", False)
        and kernel.get("launch_allowed", False)
    )


def _previous_gate_has_probe_evidence(gate: Mapping[str, Any]) -> bool:
    evidence = _probe_evidence_summary(gate)
    return bool(
        evidence["owner_native_launch_ok"]
        and evidence["owner_native_parity_ok"]
        and evidence["copyback_dispatch_validated"]
        and evidence["stream_ordering_verified"]
    )


def _previous_gate_requires_revalidation(gate: Mapping[str, Any]) -> bool:
    fallback = _as_dict(gate.get("fallback_policy"))
    recovery = _as_dict(fallback.get("runtime_recovery"))
    runtime = _as_dict(recovery.get("runtime"))
    state = _as_dict(recovery.get("state_safety"))
    return bool(
        recovery.get("disable_native_update_for_run", False)
        or runtime.get("runtime_error_observed", False)
        or state.get("state_mismatch_observed", False)
    )


def _probe_evidence_summary(gate: Mapping[str, Any]) -> dict[str, Any]:
    contract = _as_dict(gate.get("dispatch_contract"))
    evidence = _as_dict(contract.get("evidence"))
    kernel = _as_dict(gate.get("kernel_launch_plan"))
    kernel_evidence = _as_dict(kernel.get("evidence"))
    return {
        "owner_native_launch_ok": bool(evidence.get("owner_native_launch_ok", False)),
        "owner_native_parity_ok": bool(kernel_evidence.get("diagnostic_parity_ok", False)),
        "owner_native_kernel_executed": bool(kernel_evidence.get("diagnostic_kernel_executed", False)),
        "copyback_dispatch_validated": bool(evidence.get("copyback_dispatch_validated", False)),
        "native_binding_probe_present": bool(evidence.get("native_binding_probe_present", False)),
        "stream_ordering_verified": bool(evidence.get("stream_ordering_verified", False)),
        "event_chain_verified": bool(evidence.get("event_chain_verified", False)),
        "representative_performance_gate_ready": bool(
            evidence.get("representative_performance_gate_ready", False)
            or evidence.get("training_dispatch_performance_gate_ready", False)
        ),
    }


def _is_shadow_autostop_skip(shadow_report: Mapping[str, Any]) -> bool:
    after = _as_dict(shadow_report.get("after_optimizer"))
    reason = str(shadow_report.get("reason", "") or after.get("reason", "") or "")
    return reason == "auto_stopped_after_consecutive_passes"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "PROBE_CACHE_SCHEMA",
    "can_retain_native_update_probe_evidence",
    "probe_cache_summary",
    "retain_native_update_probe_evidence",
]
