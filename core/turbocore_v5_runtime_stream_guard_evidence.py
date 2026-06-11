"""Normalize runtime stream-guard evidence for V5 borrowed-stream canaries."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from core.turbocore_stream_descriptor import current_torch_stream_descriptor
from core.turbocore_v5_stream_lifetime_lease_evidence import (
    build_stream_lifetime_lease_evidence,
)


CONTRACT = "turbocore_v5_runtime_stream_guard_evidence_v0"
RUNTIME_DESCRIPTOR = "turbocore_v5_borrowed_stream_runtime_descriptor_v0"


def build_runtime_stream_guard_evidence(
    *,
    owner: Any | None = None,
    configured_descriptor: Mapping[str, Any] | None = None,
    native_binding_probe: Mapping[str, Any] | None = None,
    lifetime_lease_evidence: Mapping[str, Any] | None = None,
    runtime_context: Mapping[str, Any] | None = None,
    requested_policy: str = "borrowed_stream_event_chain",
    request_event_chain: bool = True,
) -> dict[str, Any]:
    """Build a JSON-safe evidence packet for ctx-sync-free launch decisions.

    The packet is intentionally conservative: current real probes can verify
    event ordering, but they do not claim Python/CUDA stream lifetime ownership.
    """

    source = "none"
    errors: list[str] = []
    probe = _as_dict(native_binding_probe)
    descriptor = _as_dict(configured_descriptor)
    if descriptor:
        source = "configured_descriptor"
    elif probe:
        descriptor = _extract_stream_guard(probe)
        source = "native_binding_probe"
    elif owner is not None and _owner_has_cuda_param(owner):
        try:
            from core.turbocore_update_native_binding_probe import build_update_native_binding_probe

            probe = build_update_native_binding_probe(owner, request_event_chain=bool(request_event_chain))
            descriptor = _extract_stream_guard(probe)
            source = "native_binding_probe"
        except Exception as exc:  # pragma: no cover - CUDA/native-toolchain dependent
            errors.append(f"{type(exc).__name__}: {exc}")
            descriptor = {}
            source = "native_binding_probe_error"

    if not descriptor and owner is not None:
        descriptor = _current_owner_stream_descriptor(owner, request_event_chain=bool(request_event_chain))
        source = "current_torch_stream_descriptor"

    return _normalize_evidence(
        descriptor=descriptor,
        source=source,
        source_probe=probe,
        errors=errors,
        lifetime_lease_evidence=_as_dict(lifetime_lease_evidence),
        runtime_context=_as_dict(runtime_context),
        requested_policy=str(requested_policy or "borrowed_stream_event_chain"),
        requested_event_chain=bool(request_event_chain),
    )


def stream_guard_descriptor_for_runtime_launch(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the descriptor fields consumed by the native runtime."""

    payload = dict(_as_dict(evidence.get("stream_guard_descriptor")))
    payload["runtime_stream_guard_evidence_contract"] = str(evidence.get("contract", CONTRACT) or CONTRACT)
    payload["runtime_stream_guard_evidence_ready"] = bool(evidence.get("ready_for_borrowed_stream_launch", False))
    payload["blocked_reasons"] = list(evidence.get("blocked_reasons", []) or [])
    return payload


def _normalize_evidence(
    *,
    descriptor: Mapping[str, Any],
    source: str,
    source_probe: Mapping[str, Any],
    errors: list[str],
    lifetime_lease_evidence: Mapping[str, Any],
    runtime_context: Mapping[str, Any],
    requested_policy: str,
    requested_event_chain: bool,
) -> dict[str, Any]:
    data = _as_dict(descriptor)
    stream_handle = _int_field(data, "cuda_stream_handle", "stream_handle")
    handle_field_present = "cuda_stream_handle" in data or "stream_handle" in data
    stream_handle_reported = bool(data.get("stream_handle_reported", handle_field_present)) and handle_field_present
    stream_handle_nonzero = stream_handle != 0 and bool(data.get("stream_handle_nonzero", stream_handle != 0))
    device_type = str(data.get("device_type", "cuda" if stream_handle_nonzero else "unknown") or "unknown")
    handle_kind = str(data.get("stream_handle_kind", "") or "")
    if not handle_kind:
        handle_kind = _stream_handle_kind(device_type, stream_handle_reported, stream_handle_nonzero)

    normalized = {
        "schema_version": 1,
        "descriptor": RUNTIME_DESCRIPTOR,
        "source_contract": str(data.get("contract", data.get("descriptor", "")) or ""),
        "device_type": device_type,
        "device_index": data.get("device_index"),
        "stream_kind": str(data.get("stream_kind", "") or ""),
        "stream_id": str(data.get("stream_id", "") or ""),
        "stream_source": str(data.get("stream_source", source) or source),
        "stream_capture_stage": str(data.get("stream_capture_stage", "native_adamw_runtime_step") or ""),
        "cuda_stream_handle": stream_handle,
        "stream_handle_reported": stream_handle_reported,
        "stream_handle_nonzero": stream_handle_nonzero,
        "stream_handle_kind": handle_kind,
        "event_chain_probe_requested": bool(data.get("event_chain_probe_requested", requested_event_chain)),
        "event_chain_probe_attempted": bool(data.get("event_chain_probe_attempted", False)),
        "event_chain_verified": bool(data.get("event_chain_verified", False)),
        "pre_launch_ordering_verified": bool(data.get("pre_launch_ordering_verified", False)),
        "post_launch_ordering_verified": bool(data.get("post_launch_ordering_verified", False)),
        "stream_wait_event_verified": bool(data.get("stream_wait_event_verified", False)),
        "stream_lifetime_bound": bool(data.get("stream_lifetime_bound", False)),
        "borrowed_external_stream": bool(data.get("borrowed_external_stream", stream_handle_reported)),
        "external_stream_borrow_verified": bool(data.get("external_stream_borrow_verified", False)),
        "device_match": bool(data.get("device_match", data.get("stream_device_match", False))),
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_kernel_present": False,
        "performance_test_ready": False,
    }
    lease_source = _lease_source(data, lifetime_lease_evidence)
    lease = build_stream_lifetime_lease_evidence(
        stream_guard=normalized,
        lease_evidence=lease_source,
        runtime_context=runtime_context,
        requested_policy=requested_policy,
    )
    lease_ready = bool(lease.get("ready_for_runtime_stream_guard", False))
    normalized["stream_lifetime_lease_evidence"] = lease
    normalized["runtime_stream_lifetime_lease_ready"] = lease_ready
    normalized["stream_lifetime_bound"] = bool(normalized["stream_lifetime_bound"] or lease_ready)
    normalized["native_error_recovery_verified"] = bool(lease.get("native_error_recovery_verified", False))
    blockers = _required_blockers(normalized)
    blockers.extend(_filter_lifetime_blockers(_strings(data.get("blocked_reasons")), lease_ready=lease_ready))
    blockers.extend(_filter_lifetime_blockers(_strings(source_probe.get("blocked_reasons")), lease_ready=lease_ready))
    if not lease_ready:
        blockers.extend(_strings(lease.get("blocked_reasons")))
    if errors:
        blockers.append("v5_p10_stream_guard_evidence_probe_error")
    ready = not blockers
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "ok": ready,
        "ready_for_borrowed_stream_launch": ready,
        "ctx_synchronize_free_step_allowed": ready,
        "source": source,
        "requested_event_chain": requested_event_chain,
        "default_behavior_changed": False,
        "requires_explicit_opt_in": True,
        "training_path_enabled": False,
        "default_training_path_enabled": False,
        "auto_rollout_allowed": False,
        "manual_wider_canary_allowed": False,
        "stream_guard_descriptor": normalized,
        "stream_lifetime_lease_evidence": lease,
        "source_probe_present": bool(source_probe),
        "source_probe_ok": bool(source_probe.get("ok", False)) if source_probe else None,
        "errors": errors,
        "blocked_reasons": _dedupe(blockers),
    }


def _required_blockers(descriptor: Mapping[str, Any]) -> list[str]:
    checks = {
        "v5_p10_stream_handle_reported_missing": bool(descriptor.get("stream_handle_reported", False)),
        "v5_p10_stream_handle_nonzero_missing": bool(descriptor.get("stream_handle_nonzero", False)),
        "v5_p10_external_cuda_stream_missing": str(descriptor.get("stream_handle_kind", "") or "")
        == "external_cuda_stream_handle",
        "v5_p10_event_chain_verified_missing": bool(descriptor.get("event_chain_verified", False)),
        "v5_p10_pre_launch_ordering_missing": bool(descriptor.get("pre_launch_ordering_verified", False)),
        "v5_p10_post_launch_ordering_missing": bool(descriptor.get("post_launch_ordering_verified", False)),
        "v5_p10_stream_wait_event_missing": bool(descriptor.get("stream_wait_event_verified", False)),
        "v5_p10_stream_lifetime_bound_missing": bool(descriptor.get("stream_lifetime_bound", False)),
        "v5_p11_stream_lifetime_lease_missing": bool(
            descriptor.get("runtime_stream_lifetime_lease_ready", False)
        ),
    }
    return [name for name, ok in checks.items() if not ok]


def _lease_source(
    descriptor: Mapping[str, Any],
    lifetime_lease_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    for key in (
        "stream_lifetime_lease_evidence",
        "runtime_stream_lifetime_lease_evidence",
        "lifetime_lease_evidence",
    ):
        value = _as_dict(descriptor.get(key))
        if value:
            return value
    return dict(lifetime_lease_evidence)


def _filter_lifetime_blockers(values: list[str], *, lease_ready: bool) -> list[str]:
    if not lease_ready:
        return values
    filtered = {
        "stream_lifetime_not_bound",
        "stream_lifetime_unbound",
        "cuda_stream_handle_not_owned",
        "native_error_recovery_missing",
        "stream_guard_not_ready",
    }
    return [value for value in values if value not in filtered]


def _extract_stream_guard(report: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("stream_guard_descriptor", "stream_guard_probe", "stream_guard"):
        value = _as_dict(report.get(key))
        if value:
            return value
    native_session = _as_dict(report.get("native_session"))
    for key in ("stream_guard_probe", "stream_guard_descriptor", "stream_guard"):
        value = _as_dict(native_session.get(key))
        if value:
            return value
    if any(key in report for key in ("stream_handle_nonzero", "event_chain_verified", "stream_lifetime_bound")):
        return dict(report)
    return {}


def _current_owner_stream_descriptor(owner: Any, *, request_event_chain: bool) -> dict[str, Any]:
    tensor = getattr(owner, "param_flat", None)
    device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
    try:
        return current_torch_stream_descriptor(
            device,
            capture_stage="native_adamw_runtime_step",
            request_event_chain=bool(request_event_chain),
        )
    except Exception as exc:  # pragma: no cover - CUDA runtime dependent
        return {
            "schema_version": 1,
            "descriptor": "turbocore_borrowed_cuda_stream_descriptor_v0",
            "device_type": str(device.type),
            "event_chain_probe_requested": bool(request_event_chain),
            "stream_handle_reported": False,
            "stream_handle_nonzero": False,
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": ["current_torch_stream_descriptor_error"],
        }


def _owner_has_cuda_param(owner: Any) -> bool:
    tensor = getattr(owner, "param_flat", None)
    return isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda"


def _stream_handle_kind(device_type: str, reported: bool, nonzero: bool) -> str:
    if device_type != "cuda":
        return "non_cuda_stream"
    if not reported:
        return "missing_cuda_stream_handle"
    if nonzero:
        return "external_cuda_stream_handle"
    return "cuda_default_stream_zero"


def _int_field(value: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        item = value.get(key)
        if isinstance(item, bool):
            continue
        try:
            return int(item)
        except (TypeError, ValueError):
            continue
    return 0


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
    "build_runtime_stream_guard_evidence",
    "stream_guard_descriptor_for_runtime_launch",
]
