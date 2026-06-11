"""Report-only native tensor binding probe for TurboCore update shadow."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

from core.turbocore_capabilities import probe_native_training_bridge
from core.turbocore_native_tensor_binding import build_flat_adamw_native_binding_request
from core.services.native_module_loader import load_lulynx_native, probe_lulynx_native_loader
from core.turbocore_tensor_handle_registry import (
    build_tensor_object_map_for_handles,
    register_persistent_flat_adamw_buffers,
)


def build_update_native_binding_probe(owner: Any, *, request_event_chain: bool = False) -> dict[str, Any]:
    """Validate the current-process tensor binding shape without dispatch."""

    started = time.perf_counter()
    blocked: list[str] = []
    try:
        flat_owner_capability = _flat_owner_capability()
        registry, handles, descriptor = register_persistent_flat_adamw_buffers(owner)
        tensor_map = build_tensor_object_map_for_handles(registry, handles)
        request = build_flat_adamw_native_binding_request(
            registry,
            handles,
            native_flat_owner_capability=flat_owner_capability,
        )
        readiness = dict(request.get("readiness") or {})
        native_session = _native_session_probe(request, tensor_map, request_event_chain=request_event_chain)
        blocked.extend(_strings(readiness.get("blocked_reasons")))
        blocked.extend(_strings(native_session.get("blocked_reasons")))
        stream_lifetime_bound = bool(native_session.get("stream_lifetime_bound", False))
        stream_guard_ready = bool(native_session.get("stream_guard_ready", False))
        stream_identity_ready = bool(native_session.get("stream_identity_ready", False))
        if not stream_lifetime_bound:
            blocked.append("stream_lifetime_unbound")
        if native_session.get("stream_guard_present") is not None and not stream_guard_ready:
            blocked.append("stream_guard_not_ready")
        if native_session.get("stream_guard_present") is not None and not stream_identity_ready:
            blocked.append("stream_identity_not_ready")
        request_shape_ready = bool(readiness.get("request_shape_ready", False))
        tensor_object_map_ready = len(tensor_map) == 4
        session_ok = bool(native_session.get("ok", True)) if not bool(native_session.get("attempted", False)) else bool(native_session.get("ok", False))
        ok = bool(request_shape_ready and tensor_object_map_ready and session_ok)
        payload = {
            "schema_version": 1,
            "probe": "turbocore_update_native_binding_probe_v0",
            "ok": ok,
            "training_path_enabled": False,
            "native_kernel_present": bool(readiness.get("performance_test_ready", False)) and bool(native_session.get("native_kernel_present", False)),
            "performance_test_ready": False,
            "stream_lifetime_bound": stream_lifetime_bound,
            "stream_guard_present": bool(native_session.get("stream_guard_present", False)),
            "stream_guard_ready": stream_guard_ready,
            "stream_identity_ready": stream_identity_ready,
            "stream_guard_level": str(native_session.get("stream_guard_level", "") or ""),
            "stream_handle_kind": str(native_session.get("stream_handle_kind", "") or ""),
            "stream_handle_reported": bool(native_session.get("stream_handle_reported", False)),
            "stream_handle_nonzero": bool(native_session.get("stream_handle_nonzero", False)),
            "synchronization_guard_ready": bool(native_session.get("synchronization_guard_ready", False)),
            "synchronization_strategy": str(native_session.get("synchronization_strategy", "") or ""),
            "event_chain_contract": str(native_session.get("event_chain_contract", "") or ""),
            "event_chain_state": str(native_session.get("event_chain_state", "") or ""),
            "event_chain_probe_requested": bool(native_session.get("event_chain_probe_requested", False)),
            "event_chain_probe_attempted": bool(native_session.get("event_chain_probe_attempted", False)),
            "event_chain_verified": bool(native_session.get("event_chain_verified", False)),
            "pre_launch_ordering_verified": bool(native_session.get("pre_launch_ordering_verified", False)),
            "post_launch_ordering_verified": bool(native_session.get("post_launch_ordering_verified", False)),
            "stream_wait_event_verified": bool(native_session.get("stream_wait_event_verified", False)),
            "native_launch_candidate": bool(native_session.get("native_launch_candidate", False)),
            "borrowed_external_stream": bool(native_session.get("borrowed_external_stream", False)),
            "stream_device_match": bool(native_session.get("stream_device_match", False)),
            "handle_count": len(handles),
            "tensor_object_map_ready": tensor_object_map_ready,
            "descriptor_layout": str(descriptor.get("layout", "") or ""),
            "descriptor_stream": dict(descriptor.get("stream") or {}),
            "stream_contract": dict(native_session.get("stream_contract") or {}),
            "request_shape_ready": request_shape_ready,
            "native_binding_ready": bool(readiness.get("native_binding_ready", False)) and bool(native_session.get("native_binding_ready", False)),
            "tensor_object_binding_ready": bool(native_session.get("tensor_object_binding_ready", False)),
            "launch_plan_ready": bool(native_session.get("launch_plan_ready", False)),
            "launch_plan_kind": str(native_session.get("launch_plan_kind", "") or ""),
            "launch_plan_numel": int(native_session.get("launch_plan_numel", 0) or 0),
            "stream_lease_id": int(native_session.get("stream_lease_id", 0) or 0),
            "native_session": native_session,
            "readiness": readiness,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4),
            "blocked_reasons": _dedupe(blocked),
        }
    except Exception as exc:  # pragma: no cover - defensive research probe
        payload = {
            "schema_version": 1,
            "probe": "turbocore_update_native_binding_probe_v0",
            "ok": False,
            "training_path_enabled": False,
            "native_kernel_present": False,
            "performance_test_ready": False,
            "stream_lifetime_bound": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4),
            "blocked_reasons": ["native_binding_probe_error"],
        }
    return payload


def _flat_owner_capability() -> dict[str, Any]:
    try:
        bridge = probe_native_training_bridge()
    except Exception:
        return {}
    features = bridge.get("features") if isinstance(bridge, Mapping) else {}
    optimizer = _as_dict(_as_dict(features).get("native_optimizer"))
    return _as_dict(optimizer.get("flat_owner"))


def _native_session_probe(
    request: Mapping[str, Any],
    tensor_map: Mapping[str, Any],
    *,
    request_event_chain: bool,
) -> dict[str, Any]:
    loader = probe_lulynx_native_loader()
    native = load_lulynx_native()
    if native is None:
        return {
            "schema_version": 1,
            "ok": True,
            "attempted": False,
            "skipped": True,
            "reason": "lulynx_native_not_importable",
            "loader": loader,
            "training_path_enabled": False,
            "native_kernel_present": False,
            "performance_test_ready": False,
            "stream_lifetime_bound": False,
            "blocked_reasons": ["lulynx_native_not_importable"],
        }
    required = [
        "validate_flat_adamw_tensor_binding_request",
        "probe_flat_adamw_tensor_object_binding",
        "create_flat_adamw_tensor_binding_session",
        "tensor_binding_session_snapshot",
        "tensor_binding_session_validate",
        "tensor_binding_session_launch_plan",
        "tensor_binding_session_stream_guard_probe",
        "tensor_binding_session_noop_launch",
        "destroy_tensor_binding_session",
    ]
    missing = [name for name in required if not hasattr(native, name)]
    if missing:
        return {
            "schema_version": 1,
            "ok": False,
            "attempted": True,
            "skipped": False,
            "reason": "lulynx_native_tensor_binding_entrypoints_missing",
            "loader": loader,
            "missing_entrypoints": missing,
            "training_path_enabled": False,
            "native_kernel_present": False,
            "performance_test_ready": False,
            "stream_lifetime_bound": False,
            "blocked_reasons": ["native_tensor_binding_entrypoints_missing"],
        }
    text = json.dumps(dict(request))
    session_id: int | None = None
    try:
        validation = native.validate_flat_adamw_tensor_binding_request(text)
        object_probe = native.probe_flat_adamw_tensor_object_binding(text, dict(tensor_map))
        session = native.create_flat_adamw_tensor_binding_session(text, dict(tensor_map))
        session_id = int(session.get("session_id", 0) or 0) if isinstance(session, dict) and session.get("ok") else None
        snapshot = native.tensor_binding_session_snapshot(session_id) if session_id is not None else {}
        session_validation = native.tensor_binding_session_validate(session_id) if session_id is not None else {}
        launch_plan = native.tensor_binding_session_launch_plan(session_id) if session_id is not None else {}
        stream_descriptor = _current_stream_descriptor(request, request_event_chain=request_event_chain)
        stream_guard = (
            native.tensor_binding_session_stream_guard_probe(session_id, json.dumps(stream_descriptor))
            if session_id is not None
            else {}
        )
        noop_launch = native.tensor_binding_session_noop_launch(session_id) if session_id is not None else {}
        blocked = _collect_native_blockers(validation, object_probe, session_validation, launch_plan, stream_guard, noop_launch)
        stream_contract = _stream_contract_from_reports(launch_plan, snapshot, session_validation)
        stream_lifetime_bound = bool(stream_contract.get("stream_lifetime_bound", False))
        return {
            "schema_version": 1,
            "ok": bool(validation.get("ok", False) and object_probe.get("ok", False) and session.get("ok", False)),
            "attempted": True,
            "skipped": False,
            "origin": str(loader.get("origin", "") or getattr(native, "__file__", "") or ""),
            "loader": loader,
            "training_path_enabled": False,
            "native_kernel_present": bool(launch_plan.get("native_kernel_present", False)) or bool(noop_launch.get("native_kernel_present", False)),
            "performance_test_ready": False,
            "stream_lifetime_bound": stream_lifetime_bound,
            "stream_contract": stream_contract,
            "stream_descriptor": stream_descriptor,
            "stream_guard_probe": dict(stream_guard) if isinstance(stream_guard, Mapping) else {},
            "stream_guard_present": bool(_as_dict(stream_guard).get("stream_guard_present", False)),
            "stream_guard_ready": bool(_as_dict(stream_guard).get("stream_guard_ready", False)),
            "stream_identity_ready": bool(_as_dict(stream_guard).get("stream_identity_ready", False)),
            "stream_guard_level": str(_as_dict(stream_guard).get("stream_guard_level", "") or ""),
            "stream_handle_kind": str(_as_dict(stream_guard).get("stream_handle_kind", "") or ""),
            "stream_handle_reported": bool(_as_dict(stream_guard).get("stream_handle_reported", False)),
            "stream_handle_nonzero": bool(_as_dict(stream_guard).get("stream_handle_nonzero", False)),
            "synchronization_guard_ready": bool(_as_dict(stream_guard).get("synchronization_guard_ready", False)),
            "synchronization_strategy": str(_as_dict(stream_guard).get("synchronization_strategy", "") or ""),
            "event_chain_contract": str(_as_dict(stream_guard).get("event_chain_contract", "") or ""),
            "event_chain_state": str(_as_dict(stream_guard).get("event_chain_state", "") or ""),
            "event_chain_probe_requested": bool(_as_dict(stream_guard).get("event_chain_probe_requested", False)),
            "event_chain_probe_attempted": bool(_as_dict(stream_guard).get("event_chain_probe_attempted", False)),
            "event_chain_verified": bool(_as_dict(stream_guard).get("event_chain_verified", False)),
            "pre_launch_ordering_verified": bool(_as_dict(stream_guard).get("pre_launch_ordering_verified", False)),
            "post_launch_ordering_verified": bool(_as_dict(stream_guard).get("post_launch_ordering_verified", False)),
            "stream_wait_event_verified": bool(_as_dict(stream_guard).get("stream_wait_event_verified", False)),
            "native_launch_candidate": bool(_as_dict(stream_guard).get("native_launch_candidate", False)),
            "borrowed_external_stream": bool(_as_dict(stream_guard).get("borrowed_external_stream", False)),
            "stream_device_match": bool(_as_dict(stream_guard).get("device_match", False)),
            "request_validation_ok": bool(validation.get("ok", False)),
            "tensor_object_binding_ready": bool(object_probe.get("tensor_object_binding_ready", False)),
            "session_created": bool(session.get("ok", False)),
            "holds_python_tensor_refs": bool(session.get("holds_python_tensor_refs", False)) or bool(snapshot.get("holds_python_tensor_refs", False)),
            "snapshot_ok": bool(snapshot.get("ok", False)),
            "session_validation_ok": bool(session_validation.get("ok", False)),
            "launch_plan_ready": bool(launch_plan.get("ok", False)),
            "launch_plan_kind": str(launch_plan.get("plan_kind", "") or ""),
            "launch_plan_numel": int(launch_plan.get("numel", 0) or 0),
            "stream_lease_id": int(launch_plan.get("stream_lease_id", 0) or 0),
            "launchable_by_cuda_kernel": bool(launch_plan.get("launchable_by_cuda_kernel", False)),
            "noop_launch_ready": bool(noop_launch.get("ok", False)),
            "noop_parameters_mutated": bool(noop_launch.get("parameters_mutated", False)),
            "native_binding_ready": bool(launch_plan.get("native_binding_ready", False)),
            "blocked_reasons": _dedupe(blocked),
        }
    except Exception as exc:  # pragma: no cover - depends on local native build
        return {
            "schema_version": 1,
            "ok": False,
            "attempted": True,
            "skipped": False,
            "reason": "native_tensor_binding_session_probe_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "training_path_enabled": False,
            "native_kernel_present": False,
            "performance_test_ready": False,
            "stream_lifetime_bound": False,
            "blocked_reasons": ["native_tensor_binding_session_probe_failed"],
        }
    finally:
        if session_id is not None:
            try:
                native.destroy_tensor_binding_session(session_id)
            except Exception:
                pass


def _collect_native_blockers(*reports: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    for report in reports:
        blocked.extend(_strings(_as_dict(report).get("blocked_reasons")))
    return blocked


def _current_stream_descriptor(request: Mapping[str, Any], *, request_event_chain: bool) -> dict[str, Any]:
    descriptor = _as_dict(request.get("descriptor"))
    stream = _as_dict(descriptor.get("stream"))
    device_type = str(stream.get("device_type", "unknown") or "unknown")
    device_index = stream.get("device_index")
    normalized_index = int(device_index) if isinstance(device_index, int) else None
    payload: dict[str, Any] = {
        "schema_version": 1,
        "descriptor": "turbocore_borrowed_cuda_stream_descriptor_v0",
        "device_type": device_type,
        "device_index": normalized_index,
        "stream_kind": str(stream.get("stream_kind", "default") or "default"),
        "stream_id": str(stream.get("stream_id", "") or ""),
        "stream_source": "request_descriptor",
        "stream_capture_stage": "native_binding_probe",
        "python_stream_object_alive": False,
        "python_stream_lifetime_scope": "descriptor_only",
        "cuda_stream_handle": 0,
        "stream_handle_reported": False,
        "stream_handle_nonzero": False,
        "event_chain_probe_requested": bool(request_event_chain),
        "synchronize": bool(stream.get("synchronize", False)),
        "training_path_enabled": False,
    }
    if device_type != "cuda":
        payload["stream_source"] = "cpu_no_cuda_stream"
        return payload
    try:
        import torch

        if not bool(torch.cuda.is_available()):
            payload["stream_source"] = "torch_cuda_unavailable"
            payload["blocked_reasons"] = ["torch_cuda_unavailable"]
            return payload
        device = normalized_index if normalized_index is not None else torch.cuda.current_device()
        stream_obj = torch.cuda.current_stream(device)
        handle = int(getattr(stream_obj, "cuda_stream", 0) or 0)
        payload.update(
            {
                "device_index": int(device),
                "stream_kind": "torch_current",
                "stream_id": str(handle) if handle else "",
                "stream_source": "torch.cuda.current_stream",
                "stream_capture_stage": "native_binding_probe",
                "python_stream_object_alive": True,
                "python_stream_lifetime_scope": "descriptor_only",
                "cuda_stream_handle": handle,
                "stream_handle_reported": True,
                "stream_handle_nonzero": bool(handle),
                "blocked_reasons": ["external_stream_borrow_not_verified", "stream_lifetime_not_bound"],
            }
        )
    except Exception as exc:  # pragma: no cover - depends on local CUDA runtime
        payload["stream_source"] = "torch_cuda_current_stream_error"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["blocked_reasons"] = ["torch_cuda_current_stream_error"]
    return payload


def _stream_contract_from_reports(*reports: Mapping[str, Any]) -> dict[str, Any]:
    for report in reports:
        data = _as_dict(report)
        contract = _as_dict(data.get("stream_contract"))
        if contract:
            return contract
    return {
        "schema_version": 1,
        "contract": "turbocore_tensor_binding_stream_lifetime_v0",
        "stream_lifetime_bound": False,
        "training_path_enabled": False,
        "native_kernel_present": False,
        "blocked_reasons": ["stream_lifetime_not_reported"],
    }


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


__all__ = ["build_update_native_binding_probe"]
