"""Native tensor binding request contract for TurboCore flat AdamW.

This module sits above the current-process tensor handle registry and below a
future Rust/CUDA binding layer.  It builds a JSON-safe request shape that native
code can eventually consume, while making the current limitation explicit: no
raw pointer export, no cross-process handle use, and no training dispatch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from core.turbocore_flat_buffer_descriptor import (
    ADAMW_FLAT_BUFFER_ROLES,
    validate_flat_adamw_owner_descriptor,
)
from core.turbocore_tensor_handle_registry import (
    TORCH_TENSOR_HANDLE_KIND,
    TurboCoreTensorHandleRegistry,
    build_flat_adamw_owner_descriptor_from_handles,
)


NATIVE_TENSOR_BINDING_REQUEST_SCHEMA = "turbocore_native_tensor_binding_request_v1"


@dataclass(frozen=True)
class NativeTensorBinding:
    role: str
    handle_id: str
    handle_kind: str
    numel: int
    dtype: str
    device_type: str
    device_index: int | None
    layout: str
    contiguous: bool
    alignment_bytes: int
    pointer_exported: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_flat_adamw_native_binding_request(
    registry: TurboCoreTensorHandleRegistry,
    role_handles: Mapping[str, str],
    *,
    native_flat_owner_capability: Mapping[str, Any] | None = None,
    stream_kind: str = "default",
    stream_id: str = "",
    synchronize: bool = False,
) -> dict[str, Any]:
    """Build a future native binding request from registered AdamW handles."""

    descriptor = build_flat_adamw_owner_descriptor_from_handles(
        registry,
        role_handles,
        stream_kind=stream_kind,
        stream_id=stream_id,
        synchronize=synchronize,
    )
    bindings = []
    for buffer in descriptor["buffers"]:
        handle_id = str(buffer.get("handle_id", "") or "")
        record = registry.record(handle_id)
        bindings.append(
            NativeTensorBinding(
                role=record.role,
                handle_id=record.handle_id,
                handle_kind=record.handle_kind,
                numel=record.numel,
                dtype=record.dtype,
                device_type=record.device_type,
                device_index=record.device_index,
                layout=record.layout,
                contiguous=record.contiguous,
                alignment_bytes=record.alignment_bytes,
                pointer_exported=False,
            ).as_dict()
        )
    request = {
        "schema_version": 1,
        "schema": NATIVE_TENSOR_BINDING_REQUEST_SCHEMA,
        "optimizer": "AdamW",
        "layout": "flat_contiguous_fp32_buffers",
        "descriptor_schema": "flat_adamw_owner_descriptor_v1",
        "descriptor": descriptor,
        "bindings": bindings,
        "pointer_exported": False,
        "training_path_enabled": False,
        "notes": [
            "current_process_handles_only",
            "no_raw_pointer_export",
            "no_training_dispatch",
        ],
    }
    request["readiness"] = evaluate_flat_adamw_native_binding_request(
        request,
        native_flat_owner_capability=native_flat_owner_capability,
    )
    return request


def evaluate_flat_adamw_native_binding_request(
    request: Mapping[str, Any] | None,
    *,
    native_flat_owner_capability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return readiness facts for a flat AdamW native binding request."""

    data = request if isinstance(request, Mapping) else {}
    descriptor = data.get("descriptor") if isinstance(data.get("descriptor"), dict) else {}
    descriptor_validation = validate_flat_adamw_owner_descriptor(descriptor)
    bindings = data.get("bindings") if isinstance(data.get("bindings"), list) else []
    reported_roles = [str(item.get("role", "") or "") for item in bindings if isinstance(item, dict)]
    missing_roles = [role for role in ADAMW_FLAT_BUFFER_ROLES if role not in reported_roles]
    invalid_bindings = _invalid_binding_reasons(bindings)
    pointer_exported = bool(data.get("pointer_exported", False)) or any(
        bool(item.get("pointer_exported", False)) for item in bindings if isinstance(item, dict)
    )
    shape_ready = bool(
        data
        and data.get("schema") == NATIVE_TENSOR_BINDING_REQUEST_SCHEMA
        and descriptor_validation["ok"]
        and not missing_roles
        and not invalid_bindings
        and not pointer_exported
    )
    native = _as_dict(native_flat_owner_capability)
    native_external_handles = bool(native.get("supports_external_tensor_handles", False))
    native_kernel_present = bool(native.get("native_kernel_present", False))
    native_available = bool(native.get("available", False))
    native_binding_ready = bool(shape_ready and native_available and native_external_handles)
    performance_test_ready = bool(native_binding_ready and native_kernel_present)
    return {
        "schema_version": 1,
        "validator": "turbocore_flat_adamw_native_tensor_binding_request",
        "ok": shape_ready,
        "request_shape_ready": shape_ready,
        "native_binding_ready": native_binding_ready,
        "performance_test_ready": performance_test_ready,
        "training_path_enabled": False,
        "descriptor_ok": bool(descriptor_validation["ok"]),
        "descriptor_validation": descriptor_validation,
        "required_roles": list(ADAMW_FLAT_BUFFER_ROLES),
        "reported_roles": sorted(role for role in reported_roles if role),
        "missing_roles": missing_roles,
        "invalid_bindings": invalid_bindings,
        "pointer_exported": pointer_exported,
        "native_capability": {
            "available": native_available,
            "supports_external_tensor_handles": native_external_handles,
            "native_kernel_present": native_kernel_present,
            "training_path_enabled": bool(native.get("training_path_enabled", False)),
        },
        "blocked_reasons": _blocked_reasons(
            shape_ready=shape_ready,
            native_available=native_available,
            native_external_handles=native_external_handles,
            native_kernel_present=native_kernel_present,
        ),
    }


def _invalid_binding_reasons(bindings: list[Any]) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in bindings:
        binding = item if isinstance(item, dict) else {}
        role = str(binding.get("role", "") or "")
        reasons: list[str] = []
        if role not in ADAMW_FLAT_BUFFER_ROLES:
            reasons.append("unsupported_role")
        if role in seen:
            reasons.append("duplicate_role")
        seen.add(role)
        if str(binding.get("handle_kind", "") or "") != TORCH_TENSOR_HANDLE_KIND:
            reasons.append("unsupported_handle_kind")
        if not str(binding.get("handle_id", "") or ""):
            reasons.append("missing_handle_id")
        if str(binding.get("dtype", "") or "") != "float32":
            reasons.append("unsupported_dtype")
        if str(binding.get("layout", "") or "") != "flat_contiguous":
            reasons.append("unsupported_layout")
        if bool(binding.get("contiguous", False)) is not True:
            reasons.append("not_contiguous")
        if bool(binding.get("pointer_exported", False)):
            reasons.append("raw_pointer_export_not_allowed")
        if reasons:
            invalid.append({"role": role, "reasons": reasons})
    return invalid


def _blocked_reasons(
    *,
    shape_ready: bool,
    native_available: bool,
    native_external_handles: bool,
    native_kernel_present: bool,
) -> list[str]:
    reasons: list[str] = []
    if not shape_ready:
        reasons.append("binding_request_shape_not_ready")
    if not native_available:
        reasons.append("native_flat_owner_unavailable")
    if not native_external_handles:
        reasons.append("native_external_tensor_handles_unsupported")
    if not native_kernel_present:
        reasons.append("native_kernel_missing")
    return reasons


def _as_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = [
    "NATIVE_TENSOR_BINDING_REQUEST_SCHEMA",
    "NativeTensorBinding",
    "build_flat_adamw_native_binding_request",
    "evaluate_flat_adamw_native_binding_request",
]
