"""Flat buffer descriptor contract for future TurboCore native owners.

The contract is metadata-only.  It describes buffer roles, dtype/device, layout,
and stream intent before real tensor-handle ownership is wired into training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


ADAMW_FLAT_BUFFER_ROLES = ("param_flat", "grad_flat", "exp_avg", "exp_avg_sq")
SUPPORTED_FLAT_DTYPES = ("float32",)
SUPPORTED_HANDLE_KINDS = ("json_reference", "torch_tensor", "native_tensor_handle")


@dataclass(frozen=True)
class FlatBufferDescriptor:
    role: str
    numel: int
    dtype: str = "float32"
    device_type: str = "cpu"
    device_index: int | None = None
    layout: str = "flat_contiguous"
    contiguous: bool = True
    alignment_bytes: int = 16
    handle_kind: str = "json_reference"
    handle_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StreamDescriptor:
    device_type: str = "cpu"
    device_index: int | None = None
    stream_kind: str = "default"
    stream_id: str = ""
    synchronize: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FlatAdamWOwnerDescriptor:
    schema_version: int
    layout: str
    buffers: tuple[FlatBufferDescriptor, ...]
    stream: StreamDescriptor
    owns_parameter_buffer: bool = True
    owns_gradient_buffer: bool = True
    training_path_enabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["buffers"] = [buffer.as_dict() for buffer in self.buffers]
        payload["stream"] = self.stream.as_dict()
        return payload


def build_reference_flat_adamw_owner_descriptor(
    *,
    numel: int,
    device_type: str = "cpu",
    device_index: int | None = None,
    dtype: str = "float32",
    handle_kind: str = "json_reference",
) -> dict[str, Any]:
    buffers = tuple(
        FlatBufferDescriptor(
            role=role,
            numel=max(int(numel), 0),
            dtype=str(dtype),
            device_type=str(device_type),
            device_index=device_index,
            handle_kind=str(handle_kind),
            handle_id=f"{handle_kind}:{role}",
        )
        for role in ADAMW_FLAT_BUFFER_ROLES
    )
    return FlatAdamWOwnerDescriptor(
        schema_version=1,
        layout="flat_contiguous_fp32_buffers",
        buffers=buffers,
        stream=StreamDescriptor(device_type=str(device_type), device_index=device_index),
    ).as_dict()


def validate_flat_adamw_owner_descriptor(payload: dict[str, Any] | None) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    buffers = data.get("buffers") if isinstance(data.get("buffers"), list) else []
    seen: dict[str, dict[str, Any]] = {}
    duplicate_roles: list[str] = []
    invalid_buffers: list[dict[str, Any]] = []
    for raw in buffers:
        buffer = raw if isinstance(raw, dict) else {}
        role = str(buffer.get("role", "") or "")
        if role in seen:
            duplicate_roles.append(role)
        seen[role] = buffer
        invalid_reasons = _validate_buffer(buffer)
        if invalid_reasons:
            invalid_buffers.append({"role": role, "reasons": invalid_reasons})
    missing_roles = [role for role in ADAMW_FLAT_BUFFER_ROLES if role not in seen]
    role_numels = {
        role: int(seen[role].get("numel", -1))
        for role in ADAMW_FLAT_BUFFER_ROLES
        if role in seen and _is_intish(seen[role].get("numel"))
    }
    numel_values = set(role_numels.values())
    stream = data.get("stream") if isinstance(data.get("stream"), dict) else {}
    stream_reasons = _validate_stream(stream)
    layout = str(data.get("layout", "") or "")
    ok = bool(data) and layout == "flat_contiguous_fp32_buffers" and not missing_roles and not duplicate_roles and not invalid_buffers and not stream_reasons and len(numel_values) <= 1
    return {
        "schema_version": 1,
        "validator": "turbocore_flat_adamw_owner_descriptor",
        "ok": ok,
        "layout": layout,
        "required_roles": list(ADAMW_FLAT_BUFFER_ROLES),
        "reported_roles": sorted(role for role in seen if role),
        "missing_roles": missing_roles,
        "duplicate_roles": duplicate_roles,
        "invalid_buffers": invalid_buffers,
        "role_numels": role_numels,
        "numel_mismatch": len(numel_values) > 1,
        "stream_reasons": stream_reasons,
        "training_path_enabled": bool(data.get("training_path_enabled", False)),
    }


def _validate_buffer(buffer: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    role = str(buffer.get("role", "") or "")
    if role not in ADAMW_FLAT_BUFFER_ROLES:
        reasons.append("unsupported_role")
    if not _is_intish(buffer.get("numel")) or int(buffer.get("numel", -1)) < 0:
        reasons.append("invalid_numel")
    if str(buffer.get("dtype", "") or "") not in SUPPORTED_FLAT_DTYPES:
        reasons.append("unsupported_dtype")
    if str(buffer.get("layout", "") or "") != "flat_contiguous":
        reasons.append("unsupported_layout")
    if bool(buffer.get("contiguous", False)) is not True:
        reasons.append("not_contiguous")
    handle_kind = str(buffer.get("handle_kind", "") or "")
    if handle_kind not in SUPPORTED_HANDLE_KINDS:
        reasons.append("unsupported_handle_kind")
    if handle_kind in {"torch_tensor", "native_tensor_handle"} and not str(buffer.get("handle_id", "") or ""):
        reasons.append("missing_handle_id")
    return reasons


def _validate_stream(stream: dict[str, Any]) -> list[str]:
    if not stream:
        return []
    kind = str(stream.get("stream_kind", "default") or "default")
    if kind not in {"default", "cuda_stream", "external"}:
        return ["unsupported_stream_kind"]
    return []


def _is_intish(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


__all__ = [
    "ADAMW_FLAT_BUFFER_ROLES",
    "FlatAdamWOwnerDescriptor",
    "FlatBufferDescriptor",
    "StreamDescriptor",
    "build_reference_flat_adamw_owner_descriptor",
    "validate_flat_adamw_owner_descriptor",
]
