"""Current-process tensor handle registry for TurboCore flat buffers.

The registry is a developer-only ownership boundary for the next TurboCore
optimizer stage.  Handles are opaque IDs that keep strong Python references to
``torch.Tensor`` objects in the current process.  They are not raw CUDA
pointers, are not serializable across processes, and do not enable training
dispatch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping
from uuid import uuid4

import torch

from core.turbocore_flat_buffer_descriptor import (
    ADAMW_FLAT_BUFFER_ROLES,
    SUPPORTED_FLAT_DTYPES,
    FlatAdamWOwnerDescriptor,
    FlatBufferDescriptor,
    StreamDescriptor,
    validate_flat_adamw_owner_descriptor,
)


TURBOCORE_TENSOR_HANDLE_SCHEMA = "turbocore_tensor_handle_registry_v1"
TORCH_TENSOR_HANDLE_KIND = "torch_tensor"


@dataclass(frozen=True)
class TensorHandleRecord:
    schema_version: int
    schema: str
    handle_id: str
    role: str
    numel: int
    dtype: str
    device_type: str
    device_index: int | None
    layout: str
    contiguous: bool
    alignment_bytes: int
    handle_kind: str = TORCH_TENSOR_HANDLE_KIND
    pointer_exported: bool = False
    training_path_enabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TurboCoreTensorHandleRegistry:
    """Own current-process tensor handles for native bridge experiments."""

    def __init__(self, namespace: str = "turbocore") -> None:
        self.namespace = _normalise_namespace(namespace)
        self._records: dict[str, TensorHandleRecord] = {}
        self._tensors: dict[str, torch.Tensor] = {}

    def register(
        self,
        tensor: torch.Tensor,
        *,
        role: str,
        expected_numel: int | None = None,
        expected_dtype: str | torch.dtype = "float32",
        require_contiguous: bool = True,
    ) -> TensorHandleRecord:
        """Register a tensor and return its opaque handle record."""

        role_name = _normalise_role(role)
        _validate_tensor_for_flat_handle(
            tensor,
            role=role_name,
            expected_numel=expected_numel,
            expected_dtype=expected_dtype,
            require_contiguous=require_contiguous,
        )
        handle_id = f"{TORCH_TENSOR_HANDLE_KIND}:{self.namespace}:{role_name}:{uuid4().hex}"
        record = TensorHandleRecord(
            schema_version=1,
            schema=TURBOCORE_TENSOR_HANDLE_SCHEMA,
            handle_id=handle_id,
            role=role_name,
            numel=int(tensor.numel()),
            dtype=_normalise_dtype(tensor.dtype),
            device_type=str(tensor.device.type),
            device_index=tensor.device.index,
            layout="flat_contiguous",
            contiguous=bool(tensor.is_contiguous()),
            alignment_bytes=_alignment_bytes(tensor),
        )
        self._records[handle_id] = record
        self._tensors[handle_id] = tensor
        return record

    def register_flat_adamw_buffers(
        self,
        *,
        param_flat: torch.Tensor,
        grad_flat: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
    ) -> dict[str, str]:
        """Register the four fp32 flat buffers required by AdamW ownership."""

        buffers = {
            "param_flat": param_flat,
            "grad_flat": grad_flat,
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
        }
        _validate_flat_adamw_tensor_set(buffers)
        numel = int(param_flat.numel())
        handles: dict[str, str] = {}
        try:
            for role in ADAMW_FLAT_BUFFER_ROLES:
                record = self.register(buffers[role], role=role, expected_numel=numel)
                handles[role] = record.handle_id
        except Exception:
            for handle_id in handles.values():
                self.release(handle_id)
            raise
        return handles

    def resolve(self, handle_id: str) -> torch.Tensor:
        try:
            return self._tensors[str(handle_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown TurboCore tensor handle: {handle_id}") from exc

    def record(self, handle_id: str) -> TensorHandleRecord:
        try:
            return self._records[str(handle_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown TurboCore tensor handle: {handle_id}") from exc

    def release(self, handle_id: str) -> bool:
        key = str(handle_id)
        existed = key in self._records or key in self._tensors
        self._records.pop(key, None)
        self._tensors.pop(key, None)
        return existed

    def clear(self) -> None:
        self._records.clear()
        self._tensors.clear()

    def snapshot(self) -> dict[str, Any]:
        records = [record.as_dict() for record in self._records.values()]
        return {
            "schema_version": 1,
            "schema": TURBOCORE_TENSOR_HANDLE_SCHEMA,
            "namespace": self.namespace,
            "handle_count": len(records),
            "records": records,
            "training_path_enabled": False,
            "pointer_exported": False,
        }


def build_flat_adamw_owner_descriptor_from_handles(
    registry: TurboCoreTensorHandleRegistry,
    role_handles: Mapping[str, str],
    *,
    stream_kind: str = "default",
    stream_id: str = "",
    synchronize: bool = False,
) -> dict[str, Any]:
    """Build an AdamW flat-owner descriptor from registered tensor handles."""

    records = _records_for_adamw_roles(registry, role_handles)
    _validate_flat_adamw_records(records)
    first = records[ADAMW_FLAT_BUFFER_ROLES[0]]
    buffers = tuple(
        FlatBufferDescriptor(
            role=record.role,
            numel=record.numel,
            dtype=record.dtype,
            device_type=record.device_type,
            device_index=record.device_index,
            layout=record.layout,
            contiguous=record.contiguous,
            alignment_bytes=record.alignment_bytes,
            handle_kind=record.handle_kind,
            handle_id=record.handle_id,
        )
        for record in (records[role] for role in ADAMW_FLAT_BUFFER_ROLES)
    )
    descriptor = FlatAdamWOwnerDescriptor(
        schema_version=1,
        layout="flat_contiguous_fp32_buffers",
        buffers=buffers,
        stream=StreamDescriptor(
            device_type=first.device_type,
            device_index=first.device_index,
            stream_kind=str(stream_kind or "default"),
            stream_id=str(stream_id or ""),
            synchronize=bool(synchronize),
        ),
        training_path_enabled=False,
    ).as_dict()
    validation = validate_flat_adamw_owner_descriptor(descriptor)
    if not validation["ok"]:
        raise ValueError(f"Invalid flat AdamW tensor descriptor: {validation}")
    return descriptor


def register_persistent_flat_adamw_buffers(
    owner: Any,
    registry: TurboCoreTensorHandleRegistry | None = None,
) -> tuple[TurboCoreTensorHandleRegistry, dict[str, str], dict[str, Any]]:
    """Register buffers from a PersistentFlatAdamW-like owner."""

    active_registry = registry or TurboCoreTensorHandleRegistry(namespace="persistent_flat_adamw")
    handles = active_registry.register_flat_adamw_buffers(
        param_flat=getattr(owner, "param_flat"),
        grad_flat=getattr(owner, "grad_flat"),
        exp_avg=getattr(owner, "exp_avg"),
        exp_avg_sq=getattr(owner, "exp_avg_sq"),
    )
    descriptor = build_flat_adamw_owner_descriptor_from_handles(active_registry, handles)
    return active_registry, handles, descriptor


def build_tensor_object_map_for_handles(
    registry: TurboCoreTensorHandleRegistry,
    role_handles: Mapping[str, str],
) -> dict[str, torch.Tensor]:
    """Return current-process tensor objects keyed by opaque handle ID."""

    records = _records_for_adamw_roles(registry, role_handles)
    return {
        records[role].handle_id: registry.resolve(records[role].handle_id)
        for role in ADAMW_FLAT_BUFFER_ROLES
    }


def _records_for_adamw_roles(
    registry: TurboCoreTensorHandleRegistry,
    role_handles: Mapping[str, str],
) -> dict[str, TensorHandleRecord]:
    records: dict[str, TensorHandleRecord] = {}
    for role in ADAMW_FLAT_BUFFER_ROLES:
        handle_id = str(role_handles.get(role, "") or "")
        if not handle_id:
            raise ValueError(f"Missing tensor handle for AdamW role: {role}")
        record = registry.record(handle_id)
        if record.role != role:
            raise ValueError(f"Tensor handle role mismatch: expected {role}, got {record.role}")
        records[role] = record
    return records


def _validate_flat_adamw_tensor_set(buffers: Mapping[str, torch.Tensor]) -> None:
    missing = [role for role in ADAMW_FLAT_BUFFER_ROLES if role not in buffers]
    if missing:
        raise ValueError(f"Missing flat AdamW buffers: {missing}")
    for role in ADAMW_FLAT_BUFFER_ROLES:
        _validate_tensor_for_flat_handle(buffers[role], role=role)
    numels = {int(buffers[role].numel()) for role in ADAMW_FLAT_BUFFER_ROLES}
    devices = {str(buffers[role].device) for role in ADAMW_FLAT_BUFFER_ROLES}
    if len(numels) != 1:
        raise ValueError("Flat AdamW buffers must have the same numel")
    if len(devices) != 1:
        raise ValueError("Flat AdamW buffers must live on the same device")


def _validate_flat_adamw_records(records: Mapping[str, TensorHandleRecord]) -> None:
    numels = {records[role].numel for role in ADAMW_FLAT_BUFFER_ROLES}
    devices = {(records[role].device_type, records[role].device_index) for role in ADAMW_FLAT_BUFFER_ROLES}
    dtypes = {records[role].dtype for role in ADAMW_FLAT_BUFFER_ROLES}
    if len(numels) != 1:
        raise ValueError("Flat AdamW tensor handles must have the same numel")
    if len(devices) != 1:
        raise ValueError("Flat AdamW tensor handles must live on the same device")
    if dtypes != {"float32"}:
        raise ValueError("Flat AdamW tensor handles must be float32")
    if any(not records[role].contiguous for role in ADAMW_FLAT_BUFFER_ROLES):
        raise ValueError("Flat AdamW tensor handles must be contiguous")


def _validate_tensor_for_flat_handle(
    tensor: torch.Tensor,
    *,
    role: str,
    expected_numel: int | None = None,
    expected_dtype: str | torch.dtype = "float32",
    require_contiguous: bool = True,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"TurboCore tensor handle requires torch.Tensor for {role}")
    dtype = _normalise_dtype(tensor.dtype)
    target_dtype = _normalise_dtype(expected_dtype)
    if target_dtype not in SUPPORTED_FLAT_DTYPES:
        raise ValueError(f"Unsupported expected dtype for TurboCore flat handle: {target_dtype}")
    if dtype != target_dtype:
        raise ValueError(f"Tensor {role} has dtype {dtype}, expected {target_dtype}")
    if expected_numel is not None and int(tensor.numel()) != int(expected_numel):
        raise ValueError(f"Tensor {role} has numel {int(tensor.numel())}, expected {int(expected_numel)}")
    if require_contiguous and not bool(tensor.is_contiguous()):
        raise ValueError(f"Tensor {role} must be contiguous")


def _normalise_dtype(dtype: str | torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _normalise_role(role: str) -> str:
    value = str(role or "").strip()
    if not value:
        raise ValueError("Tensor handle role is required")
    return value


def _normalise_namespace(namespace: str) -> str:
    value = str(namespace or "turbocore").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    return safe or "turbocore"


def _alignment_bytes(tensor: torch.Tensor) -> int:
    pointer = int(tensor.data_ptr())
    for alignment in (256, 128, 64, 32, 16, 8, 4, 2):
        if pointer % alignment == 0:
            return alignment
    return 1


__all__ = [
    "TORCH_TENSOR_HANDLE_KIND",
    "TURBOCORE_TENSOR_HANDLE_SCHEMA",
    "TensorHandleRecord",
    "TurboCoreTensorHandleRegistry",
    "build_flat_adamw_owner_descriptor_from_handles",
    "build_tensor_object_map_for_handles",
    "register_persistent_flat_adamw_buffers",
]
