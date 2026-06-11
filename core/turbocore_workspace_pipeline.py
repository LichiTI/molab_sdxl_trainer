"""TurboCore workspace/data-pipeline nativeization prototype.

This module is intentionally Python-only and research-only.  It models the ABI
shape we want from a future native TurboCore data path: bounded staging queue,
explicit workspace acquisition, explicit release, and teardown diagnostics.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, Mapping

from core.turbocore_optimizer_abi import build_native_optimizer_stateful_capability_stub
from core.turbocore_phase1 import NativeWorkspacePool


_NATIVE_PIPELINE_DISABLE_ENV = "LULYNX_DISABLE_NATIVE_TURBOCORE_PIPELINE"


@dataclass(frozen=True)
class WorkspaceBufferSpec:
    name: str
    shape: tuple[int, ...]
    dtype: Any
    device: Any = "cpu"

    @classmethod
    def from_shape(
        cls,
        name: str,
        shape: Iterable[int],
        *,
        dtype: Any,
        device: Any = "cpu",
    ) -> "WorkspaceBufferSpec":
        return cls(
            name=str(name),
            shape=tuple(int(dim) for dim in shape),
            dtype=dtype,
            device=device,
        )


@dataclass(frozen=True)
class StagedBatch:
    batch_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    workspace: tuple[WorkspaceBufferSpec, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStats:
    submitted: int = 0
    consumed: int = 0
    released: int = 0
    queue_full_stalls: int = 0
    queue_empty_stalls: int = 0
    close_released: int = 0
    peak_ready: int = 0
    peak_in_flight: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "submitted": self.submitted,
            "consumed": self.consumed,
            "released": self.released,
            "queue_full_stalls": self.queue_full_stalls,
            "queue_empty_stalls": self.queue_empty_stalls,
            "close_released": self.close_released,
            "peak_ready": self.peak_ready,
            "peak_in_flight": self.peak_in_flight,
        }


@dataclass
class _PreparedBatch:
    batch: StagedBatch
    buffers: Dict[str, Any]


class PipelineLease:
    """Explicit ownership token for a consumed staged batch."""

    def __init__(self, owner: "NativeDataPipelinePrototype", prepared: _PreparedBatch) -> None:
        self._owner = owner
        self.batch = prepared.batch
        self.buffers = prepared.buffers
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self._owner._release_lease(self)

    def __enter__(self) -> "PipelineLease":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.release()


class NativeDataPipelinePrototype:
    """Bounded staging queue with explicit workspace buffer lifecycle."""

    def __init__(
        self,
        *,
        prefetch_depth: int = 2,
        workspace_pool: NativeWorkspacePool | None = None,
    ) -> None:
        self.prefetch_depth = max(int(prefetch_depth), 1)
        self.workspace_pool = workspace_pool or NativeWorkspacePool()
        self._ready: Deque[_PreparedBatch] = deque()
        self._in_flight: set[int] = set()
        self._closed = False
        self.stats = PipelineStats()

    @property
    def closed(self) -> bool:
        return self._closed

    def submit(self, batch: StagedBatch) -> bool:
        if self._closed:
            raise RuntimeError("NativeDataPipelinePrototype is closed")
        if len(self._ready) >= self.prefetch_depth:
            self.stats.queue_full_stalls += 1
            return False
        buffers = {
            spec.name: self.workspace_pool.acquire(spec.shape, dtype=spec.dtype, device=spec.device)
            for spec in batch.workspace
        }
        self._ready.append(_PreparedBatch(batch=batch, buffers=buffers))
        self.stats.submitted += 1
        self.stats.peak_ready = max(self.stats.peak_ready, len(self._ready))
        return True

    def consume(self) -> PipelineLease | None:
        if not self._ready:
            self.stats.queue_empty_stalls += 1
            return None
        prepared = self._ready.popleft()
        lease = PipelineLease(self, prepared)
        self._in_flight.add(id(lease))
        self.stats.consumed += 1
        self.stats.peak_in_flight = max(self.stats.peak_in_flight, len(self._in_flight))
        return lease

    def _release_lease(self, lease: PipelineLease) -> None:
        lease_id = id(lease)
        if lease_id not in self._in_flight:
            return
        for tensor in lease.buffers.values():
            self.workspace_pool.release(tensor)
        self._in_flight.remove(lease_id)
        self.stats.released += 1

    def close(self) -> Dict[str, Any]:
        if self._closed:
            return self.snapshot()
        self._closed = True
        while self._ready:
            prepared = self._ready.popleft()
            for tensor in prepared.buffers.values():
                self.workspace_pool.release(tensor)
            self.stats.close_released += 1
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "prototype": "turbocore_native_data_pipeline",
            "training_path_enabled": False,
            "prefetch_depth": self.prefetch_depth,
            "closed": self._closed,
            "ready": len(self._ready),
            "in_flight": len(self._in_flight),
            "stats": self.stats.as_dict(),
            "workspace_pool": self.workspace_pool.stats(),
        }


def build_workspace_pipeline_prototype_report(
    *,
    prefetch_depth: int = 2,
    workspace_mb: int = 0,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "prototype": "turbocore_workspace_data_pipeline",
        "status": "python_abi_prototype",
        "training_path_enabled": False,
        "workspace_mb": max(int(workspace_mb), 0),
        "prefetch_depth": max(int(prefetch_depth), 1),
        "features": {
            "bounded_prefetch_queue": True,
            "explicit_workspace_lease": True,
            "explicit_release": True,
            "teardown_snapshot": True,
            "native_runtime": False,
        },
        "next_native_abi": [
            "create_workspace_pool(max_bytes)",
            "create_data_pipeline(prefetch_depth, workspace_pool)",
            "submit_staged_batch(batch_descriptor)",
            "consume_ready_batch() -> lease",
            "release_batch_lease(lease)",
            "close_data_pipeline() -> teardown_stats",
        ],
    }


def build_workspace_pipeline_native_capability_stub() -> Dict[str, Any]:
    """Return the expected native capability shape for this prototype.

    This is a contract sample for the future ``lulynx_native`` report, not an
    active capability.
    """

    return {
        "schema_version": 1,
        "status": "expected_native_schema",
        "training_path_enabled": False,
        "features": {
            "workspace_pool": {
                "available": False,
                "status": "python_abi_prototype",
                "reason": "native_workspace_pool_not_implemented",
                "required_entrypoints": [
                    "create_workspace_pool",
                    "workspace_acquire",
                    "workspace_release",
                    "workspace_stats",
                    "destroy_workspace_pool",
                ],
            },
            "data_pipeline": {
                "available": False,
                "status": "python_abi_prototype",
                "reason": "native_data_pipeline_not_implemented",
                "required_entrypoints": [
                    "create_data_pipeline",
                    "submit_staged_batch",
                    "consume_ready_batch",
                    "release_batch_lease",
                    "close_data_pipeline",
                ],
            },
        },
    }


def build_turbocore_native_training_capability_stub() -> Dict[str, Any]:
    """Return a complete inactive ``lulynx_native`` training capability report.

    Future native bindings should expose this shape from
    ``get_turbocore_training_capabilities()`` before any training route is
    allowed to use native execution.  Entry points are reported as present so
    the Python ABI validator can distinguish "schema wired" from "runtime
    active"; every feature remains unavailable.
    """

    expected = build_workspace_pipeline_native_capability_stub()
    expected_features = expected["features"]
    workspace_entrypoints = list(expected_features["workspace_pool"]["required_entrypoints"])
    data_pipeline_entrypoints = list(expected_features["data_pipeline"]["required_entrypoints"])
    native_optimizer = build_native_optimizer_stateful_capability_stub()
    return {
        "schema_version": 1,
        "training_path_enabled": False,
        "training_bridge": {
            "available": True,
            "status": "capability_stub",
            "reason": "native_training_capability_stub",
        },
        "features": {
            "lora_fused": {
                "available": False,
                "status": "capability_stub",
                "reason": "native_lora_fused_kernel_not_implemented",
            },
            "native_optimizer": {
                **native_optimizer,
                "status": "capability_stub",
                "reason": "native_optimizer_not_implemented",
            },
            "workspace_pool": {
                "available": False,
                "status": "capability_stub",
                "reason": "native_workspace_pool_not_implemented",
                "entrypoints": workspace_entrypoints,
            },
            "data_pipeline": {
                "available": False,
                "status": "capability_stub",
                "reason": "native_data_pipeline_not_implemented",
                "entrypoints": data_pipeline_entrypoints,
            },
            "cuda_nvrtc_compile_probe": {
                "available": False,
                "status": "probe_only_stub",
                "reason": "native_nvrtc_compile_probe_not_loaded",
                "entrypoints": ["probe_adamw_cuda_nvrtc_compile_py"],
                "native_kernel_present": False,
                "training_path_enabled": False,
                "performance_test_ready": False,
                "artifact_only": True,
            },
            "cuda_driver_ptx_probe": {
                "available": False,
                "status": "probe_only_stub",
                "reason": "native_driver_ptx_probe_not_loaded",
                "entrypoints": ["probe_adamw_cuda_driver_ptx_load_py"],
                "native_kernel_present": False,
                "training_path_enabled": False,
                "performance_test_ready": False,
                "artifact_only": True,
                "kernel_executed": False,
            },
            "cuda_scratch_launch_probe": {
                "available": False,
                "status": "probe_only_stub",
                "reason": "native_scratch_launch_probe_not_loaded",
                "entrypoints": ["probe_adamw_cuda_scratch_launch_py"],
                "scratch_buffers_only": True,
                "training_tensor_binding": False,
                "native_kernel_present": False,
                "training_path_enabled": False,
                "performance_test_ready": False,
            },
            "cuda_adamw_runtime": {
                "available": False,
                "status": "probe_only_stub",
                "reason": "native_adamw_runtime_session_not_loaded",
                "entrypoints": [
                    "create_adamw_cuda_kernel_runtime_session_py",
                    "adamw_cuda_kernel_runtime_session_snapshot_py",
                    "destroy_adamw_cuda_kernel_runtime_session_py",
                    "tensor_binding_session_cuda_adamw_runtime_probe",
                    "benchmark_adamw_cuda_kernel_runtime_session_py",
                ],
                "runtime_session": True,
                "training_dispatch": False,
                "stream_lifetime_bound": False,
                "native_kernel_present": False,
                "training_path_enabled": False,
                "performance_test_ready": False,
            },
        },
    }


def _truthy_env(name: str) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on", "enable", "enabled"}


def _inject_native_artifact_dir_from_env() -> None:
    raw = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    if not raw:
        return
    path = os.path.abspath(os.path.expanduser(raw))
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def _load_native_pipeline_api() -> Any:
    if _truthy_env(_NATIVE_PIPELINE_DISABLE_ENV):
        raise RuntimeError("native_turbocore_pipeline_disabled_by_env")
    _inject_native_artifact_dir_from_env()
    if importlib.util.find_spec("lulynx_native") is None:
        raise RuntimeError("lulynx_native_not_importable")
    native = importlib.import_module("lulynx_native")
    required = (
        "create_workspace_pool",
        "workspace_acquire",
        "workspace_release",
        "workspace_stats",
        "destroy_workspace_pool",
        "create_data_pipeline",
        "submit_indexed_batches",
        "consume_and_release_counted_batches",
        "close_data_pipeline",
    )
    missing = [name for name in required if not callable(getattr(native, name, None))]
    if missing:
        raise RuntimeError("native_pipeline_entrypoints_missing:" + ",".join(missing))
    return native


def run_native_workspace_pipeline_lifecycle_probe(
    *,
    batches: int = 4,
    prefetch_depth: int = 2,
    workspace_mb: int = 0,
    dtype: Any = None,
    device: Any = "cpu",
    chunk_size: int = 256,
) -> Dict[str, Any]:
    """Run the Rust count/handle lifecycle probe through the Python bridge."""

    native = _load_native_pipeline_api()
    requested_batches = max(int(batches), 0)
    chunk = max(int(chunk_size), 1)
    requested_prefetch = max(int(prefetch_depth), 1)
    effective_prefetch = max(requested_prefetch, chunk)
    pool_id = int(native.create_workspace_pool(max(int(workspace_mb), 0) * 1024 * 1024))
    if pool_id <= 0:
        raise RuntimeError("native_workspace_pool_create_failed")

    pipeline_id = 0
    close_stats: Dict[str, Any] = {}
    try:
        workspace_key = f"{device}:{dtype or 'float32'}:latents:1x4x8x8"
        native.workspace_acquire(pool_id, workspace_key)
        native.workspace_release(pool_id, workspace_key)
        native.workspace_acquire(pool_id, workspace_key)
        native.workspace_release(pool_id, workspace_key)
        workspace_stats = native.workspace_stats(pool_id)
        if not isinstance(workspace_stats, dict):
            workspace_stats = {"ok": False, "reason": "native_workspace_stats_invalid"}

        pipeline_id = int(native.create_data_pipeline(effective_prefetch, pool_id))
        if pipeline_id <= 0:
            raise RuntimeError("native_data_pipeline_create_failed")

        submitted = 0
        consumed = 0
        for cursor in range(0, requested_batches, chunk):
            count = min(chunk, requested_batches - cursor)
            submitted += int(native.submit_indexed_batches(pipeline_id, cursor, count))
            consumed += int(native.consume_and_release_counted_batches(pipeline_id, chunk))
        while True:
            drained = int(native.consume_and_release_counted_batches(pipeline_id, chunk))
            if not drained:
                break
            consumed += drained

        raw_close = native.close_data_pipeline(pipeline_id)
        pipeline_id = 0
        close_stats = raw_close if isinstance(raw_close, dict) else {"ok": False, "reason": "native_close_stats_invalid"}
        stats = close_stats.get("stats") if isinstance(close_stats.get("stats"), dict) else {}
        return {
            "schema_version": 1,
            "probe": "turbocore_workspace_pipeline_lifecycle",
            "provider": "native_indexed_count_queue",
            "native_runtime": True,
            "training_path_enabled": False,
            "requested_batches": requested_batches,
            "submitted_batches": submitted,
            "consumed_batches": consumed,
            "workspace_mb": max(int(workspace_mb), 0),
            "prefetch_depth": effective_prefetch,
            "requested_prefetch_depth": requested_prefetch,
            "chunk_size": chunk,
            "closed": bool(close_stats.get("closed", False)),
            "ready": int(close_stats.get("ready", 0) or 0),
            "in_flight": int(close_stats.get("in_flight", 0) or 0),
            "stats": dict(stats),
            "workspace_pool": workspace_stats,
            "ok": submitted == requested_batches
            and consumed == submitted
            and int(close_stats.get("in_flight", 0) or 0) == 0,
        }
    finally:
        if pipeline_id:
            try:
                native.close_data_pipeline(pipeline_id)
            except Exception:
                pass
        try:
            native.destroy_workspace_pool(pool_id)
        except Exception:
            pass


def _run_python_workspace_pipeline_lifecycle_probe(
    *,
    batches: int = 4,
    prefetch_depth: int = 2,
    workspace_mb: int = 0,
    dtype: Any = None,
    device: Any = "cpu",
) -> Dict[str, Any]:
    """Run a tiny lifecycle probe against the Python ABI prototype."""

    import torch

    resolved_dtype = dtype or torch.float32
    pipeline = NativeDataPipelinePrototype(prefetch_depth=prefetch_depth)
    specs = (
        WorkspaceBufferSpec.from_shape("latents", (1, 4, 8, 8), dtype=resolved_dtype, device=device),
        WorkspaceBufferSpec.from_shape("prompt", (1, 16, 32), dtype=resolved_dtype, device=device),
    )
    submitted = 0
    consumed = 0
    for index in range(max(int(batches), 0)):
        batch = StagedBatch(batch_id=f"probe-{index}", workspace=specs)
        if not pipeline.submit(batch):
            lease = pipeline.consume()
            if lease is not None:
                with lease:
                    consumed += 1
            if pipeline.submit(batch):
                submitted += 1
        else:
            submitted += 1

    while True:
        lease = pipeline.consume()
        if lease is None:
            break
        with lease:
            consumed += 1
    snapshot = pipeline.close()
    snapshot.update({
        "probe": "turbocore_workspace_pipeline_lifecycle",
        "provider": "python_prototype",
        "native_runtime": False,
        "requested_batches": max(int(batches), 0),
        "submitted_batches": submitted,
        "consumed_batches": consumed,
        "workspace_mb": max(int(workspace_mb), 0),
        "ok": submitted == max(int(batches), 0) and consumed == submitted and snapshot["in_flight"] == 0,
    })
    return snapshot


def run_workspace_pipeline_lifecycle_probe(
    *,
    batches: int = 4,
    prefetch_depth: int = 2,
    workspace_mb: int = 0,
    dtype: Any = None,
    device: Any = "cpu",
    prefer_native: bool = True,
    chunk_size: int = 256,
) -> Dict[str, Any]:
    """Run lifecycle evidence, preferring the Rust count/handle ABI when present."""

    native_error = ""
    if prefer_native:
        try:
            return run_native_workspace_pipeline_lifecycle_probe(
                batches=batches,
                prefetch_depth=prefetch_depth,
                workspace_mb=workspace_mb,
                dtype=dtype,
                device=device,
                chunk_size=chunk_size,
            )
        except Exception as exc:
            native_error = f"{type(exc).__name__}: {exc}"

    payload = _run_python_workspace_pipeline_lifecycle_probe(
        batches=batches,
        prefetch_depth=prefetch_depth,
        workspace_mb=workspace_mb,
        dtype=dtype,
        device=device,
    )
    payload["native_attempted"] = bool(prefer_native)
    if native_error:
        payload["native_fallback_reason"] = native_error
    return payload


__all__ = [
    "NativeDataPipelinePrototype",
    "PipelineLease",
    "PipelineStats",
    "StagedBatch",
    "WorkspaceBufferSpec",
    "build_turbocore_native_training_capability_stub",
    "build_workspace_pipeline_native_capability_stub",
    "build_workspace_pipeline_prototype_report",
    "run_native_workspace_pipeline_lifecycle_probe",
    "run_workspace_pipeline_lifecycle_probe",
]
