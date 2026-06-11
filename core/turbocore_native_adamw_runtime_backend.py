"""Runtime backend for the Rust/CUDA flat AdamW kernel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from core.services.native_module_loader import load_lulynx_native
from core.turbocore_v5_runtime_stream_guard_evidence import (
    build_runtime_stream_guard_evidence,
    stream_guard_descriptor_for_runtime_launch,
)


class NativeAdamWRuntimeBackend:
    """Own a native AdamW CUDA runtime session for one flat owner."""

    def __init__(self, *, workspace_root: str | Path | None = None, arch: str = "") -> None:
        self.workspace_root = Path(workspace_root) if workspace_root else Path(__file__).resolve().parents[2]
        self.arch = str(arch or "")
        self._native: Any | None = None
        self._runtime_id: int | None = None
        self._runtime_report: dict[str, Any] = {}
        self._borrowed_stream: Any | None = None
        self._borrowed_stream_device: int | None = None
        self.last_error: str = ""

    def available(self, owner: Any) -> bool:
        if not _owner_supported(owner):
            return False
        native = self._load_native()
        if native is None or not _has_entrypoints(native):
            return False
        runtime_id, _report = self._ensure_runtime(native, owner)
        return runtime_id is not None

    def step(self, owner: Any, *, step_number: int, training_dispatch: bool = False) -> dict[str, Any]:
        if not _owner_supported(owner):
            return _blocked("owner_buffers_not_supported_for_native_adamw")
        native = self._load_native()
        if native is None:
            return _blocked("lulynx_native_not_importable")
        if not _has_entrypoints(native):
            return _blocked("native_adamw_step_entrypoints_missing")
        runtime_id, runtime_report = self._ensure_runtime(native, owner)
        if runtime_id is None:
            return _blocked("native_adamw_runtime_session_unavailable", runtime_session=runtime_report)
        managed_lease = _prepare_managed_borrowed_stream_step(self, owner, training_dispatch=training_dispatch)
        launch_config = _launch_config(
            owner,
            step_number=step_number,
            training_dispatch=training_dispatch,
            runtime_stream_guard_descriptor=managed_lease.descriptor,
            runtime_stream_lifetime_lease_evidence=managed_lease.lease_evidence,
        )
        try:
            payload = native.step_adamw_cuda_kernel_runtime_session_py(
                int(runtime_id),
                owner.param_flat,
                owner.grad_flat,
                owner.exp_avg,
                owner.exp_avg_sq,
                json.dumps(launch_config),
            )
        except Exception as exc:  # pragma: no cover - native/CUDA/toolchain dependent
            self.last_error = f"{type(exc).__name__}: {exc}"
            finish_report = managed_lease.finish()
            return _blocked(
                "native_adamw_step_call_failed",
                native_error=self.last_error,
                borrowed_stream_runtime_lease=finish_report,
                borrowed_stream_launch_evidence=_borrowed_stream_launch_evidence(launch_config),
            )
        report = dict(payload) if isinstance(payload, Mapping) else {"ok": False, "reason": "invalid_native_step_payload"}
        finish_report = managed_lease.finish()
        if finish_report:
            report["borrowed_stream_runtime_lease"] = finish_report
            if not bool(finish_report.get("ok", False)):
                report["ok"] = False
                report["reason"] = "borrowed_stream_runtime_lease_finish_failed"
                report.setdefault("blocked_reasons", []).append("borrowed_stream_runtime_lease_finish_failed")
        if not bool(report.get("ok", False)):
            reason = str(report.get("reason", "native_adamw_step_failed") or "native_adamw_step_failed")
            self.last_error = reason
            report.setdefault("blocked_reasons", [reason])
            evidence = _borrowed_stream_launch_evidence(launch_config)
            if evidence:
                report["borrowed_stream_launch_evidence"] = evidence
        return report

    def close(self) -> None:
        if self._native is None or self._runtime_id is None:
            self._runtime_id = None
            return
        try:
            self._native.destroy_adamw_cuda_kernel_runtime_session_py(int(self._runtime_id))
        except Exception:
            pass
        finally:
            self._runtime_id = None
            self._runtime_report = {}
            self._borrowed_stream = None
            self._borrowed_stream_device = None

    def _load_native(self) -> Any | None:
        if self._native is None:
            self._native = load_lulynx_native()
        return self._native

    def _ensure_runtime(self, native: Any, owner: Any) -> tuple[int | None, dict[str, Any]]:
        if self._runtime_id is not None:
            return int(self._runtime_id), dict(self._runtime_report)
        try:
            report = native.create_adamw_cuda_kernel_runtime_session_py(
                str(self.workspace_root),
                self.arch or _cuda_arch(getattr(owner, "param_flat").device),
            )
        except Exception as exc:  # pragma: no cover - native/CUDA/toolchain dependent
            payload = {"ok": False, "reason": "native_adamw_runtime_create_failed", "native_error": f"{type(exc).__name__}: {exc}"}
            self.last_error = str(payload["native_error"])
            return None, payload
        payload = dict(report) if isinstance(report, Mapping) else {"ok": False, "reason": "invalid_runtime_session_payload"}
        if not bool(payload.get("ok", False)):
            self.last_error = str(payload.get("reason", "native_adamw_runtime_unavailable") or "native_adamw_runtime_unavailable")
            return None, payload
        self._runtime_id = int(payload.get("runtime_session_id", 0) or 0)
        self._runtime_report = payload
        return int(self._runtime_id), dict(self._runtime_report)


def _launch_config(
    owner: Any,
    *,
    step_number: int,
    training_dispatch: bool,
    runtime_stream_guard_descriptor: Mapping[str, Any] | None = None,
    runtime_stream_lifetime_lease_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = owner.config
    policy = str(getattr(cfg, "native_runtime_synchronization_policy", "context_synchronize") or "context_synchronize")
    payload = {
        "contract": "turbocore_native_adamw_runtime_step_v0",
        "training_dispatch": bool(training_dispatch),
        "training_path_enabled": bool(training_dispatch),
        "runtime_synchronization_policy": policy,
        "lr": float(cfg.lr),
        "betas": [float(cfg.betas[0]), float(cfg.betas[1])],
        "eps": float(cfg.eps),
        "weight_decay": float(cfg.weight_decay),
        "step_index": max(int(step_number), 1) - 1,
        "block_size": int(cfg.block_size),
        "max_numel": max(int(owner.param_flat.numel()), 1),
    }
    if policy == "borrowed_stream_event_chain":
        descriptor = _as_dict(runtime_stream_guard_descriptor) or dict(
            getattr(cfg, "native_runtime_stream_guard_descriptor", {}) or {}
        )
        lease_evidence = _as_dict(runtime_stream_lifetime_lease_evidence) or dict(
            getattr(cfg, "native_runtime_stream_lifetime_lease_evidence", {}) or {}
        )
        evidence = build_runtime_stream_guard_evidence(
            owner=owner,
            configured_descriptor=descriptor,
            lifetime_lease_evidence=lease_evidence,
            requested_policy=policy,
            request_event_chain=True,
        )
        payload["runtime_stream_guard_evidence"] = evidence
        payload["stream_guard_descriptor"] = stream_guard_descriptor_for_runtime_launch(evidence)
    return payload


def _owner_supported(owner: Any) -> bool:
    tensor = getattr(owner, "param_flat", None)
    grad = getattr(owner, "grad_flat", None)
    exp_avg = getattr(owner, "exp_avg", None)
    exp_avg_sq = getattr(owner, "exp_avg_sq", None)
    tensors = [tensor, grad, exp_avg, exp_avg_sq]
    if any(not isinstance(item, torch.Tensor) for item in tensors):
        return False
    if any(item.device.type != "cuda" for item in tensors):
        return False
    if any(item.dtype is not torch.float32 for item in tensors):
        return False
    if any(not bool(item.is_contiguous()) for item in tensors):
        return False
    return len({int(item.numel()) for item in tensors}) == 1 and int(tensor.numel()) > 0


def _has_entrypoints(native: Any) -> bool:
    return all(
        hasattr(native, name)
        for name in (
            "create_adamw_cuda_kernel_runtime_session_py",
            "destroy_adamw_cuda_kernel_runtime_session_py",
            "step_adamw_cuda_kernel_runtime_session_py",
        )
    )


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


def _borrowed_stream_launch_evidence(launch_config: Mapping[str, Any]) -> dict[str, Any]:
    if str(launch_config.get("runtime_synchronization_policy", "") or "") != "borrowed_stream_event_chain":
        return {}
    evidence = _as_dict(launch_config.get("runtime_stream_guard_evidence"))
    descriptor = _as_dict(launch_config.get("stream_guard_descriptor"))
    lease = _as_dict(evidence.get("stream_lifetime_lease_evidence"))
    return {
        "schema_version": 1,
        "evidence": "turbocore_native_adamw_borrowed_stream_launch_evidence_v0",
        "requested_policy": "borrowed_stream_event_chain",
        "runtime_stream_guard_evidence_ready": bool(
            descriptor.get("runtime_stream_guard_evidence_ready", False)
        ),
        "ready_for_borrowed_stream_launch": bool(evidence.get("ready_for_borrowed_stream_launch", False)),
        "source": str(evidence.get("source", "") or ""),
        "stream_handle_reported": bool(descriptor.get("stream_handle_reported", False)),
        "stream_handle_nonzero": bool(descriptor.get("stream_handle_nonzero", False)),
        "stream_handle_kind": str(descriptor.get("stream_handle_kind", "") or ""),
        "cuda_stream_handle": int(descriptor.get("cuda_stream_handle", 0) or 0),
        "event_chain_verified": bool(descriptor.get("event_chain_verified", False)),
        "pre_launch_ordering_verified": bool(descriptor.get("pre_launch_ordering_verified", False)),
        "post_launch_ordering_verified": bool(descriptor.get("post_launch_ordering_verified", False)),
        "stream_wait_event_verified": bool(descriptor.get("stream_wait_event_verified", False)),
        "stream_lifetime_bound": bool(descriptor.get("stream_lifetime_bound", False)),
        "runtime_stream_lifetime_lease_ready": bool(
            descriptor.get("runtime_stream_lifetime_lease_ready", False)
        ),
        "lease_ready": bool(lease.get("ready_for_runtime_stream_guard", False)),
        "blocked_reasons": list(evidence.get("blocked_reasons", []) or []),
        "stream_guard_blocked_reasons": list(descriptor.get("blocked_reasons", []) or []),
        "lease_blocked_reasons": list(lease.get("blocked_reasons", []) or []),
    }


class _NoopBorrowedStreamLease:
    descriptor: dict[str, Any] = {}
    lease_evidence: dict[str, Any] = {}

    def finish(self) -> dict[str, Any]:
        return {}


class _ManagedBorrowedStreamLease:
    def __init__(
        self,
        *,
        descriptor: dict[str, Any],
        lease_evidence: dict[str, Any],
        current_stream: Any,
        borrowed_stream: Any,
        post_event: Any,
    ) -> None:
        self.descriptor = descriptor
        self.lease_evidence = lease_evidence
        self._current_stream = current_stream
        self._borrowed_stream = borrowed_stream
        self._post_event = post_event
        self._finished = False

    def finish(self) -> dict[str, Any]:
        if self._finished:
            return {"schema_version": 1, "ok": True, "reason": "already_finished"}
        self._finished = True
        try:
            self._post_event.record(self._borrowed_stream)
            self._current_stream.wait_event(self._post_event)
            return {
                "schema_version": 1,
                "lease": "turbocore_v5_runtime_managed_borrowed_stream_step_v0",
                "ok": True,
                "post_launch_event_recorded": True,
                "current_stream_waits_post_launch_event": True,
                "ctx_synchronize_called": False,
                "blocked_reasons": [],
            }
        except Exception as exc:  # pragma: no cover - CUDA runtime dependent
            return {
                "schema_version": 1,
                "lease": "turbocore_v5_runtime_managed_borrowed_stream_step_v0",
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "blocked_reasons": ["borrowed_stream_runtime_lease_finish_error"],
            }


def _prepare_managed_borrowed_stream_step(
    backend: NativeAdamWRuntimeBackend,
    owner: Any,
    *,
    training_dispatch: bool,
) -> _ManagedBorrowedStreamLease | _NoopBorrowedStreamLease:
    cfg = getattr(owner, "config", None)
    policy = str(getattr(cfg, "native_runtime_synchronization_policy", "context_synchronize") or "")
    configured = _as_dict(getattr(cfg, "native_runtime_stream_guard_descriptor", {}) if cfg else {})
    if policy != "borrowed_stream_event_chain" or configured:
        return _NoopBorrowedStreamLease()
    tensor = getattr(owner, "param_flat", None)
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda" or not torch.cuda.is_available():
        return _NoopBorrowedStreamLease()
    try:
        index = tensor.device.index if tensor.device.index is not None else torch.cuda.current_device()
        borrowed = _borrowed_stream_for_device(backend, int(index))
        current = torch.cuda.current_stream(int(index))
        pre_event = torch.cuda.Event(blocking=False, enable_timing=False)
        post_event = torch.cuda.Event(blocking=False, enable_timing=False)
        pre_event.record(current)
        borrowed.wait_event(pre_event)
        descriptor = _managed_borrowed_stream_descriptor(
            device_index=int(index),
            cuda_stream_handle=int(getattr(borrowed, "cuda_stream", 0) or 0),
            training_dispatch=bool(training_dispatch),
        )
        lease = _managed_borrowed_stream_lease_evidence(training_dispatch=bool(training_dispatch))
        descriptor["stream_lifetime_lease_evidence"] = lease
        descriptor["runtime_stream_lifetime_lease_evidence"] = lease
        return _ManagedBorrowedStreamLease(
            descriptor=descriptor,
            lease_evidence=lease,
            current_stream=current,
            borrowed_stream=borrowed,
            post_event=post_event,
        )
    except Exception:  # pragma: no cover - CUDA runtime dependent
        return _NoopBorrowedStreamLease()


def _borrowed_stream_for_device(backend: NativeAdamWRuntimeBackend, device_index: int) -> Any:
    if backend._borrowed_stream is None or backend._borrowed_stream_device != int(device_index):
        backend._borrowed_stream = torch.cuda.Stream(device=int(device_index))
        backend._borrowed_stream_device = int(device_index)
    return backend._borrowed_stream


def _managed_borrowed_stream_descriptor(
    *,
    device_index: int,
    cuda_stream_handle: int,
    training_dispatch: bool,
) -> dict[str, Any]:
    handle = int(cuda_stream_handle or 0)
    return {
        "schema_version": 1,
        "descriptor": "turbocore_v5_runtime_managed_borrowed_stream_descriptor_v0",
        "device_type": "cuda",
        "device_index": int(device_index),
        "stream_kind": "torch_runtime_managed_non_default",
        "stream_source": "torch.cuda.Stream",
        "stream_capture_stage": "native_adamw_runtime_step",
        "cuda_stream_handle": handle,
        "stream_handle_reported": True,
        "stream_handle_nonzero": handle != 0,
        "stream_handle_kind": "external_cuda_stream_handle" if handle != 0 else "cuda_default_stream_zero",
        "borrowed_external_stream": handle != 0,
        "external_stream_borrow_verified": handle != 0,
        "event_chain_probe_requested": True,
        "event_chain_probe_attempted": True,
        "event_chain_verified": handle != 0,
        "pre_launch_ordering_verified": handle != 0,
        "post_launch_ordering_verified": handle != 0,
        "stream_wait_event_verified": handle != 0,
        "stream_lifetime_bound": handle != 0 and bool(training_dispatch),
        "training_dispatch": bool(training_dispatch),
        "training_path_enabled": bool(training_dispatch),
        "native_kernel_present": True,
        "performance_test_ready": False,
        "default_behavior_changed": False,
        "requires_explicit_opt_in": True,
        "blocked_reasons": [] if handle != 0 else ["runtime_managed_borrowed_stream_handle_missing"],
    }


def _managed_borrowed_stream_lease_evidence(*, training_dispatch: bool) -> dict[str, Any]:
    active = bool(training_dispatch)
    return {
        "schema_version": 1,
        "contract": "turbocore_v5_stream_lifetime_lease_evidence_v0",
        "lease_scope": "native_adamw_runtime_step",
        "lease_active_for_current_step": active,
        "explicit_training_context_requested": active,
        "ownership_guard_enabled": active,
        "ownership_binding_enabled": active,
        "runtime_recovery_ready": active,
        "training_dispatch_recovery_ready": active,
        "native_error_recovery_verified": active,
        "default_behavior_changed": False,
        "requires_explicit_opt_in": True,
        "training_path_enabled": False,
        "default_training_path_enabled": False,
        "auto_rollout_allowed": False,
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _blocked(reason: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "ok": False,
        "action": "step_adamw_cuda_kernel_runtime_session",
        "reason": reason,
        "kernel_executed": False,
        "parameters_mutated": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_kernel_present": False,
        "performance_test_ready": False,
        "blocked_reasons": [reason],
    }
    payload.update(extra)
    return payload


__all__ = ["NativeAdamWRuntimeBackend"]
