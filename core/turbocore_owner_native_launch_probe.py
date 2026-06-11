"""Report-only native launch probe for PersistentFlatAdamW owner buffers.

This probe runs the dev-only CUDA AdamW runtime against cloned owner buffers,
then compares the clone with the normal Python/Torch owner result. It never
binds real training parameters and never enables optimizer dispatch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from core.turbocore_flat_adamw_state import PersistentFlatAdamW
from core.turbocore_native_update_kernel_launcher import build_native_update_adamw_launch_config
from core.turbocore_native_tensor_binding import build_flat_adamw_native_binding_request
from core.services.native_module_loader import load_lulynx_native, probe_lulynx_native_loader
from core.turbocore_tensor_handle_registry import (
    build_tensor_object_map_for_handles,
    register_persistent_flat_adamw_buffers,
)


DEFAULT_OWNER_NATIVE_LAUNCH_MAX_NUMEL = 1_048_576


class TurboCoreOwnerNativeLaunchProbe:
    """Owns a cached dev-only CUDA AdamW runtime session for probe launches."""

    def __init__(self, *, workspace_root: Path | str | None = None, arch: str = "", event_chain_probe: bool = False) -> None:
        self.workspace_root = Path(workspace_root) if workspace_root else Path(__file__).resolve().parents[2]
        self.arch = str(arch or "")
        self.event_chain_probe = bool(event_chain_probe)
        self._native: Any | None = None
        self._runtime_id: int | None = None
        self._runtime_report: dict[str, Any] = {}
        self._probe_owner: PersistentFlatAdamW | None = None
        self._binding_session_id: int | None = None
        self._binding_signature = ""
        self._binding_report: dict[str, Any] = {}
        self._prepared_report: dict[str, Any] = {}

    def close(self) -> None:
        self._destroy_binding_session()
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
            self._probe_owner = None
            self._prepared_report = {}

    def run(
        self,
        *,
        owner_state_before: Mapping[str, Any] | None,
        expected_owner_after: Any,
        owner_step_report: Mapping[str, Any] | None,
        max_numel: int = DEFAULT_OWNER_NATIVE_LAUNCH_MAX_NUMEL,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        step_report = dict(owner_step_report or {})
        if not owner_state_before:
            return build_owner_native_launch_probe_skip(
                reason="owner_state_before_missing",
                max_numel=max_numel,
                started=started,
            )
        if bool(step_report.get("skipped", False)):
            return build_owner_native_launch_probe_skip(
                reason="owner_step_skipped",
                max_numel=max_numel,
                owner_state=owner_state_before,
                started=started,
            )
        try:
            source_owner = PersistentFlatAdamW.from_state_dict(dict(owner_state_before))
            prepared = self.prepare_from_owner(source_owner, max_numel=max_numel, started=started)
            if not bool(prepared.get("prepared", False)):
                return prepared
            return self.run_prepared(
                expected_owner_after=expected_owner_after,
                owner_step_report=step_report,
                max_numel=max_numel,
            )
        except Exception as exc:  # pragma: no cover - local native/CUDA/toolchain dependent
            return _error_payload(
                "native_owner_launch_probe_error",
                started,
                owner_state_before,
                max_numel=max_numel,
                error=f"{type(exc).__name__}: {exc}",
            )

    def prepare_from_owner(
        self,
        owner: Any,
        *,
        max_numel: int = DEFAULT_OWNER_NATIVE_LAUNCH_MAX_NUMEL,
        started: float | None = None,
    ) -> dict[str, Any]:
        started = started or time.perf_counter()
        skip = _owner_skip_reason(owner, max_numel=max_numel)
        if skip:
            self._prepared_report = {}
            return build_owner_native_launch_probe_skip(reason=skip, max_numel=max_numel, owner_state=owner, started=started)
        native = self._load_native()
        if native is None:
            self._prepared_report = {}
            return build_owner_native_launch_probe_skip(
                reason="lulynx_native_not_importable",
                max_numel=max_numel,
                owner_state=owner,
                loader=probe_lulynx_native_loader(),
                started=started,
            )
        missing = _missing_entrypoints(native)
        if missing:
            self._prepared_report = {}
            return _error_payload("native_owner_launch_entrypoints_missing", started, owner, max_numel=max_numel, missing_entrypoints=missing)
        runtime_id, runtime_report = self._ensure_runtime(native, owner.param_flat.device)
        if runtime_id is None:
            self._prepared_report = {}
            return _error_payload("native_owner_launch_runtime_unavailable", started, owner, max_numel=max_numel, runtime_session=runtime_report)
        probe_owner_reused = self._ensure_probe_owner(owner)
        session_id, binding_reused, binding_report = self._ensure_binding_session(native, self._probe_owner)
        if session_id is None:
            self._prepared_report = {}
            return _error_payload("native_owner_launch_binding_session_unavailable", started, owner, max_numel=max_numel, binding_session=binding_report)
        payload = _base_payload(started, owner, max_numel=max_numel)
        payload.update(
            {
                "ok": True,
                "prepared": True,
                "skipped": False,
                "attempted": False,
                "reason": "prepared",
                "state_transfer_mode": "persistent_probe_owner_copy",
                "owner_state_cloned": False,
                "probe_owner_reused": probe_owner_reused,
                "binding_session_reused": binding_reused,
                "binding_session_id": int(session_id),
                "binding_session": binding_report,
                "runtime_session": dict(runtime_report),
                "runtime_session_id": int(runtime_id),
                "native_launch_attempted": False,
                "native_launch_ok": False,
                "kernel_executed": False,
                "persistent_owner_mutated": False,
                "blocked_reasons": [],
            }
        )
        self._prepared_report = payload
        return payload

    def run_prepared(
        self,
        *,
        expected_owner_after: Any,
        owner_step_report: Mapping[str, Any] | None,
        max_numel: int = DEFAULT_OWNER_NATIVE_LAUNCH_MAX_NUMEL,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        prepared = dict(self._prepared_report)
        self._prepared_report = {}
        step_report = dict(owner_step_report or {})
        if not prepared or not bool(prepared.get("prepared", False)) or self._probe_owner is None:
            return build_owner_native_launch_probe_skip(reason="owner_native_launch_not_prepared", max_numel=max_numel, owner_state=expected_owner_after, started=started)
        native = self._load_native()
        runtime_id = int(prepared.get("runtime_session_id", 0) or 0)
        session_id = int(prepared.get("binding_session_id", 0) or 0)
        launch = self._launch_prepared_owner(native, session_id, runtime_id, self._probe_owner, max_numel=max_numel)
        stream_guard = dict(launch.get("stream_guard_probe") or {})
        parity = _compare_owner_buffers(self._probe_owner, expected_owner_after)
        launch_ok = bool(launch.get("ok", False))
        parity_ok = bool(parity.get("parity_ok", False))
        prepare_elapsed = float(prepared.get("elapsed_ms", 0.0) or 0.0)
        launch_elapsed = (time.perf_counter() - started) * 1000.0
        prepared.update(
            {
                "ok": bool(launch_ok and parity_ok),
                "prepared": True,
                "attempted": True,
                "reason": "launched" if launch_ok else str(launch.get("reason", "native_launch_failed") or "native_launch_failed"),
                "native_launch_attempted": True,
                "native_launch_ok": launch_ok,
                "kernel_executed": bool(launch.get("kernel_executed", False)),
                "probe_clone_parameters_mutated": bool(launch.get("parameters_mutated", False)),
                "persistent_owner_mutated": False,
                "runtime_session_reused": int(launch.get("launch_count", 0) or 0) > 1,
                "runtime_launch_contract": dict(launch.get("runtime_launch_contract") or {}),
                "runtime_launch_stream_binding": str(launch.get("runtime_launch_stream_binding", "") or ""),
                "runtime_synchronization": str(launch.get("runtime_synchronization", "") or ""),
                "stream_guard_probe": stream_guard,
                "event_chain_probe_requested": bool(stream_guard.get("event_chain_probe_requested", False)),
                "event_chain_probe_attempted": bool(stream_guard.get("event_chain_probe_attempted", False)),
                "event_chain_verified": bool(stream_guard.get("event_chain_verified", False)),
                "pre_launch_ordering_verified": bool(stream_guard.get("pre_launch_ordering_verified", False)),
                "post_launch_ordering_verified": bool(stream_guard.get("post_launch_ordering_verified", False)),
                "stream_wait_event_verified": bool(stream_guard.get("stream_wait_event_verified", False)),
                "launch": launch,
                "parity": parity,
                "parity_ok": parity_ok,
                "max_abs_diff": float(parity.get("max_abs_diff", float("inf"))),
                "max_rel_diff": float(parity.get("max_rel_diff", float("inf"))),
                "step_index_before": int(step_report.get("step_index", _owner_step_index(self._probe_owner)) or 0) - 1,
                "step_index_after_expected": int(step_report.get("step_index", 0) or 0),
                "owner_step_backend": str(step_report.get("backend", "") or ""),
                "owner_step_native_kernel_present": bool(step_report.get("native_kernel_present", False)),
                "prepare_elapsed_ms": round(prepare_elapsed, 4),
                "launch_elapsed_ms": round(launch_elapsed, 4),
                "elapsed_ms": round(prepare_elapsed + launch_elapsed, 4),
                "blocked_reasons": _blocked_reasons(launch=launch, parity=parity),
            }
        )
        return prepared

    def _load_native(self) -> Any | None:
        if self._native is None:
            self._native = load_lulynx_native()
        return self._native

    def _ensure_runtime(self, native: Any, device: torch.device) -> tuple[int | None, dict[str, Any]]:
        if self._runtime_id is not None:
            return int(self._runtime_id), dict(self._runtime_report)
        arch = self.arch or _cuda_arch(device)
        report = native.create_adamw_cuda_kernel_runtime_session_py(str(self.workspace_root), arch)
        if not isinstance(report, Mapping) or not bool(report.get("ok", False)):
            return None, dict(report) if isinstance(report, Mapping) else {"raw_type": type(report).__name__}
        self._runtime_id = int(report.get("runtime_session_id", 0) or 0)
        self._runtime_report = dict(report)
        return int(self._runtime_id), dict(self._runtime_report)

    def _ensure_probe_owner(self, source: Any) -> bool:
        signature = _owner_signature(source)
        reused = self._probe_owner is not None and _owner_signature(self._probe_owner) == signature
        if not reused:
            self._destroy_binding_session()
            self._probe_owner = PersistentFlatAdamW.from_state_dict(source.state_dict())
        else:
            self._probe_owner.load_state_dict(source.state_dict())
        return reused

    def _ensure_binding_session(self, native: Any, owner: PersistentFlatAdamW | None) -> tuple[int | None, bool, dict[str, Any]]:
        if owner is None:
            return None, False, {"ok": False, "reason": "probe_owner_missing"}
        signature = _owner_signature(owner)
        if self._binding_session_id is not None and self._binding_signature == signature:
            return int(self._binding_session_id), True, dict(self._binding_report)
        self._destroy_binding_session()
        registry, handles, _descriptor = register_persistent_flat_adamw_buffers(owner)
        request = build_flat_adamw_native_binding_request(registry, handles)
        tensor_map = build_tensor_object_map_for_handles(registry, handles)
        session = native.create_flat_adamw_tensor_binding_session(json.dumps(request), tensor_map)
        if not isinstance(session, Mapping) or not bool(session.get("ok", False)):
            return None, False, dict(session) if isinstance(session, Mapping) else {"ok": False, "reason": "binding_session_invalid"}
        self._binding_session_id = int(session.get("session_id", 0) or 0)
        self._binding_signature = signature
        self._binding_report = dict(session)
        return int(self._binding_session_id), False, dict(self._binding_report)

    def _destroy_binding_session(self) -> None:
        if self._native is not None and self._binding_session_id is not None:
            try:
                self._native.destroy_tensor_binding_session(int(self._binding_session_id))
            except Exception:
                pass
        self._binding_session_id = None
        self._binding_signature = ""
        self._binding_report = {}

    def _launch_prepared_owner(
        self,
        native: Any,
        session_id: int,
        runtime_id: int,
        owner: PersistentFlatAdamW,
        *,
        max_numel: int,
    ) -> dict[str, Any]:
        if native is None:
            return {"ok": False, "reason": "lulynx_native_not_importable"}
        if session_id <= 0:
            return {"ok": False, "reason": "binding_session_missing"}
        if runtime_id <= 0:
            return {"ok": False, "reason": "runtime_session_missing"}
        launch = native.tensor_binding_session_cuda_adamw_runtime_probe(
            int(session_id),
            int(runtime_id),
            json.dumps(
                build_native_update_adamw_launch_config(
                    owner,
                    max_numel=max_numel,
                    event_chain_probe=self.event_chain_probe,
                    capture_stage="owner_native_launch_probe",
                )
            ),
        )
        return dict(launch) if isinstance(launch, Mapping) else {"ok": False, "reason": "native_launch_invalid"}

    def _launch_native_owner(
        self,
        native: Any,
        runtime_id: int,
        owner: PersistentFlatAdamW,
        *,
        max_numel: int,
    ) -> dict[str, Any]:
        registry, handles, _descriptor = register_persistent_flat_adamw_buffers(owner)
        request = build_flat_adamw_native_binding_request(registry, handles)
        tensor_map = build_tensor_object_map_for_handles(registry, handles)
        session = native.create_flat_adamw_tensor_binding_session(json.dumps(request), tensor_map)
        if not isinstance(session, Mapping) or not bool(session.get("ok", False)):
            return dict(session) if isinstance(session, Mapping) else {"ok": False, "reason": "binding_session_invalid"}
        session_id = int(session.get("session_id", 0) or 0)
        try:
            config = build_native_update_adamw_launch_config(
                owner,
                max_numel=max_numel,
                event_chain_probe=self.event_chain_probe,
                capture_stage="owner_native_launch_probe",
            )
            launch = native.tensor_binding_session_cuda_adamw_runtime_probe(
                session_id,
                int(runtime_id),
                json.dumps(config),
            )
            return dict(launch) if isinstance(launch, Mapping) else {"ok": False, "reason": "native_launch_invalid"}
        finally:
            try:
                native.destroy_tensor_binding_session(session_id)
            except Exception:
                pass


def build_owner_native_launch_probe_skip(
    *,
    reason: str,
    max_numel: int = DEFAULT_OWNER_NATIVE_LAUNCH_MAX_NUMEL,
    owner_state: Any = None,
    loader: Mapping[str, Any] | None = None,
    started: float | None = None,
) -> dict[str, Any]:
    payload = _base_payload(started or time.perf_counter(), owner_state, max_numel=max_numel)
    payload.update(
        {
            "ok": False,
            "skipped": True,
            "attempted": False,
            "reason": str(reason or "skipped"),
            "native_launch_attempted": False,
            "native_launch_ok": False,
            "kernel_executed": False,
            "probe_clone_parameters_mutated": False,
            "persistent_owner_mutated": False,
            "loader": dict(loader or {}),
            "parity_ok": False,
            "blocked_reasons": [str(reason or "skipped")],
        }
    )
    return payload


def _base_payload(started: float, owner_state: Any, *, max_numel: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "probe": "turbocore_owner_native_launch_probe_v1",
        "contract": "turbocore_owner_buffer_native_launch_v1",
        "owner_numel": _state_numel(owner_state),
        "owner_device": _state_device(owner_state),
        "max_numel": max(int(max_numel or 0), 1),
        "runtime_diagnostic_launch": True,
        "owner_state_cloned": bool(owner_state),
        "persistent_runtime_session": True,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_kernel_present": False,
        "performance_test_ready": False,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4),
    }


def _error_payload(
    reason: str,
    started: float,
    owner_state: Any,
    *,
    max_numel: int,
    **extra: Any,
) -> dict[str, Any]:
    payload = _base_payload(started, owner_state, max_numel=max_numel)
    payload.update(
        {
            "ok": False,
            "skipped": False,
            "attempted": True,
            "reason": reason,
            "native_launch_attempted": False,
            "native_launch_ok": False,
            "kernel_executed": False,
            "probe_clone_parameters_mutated": False,
            "persistent_owner_mutated": False,
            "parity_ok": False,
            "blocked_reasons": [reason],
        }
    )
    payload.update(extra)
    return payload

def _compare_owner_buffers(native_owner: PersistentFlatAdamW, expected_owner: Any) -> dict[str, Any]:
    diffs = {
        role: _max_abs_diff(getattr(native_owner, role), getattr(expected_owner, role, None))
        for role in ("param_flat", "grad_flat", "exp_avg", "exp_avg_sq")
    }
    rel_diffs = {
        role: _max_rel_diff(getattr(native_owner, role), getattr(expected_owner, role, None))
        for role in ("param_flat", "grad_flat", "exp_avg", "exp_avg_sq")
    }
    parity_ok = (
        (diffs["param_flat"] <= 3e-4 or rel_diffs["param_flat"] <= 5e-6)
        and (diffs["grad_flat"] <= 1e-7 or rel_diffs["grad_flat"] <= 1e-7)
        and (diffs["exp_avg"] <= 5e-6 or rel_diffs["exp_avg"] <= 5e-6)
        and (diffs["exp_avg_sq"] <= 5e-5 or rel_diffs["exp_avg_sq"] <= 2e-5)
    )
    return {
        "schema_version": 1,
        "parity_ok": bool(parity_ok),
        "max_abs_diff": max(diffs.values()) if diffs else 0.0,
        "max_rel_diff": max(rel_diffs.values()) if rel_diffs else 0.0,
        "diffs": diffs,
        "rel_diffs": rel_diffs,
    }


def _max_abs_diff(left: Any, right: Any) -> float:
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor) or left.numel() != right.numel():
        return float("inf")
    return float((left.detach().float() - right.detach().float()).abs().max().cpu().item()) if left.numel() else 0.0


def _max_rel_diff(left: Any, right: Any) -> float:
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor) or left.numel() != right.numel():
        return float("inf")
    if left.numel() == 0:
        return 0.0
    diff = (left.detach().float() - right.detach().float()).abs()
    denom = right.detach().float().abs().clamp_min(1e-12)
    return float((diff / denom).max().cpu().item())


def _blocked_reasons(*, launch: Mapping[str, Any], parity: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not bool(launch.get("ok", False)):
        reasons.append(str(launch.get("reason", "native_launch_failed") or "native_launch_failed"))
    if not bool(parity.get("parity_ok", False)):
        reasons.append("owner_native_launch_parity_failed")
    if bool(launch.get("training_dispatch", False)) or bool(launch.get("training_path_enabled", False)):
        reasons.append("native_launch_training_dispatch_unexpected")
    return _dedupe(reasons)


def _missing_entrypoints(native: Any) -> list[str]:
    required = [
        "create_flat_adamw_tensor_binding_session",
        "destroy_tensor_binding_session",
        "create_adamw_cuda_kernel_runtime_session_py",
        "destroy_adamw_cuda_kernel_runtime_session_py",
        "tensor_binding_session_cuda_adamw_runtime_probe",
    ]
    return [name for name in required if not hasattr(native, name)]


def _owner_skip_reason(owner: Any, *, max_numel: int) -> str:
    numel = _state_numel(owner)
    if numel <= 0:
        return "owner_numel_empty"
    if numel > max(int(max_numel or 0), 1):
        return "owner_numel_exceeds_probe_limit"
    tensor = getattr(owner, "param_flat", None)
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
        return "owner_buffers_not_cuda"
    if not torch.cuda.is_available():
        return "torch_cuda_unavailable"
    return ""


def _owner_signature(owner: Any) -> str:
    layout = getattr(owner, "layout", None)
    if layout is not None:
        shapes = getattr(layout, "shapes", ())
        numels = getattr(layout, "numels", ())
        device = str(getattr(layout, "device", "") or getattr(getattr(owner, "param_flat", None), "device", ""))
        return json.dumps({"shapes": [list(shape) for shape in shapes], "numels": list(numels), "device": device})
    data = dict(owner or {}) if isinstance(owner, Mapping) else {}
    return json.dumps(data.get("layout") or {}, sort_keys=True, default=str)


def _state_numel(state: Any) -> int:
    owner_tensor = getattr(state, "param_flat", None)
    if isinstance(owner_tensor, torch.Tensor):
        return int(owner_tensor.numel())
    data = dict(state) if isinstance(state, Mapping) else {}
    layout = data.get("layout") if isinstance(data.get("layout"), Mapping) else {}
    if layout:
        return int(layout.get("total_numel", 0) or 0)
    tensor = data.get("param_flat")
    return int(tensor.numel()) if isinstance(tensor, torch.Tensor) else 0


def _state_step_index(state: Any) -> int:
    try:
        if hasattr(state, "step_index"):
            return int(getattr(state, "step_index", 0) or 0)
        return int((state.get("step_index", 0) if isinstance(state, Mapping) else 0) or 0)
    except (TypeError, ValueError):
        return 0


def _owner_step_index(owner: Any) -> int:
    try:
        return int(getattr(owner, "step_index", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _state_device(state: Any) -> str:
    owner_tensor = getattr(state, "param_flat", None)
    if isinstance(owner_tensor, torch.Tensor):
        return str(owner_tensor.device)
    data = dict(state) if isinstance(state, Mapping) else {}
    layout = data.get("layout") if isinstance(data.get("layout"), Mapping) else {}
    if layout.get("device"):
        return str(layout.get("device"))
    tensor = data.get("param_flat")
    return str(tensor.device) if isinstance(tensor, torch.Tensor) else ""


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = [
    "DEFAULT_OWNER_NATIVE_LAUNCH_MAX_NUMEL",
    "TurboCoreOwnerNativeLaunchProbe",
    "build_owner_native_launch_probe_skip",
]
