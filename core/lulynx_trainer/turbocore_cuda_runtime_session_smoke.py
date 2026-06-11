"""Smoke for dev-only cached CUDA AdamW kernel runtime sessions."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_native_tensor_binding import build_flat_adamw_native_binding_request  # noqa: E402
from core.services.native_module_loader import ensure_lulynx_native_artifact_path  # noqa: E402
from core.turbocore_tensor_handle_registry import (  # noqa: E402
    TurboCoreTensorHandleRegistry,
    build_tensor_object_map_for_handles,
)


def _inject_native_artifact_dir_from_env() -> None:
    ensure_lulynx_native_artifact_path()


def _current_stream_descriptor() -> dict[str, Any]:
    stream = torch.cuda.current_stream()
    handle = int(getattr(stream, "cuda_stream", 0) or 0)
    return {
        "schema_version": 1,
        "descriptor": "turbocore_borrowed_cuda_stream_descriptor_v0",
        "device_type": "cuda",
        "device_index": int(torch.cuda.current_device()),
        "stream_kind": "torch_current",
        "stream_id": str(handle) if handle else "",
        "stream_source": "torch.cuda.current_stream",
        "stream_capture_stage": "runtime_session_smoke",
        "python_stream_object_alive": True,
        "python_stream_lifetime_scope": "descriptor_only",
        "cuda_stream_handle": handle,
        "stream_handle_reported": True,
        "stream_handle_nonzero": bool(handle),
        "training_path_enabled": False,
    }


def _make_cuda_request(numel: int) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    device = torch.device("cuda")
    registry = TurboCoreTensorHandleRegistry(namespace="cuda_runtime_session_smoke")
    param = (torch.arange(numel, dtype=torch.float32, device=device) * 0.0005 + 0.5).contiguous()
    grad = (torch.arange(numel, dtype=torch.float32, device=device) * 0.00001 + 0.01).contiguous()
    exp_avg = torch.zeros(numel, dtype=torch.float32, device=device).contiguous()
    exp_avg_sq = torch.zeros(numel, dtype=torch.float32, device=device).contiguous()
    handles = registry.register_flat_adamw_buffers(
        param_flat=param,
        grad_flat=grad,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
    )
    request = build_flat_adamw_native_binding_request(registry, handles)
    tensor_map = build_tensor_object_map_for_handles(registry, handles)
    role_tensors = {binding["role"]: tensor_map[binding["handle_id"]] for binding in request["bindings"]}
    return request, tensor_map, role_tensors


def _reference_step(role_tensors: dict[str, torch.Tensor], config: dict[str, Any]) -> None:
    param = role_tensors["ref_param"]
    grad = role_tensors["ref_grad"]
    exp_avg = role_tensors["ref_exp_avg"]
    exp_avg_sq = role_tensors["ref_exp_avg_sq"]
    beta1, beta2 = [float(item) for item in config["betas"]]
    lr = float(config["lr"])
    eps = float(config["eps"])
    weight_decay = float(config["weight_decay"])
    step_number = int(config["step_index"]) + 1
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    if weight_decay:
        param.mul_(1.0 - lr * weight_decay)
    step_size = lr / (1.0 - beta1**step_number)
    denom = exp_avg_sq.sqrt().div_((1.0 - beta2**step_number) ** 0.5).add_(eps)
    param.addcdiv_(exp_avg, denom, value=-step_size)


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def run_smoke(numel: int = 4096, steps: int = 8) -> dict[str, Any]:
    _inject_native_artifact_dir_from_env()
    spec = importlib.util.find_spec("lulynx_native")
    if spec is None:
        return {"schema_version": 1, "probe": "turbocore_cuda_runtime_session_smoke", "ok": True, "skipped": True, "reason": "lulynx_native_not_importable"}
    if not torch.cuda.is_available():
        return {"schema_version": 1, "probe": "turbocore_cuda_runtime_session_smoke", "ok": True, "skipped": True, "reason": "torch_cuda_unavailable"}

    import lulynx_native  # type: ignore

    required = [
        "create_adamw_cuda_kernel_runtime_session_py",
        "adamw_cuda_kernel_runtime_session_snapshot_py",
        "destroy_adamw_cuda_kernel_runtime_session_py",
        "tensor_binding_session_cuda_adamw_runtime_probe",
    ]
    missing = [name for name in required if not hasattr(lulynx_native, name)]
    assert not missing, missing

    request, tensor_map, role_tensors = _make_cuda_request(numel)
    role_tensors["ref_param"] = role_tensors["param_flat"].detach().clone()
    role_tensors["ref_grad"] = role_tensors["grad_flat"].detach().clone()
    role_tensors["ref_exp_avg"] = role_tensors["exp_avg"].detach().clone()
    role_tensors["ref_exp_avg_sq"] = role_tensors["exp_avg_sq"].detach().clone()
    runtime = lulynx_native.create_adamw_cuda_kernel_runtime_session_py(str(PROJECT_ROOT), "compute_89")
    assert runtime["ok"] is True, runtime
    runtime_id = int(runtime["runtime_session_id"])
    session = lulynx_native.create_flat_adamw_tensor_binding_session(json.dumps(request), tensor_map)
    assert session["ok"] is True, session
    session_id = int(session["session_id"])
    launches: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for step in range(steps):
            config = {
                "lr": 1e-3,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
                "step_index": step,
                "block_size": 256,
                "max_numel": max(numel, 4096),
                "stream_guard_descriptor": _current_stream_descriptor(),
            }
            _reference_step(role_tensors, config)
            launch = lulynx_native.tensor_binding_session_cuda_adamw_runtime_probe(session_id, runtime_id, json.dumps(config))
            assert launch["ok"] is True, launch
            assert launch["kernel_executed"] is True, launch
            assert launch["training_dispatch"] is False, launch
            assert launch["training_path_enabled"] is False, launch
            assert launch["runtime_diagnostic_launch"] is True, launch
            assert launch["runtime_launch_stream_binding"] == "cuda_driver_default_stream_null", launch
            assert launch["runtime_synchronization"] == "cuCtxSynchronize_diagnostic_only", launch
            stream_guard = launch["stream_guard_probe"]
            assert stream_guard["contract"] == "turbocore_tensor_binding_stream_guard_v2", launch
            assert stream_guard["legacy_contract"] == "turbocore_tensor_binding_stream_guard_v1", launch
            assert stream_guard["stream_identity_ready"] is True, launch
            assert stream_guard["stream_guard_ready"] is False, launch
            assert stream_guard["event_chain_contract"] == "turbocore_stream_event_chain_guard_v2", launch
            assert stream_guard["event_chain_probe_requested"] is False, launch
            assert stream_guard["event_chain_probe_attempted"] is False, launch
            assert stream_guard["event_chain_state"] == "not_attempted", launch
            assert stream_guard["event_chain_verified"] is False, launch
            assert stream_guard["stream_wait_event_verified"] is False, launch
            runtime_contract = launch["runtime_launch_contract"]
            assert runtime_contract["contract"] == "turbocore_tensor_binding_runtime_launch_v0", launch
            assert runtime_contract["runtime_diagnostic_launch"] is True, launch
            assert runtime_contract["stream_identity_ready"] is True, launch
            assert runtime_contract["stream_guard_ready"] is False, launch
            assert runtime_contract["synchronization_guard_ready"] is False, launch
            assert runtime_contract["event_chain_state"] == "not_attempted", launch
            assert runtime_contract["event_chain_probe_requested"] is False, launch
            assert runtime_contract["native_launch_candidate"] is False, launch
            launches.append({"step": step, "launch_count": int(launch["launch_count"])})
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        diffs = {
            "param_flat": _max_abs_diff(role_tensors["param_flat"], role_tensors["ref_param"]),
            "exp_avg": _max_abs_diff(role_tensors["exp_avg"], role_tensors["ref_exp_avg"]),
            "exp_avg_sq": _max_abs_diff(role_tensors["exp_avg_sq"], role_tensors["ref_exp_avg_sq"]),
        }
        max_diff = max(diffs.values())
        assert max_diff <= 5e-6, {"diffs": diffs, "launches": launches}
        snapshot = lulynx_native.adamw_cuda_kernel_runtime_session_snapshot_py(runtime_id)
        assert snapshot["ok"] is True, snapshot
        assert int(snapshot["launch_count"]) == steps, snapshot
    finally:
        destroyed_session = lulynx_native.destroy_tensor_binding_session(session_id)
        destroyed_runtime = lulynx_native.destroy_adamw_cuda_kernel_runtime_session_py(runtime_id)
        assert destroyed_session["ok"] is True, destroyed_session
        assert destroyed_runtime["ok"] is True, destroyed_runtime

    return {
        "schema_version": 1,
        "probe": "turbocore_cuda_runtime_session_smoke",
        "ok": True,
        "skipped": False,
        "origin": str(getattr(spec, "origin", "")),
        "device": torch.cuda.get_device_name(0),
        "numel": numel,
        "steps": steps,
        "elapsed_ms": elapsed_ms,
        "avg_launch_ms_including_python": elapsed_ms / max(steps, 1),
        "max_abs_diff": max_diff,
        "diffs": diffs,
        "launches": launches,
        "runtime_session": True,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_kernel_present": False,
        "performance_test_ready": False,
    }


if __name__ == "__main__":
    result = run_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not bool(result.get("ok", False)):
        raise SystemExit(1)
