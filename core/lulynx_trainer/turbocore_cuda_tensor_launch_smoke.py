"""Smoke for the dev-only native AdamW launch against temporary CUDA tensors."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
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
from core.turbocore_tensor_handle_registry import (  # noqa: E402
    TurboCoreTensorHandleRegistry,
    build_tensor_object_map_for_handles,
)


def _inject_native_artifact_dir_from_env() -> None:
    raw = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    if not raw:
        return
    path = Path(raw).expanduser()
    if path.is_dir():
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _make_cuda_request() -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    device = torch.device("cuda")
    registry = TurboCoreTensorHandleRegistry(namespace="cuda_tensor_launch_smoke")
    param = (torch.arange(24, dtype=torch.float32, device=device) * 0.125 + 0.5).contiguous()
    grad = (torch.arange(24, dtype=torch.float32, device=device) * 0.001 + 0.01).contiguous()
    exp_avg = torch.zeros(24, dtype=torch.float32, device=device).contiguous()
    exp_avg_sq = torch.zeros(24, dtype=torch.float32, device=device).contiguous()
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


def _reference(role_tensors: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    param = role_tensors["param_flat"].detach().clone()
    grad = role_tensors["grad_flat"].detach().clone()
    exp_avg = role_tensors["exp_avg"].detach().clone()
    exp_avg_sq = role_tensors["exp_avg_sq"].detach().clone()
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
    return {"param_flat": param, "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def run_smoke() -> dict[str, Any]:
    _inject_native_artifact_dir_from_env()
    spec = importlib.util.find_spec("lulynx_native")
    if spec is None:
        return {
            "schema_version": 1,
            "probe": "turbocore_cuda_tensor_launch_smoke",
            "ok": True,
            "skipped": True,
            "reason": "lulynx_native_not_importable",
            "training_path_enabled": False,
            "performance_test_ready": False,
        }
    if not torch.cuda.is_available():
        return {
            "schema_version": 1,
            "probe": "turbocore_cuda_tensor_launch_smoke",
            "ok": True,
            "skipped": True,
            "reason": "torch_cuda_unavailable",
            "training_path_enabled": False,
            "performance_test_ready": False,
        }

    import lulynx_native  # type: ignore

    if not hasattr(lulynx_native, "tensor_binding_session_cuda_adamw_tensor_probe"):
        raise AssertionError("tensor_binding_session_cuda_adamw_tensor_probe missing from lulynx_native")
    request, tensor_map, role_tensors = _make_cuda_request()
    config = {
        "workspace_root": str(PROJECT_ROOT),
        "arch": "compute_89",
        "lr": 1e-3,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "step_index": 0,
        "block_size": 128,
        "max_numel": 64,
    }
    reference = _reference(role_tensors, config)
    session = lulynx_native.create_flat_adamw_tensor_binding_session(json.dumps(request), tensor_map)
    assert session["ok"] is True, session
    session_id = int(session["session_id"])
    try:
        launch = lulynx_native.tensor_binding_session_cuda_adamw_tensor_probe(session_id, json.dumps(config))
        assert launch["ok"] is True, launch
        assert launch["kernel_executed"] is True, launch
        assert launch["parameters_mutated"] is True, launch
        assert launch["training_tensor_binding"] is True, launch
        assert launch["training_dispatch"] is False, launch
        assert launch["training_path_enabled"] is False, launch
        assert launch["native_kernel_present"] is False, launch
        assert launch["performance_test_ready"] is False, launch
        torch.cuda.synchronize()
        diffs = {
            "param_flat": _max_abs_diff(role_tensors["param_flat"], reference["param_flat"]),
            "exp_avg": _max_abs_diff(role_tensors["exp_avg"], reference["exp_avg"]),
            "exp_avg_sq": _max_abs_diff(role_tensors["exp_avg_sq"], reference["exp_avg_sq"]),
        }
        max_diff = max(diffs.values())
        assert max_diff <= 5e-6, {"diffs": diffs, "launch": launch}
    finally:
        destroyed = lulynx_native.destroy_tensor_binding_session(session_id)
        assert destroyed["ok"] is True, destroyed

    return {
        "schema_version": 1,
        "probe": "turbocore_cuda_tensor_launch_smoke",
        "ok": True,
        "skipped": False,
        "origin": str(getattr(spec, "origin", "")),
        "device": torch.cuda.get_device_name(0),
        "numel": 24,
        "max_abs_diff": max_diff,
        "diffs": diffs,
        "kernel_executed": True,
        "training_tensor_binding": True,
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
