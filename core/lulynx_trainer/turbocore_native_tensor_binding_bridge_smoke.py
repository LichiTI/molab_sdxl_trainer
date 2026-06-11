"""Smoke for native .pyd tensor binding request validation."""

from __future__ import annotations

import importlib.util
import json
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
from core.services.native_module_loader import ensure_lulynx_native_artifact_path  # noqa: E402
from core.turbocore_tensor_handle_registry import (  # noqa: E402
    TurboCoreTensorHandleRegistry,
    build_tensor_object_map_for_handles,
)


def _inject_native_artifact_dir_from_env() -> None:
    ensure_lulynx_native_artifact_path()


def _make_request_and_tensors() -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    registry = TurboCoreTensorHandleRegistry(namespace="native_bridge_smoke")
    handles = registry.register_flat_adamw_buffers(
        param_flat=torch.arange(24, dtype=torch.float32).contiguous(),
        grad_flat=torch.full((24,), 0.01, dtype=torch.float32).contiguous(),
        exp_avg=torch.zeros(24, dtype=torch.float32).contiguous(),
        exp_avg_sq=torch.zeros(24, dtype=torch.float32).contiguous(),
    )
    request = build_flat_adamw_native_binding_request(registry, handles)
    return request, build_tensor_object_map_for_handles(registry, handles)


def _role_tensors(request: dict[str, Any], tensor_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {binding["role"]: tensor_map[binding["handle_id"]] for binding in request["bindings"]}


def _python_adamw_preview(role_tensors: dict[str, torch.Tensor], config: dict[str, Any]) -> list[float]:
    param = role_tensors["param_flat"].detach().double().reshape(-1).clone()
    grad = role_tensors["grad_flat"].detach().double().reshape(-1).clone()
    exp_avg = role_tensors["exp_avg"].detach().double().reshape(-1).clone()
    exp_avg_sq = role_tensors["exp_avg_sq"].detach().double().reshape(-1).clone()
    max_grad_norm = float(config.get("max_grad_norm", 0.0))
    norm = float(torch.linalg.vector_norm(grad.float(), ord=2).item()) if grad.numel() else 0.0
    if max_grad_norm > 0.0 and norm > max_grad_norm:
        grad.mul_(max_grad_norm / max(norm, 1e-12))
    beta1, beta2 = [float(item) for item in config.get("betas", [0.9, 0.999])]
    lr = float(config.get("lr", 1e-4))
    eps = float(config.get("eps", 1e-8))
    weight_decay = float(config.get("weight_decay", 0.01))
    step_number = int(config.get("step_index", 0)) + 1
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    if weight_decay:
        param.mul_(1.0 - lr * weight_decay)
    step_size = lr / (1.0 - beta1**step_number)
    denom = exp_avg_sq.sqrt().div_((1.0 - beta2**step_number) ** 0.5).add_(eps)
    param.addcdiv_(exp_avg, denom, value=-step_size)
    limit = int(config.get("preview_limit", 8))
    return [float(item) for item in param[:limit].tolist()]


def _assert_close_list(actual: list[float], expected: list[float], *, atol: float = 5e-7) -> None:
    assert len(actual) == len(expected), {"actual": actual, "expected": expected}
    diffs = [abs(float(left) - float(right)) for left, right in zip(actual, expected)]
    assert max(diffs, default=0.0) <= atol, {"max_diff": max(diffs, default=0.0), "actual": actual, "expected": expected}


def run_smoke() -> dict[str, Any]:
    _inject_native_artifact_dir_from_env()
    spec = importlib.util.find_spec("lulynx_native")
    if spec is None:
        return {
            "schema_version": 1,
            "probe": "turbocore_native_tensor_binding_bridge_smoke",
            "ok": True,
            "skipped": True,
            "reason": "lulynx_native_not_importable",
            "training_path_enabled": False,
            "performance_test_ready": False,
        }
    import lulynx_native  # type: ignore

    assert hasattr(lulynx_native, "probe_cuda_toolchain_py"), "probe_cuda_toolchain_py missing from lulynx_native"
    toolchain = lulynx_native.probe_cuda_toolchain_py()
    assert toolchain["training_path_enabled"] is False, toolchain
    assert toolchain["native_kernel_present"] is False, toolchain
    assert "recommended_strategy" in toolchain, toolchain

    request, tensor_map = _make_request_and_tensors()
    role_tensors = _role_tensors(request, tensor_map)
    param_before = role_tensors["param_flat"].detach().clone()
    if not hasattr(lulynx_native, "validate_flat_adamw_tensor_binding_request"):
        raise AssertionError("validate_flat_adamw_tensor_binding_request missing from lulynx_native")
    validation = lulynx_native.validate_flat_adamw_tensor_binding_request(json.dumps(request))
    assert validation["ok"] is True, validation
    assert validation["request_shape_ready"] is True, validation
    assert validation["native_binding_ready"] is False, validation
    assert validation["performance_test_ready"] is False, validation
    assert validation["training_path_enabled"] is False, validation
    assert "native_external_tensor_handles_unsupported" in validation["blocked_reasons"], validation

    bad_request = dict(request)
    bad_request["bindings"] = [dict(item) for item in request["bindings"]]
    bad_request["bindings"][0]["pointer_exported"] = True
    bad_validation = lulynx_native.validate_flat_adamw_tensor_binding_request(json.dumps(bad_request))
    assert bad_validation["ok"] is False, bad_validation
    assert bad_validation["pointer_exported"] is True, bad_validation

    if not hasattr(lulynx_native, "probe_flat_adamw_tensor_object_binding"):
        raise AssertionError("probe_flat_adamw_tensor_object_binding missing from lulynx_native")
    object_probe = lulynx_native.probe_flat_adamw_tensor_object_binding(json.dumps(request), tensor_map)
    assert object_probe["ok"] is True, object_probe
    assert object_probe["tensor_object_binding_ready"] is True, object_probe
    assert object_probe["native_binding_ready"] is False, object_probe
    assert object_probe["performance_test_ready"] is False, object_probe
    assert object_probe["tensor_probe_count"] == 4, object_probe
    assert not object_probe["missing_tensor_handles"], object_probe
    for probe in object_probe["tensor_probes"]:
        assert probe["ok"] is True, object_probe
        assert probe["metadata"]["data_ptr_nonzero"] is True, object_probe

    missing_map = dict(tensor_map)
    missing_map.pop(request["bindings"][0]["handle_id"])
    missing_probe = lulynx_native.probe_flat_adamw_tensor_object_binding(json.dumps(request), missing_map)
    assert missing_probe["ok"] is False, missing_probe
    assert missing_probe["missing_tensor_handles"], missing_probe

    wrong_tensor_map = dict(tensor_map)
    wrong_tensor_map[request["bindings"][0]["handle_id"]] = torch.zeros(25, dtype=torch.float32)
    wrong_probe = lulynx_native.probe_flat_adamw_tensor_object_binding(json.dumps(request), wrong_tensor_map)
    assert wrong_probe["ok"] is False, wrong_probe
    assert wrong_probe["invalid_tensor_bindings"], wrong_probe

    required_session_entrypoints = [
        "create_flat_adamw_tensor_binding_session",
        "tensor_binding_session_snapshot",
        "tensor_binding_session_validate",
        "tensor_binding_session_launch_plan",
        "tensor_binding_session_stream_guard_probe",
        "tensor_binding_session_noop_launch",
        "tensor_binding_session_cpu_reference_guard",
        "tensor_binding_session_cuda_stub_launch",
        "tensor_binding_session_cuda_adamw_tensor_probe",
        "tensor_binding_session_cuda_adamw_runtime_probe",
        "get_adamw_cuda_kernel_contract",
        "destroy_tensor_binding_session",
    ]
    for name in required_session_entrypoints:
        if not hasattr(lulynx_native, name):
            raise AssertionError(f"{name} missing from lulynx_native")
    session = lulynx_native.create_flat_adamw_tensor_binding_session(json.dumps(request), tensor_map)
    assert session["ok"] is True, session
    assert session["tensor_object_binding_ready"] is True, session
    assert session["holds_python_tensor_refs"] is True, session
    assert session["training_path_enabled"] is False, session
    session_id = int(session["session_id"])
    snapshot = lulynx_native.tensor_binding_session_snapshot(session_id)
    assert snapshot["ok"] is True, snapshot
    assert snapshot["tensor_ref_count"] == 4, snapshot
    assert snapshot["native_binding_ready"] is False, snapshot
    assert snapshot["stream_contract"]["contract"] == "turbocore_tensor_binding_stream_lifetime_v0", snapshot
    assert snapshot["stream_lifetime_bound"] is False, snapshot
    validation_session = lulynx_native.tensor_binding_session_validate(session_id)
    assert validation_session["ok"] is True, validation_session
    assert validation_session["tensor_object_binding_ready"] is True, validation_session
    assert validation_session["stream_contract"]["python_tensor_lifetime_held"] is True, validation_session
    launch_plan = lulynx_native.tensor_binding_session_launch_plan(session_id)
    assert launch_plan["ok"] is True, launch_plan
    assert launch_plan["plan_kind"] == "adamw_flat_fp32_launch_plan_v0", launch_plan
    assert launch_plan["numel"] == 24, launch_plan
    assert launch_plan["grid_blocks"] == 1, launch_plan
    assert launch_plan["launchable_by_cuda_kernel"] is False, launch_plan
    assert launch_plan["native_binding_ready"] is False, launch_plan
    assert launch_plan["performance_test_ready"] is False, launch_plan
    assert launch_plan["stream_contract"]["lease_kind"] == "launch_plan_metadata_only", launch_plan
    assert launch_plan["stream_lifetime_bound"] is False, launch_plan
    assert launch_plan["stream_lease_id"] == launch_plan["lease_id"], launch_plan
    assert "stream_lifetime_not_bound" in launch_plan["stream_contract"]["blocked_reasons"], launch_plan
    stream_guard = lulynx_native.tensor_binding_session_stream_guard_probe(
        session_id,
        json.dumps(
            {
                "device_type": "cpu",
                "device_index": None,
                "stream_kind": "cpu_no_cuda_stream",
                "stream_id": "",
                "stream_source": "smoke_cpu",
                "cuda_stream_handle": 0,
                "stream_handle_reported": True,
            }
        ),
    )
    assert stream_guard["ok"] is True, stream_guard
    assert stream_guard["contract"] == "turbocore_tensor_binding_stream_guard_v2", stream_guard
    assert stream_guard["legacy_contract"] == "turbocore_tensor_binding_stream_guard_v1", stream_guard
    assert stream_guard["stream_guard_present"] is True, stream_guard
    assert stream_guard["stream_guard_ready"] is False, stream_guard
    assert stream_guard["stream_identity_ready"] is True, stream_guard
    assert stream_guard["stream_guard_level"] == "identity_verified_sync_blocked", stream_guard
    assert stream_guard["stream_handle_kind"] == "non_cuda_stream", stream_guard
    assert stream_guard["stream_handle_reported"] is True, stream_guard
    assert stream_guard["stream_handle_nonzero"] is False, stream_guard
    assert stream_guard["device_match"] is True, stream_guard
    assert stream_guard["synchronization_guard_ready"] is False, stream_guard
    assert stream_guard["event_chain_contract"] == "turbocore_stream_event_chain_guard_v2", stream_guard
    assert stream_guard["event_chain_state"] == "not_attempted", stream_guard
    assert stream_guard["event_chain_probe_requested"] is False, stream_guard
    assert stream_guard["event_chain_probe_attempted"] is False, stream_guard
    assert stream_guard["synchronization_strategy"] == "non_cuda_no_stream_sync", stream_guard
    assert stream_guard["event_chain_verified"] is False, stream_guard
    assert stream_guard["stream_wait_event_verified"] is False, stream_guard
    assert stream_guard["native_launch_candidate"] is False, stream_guard
    assert "event_chain_probe_not_requested" in stream_guard["blocked_reasons"], stream_guard
    assert "synchronization_guard_not_ready" in stream_guard["blocked_reasons"], stream_guard
    noop_launch = lulynx_native.tensor_binding_session_noop_launch(session_id)
    assert noop_launch["ok"] is True, noop_launch
    assert noop_launch["dry_run"] is True, noop_launch
    assert noop_launch["would_launch_kernel"] is True, noop_launch
    assert noop_launch["kernel_executed"] is False, noop_launch
    assert noop_launch["parameters_mutated"] is False, noop_launch
    assert noop_launch["native_kernel_present"] is False, noop_launch
    assert noop_launch["training_path_enabled"] is False, noop_launch
    assert noop_launch["performance_test_ready"] is False, noop_launch
    guard_config = {
        "lr": 1e-3,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "max_grad_norm": 0.0,
        "finite_check": True,
        "step_index": 0,
        "preview_limit": 8,
        "max_numel": 64,
    }
    cpu_guard = lulynx_native.tensor_binding_session_cpu_reference_guard(session_id, json.dumps(guard_config))
    assert cpu_guard["ok"] is True, cpu_guard
    assert cpu_guard["reference_backend"] == "cpu_reference_no_mutate", cpu_guard
    assert cpu_guard["kernel_executed"] is False, cpu_guard
    assert cpu_guard["parameters_mutated"] is False, cpu_guard
    assert cpu_guard["native_kernel_present"] is False, cpu_guard
    _assert_close_list(cpu_guard["param_preview"], _python_adamw_preview(role_tensors, guard_config))
    assert torch.equal(role_tensors["param_flat"], param_before), "CPU guard mutated param_flat"
    cuda_stub = lulynx_native.tensor_binding_session_cuda_stub_launch(session_id)
    assert cuda_stub["ok"] is True, cuda_stub
    assert cuda_stub["stub"] is True, cuda_stub
    assert cuda_stub["dry_run"] is True, cuda_stub
    assert cuda_stub["kernel_executed"] is False, cuda_stub
    assert cuda_stub["parameters_mutated"] is False, cuda_stub
    assert cuda_stub["native_kernel_present"] is False, cuda_stub
    assert cuda_stub["training_path_enabled"] is False, cuda_stub
    assert cuda_stub["performance_test_ready"] is False, cuda_stub
    assert cuda_stub["tensor_devices"] == ["cpu", "cpu", "cpu", "cpu"], cuda_stub
    assert "session_tensors_not_all_cuda" in cuda_stub["blocked_reasons"], cuda_stub
    kernel_contract = lulynx_native.get_adamw_cuda_kernel_contract()
    assert kernel_contract["contract"] == "adamw_flat_fp32_cuda_kernel_v0", kernel_contract
    assert kernel_contract["validation"]["ok"] is True, kernel_contract
    assert kernel_contract["native_kernel_present"] is False, kernel_contract
    assert kernel_contract["training_path_enabled"] is False, kernel_contract
    destroyed = lulynx_native.destroy_tensor_binding_session(session_id)
    assert destroyed["ok"] is True and destroyed["destroyed"] is True, destroyed
    after_destroy = lulynx_native.tensor_binding_session_snapshot(session_id)
    assert after_destroy["ok"] is False, after_destroy
    assert after_destroy["reason"] == "unknown_session", after_destroy

    bad_session = lulynx_native.create_flat_adamw_tensor_binding_session(json.dumps(request), missing_map)
    assert bad_session["ok"] is False, bad_session
    assert bad_session["reason"] == "tensor_object_probe_failed", bad_session
    return {
        "schema_version": 1,
        "probe": "turbocore_native_tensor_binding_bridge_smoke",
        "ok": True,
        "skipped": False,
        "origin": str(getattr(spec, "origin", "")),
        "cuda_toolchain_probe": bool(toolchain),
        "training_path_enabled": False,
        "tensor_object_binding_ready": True,
        "tensor_binding_session_ready": True,
        "launch_plan_ready": True,
        "stream_guard_probe_ready": True,
        "noop_launch_ready": True,
        "cpu_reference_guard_ready": True,
        "cuda_stub_ready": True,
        "kernel_contract_ready": True,
        "native_binding_ready": False,
        "performance_test_ready": False,
    }


if __name__ == "__main__":
    result = run_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not bool(result.get("ok", False)):
        raise SystemExit(1)
