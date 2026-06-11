"""Smoke tests for TurboCore native ABI validation helpers."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_native_abi import (  # noqa: E402
    validate_flat_adamw_owner_capability,
    validate_native_optimizer_stateful_capability,
    validate_workspace_pipeline_native_capabilities,
)
from core.turbocore_workspace_pipeline import (  # noqa: E402
    build_turbocore_native_training_capability_stub,
    build_workspace_pipeline_native_capability_stub,
)


def test_inactive_stub_reports_missing_entrypoints() -> None:
    stub = build_workspace_pipeline_native_capability_stub()
    payload = validate_workspace_pipeline_native_capabilities(stub)
    assert payload["training_path_enabled"] is False
    assert payload["ok"] is False
    assert payload["features"]["workspace_pool"]["missing_entrypoints"]
    assert payload["features"]["data_pipeline"]["missing_entrypoints"]


def test_complete_report_passes_schema_validator() -> None:
    payload = validate_workspace_pipeline_native_capabilities(
        {
            "features": {
                "workspace_pool": {
                    "available": False,
                    "status": "capability_stub",
                    "entrypoints": [
                        "create_workspace_pool",
                        "workspace_acquire",
                        "workspace_release",
                        "workspace_stats",
                        "destroy_workspace_pool",
                    ],
                },
                "data_pipeline": {
                    "available": False,
                    "status": "capability_stub",
                    "entrypoints": [
                        "create_data_pipeline",
                        "submit_staged_batch",
                        "consume_ready_batch",
                        "release_batch_lease",
                        "close_data_pipeline",
                    ],
                },
            },
        }
    )
    assert payload["ok"] is True
    assert payload["features"]["workspace_pool"]["available"] is False
    assert payload["features"]["data_pipeline"]["available"] is False


def test_stateful_optimizer_capability_validator() -> None:
    report = build_turbocore_native_training_capability_stub()
    payload = validate_native_optimizer_stateful_capability(report)
    assert payload["ok"] is True, payload
    native_optimizer = payload["features"]["native_optimizer"]
    assert native_optimizer["stateful"] is True, native_optimizer
    assert "AdamW" in native_optimizer["supported_optimizers"], native_optimizer
    assert not native_optimizer["missing_entrypoints"], native_optimizer
    flat_owner = native_optimizer["flat_owner"]
    assert flat_owner["ok"] is True, flat_owner
    assert flat_owner["available"] is False, flat_owner
    assert flat_owner["training_path_enabled"] is False, flat_owner
    assert flat_owner["native_kernel_present"] is False, flat_owner
    assert flat_owner["layout"] == "flat_contiguous_fp32_buffers", flat_owner
    assert not flat_owner["missing_buffers"], flat_owner
    assert flat_owner["supports_stream_descriptor"] is True, flat_owner
    assert flat_owner["descriptor_schema"] == "flat_adamw_owner_descriptor_v1", flat_owner
    assert not flat_owner["missing_descriptor_fields"], flat_owner
    binding_request = flat_owner["binding_request"]
    assert binding_request["ok"] is True, binding_request
    assert binding_request["present"] is True, binding_request
    assert binding_request["available"] is False, binding_request
    assert binding_request["binding_request_schema"] == "turbocore_native_tensor_binding_request_v1", binding_request
    assert "validate_flat_adamw_tensor_binding_request" in binding_request["entrypoints"], binding_request
    assert "probe_flat_adamw_tensor_object_binding" in binding_request["entrypoints"], binding_request
    assert "create_flat_adamw_tensor_binding_session" in binding_request["entrypoints"], binding_request
    assert "tensor_binding_session_snapshot" in binding_request["entrypoints"], binding_request
    assert "tensor_binding_session_validate" in binding_request["entrypoints"], binding_request
    assert "tensor_binding_session_launch_plan" in binding_request["entrypoints"], binding_request
    assert "tensor_binding_session_stream_guard_probe" in binding_request["entrypoints"], binding_request
    assert "tensor_binding_session_noop_launch" in binding_request["entrypoints"], binding_request
    assert "tensor_binding_session_cpu_reference_guard" in binding_request["entrypoints"], binding_request
    assert "tensor_binding_session_cuda_stub_launch" in binding_request["entrypoints"], binding_request
    assert "tensor_binding_session_cuda_adamw_tensor_probe" in binding_request["entrypoints"], binding_request
    assert "tensor_binding_session_cuda_adamw_runtime_probe" in binding_request["entrypoints"], binding_request
    assert "get_adamw_cuda_kernel_contract" in binding_request["entrypoints"], binding_request
    assert "destroy_tensor_binding_session" in binding_request["entrypoints"], binding_request
    assert binding_request["supports_external_tensor_handles"] is False, binding_request
    assert binding_request["supports_tensor_object_sessions"] is True, binding_request
    assert binding_request["supports_launch_plan"] is True, binding_request
    kernel_registry = binding_request["kernel_registry"]
    assert kernel_registry["ok"] is True, kernel_registry
    assert kernel_registry["dry_run_launch_supported"] is True, kernel_registry
    assert kernel_registry["cpu_reference_guard_supported"] is True, kernel_registry
    assert kernel_registry["cuda_stub_launch_supported"] is True, kernel_registry
    kernel_contract = kernel_registry["kernel_contract"]
    assert kernel_contract["ok"] is True, kernel_contract
    assert kernel_contract["contract"] == "adamw_flat_fp32_cuda_kernel_v0", kernel_contract
    assert kernel_contract["native_kernel_present"] is False, kernel_contract
    assert kernel_contract["training_path_enabled"] is False, kernel_contract
    assert kernel_registry["native_kernel_present"] is False, kernel_registry
    assert kernel_registry["training_path_enabled"] is False, kernel_registry
    assert binding_request["native_kernel_present"] is False, binding_request
    assert binding_request["training_path_enabled"] is False, binding_request
    nvrtc_probe = report["features"].get("cuda_nvrtc_compile_probe", {})
    assert nvrtc_probe["entrypoints"] == ["probe_adamw_cuda_nvrtc_compile_py"], nvrtc_probe
    assert nvrtc_probe["native_kernel_present"] is False, nvrtc_probe
    assert nvrtc_probe["training_path_enabled"] is False, nvrtc_probe
    assert nvrtc_probe["performance_test_ready"] is False, nvrtc_probe
    assert nvrtc_probe["artifact_only"] is True, nvrtc_probe
    driver_probe = report["features"].get("cuda_driver_ptx_probe", {})
    assert driver_probe["entrypoints"] == ["probe_adamw_cuda_driver_ptx_load_py"], driver_probe
    assert driver_probe["native_kernel_present"] is False, driver_probe
    assert driver_probe["training_path_enabled"] is False, driver_probe
    assert driver_probe["performance_test_ready"] is False, driver_probe
    assert driver_probe["artifact_only"] is True, driver_probe
    assert driver_probe["kernel_executed"] is False, driver_probe
    scratch_probe = report["features"].get("cuda_scratch_launch_probe", {})
    assert scratch_probe["entrypoints"] == ["probe_adamw_cuda_scratch_launch_py"], scratch_probe
    assert scratch_probe["scratch_buffers_only"] is True, scratch_probe
    assert scratch_probe["training_tensor_binding"] is False, scratch_probe
    assert scratch_probe["native_kernel_present"] is False, scratch_probe
    assert scratch_probe["training_path_enabled"] is False, scratch_probe
    assert scratch_probe["performance_test_ready"] is False, scratch_probe
    runtime_probe = report["features"].get("cuda_adamw_runtime", {})
    assert "benchmark_adamw_cuda_kernel_runtime_session_py" in runtime_probe["entrypoints"], runtime_probe
    assert runtime_probe["training_dispatch"] is False, runtime_probe
    assert runtime_probe["stream_lifetime_bound"] is False, runtime_probe
    assert runtime_probe["native_kernel_present"] is False, runtime_probe
    assert runtime_probe["training_path_enabled"] is False, runtime_probe
    assert runtime_probe["performance_test_ready"] is False, runtime_probe


def test_flat_adamw_owner_validator_rejects_incomplete_contract() -> None:
    payload = validate_flat_adamw_owner_capability({"flat_owner": {"entrypoints": []}})
    assert payload["ok"] is False, payload
    assert payload["missing_entrypoints"], payload
    assert payload["missing_buffers"], payload


if __name__ == "__main__":
    test_inactive_stub_reports_missing_entrypoints()
    test_complete_report_passes_schema_validator()
    test_stateful_optimizer_capability_validator()
    test_flat_adamw_owner_validator_rejects_incomplete_contract()
    print("turbocore_native_abi_smoke: ok")
