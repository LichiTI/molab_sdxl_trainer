"""Smoke probe for the minimal lulynx_native TurboCore capability bridge."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_capabilities import probe_native_training_bridge  # noqa: E402
from core.turbocore_native_abi import (  # noqa: E402
    validate_native_optimizer_stateful_capability,
    validate_workspace_pipeline_native_capabilities,
)
from core.services.native_module_loader import ensure_lulynx_native_artifact_path  # noqa: E402
from core.turbocore_workspace_pipeline import build_turbocore_native_training_capability_stub  # noqa: E402


def _inject_native_artifact_dir_from_env() -> None:
    ensure_lulynx_native_artifact_path()


def run_smoke() -> dict[str, Any]:
    _inject_native_artifact_dir_from_env()
    spec = importlib.util.find_spec("lulynx_native")
    bridge = probe_native_training_bridge()
    features = bridge.get("features") if isinstance(bridge.get("features"), dict) else {}
    report = {
        "schema_version": int(bridge.get("schema_version", 1) or 1),
        "training_path_enabled": bool(bridge.get("training_path_enabled", False)),
        "training_bridge": {
            "available": bool(bridge.get("available", False)),
            "status": str(bridge.get("status", "unknown") or "unknown"),
            "reason": str(bridge.get("reason", "") or ""),
        },
        "features": dict(features),
    }
    probe = bridge.get("diagnostic")
    provider = str(((probe or {}).get("provider", "python_stub"))) if isinstance(probe, dict) else "python_stub"
    used_fallback = provider != "native_module"
    if not bool(report["training_bridge"]["available"]) or not bool(report["features"]):
        report = build_turbocore_native_training_capability_stub()
    if isinstance(probe, dict):
        report["native_probe"] = probe

    validation = validate_workspace_pipeline_native_capabilities(report)
    optimizer_validation = validate_native_optimizer_stateful_capability(report)
    native_optimizer = (report.get("features") or {}).get("native_optimizer", {}) if isinstance(report.get("features"), dict) else {}
    flat_owner = (native_optimizer or {}).get("flat_owner", {}) if isinstance(native_optimizer, dict) else {}
    flat_owner_validation = (
        (optimizer_validation.get("features") or {})
        .get("native_optimizer", {})
        .get("flat_owner", {})
        if isinstance(optimizer_validation.get("features"), dict)
        else {}
    )
    payload = {
        "schema_version": 1,
        "probe": "turbocore_native_bridge_build_smoke",
        "importable": spec is not None,
        "origin": str(getattr(spec, "origin", "")) if spec is not None else "",
        "provider": str(((report.get("native_probe") or {}).get("provider", provider))),
        "native_probe_reason": str(((report.get("native_probe") or {}).get("reason", ""))),
        "native_probe_error": str(((report.get("native_probe") or {}).get("error", ""))),
        "used_fallback_stub": used_fallback,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_abi_validation_ok": bool(validation.get("ok", False)),
        "native_optimizer_validation_ok": bool(optimizer_validation.get("ok", False)),
        "native_optimizer_stateful": bool((native_optimizer or {}).get("stateful", False)) if isinstance(native_optimizer, dict) else False,
        "flat_adamw_owner_validation_ok": bool(flat_owner_validation.get("ok", False)) if isinstance(flat_owner_validation, dict) else False,
        "flat_adamw_owner_available": bool((flat_owner or {}).get("available", False)) if isinstance(flat_owner, dict) else False,
        "flat_adamw_owner_training_enabled": bool((flat_owner or {}).get("training_path_enabled", False)) if isinstance(flat_owner, dict) else False,
        "training_bridge_status": str((report.get("training_bridge") or {}).get("status", "unknown")),
        "training_bridge_reason": str((report.get("training_bridge") or {}).get("reason", "")),
        "ok": bool(validation.get("ok", False))
        and bool(optimizer_validation.get("ok", False))
        and not bool(report.get("training_path_enabled", False)),
    }
    nvrtc_probe = (report.get("features") or {}).get("cuda_nvrtc_compile_probe", {}) if isinstance(report.get("features"), dict) else {}
    if isinstance(nvrtc_probe, dict) and nvrtc_probe:
        payload["nvrtc_compile_probe_reported"] = True
        payload["nvrtc_compile_probe_training_enabled"] = bool(nvrtc_probe.get("training_path_enabled", False))
        payload["nvrtc_compile_probe_native_kernel_present"] = bool(nvrtc_probe.get("native_kernel_present", False))
        payload["nvrtc_compile_probe_artifact_only"] = bool(nvrtc_probe.get("artifact_only", False))
        payload["ok"] = bool(
            payload["ok"]
            and not payload["nvrtc_compile_probe_training_enabled"]
            and not payload["nvrtc_compile_probe_native_kernel_present"]
            and payload["nvrtc_compile_probe_artifact_only"]
        )
    driver_probe = (report.get("features") or {}).get("cuda_driver_ptx_probe", {}) if isinstance(report.get("features"), dict) else {}
    if isinstance(driver_probe, dict) and driver_probe:
        payload["driver_ptx_probe_reported"] = True
        payload["driver_ptx_probe_training_enabled"] = bool(driver_probe.get("training_path_enabled", False))
        payload["driver_ptx_probe_native_kernel_present"] = bool(driver_probe.get("native_kernel_present", False))
        payload["driver_ptx_probe_artifact_only"] = bool(driver_probe.get("artifact_only", False))
        payload["driver_ptx_probe_kernel_executed"] = bool(driver_probe.get("kernel_executed", False))
        payload["ok"] = bool(
            payload["ok"]
            and not payload["driver_ptx_probe_training_enabled"]
            and not payload["driver_ptx_probe_native_kernel_present"]
            and payload["driver_ptx_probe_artifact_only"]
            and not payload["driver_ptx_probe_kernel_executed"]
        )
    scratch_probe = (report.get("features") or {}).get("cuda_scratch_launch_probe", {}) if isinstance(report.get("features"), dict) else {}
    if isinstance(scratch_probe, dict) and scratch_probe:
        payload["scratch_launch_probe_reported"] = True
        payload["scratch_launch_probe_training_enabled"] = bool(scratch_probe.get("training_path_enabled", False))
        payload["scratch_launch_probe_native_kernel_present"] = bool(scratch_probe.get("native_kernel_present", False))
        payload["scratch_launch_probe_scratch_only"] = bool(scratch_probe.get("scratch_buffers_only", False))
        payload["scratch_launch_probe_training_tensor_binding"] = bool(scratch_probe.get("training_tensor_binding", False))
        payload["ok"] = bool(
            payload["ok"]
            and not payload["scratch_launch_probe_training_enabled"]
            and not payload["scratch_launch_probe_native_kernel_present"]
            and payload["scratch_launch_probe_scratch_only"]
            and not payload["scratch_launch_probe_training_tensor_binding"]
        )
    runtime_probe = (report.get("features") or {}).get("cuda_adamw_runtime", {}) if isinstance(report.get("features"), dict) else {}
    if isinstance(runtime_probe, dict) and runtime_probe:
        runtime_entrypoints = runtime_probe.get("entrypoints") if isinstance(runtime_probe.get("entrypoints"), list) else []
        payload["runtime_session_probe_reported"] = True
        payload["runtime_session_benchmark_entrypoint_reported"] = "benchmark_adamw_cuda_kernel_runtime_session_py" in runtime_entrypoints
        payload["runtime_session_training_enabled"] = bool(runtime_probe.get("training_path_enabled", False))
        payload["runtime_session_native_kernel_present"] = bool(runtime_probe.get("native_kernel_present", False))
        payload["runtime_session_performance_ready"] = bool(runtime_probe.get("performance_test_ready", False))
        payload["ok"] = bool(
            payload["ok"]
            and payload["runtime_session_benchmark_entrypoint_reported"]
            and not payload["runtime_session_training_enabled"]
            and not payload["runtime_session_native_kernel_present"]
            and not payload["runtime_session_performance_ready"]
        )
    lora_probe = (report.get("features") or {}).get("lora_fused", {}) if isinstance(report.get("features"), dict) else {}
    if isinstance(lora_probe, dict) and lora_probe:
        lora_entrypoints = lora_probe.get("entrypoints") if isinstance(lora_probe.get("entrypoints"), list) else []
        lora_scratch = lora_probe.get("scratch_kernel_probe") if isinstance(lora_probe.get("scratch_kernel_probe"), dict) else {}
        payload["lora_native_abi_reported"] = True
        payload["lora_native_abi_contract_available"] = bool(lora_probe.get("abi_contract_available", False))
        payload["lora_scratch_kernel_probe_reported"] = bool(lora_scratch)
        payload["lora_scratch_kernel_probe_only"] = bool(lora_probe.get("scratch_kernel_probe_only", False))
        payload["lora_runtime_session_available"] = bool(lora_probe.get("runtime_session_available", False))
        payload["lora_scratch_kernel_training_enabled"] = bool(lora_scratch.get("training_path_enabled", False)) if isinstance(lora_scratch, dict) else False
        payload["lora_scratch_kernel_training_dispatch"] = bool(lora_scratch.get("training_dispatch", False)) if isinstance(lora_scratch, dict) else False
        payload["lora_scratch_kernel_training_tensor_binding"] = bool(lora_scratch.get("training_tensor_binding", False)) if isinstance(lora_scratch, dict) else False
        payload["lora_native_abi_entrypoints_present_in_report"] = all(
            name in lora_entrypoints
            for name in [
                "get_lora_fused_kernel_contract",
                "build_lora_fused_launch_plan",
                "validate_lora_fused_launch_plan",
                "create_lora_cuda_kernel_runtime_session_py",
                "launch_lora_delta_f32_runtime_session_py",
                "destroy_lora_cuda_kernel_runtime_session_py",
            ]
        )
        payload["lora_native_kernel_present"] = bool(lora_probe.get("native_kernel_present", False))
        payload["lora_training_dispatch"] = bool(lora_probe.get("training_dispatch", False))
        payload["lora_training_path_enabled"] = bool(lora_probe.get("training_path_enabled", False))
        payload["ok"] = bool(
            payload["ok"]
            and payload["lora_native_abi_contract_available"]
            and payload["lora_native_abi_entrypoints_present_in_report"]
            and payload["lora_scratch_kernel_probe_reported"]
            and payload["lora_runtime_session_available"]
            and not payload["lora_scratch_kernel_training_enabled"]
            and not payload["lora_scratch_kernel_training_dispatch"]
            and not payload["lora_scratch_kernel_training_tensor_binding"]
            and payload["lora_native_kernel_present"]
            and payload["lora_training_dispatch"]
            and payload["lora_training_path_enabled"]
        )
    if spec is not None and provider == "native_module":
        try:
            import lulynx_native  # type: ignore

            payload["native_optimizer_entrypoints_present"] = all(
                hasattr(lulynx_native, name)
                for name in [
                    "create_stateful_adamw_optimizer",
                    "optimizer_step",
                    "optimizer_zero_grad",
                    "optimizer_state_dict",
                    "optimizer_load_state_dict",
                    "optimizer_snapshot",
                    "destroy_optimizer",
                ]
            )
            payload["flat_adamw_owner_entrypoints_contract_reported"] = bool(
                flat_owner_validation.get("ok", False)
            ) if isinstance(flat_owner_validation, dict) else False
            payload["flat_adamw_owner_entrypoints_present"] = all(
                hasattr(lulynx_native, name)
                for name in [
                    "create_flat_adamw_owner",
                    "flat_adamw_set_grad_buffer",
                    "flat_adamw_step",
                    "flat_adamw_zero_grad",
                    "flat_adamw_state_dict",
                    "flat_adamw_load_state_dict",
                    "flat_adamw_snapshot",
                    "destroy_flat_adamw_owner",
                ]
            )
            payload["lora_native_abi_entrypoints_present"] = all(
                hasattr(lulynx_native, name)
                for name in [
                    "get_lora_fused_kernel_contract",
                    "build_lora_fused_launch_plan",
                    "validate_lora_fused_launch_plan",
                    "create_lora_cuda_kernel_runtime_session_py",
                    "lora_cuda_kernel_runtime_session_snapshot_py",
                    "launch_lora_delta_f32_runtime_session_py",
                    "destroy_lora_cuda_kernel_runtime_session_py",
                    "probe_lora_cuda_scratch_launch_py",
                    "probe_lora_cuda_scratch_matrix_py",
                ]
            )
            payload["optimizer_family_kernel_contract_entrypoint_present"] = hasattr(
                lulynx_native,
                "get_optimizer_family_kernel_contracts",
            )
            family_contracts = (
                lulynx_native.get_optimizer_family_kernel_contracts()
                if payload["optimizer_family_kernel_contract_entrypoint_present"]
                else {}
            )
            payload["optimizer_family_kernel_contract_count"] = (
                int(family_contracts.get("optimizer_family_contract_count", 0) or 0)
                if isinstance(family_contracts, dict)
                else 0
            )
            payload["optimizer_family_kernel_contract_training_enabled"] = (
                bool(family_contracts.get("training_path_enabled", False))
                if isinstance(family_contracts, dict)
                else True
            )
            payload["optimizer_family_kernel_contract_native_dispatch_allowed"] = (
                bool(family_contracts.get("native_dispatch_allowed", False))
                if isinstance(family_contracts, dict)
                else True
            )
            create_result = lulynx_native.create_stateful_adamw_optimizer("{}")
            payload["native_optimizer_training_disabled"] = (
                isinstance(create_result, dict)
                and create_result.get("training_path_enabled") is False
                and create_result.get("native_kernel_present") is False
            )
            payload["native_optimizer_reference_microkernel"] = (
                isinstance(create_result, dict)
                and (create_result.get("reference_microkernel") is True or create_result.get("ok") is False)
            )
            payload["ok"] = bool(
                payload["ok"]
                and payload["native_optimizer_entrypoints_present"]
                and payload["flat_adamw_owner_entrypoints_contract_reported"]
                and payload["flat_adamw_owner_entrypoints_present"]
                and payload["lora_native_abi_entrypoints_present"]
                and payload["optimizer_family_kernel_contract_entrypoint_present"]
                and payload["optimizer_family_kernel_contract_count"] == 10
                and not payload["optimizer_family_kernel_contract_training_enabled"]
                and not payload["optimizer_family_kernel_contract_native_dispatch_allowed"]
                and payload["native_optimizer_training_disabled"]
                and payload["native_optimizer_reference_microkernel"]
            )
        except Exception as exc:
            payload["native_optimizer_entrypoints_present"] = False
            payload["flat_adamw_owner_entrypoints_contract_reported"] = False
            payload["flat_adamw_owner_entrypoints_present"] = False
            payload["lora_native_abi_entrypoints_present"] = False
            payload["optimizer_family_kernel_contract_entrypoint_present"] = False
            payload["optimizer_family_kernel_contract_count"] = 0
            payload["optimizer_family_kernel_contract_training_enabled"] = False
            payload["optimizer_family_kernel_contract_native_dispatch_allowed"] = False
            payload["native_optimizer_training_disabled"] = False
            payload["native_optimizer_reference_microkernel"] = False
            payload["native_optimizer_stub_error"] = f"{type(exc).__name__}: {exc}"
            payload["ok"] = False
    return payload


if __name__ == "__main__":
    result = run_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not bool(result.get("ok", False)):
        raise SystemExit(1)
