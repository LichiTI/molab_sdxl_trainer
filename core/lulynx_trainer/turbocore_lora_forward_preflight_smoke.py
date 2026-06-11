"""Smoke checks for the report-only LoRA forward dispatch preflight."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_lora_forward_preflight import build_lora_forward_dispatch_preflight  # noqa: E402


def _scratch_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "report": "turbocore_lora_cuda_scratch_kernel_probe_v0",
        "ok": True,
        "scratch_kernel_present": True,
        "native_kernel_present": True,
        "case_count": 4,
        "passed_case_count": 4,
        "rank_count": 4,
        "max_abs_diff": 0.0,
        "native_candidate_repeated_validation_seen": True,
        "scratch_matrix_representative": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "training_tensor_binding": False,
        "blocked_reasons": [],
    }


def _native_abi_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "report": "turbocore_lora_native_abi_probe_v0",
        "ok": True,
        "abi_contract_available": True,
        "candidate": "rust_cuda_lora_delta_v0",
        "native_kernel_present": True,
        "native_dispatch_allowed": True,
        "training_dispatch": True,
        "training_path_enabled": True,
        "launch_plan": {
            "plan_kind": "lora_delta_add_launch_plan_v0",
            "shape_contract_ok": True,
            "training_path_enabled": True,
        },
        "blocked_reasons": [],
    }


def _native_training_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "report": "turbocore_lora_native_training_dispatch_probe_v0",
        "ok": True,
        "native_kernel_present": True,
        "kernel_executed": True,
        "output_mutated": True,
        "training_tensor_binding": True,
        "training_dispatch": True,
        "training_path_enabled": True,
        "autograd_binding": True,
        "forward_backward_training_integration": True,
        "forward_parity_ok": True,
        "backward_parity_ok": True,
        "stream_lifetime_bound": True,
        "runtime_recovery_ready": True,
        "fallback_to_pytorch_lora": False,
        "blocked_reasons": [],
    }


def test_preflight_stays_closed_without_training_integration() -> None:
    report = build_lora_forward_dispatch_preflight(
        x_shape=(2, 64, 320),
        dtype="float32",
        rank=4,
        native_abi_report=_native_abi_report(),
        native_scratch_report=_scratch_report(),
        request_training_dispatch=True,
        allow_experimental_native=True,
    )
    assert report["ok"] is True, report
    assert report["shape_contract_ok"] is True, report
    assert report["abi_contract_available"] is True, report
    assert report["native_validation_ok"] is True, report
    assert report["native_candidate_repeated_validation_seen"] is True, report
    assert report["would_allow_native_forward"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_dispatch"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["fallback_to_pytorch_lora"] is True, report
    assert "lora_forward_backward_training_integration_missing" in report["blocked_reasons"], report
    assert "lora_forward_scratch_kernel_not_representative" in report["blocked_reasons"], report


def test_preflight_allows_native_training_dispatch_with_runtime_evidence() -> None:
    report = build_lora_forward_dispatch_preflight(
        x_shape=(2, 64, 320),
        dtype="float32",
        rank=4,
        native_abi_report=_native_abi_report(),
        native_scratch_report=_scratch_report(),
        native_training_report=_native_training_report(),
        request_training_dispatch=True,
        allow_experimental_native=True,
    )
    assert report["ok"] is True, report
    assert report["shape_contract_ok"] is True, report
    assert report["would_allow_native_forward"] is True, report
    assert report["native_dispatch_allowed"] is True, report
    assert report["training_dispatch"] is True, report
    assert report["training_path_enabled"] is True, report
    assert report["fallback_to_pytorch_lora"] is False, report
    assert report["promotion_blockers"] == [], report
    assert report["blocked_reasons"] == [], report


def test_preflight_reports_shape_blockers() -> None:
    report = build_lora_forward_dispatch_preflight(
        x_shape=(2, 64, 320),
        dtype="float16",
        rank=3,
        native_abi_report=_native_abi_report(),
        native_scratch_report={},
    )
    assert report["shape_contract_ok"] is False, report
    assert "lora_forward_unsupported_rank" in report["blocked_reasons"], report
    assert "lora_forward_unsupported_dtype" in report["blocked_reasons"], report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report


def main() -> int:
    test_preflight_stays_closed_without_training_integration()
    test_preflight_allows_native_training_dispatch_with_runtime_evidence()
    test_preflight_reports_shape_blockers()
    print("turbocore_lora_forward_preflight_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
