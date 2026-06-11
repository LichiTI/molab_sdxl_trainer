"""Smoke tests for TurboCore evidence and benchmark scaffolding."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_capabilities import build_turbocore_capability_report  # noqa: E402
from core.turbocore_evidence import build_turbocore_validation_status  # noqa: E402
from core.lulynx_trainer.turbocore_lora_fused_benchmark import run_benchmark  # noqa: E402
from core.lulynx_trainer.turbocore_native_optimizer_benchmark import (  # noqa: E402
    run_benchmark as run_optimizer_benchmark,
)


def test_capability_report_includes_evidence_and_gates() -> None:
    report = build_turbocore_capability_report(
        {
            "model_type": "anima",
            "training_type": "lora",
            "execution_core": "turbo",
            "turbocore_features": ["lora_fused", "workspace_pool", "data_pipeline"],
            "turbocore_workspace_mb": 256,
            "turbocore_prefetch_depth": 3,
            "enable_mixed_resolution_training": True,
            "pcie_transfer_format": "fp8_e4m3",
        },
        resolution={
            "requested_execution_core": "turbo",
            "effective_execution_core": "standard",
            "turbocore_features_requested": ["lora_fused"],
            "turbocore_features_active": [],
            "turbocore_features_disabled": [{"feature": "lora_fused", "reason": "turbocore_not_implemented"}],
        },
    )
    assert "evidence" in report
    assert "validation_status" in report
    assert "native_abi_validation" in report
    assert report["native_abi_validation"]["training_path_enabled"] is False
    assert report["evidence"]["route"]["anima_staged_resolution_enabled"] is True
    assert report["validation_status"]["summary"]["ready_for_ui"] is False
    assert report["evidence"]["candidate_registry"]["available"] is True
    assert "rust_cuda_lora_delta_v0" in report["evidence"]["candidate_registry"]["reserved_native_candidates"]
    assert report["evidence"]["workspace_pool"]["requested_mb"] == 256
    assert report["evidence"]["workspace_pool"]["status"] == "prototype"
    assert report["evidence"]["native_data_pipeline"]["status"] == "prototype"
    assert report["evidence"]["native_data_pipeline"]["prototype"]["prefetch_depth"] == 3
    assert report["evidence"]["native_data_pipeline"]["prototype"]["training_path_enabled"] is False
    assert report["evidence"]["workspace_pool"]["native_capability_schema"]["available"] is False
    assert report["evidence"]["native_data_pipeline"]["native_capability_schema"]["available"] is False


def test_validation_status_stays_developer_only() -> None:
    status = build_turbocore_validation_status({"model_type": "sdxl", "training_type": "lora"})
    gates = {gate["gate"]: gate["status"] for gate in status["gates"]}
    assert gates["product_readiness"] == "blocked"
    assert status["summary"]["ready_for_ui"] is False


def test_lora_benchmark_tiny_cpu_path() -> None:
    payload = run_benchmark(
        preset="tiny",
        ranks=[4],
        dtype=torch.float32,
        device=torch.device("cpu"),
        iters=1,
        warmup=0,
    )
    assert payload["benchmark"] == "turbocore_lora_fused_delta_reference"
    assert payload["summary"]["native_kernel_present"] is False
    assert payload["results"][0]["max_abs_error"] <= 1e-6


def test_native_optimizer_benchmark_tiny_cpu_path() -> None:
    payload = run_optimizer_benchmark(
        preset="tiny",
        ranks=[4],
        dtype=torch.float32,
        device=torch.device("cpu"),
        iters=1,
        warmup=0,
    )
    assert payload["benchmark"] == "turbocore_native_optimizer_reference"
    assert payload["summary"]["native_kernel_present"] is False
    assert payload["results"][0]["parameter_tensors"] == 16
    assert payload["results"][0]["adamw_step_ms"] >= 0.0


if __name__ == "__main__":
    test_capability_report_includes_evidence_and_gates()
    test_validation_status_stays_developer_only()
    test_lora_benchmark_tiny_cpu_path()
    test_native_optimizer_benchmark_tiny_cpu_path()
    print("turbocore_evidence_smoke: ok")
