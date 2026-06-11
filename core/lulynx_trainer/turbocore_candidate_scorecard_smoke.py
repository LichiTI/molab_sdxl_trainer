"""Smoke test for the TurboCore candidate scorecard prototype."""

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

from core.lulynx_trainer.turbocore_candidate_scorecard import build_candidate_scorecard  # noqa: E402
from core.lulynx_trainer.turbocore_readiness_probe import build_readiness_report  # noqa: E402


def test_scorecard() -> None:
    payload = build_candidate_scorecard(
        device=torch.device("cpu"),
        dtype=torch.float32,
        preset="tiny",
        rank=4,
        iters=1,
        warmup=0,
    )
    assert payload["prototype"] == "turbocore_candidate_scorecard"
    assert payload["summary"]["candidate_count"] >= 4
    assert payload["summary"]["available_candidate_count"] >= 2
    assert payload["summary"]["ready_for_training_activation"] is False
    assert payload["summary"]["training_activation_blockers"]
    assert "experimental_probe_skipped" in payload["summary"]["gate_explanations"]
    gates = payload["summary"]["gate_counts"]
    assert gates.get("reference_baseline", 0) >= 2
    assert gates.get("discovery_only", 0) >= 2
    assert gates.get("device_incompatible", 0) >= 0
    assert gates.get("experimental_probe_skipped", 0) >= 1
    for row in payload["rows"]:
        assert row.get("gate_explanation")


def test_scorecard_shape_policy_skips_dit_v1() -> None:
    payload = build_candidate_scorecard(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float32,
        preset="dit_short",
        rank=4,
        iters=1,
        warmup=0,
        shape_policy="auto",
    )
    assert payload["shape_policy"] == "auto"
    gates = payload["summary"]["gate_counts"]
    if torch.cuda.is_available():
        assert gates.get("shape_policy_skipped", 0) >= 1


def test_readiness_scorecard_section() -> None:
    payload = build_readiness_report(
        device=torch.device("cpu"),
        dtype=torch.float32,
        preset="tiny",
        rank=4,
        iters=1,
        warmup=0,
        include_torch_compile=False,
    )
    assert payload["sections"]["candidate_scorecard"]["ok"] is True
    assert payload["sections"]["lora_promotion_scorecard"]["ok"] is True
    assert payload["sections"]["lora_promotion_scorecard"]["promotion_ready"] is False
    assert payload["sections"]["native_update_promotion_scorecard"]["ok"] is True
    assert payload["sections"]["native_update_promotion_scorecard"]["promotion_ready"] is False
    assert payload["sections"]["workspace_data_pipeline_prototype"]["ok"] is True
    assert payload["sections"]["workspace_data_pipeline_prototype"]["training_path_enabled"] is False
    assert payload["sections"]["workspace_data_pipeline_lifecycle"]["ok"] is True
    assert payload["sections"]["workspace_data_pipeline_lifecycle"]["training_path_enabled"] is False
    assert payload["sections"]["native_capability_stub"]["ok"] is True
    assert payload["sections"]["native_capability_stub"]["training_path_enabled"] is False
    assert payload["sections"]["native_abi_validation"]["ok"] is True
    assert payload["summary"]["workspace_data_pipeline_prototype"] == "python_abi_prototype"
    assert payload["summary"]["workspace_data_pipeline_lifecycle_ok"] is True
    assert payload["summary"]["native_stub_schema_complete"] is True
    assert payload["summary"]["native_training_path_locked"] is True
    assert payload["summary"]["native_stub_gate"] == "stub_complete_training_locked"
    assert payload["summary"]["candidate_gate_counts"]
    assert payload["summary"]["lora_promotion_ready"] is False
    assert payload["summary"]["lora_native_abi_contract_available"] is True
    assert payload["summary"]["lora_native_kernel_present"] is True
    assert payload["summary"]["native_update_promotion_ready"] is False
    assert "native_lora_training_kernel_not_promoted" in payload["summary"]["lora_promotion_blockers"]
    native_update_scorecard = payload["sections"]["native_update_promotion_scorecard"]
    assert native_update_scorecard["primary_promotion_blockers"], native_update_scorecard
    assert native_update_scorecard["derived_promotion_blockers"], native_update_scorecard
    assert "owner_gradient_sync_default_off" in native_update_scorecard["primary_promotion_blockers"], native_update_scorecard
    assert "native_dispatch_runtime_not_implemented" in native_update_scorecard["derived_promotion_blockers"], native_update_scorecard
    assert payload["summary"]["native_update_promotion_blockers"] == native_update_scorecard["primary_promotion_blockers"][:8]


def test_readiness_scorecard_accepts_native_update_performance_report() -> None:
    payload = build_readiness_report(
        device=torch.device("cpu"),
        dtype=torch.float32,
        preset="tiny",
        rank=4,
        iters=1,
        warmup=0,
        include_torch_compile=False,
        native_update_performance_report={
            "native_update_performance_report": {
                "performance_gate": _native_update_performance_gate(),
            }
        },
    )
    summary = payload["summary"]
    scorecard = payload["sections"]["native_update_promotion_scorecard"]
    performance_gate = scorecard["performance_gate"]
    assert summary["native_update_representative_performance_ready"] is True
    assert summary["native_update_performance_blockers"] == []
    assert performance_gate["representative_performance_gate_ready"] is True
    assert "representative_performance_gate_missing" not in scorecard["primary_promotion_blockers"]
    assert "optimizer_microbenchmark_missing" not in scorecard["primary_promotion_blockers"]
    assert "representative_training_matrix_missing" not in scorecard["primary_promotion_blockers"]
    assert summary["ready_for_ui"] is False
    assert summary["native_training_path_locked"] is True


def _native_update_matrix_report() -> dict:
    return {
        "schema_version": 1,
        "matrix": "turbocore_update_benchmark_matrix_v0",
        "run": True,
        "cases": [
            {
                "case": {"name": "baseline_phase"},
                "returncode": 0,
                "summary": {
                    "success": True,
                    "steps_completed": 20,
                    "steady_mean_step_ms": 640.0,
                },
            },
            {
                "case": {"name": "native_update_dispatch_promotion_perf"},
                "returncode": 0,
                "summary": {
                    "success": True,
                    "steps_completed": 20,
                    "steady_mean_step_ms": 600.0,
                    "native_dispatch_executed": True,
                },
            },
        ],
        "summary": {
            "executed_count": 2,
            "all_success": True,
            "mean_step_ms_by_case": {
                "baseline_phase": 640.0,
                "native_update_dispatch_promotion_perf": 600.0,
            },
        },
        "optimizer_performance_gate": {
            "gate": "turbocore_optimizer_performance_gate",
            "ok": True,
            "promotion_gate_ok": True,
            "runtime_dispatch_allowed": False,
            "evidence_quality": "promotion_benchmark",
            "best_candidate": {
                "optimizer": "triton_adamw_flat_v0",
                "speedup_vs_baseline": 6.5,
            },
        },
    }


def _native_update_performance_gate() -> dict:
    from core.turbocore_native_update_performance import build_native_update_performance_gate

    return build_native_update_performance_gate(
        shadow_report={
            "owner_native_launch_probe": {
                "ok": True,
                "attempted": True,
                "kernel_executed": True,
                "parity_ok": True,
                "persistent_owner_mutated": False,
                "owner_numel": 2171392,
            }
        },
        performance_report=_native_update_matrix_report(),
    )


if __name__ == "__main__":
    test_scorecard()
    test_scorecard_shape_policy_skips_dit_v1()
    test_readiness_scorecard_section()
    test_readiness_scorecard_accepts_native_update_performance_report()
    print("turbocore_candidate_scorecard_smoke: ok")
