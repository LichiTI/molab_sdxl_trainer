"""Smoke checks for TurboCore LoRA fused promotion scorecard."""

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

from core.lulynx_trainer.turbocore_lora_promotion_scorecard import build_lora_fused_promotion_scorecard  # noqa: E402


def _promotion_matrix() -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark": "turbocore_lora_candidate_matrix",
        "summary": {
            "quality": {"evidence_level": "benchmark", "smoke_only": False, "warnings": []},
            "ready_for_training_activation": False,
            "candidate_summaries": [
                {
                    "candidate": "triton_lora_delta_v3_dispatch",
                    "ok": True,
                    "case_count": 5,
                    "win_count": 5,
                    "loss_count": 0,
                    "avg_speedup_vs_reference": 1.12,
                    "best_speedup_vs_reference": 1.2,
                    "worst_speedup_vs_reference": 1.06,
                }
            ],
        },
    }


def _candidate_scorecard() -> dict[str, object]:
    return {
        "schema_version": 1,
        "prototype": "turbocore_candidate_scorecard",
        "summary": {
            "ready_for_training_activation": False,
            "gate_counts": {"discovery_only": 1},
            "training_activation_blockers": ["TurboCore candidates are research-only and are not wired to training activation."],
        },
        "rows": [
            {
                "feature": "lora_fused",
                "candidate": {
                    "name": "rust_cuda_lora_delta_v0",
                    "native": True,
                    "available": False,
                },
                "gate": "discovery_only",
            }
        ],
    }


def _assert_closed(report: dict[str, object]) -> None:
    assert report["promotion_ready"] is False, report
    assert report["training_dispatch"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["returns_training_tensor_payloads"] is False, report
    assert report["pytorch_lora_path_authoritative"] is True, report
    assert report["fallback_to_pytorch_lora"] is True, report
    assert report["fallback_to_existing_training_path"] is True, report


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
        "contract": {"contract": "lora_delta_add_cuda_kernel_v0"},
        "launch_plan": {
            "plan_kind": "lora_delta_add_launch_plan_v0",
            "shape_contract_ok": True,
            "native_kernel_present": True,
            "training_path_enabled": True,
            "launch_allowed": True,
        },
        "launch_plan_validation": {"ok": True},
        "blocked_reasons": [],
    }


def _scratch_report(*, ok: bool = True) -> dict[str, object]:
    blocked = [] if ok else ["nvrtc_compile_not_ready"]
    return {
        "schema_version": 1,
        "report": "turbocore_lora_cuda_scratch_kernel_probe_v0",
        "ok": ok,
        "present": True,
        "entrypoint_present": True,
        "scratch_kernel_probe_available": True,
        "scratch_kernel_present": ok,
        "native_kernel_present": ok,
        "kernel_executed": ok,
        "case_count": 4 if ok else 0,
        "passed_case_count": 4 if ok else 0,
        "rank_count": 4 if ok else 0,
        "native_candidate_repeated_validation_seen": ok,
        "scratch_matrix_representative": False,
        "parity_ok": ok,
        "max_abs_diff": 0.0 if ok else None,
        "scratch_buffers_only": True,
        "training_tensor_binding": False,
        "parameters_mutated": False,
        "training_parameters_mutated": False,
        "native_dispatch_allowed": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "performance_test_ready": False,
        "blocked_reasons": blocked,
    }


def _native_training_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "report": "turbocore_lora_native_training_dispatch_probe_v0",
        "ok": True,
        "candidate": "rust_cuda_lora_delta_v0",
        "native_kernel_present": True,
        "kernel_executed": True,
        "kernel_launch_count": 2,
        "output_mutated": True,
        "training_tensor_binding": True,
        "training_dispatch": True,
        "training_path_enabled": True,
        "autograd_binding": True,
        "forward_backward_training_integration": True,
        "forward_parity_ok": True,
        "backward_parity_ok": True,
        "max_abs_forward_diff": 1.0e-5,
        "max_abs_grad_diff": 1.0e-7,
        "stream_lifetime_bound": True,
        "runtime_recovery_ready": True,
        "fallback_to_pytorch_lora": False,
        "pytorch_lora_path_authoritative": False,
        "dtype": "float32",
        "device": "cuda",
        "blocked_reasons": [],
    }


def test_scorecard_blocks_without_benchmark_matrix() -> None:
    report = build_lora_fused_promotion_scorecard(
        candidate_scorecard=None,
        benchmark_matrix=None,
        dtype=torch.float16,
    )
    reasons = set(report["blocked_reasons"])
    blockers = set(report["promotion_blockers"])
    assert report["scorecard"] == "turbocore_lora_fused_promotion_scorecard_v0", report
    assert report["ok"] is True, report
    _assert_closed(report)
    assert report["registry"]["target_registered"] is True, report
    assert report["registry"]["target_available"] is False, report
    assert report["registry"]["target_abi_contract_available"] is True, report
    assert report["native_abi"]["abi_contract_available"] is True, report
    assert report["native_abi"]["native_kernel_present"] is True, report
    assert report["native_scratch_kernel"]["present"] is False, report
    assert "lora_cuda_scratch_kernel_probe_missing" in reasons, report
    assert "rust_cuda_lora_native_abi_not_available" not in reasons, report
    assert "rust_cuda_lora_abi_contract_not_available" not in reasons, report
    assert "native_lora_kernel_not_registered" not in reasons, report
    assert "representative_lora_benchmark_matrix_missing" in reasons, report
    assert "lora_candidate_scorecard_missing" in reasons, report
    assert "rust_cuda_lora_abi_not_promoted" not in blockers, report
    assert "native_lora_training_kernel_not_promoted" in blockers, report
    assert "lora_fused_training_dispatch_not_implemented" in blockers, report


def test_scorecard_keeps_dispatch_closed_with_positive_research_matrix() -> None:
    report = build_lora_fused_promotion_scorecard(
        candidate_scorecard=_candidate_scorecard(),
        benchmark_matrix=_promotion_matrix(),
        native_abi_report=_native_abi_report(),
        native_scratch_report=_scratch_report(ok=True),
        dtype=torch.float16,
        presets=["tiny"],
        ranks=[4],
    )
    reasons = set(report["blocked_reasons"])
    _assert_closed(report)
    assert report["benchmark_matrix"]["evidence_level"] == "benchmark", report
    assert report["benchmark_matrix"]["avg_speedup_vs_reference"] == 1.12, report
    assert report["benchmark_matrix"]["loss_count"] == 0, report
    assert "representative_lora_benchmark_matrix_missing" not in reasons, report
    assert "representative_lora_speedup_below_threshold" not in reasons, report
    assert "rust_cuda_lora_native_abi_not_available" not in reasons, report
    assert "rust_cuda_lora_abi_contract_not_available" not in reasons, report
    assert report["registry"]["target_abi_contract_available"] is True, report
    assert report["native_abi"]["abi_contract_available"] is True, report
    assert report["native_abi"]["native_kernel_present"] is True, report
    assert report["native_scratch_kernel"]["scratch_kernel_present"] is True, report
    assert report["native_scratch_kernel"]["kernel_executed"] is True, report
    assert report["native_scratch_kernel"]["training_path_enabled"] is False, report
    assert report["forward_preflight"]["present"] is True, report
    assert report["forward_preflight"]["would_allow_native_forward"] is False, report
    assert report["forward_preflight"]["training_path_enabled"] is False, report
    assert "native_lora_kernel_not_registered" not in reasons, report
    assert "native_lora_training_kernel_not_promoted" in reasons, report
    assert "lora_scratch_kernel_not_representative" in reasons, report
    assert "native_lora_candidate_has_not_passed_repeated_validation" in reasons, report
    assert report["shape_policy"]["case_count"] > 0, report
    assert report["shape_policy"]["route_counts"], report


def test_scorecard_promotes_with_native_training_dispatch_evidence() -> None:
    report = build_lora_fused_promotion_scorecard(
        candidate_scorecard=_candidate_scorecard(),
        benchmark_matrix=_promotion_matrix(),
        native_abi_report=_native_abi_report(),
        native_scratch_report=_scratch_report(ok=True),
        native_training_report=_native_training_report(),
        dtype=torch.float32,
        presets=["tiny"],
        ranks=[4],
    )
    assert report["promotion_ready"] is True, report
    assert report["promotion_blockers"] == [], report
    assert report["blocked_reasons"] == [], report
    assert report["native_kernel_present"] is True, report
    assert report["native_dispatch_allowed"] is True, report
    assert report["training_dispatch"] is True, report
    assert report["training_path_enabled"] is True, report
    assert report["returns_training_tensor_payloads"] is True, report
    assert report["pytorch_lora_path_authoritative"] is False, report
    assert report["fallback_to_pytorch_lora"] is False, report
    assert report["native_training_dispatch"]["autograd_binding"] is True, report
    assert report["forward_preflight"]["native_dispatch_allowed"] is True, report


def main() -> int:
    test_scorecard_blocks_without_benchmark_matrix()
    test_scorecard_keeps_dispatch_closed_with_positive_research_matrix()
    test_scorecard_promotes_with_native_training_dispatch_evidence()
    print("turbocore_lora_promotion_scorecard_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
