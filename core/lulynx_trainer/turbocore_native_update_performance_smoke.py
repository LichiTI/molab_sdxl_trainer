"""Smoke checks for TurboCore native update representative performance gate."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_native_update_performance import build_native_update_performance_gate  # noqa: E402


def _owner_probe() -> dict[str, object]:
    return {
        "ok": True,
        "attempted": True,
        "kernel_executed": True,
        "parity_ok": True,
        "persistent_owner_mutated": False,
        "owner_numel": 4096,
        "elapsed_ms": 4.0,
    }


def _optimizer_gate() -> dict[str, object]:
    return {
        "gate": "turbocore_optimizer_performance_gate",
        "ok": True,
        "promotion_gate_ok": True,
        "runtime_dispatch_allowed": False,
        "evidence_quality": "promotion_benchmark",
        "best_candidate": {
            "optimizer": "turbocore_cuda_adamw_v0",
            "speedup_vs_baseline": 1.35,
        },
    }


def _matrix(*, native_ms: float = 970.0, steps: int = 24, native_executed: bool = True) -> dict[str, object]:
    return {
        "matrix": "turbocore_update_benchmark_matrix_v0",
        "run": True,
        "summary": {
            "executed_count": 2,
            "all_success": True,
            "mean_step_ms_by_case": {
                "baseline_phase": 1000.0,
                "native_update_dispatch": native_ms,
            },
        },
        "cases": [
            {"case": {"name": "baseline_phase"}, "summary": {"steps_completed": steps, "mean_step_ms": 1000.0}},
            {
                "case": {"name": "native_update_dispatch"},
                "summary": {
                    "steps_completed": steps,
                    "mean_step_ms": native_ms,
                    "native_dispatch_executed": native_executed,
                },
            },
        ],
    }


def test_microbench_without_training_matrix_blocks() -> None:
    report = build_native_update_performance_gate(
        shadow_report={"owner_native_launch_probe": _owner_probe(), "optimizer_performance_gate": _optimizer_gate()}
    )
    reasons = set(report["blocked_reasons"])
    assert report["performance_test_ready"] is False, report
    assert report["evidence"]["owner_native_kernel"]["ok"] is True, report
    assert "representative_training_matrix_missing" in reasons, report


def test_short_training_matrix_blocks_promotion() -> None:
    report = build_native_update_performance_gate(
        shadow_report={
            "owner_native_launch_probe": _owner_probe(),
            "optimizer_performance_gate": _optimizer_gate(),
            "benchmark_matrix": _matrix(steps=2),
        }
    )
    assert report["performance_test_ready"] is False, report
    assert "representative_training_steps_too_low" in set(report["blocked_reasons"]), report


def test_named_native_case_must_execute_native_dispatch() -> None:
    report = build_native_update_performance_gate(
        shadow_report={
            "owner_native_launch_probe": _owner_probe(),
            "optimizer_performance_gate": _optimizer_gate(),
            "benchmark_matrix": _matrix(native_ms=960.0, steps=24, native_executed=False),
        }
    )
    assert report["performance_test_ready"] is False, report
    assert "native_dispatch_not_executed_in_benchmark_case" in set(report["blocked_reasons"]), report


def test_representative_matrix_can_pass_report_only() -> None:
    report = build_native_update_performance_gate(
        shadow_report={
            "owner_native_launch_probe": _owner_probe(),
            "optimizer_performance_gate": _optimizer_gate(),
            "benchmark_matrix": _matrix(native_ms=960.0, steps=24),
        }
    )
    assert report["performance_test_ready"] is True, report
    assert report["representative_performance_gate_ready"] is True, report
    assert report["runtime_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["evidence"]["training_matrix"]["end_to_end_speedup"] >= 1.03, report


def main() -> int:
    test_microbench_without_training_matrix_blocks()
    test_short_training_matrix_blocks_promotion()
    test_named_native_case_must_execute_native_dispatch()
    test_representative_matrix_can_pass_report_only()
    print("turbocore_native_update_performance_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
