"""Smoke checks for native-runtime profile performance report conversion."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_native_update_performance_report import (  # noqa: E402
    build_native_update_benchmark_matrix_from_profile_summary,
    build_native_update_profile_performance_report,
)


def _profile_summary(*, steps: int = 2) -> dict[str, object]:
    return {
        "benchmark": {"family": "anima"},
        "runs": {
            "standard": {
                "profile": "standard",
                "success": True,
                "steps_completed": steps,
                "mean_step_ms": 1000.0,
                "steady_mean_step_ms": 980.0,
                "peak_vram_mb": 4096.0,
                "optimizer_type": "AdamW",
                "update_shadow_reports": [
                    {
                        "owner_native_launch_probe": {
                            "ok": True,
                            "attempted": True,
                            "kernel_executed": True,
                            "parity_ok": True,
                            "persistent_owner_mutated": False,
                            "owner_numel": 4096,
                        }
                    }
                ],
                "native_update_dispatch_runtime_reports": [
                    {"requested": True, "native_step_executed": False}
                ],
                "native_update_gate_reports": [
                    {"blocked_reasons": ["native_dispatch_runtime_not_implemented"]}
                ],
            }
        },
    }


def _profile_summary_with_native_case(*, steps: int = 2) -> dict[str, object]:
    summary = _profile_summary(steps=steps)
    runs = summary["runs"]
    assert isinstance(runs, dict)
    runs["native_update_dispatch"] = {
        "profile": "native_update_dispatch",
        "success": True,
        "steps_completed": steps,
        "mean_step_ms": 900.0,
        "steady_mean_step_ms": 880.0,
        "peak_vram_mb": 4096.0,
        "optimizer_type": "AdamW",
        "native_update_dispatch_runtime_reports": [
            {"requested": True, "native_step_executed": True}
        ],
    }
    return summary


def test_profile_summary_becomes_baseline_matrix() -> None:
    matrix = build_native_update_benchmark_matrix_from_profile_summary(_profile_summary())
    assert matrix["matrix"] == "turbocore_update_benchmark_matrix_v0", matrix
    assert matrix["summary"]["case_count"] == 1, matrix
    assert matrix["cases"][0]["case"]["name"] == "baseline_phase", matrix
    assert matrix["summary"]["native_dispatch_case_present"] is False, matrix


def test_short_profile_report_blocks_representative_promotion() -> None:
    report = build_native_update_profile_performance_report(_profile_summary(steps=2))
    reasons = set(report["blocked_reasons"])
    assert report["report"] == "turbocore_native_update_profile_performance_report_v0", report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_allowed"] is False, report
    assert "native_dispatch_benchmark_case_missing" in reasons, report
    assert "representative_training_steps_missing" in reasons, report
    assert "optimizer_microbenchmark_missing" in reasons, report


def test_short_native_case_blocks_on_step_count() -> None:
    report = build_native_update_profile_performance_report(_profile_summary_with_native_case(steps=2))
    reasons = set(report["blocked_reasons"])
    assert "native_dispatch_benchmark_case_missing" not in reasons, report
    assert "representative_training_steps_too_low" in reasons, report
    assert report["benchmark_matrix"]["summary"]["native_dispatch_case_present"] is True, report
    assert report["benchmark_matrix"]["summary"]["native_dispatch_executed"] is True, report


def main() -> int:
    test_profile_summary_becomes_baseline_matrix()
    test_short_profile_report_blocks_representative_promotion()
    test_short_native_case_blocks_on_step_count()
    print("turbocore_native_update_performance_report_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
