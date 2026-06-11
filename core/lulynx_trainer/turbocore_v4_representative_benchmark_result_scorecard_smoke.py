"""Smoke checks for V4 representative benchmark result ingestion."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v4_representative_benchmark_result_scorecard import (  # noqa: E402
    build_v4_representative_benchmark_result_scorecard,
)


def run_smoke() -> dict[str, Any]:
    accepted = build_v4_representative_benchmark_result_scorecard(
        matrix_payload=_representative_payload(),
        p0_audit={"milestone_completed": True},
        require_full_promotion_gate=True,
    )
    assert accepted["ok"] is True, accepted
    assert accepted["promotion_performance_gate_ready"] is True, accepted
    gates = accepted["progress_gates"]
    assert all(gates.values()), accepted
    assert accepted["default_training_path_enabled"] is False, accepted
    assert accepted["default_rollout_allowed"] is False, accepted
    assert accepted["auto_rollout_allowed"] is False, accepted

    perf_blocked = build_v4_representative_benchmark_result_scorecard(
        matrix_payload=_performance_blocked_payload(),
        p0_audit={"milestone_completed": True},
        require_full_promotion_gate=True,
    )
    assert perf_blocked["ok"] is False, perf_blocked
    assert perf_blocked["benchmark_result_ready"] is True, perf_blocked
    assert perf_blocked["real_benchmark_input_present"] is True, perf_blocked
    assert perf_blocked["real_benchmark_executed"] is True, perf_blocked
    assert perf_blocked["real_benchmark_contract_ready"] is True, perf_blocked
    assert perf_blocked["real_benchmark_status"] == "performance_gate_blocked", perf_blocked
    assert "end_to_end_speedup_below_threshold" in perf_blocked["real_benchmark_performance_blockers"], perf_blocked

    pending = build_v4_representative_benchmark_result_scorecard(
        matrix_payload=_dry_run_payload(),
        p0_audit={"milestone_completed": True},
    )
    assert pending["ok"] is False, pending
    assert pending["benchmark_result_ready"] is False, pending
    assert "v4_p1_matrix_executed_missing" in pending["blocked_reasons"], pending

    missing = build_v4_representative_benchmark_result_scorecard(
        p0_audit={"milestone_completed": True},
    )
    assert missing["ok"] is False, missing
    assert "v4_p1_result_input_missing" in missing["blocked_reasons"], missing

    with tempfile.TemporaryDirectory() as tmp:
        summary_path = Path(tmp) / "matrix_summary.json"
        summary_path.write_text(json.dumps(_representative_payload(), ensure_ascii=False), encoding="utf-8")
        loaded = build_v4_representative_benchmark_result_scorecard(
            matrix_summary_path=summary_path,
            p0_audit={"milestone_completed": True},
        )
    assert loaded["ok"] is True, loaded
    assert loaded["source"] == "matrix_summary_path", loaded

    return {
        "schema_version": 1,
        "probe": "turbocore_v4_representative_benchmark_result_scorecard_smoke",
        "ok": True,
        "accepted_progress_gates": gates,
        "perf_blocked_status": perf_blocked["real_benchmark_status"],
        "pending_blocked_reasons": pending["blocked_reasons"],
        "missing_blocked_reasons": missing["blocked_reasons"],
        "recommended_next_step": accepted["recommended_next_step"],
    }


def _representative_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "matrix": "turbocore_update_benchmark_matrix_v0",
        "run": True,
        "family": "anima",
        "cases": [
            {
                "case": {"name": "baseline_phase", "performance_sample": False},
                "returncode": 0,
                "summary": {
                    "success": True,
                    "steps_completed": 24,
                    "mean_step_ms": 1000.0,
                    "steady_mean_step_ms": 1000.0,
                    "peak_vram_mb": 4096.0,
                },
            },
            {
                "case": {
                    "name": "native_update_dispatch_perf",
                    "performance_sample": True,
                },
                "returncode": 0,
                "summary": {
                    "success": True,
                    "steps_completed": 24,
                    "mean_step_ms": 900.0,
                    "steady_mean_step_ms": 900.0,
                    "peak_vram_mb": 4096.0,
                    "native_dispatch_executed": True,
                    "owner_native_launch_probe_present": True,
                    "owner_native_launch_attempted": True,
                    "owner_native_launch_ok": True,
                    "owner_native_kernel_executed": True,
                    "owner_native_parity_ok": True,
                    "owner_native_numel": 4096,
                },
            },
        ],
        "summary": {
            "case_count": 2,
            "executed_count": 2,
            "all_success": True,
            "mean_step_ms_by_case": {
                "baseline_phase": 1000.0,
                "native_update_dispatch_perf": 900.0,
            },
        },
        "optimizer_performance_artifact": {
            "optimizer_performance_gate": {
                "gate": "turbocore_optimizer_performance_gate",
                "ok": True,
                "promotion_gate_ok": True,
                "runtime_dispatch_allowed": False,
                "evidence_quality": "promotion_benchmark",
                "best_candidate": {
                    "optimizer": "triton_adamw_flat_v0",
                    "speedup_vs_baseline": 1.25,
                    "promotion_gate_ok": True,
                },
            }
        },
    }


def _dry_run_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "matrix": "turbocore_update_benchmark_matrix_v0",
        "run": False,
        "family": "anima",
        "cases": [
            {"case": {"name": "baseline_phase"}, "summary_path": "baseline/anima_summary.json"},
            {"case": {"name": "native_update_dispatch_perf"}, "summary_path": "native/anima_summary.json"},
        ],
        "summary": {
            "case_count": 2,
            "executed_count": 0,
            "all_success": None,
        },
    }


def _performance_blocked_payload() -> dict[str, Any]:
    payload = _representative_payload()
    for item in payload["cases"]:
        if item["case"]["name"] == "native_update_dispatch_perf":
            item["summary"]["mean_step_ms"] = 1200.0
            item["summary"]["steady_mean_step_ms"] = 1200.0
    payload["summary"]["mean_step_ms_by_case"]["native_update_dispatch_perf"] = 1200.0
    return payload


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
