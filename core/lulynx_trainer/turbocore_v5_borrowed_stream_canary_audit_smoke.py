"""Smoke checks for V5-P13 borrowed-stream canary blocker auditing."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lulynx_trainer.turbocore_update_benchmark_matrix import (  # noqa: E402
    _summarize_benchmark_summary,
    _summarize_matrix,
)


def _runtime_report() -> dict:
    return {
        "schema_version": 1,
        "requested": True,
        "native_step_executed": False,
        "blocked_reasons": ["native_update_training_executor_error"],
        "training_executor": {
            "attempted": True,
            "called": True,
            "ok": False,
            "reason": "native_update_training_executor_error",
            "native_step_executed": False,
            "result": {"error": "RuntimeError: borrowed_stream_policy_not_allowed"},
            "blocked_reasons": ["native_update_training_executor_error"],
        },
    }


def test_case_summary_surfaces_training_executor_borrowed_stream_blocker() -> None:
    summary = {
        "benchmark": {
            "turbocore_native_update_runtime_synchronization_policy": "borrowed_stream_event_chain"
        },
        "runs": {
            "standard": {
                "success": True,
                "steps_completed": 1,
                "mean_step_ms": 1.0,
                "steady_mean_step_ms": 1.0,
                "peak_vram_mb": 1.0,
                "native_update_dispatch_runtime_reports": [_runtime_report()],
            }
        },
    }
    case_summary = _summarize_benchmark_summary(summary)
    assert case_summary["native_dispatch_requested_borrowed_stream_event_chain"] is True, case_summary
    assert case_summary["native_dispatch_requested"] is True, case_summary
    assert case_summary["native_dispatch_executed"] is False, case_summary
    assert case_summary["native_dispatch_borrowed_stream_blocked_before_native_step"] is True, case_summary
    assert case_summary["native_dispatch_borrowed_stream_policy_not_allowed"] is True, case_summary
    assert "borrowed_stream_policy_not_allowed" in case_summary[
        "native_dispatch_training_executor_blocked_reasons"
    ], case_summary
    assert case_summary["native_dispatch_training_executor_last_error"].endswith(
        "borrowed_stream_policy_not_allowed"
    ), case_summary


def test_matrix_summary_groups_borrowed_stream_canary_blockers() -> None:
    payload = {
        "matrix": "turbocore_update_benchmark_matrix_v0",
        "run": True,
        "cases": [
            {
                "case": {"name": "native_update_dispatch_ctx_sync_free_canary"},
                "returncode": 0,
                "summary": {
                    "success": True,
                    "native_dispatch_requested_runtime_synchronization_policy": "borrowed_stream_event_chain",
                    "native_dispatch_requested_borrowed_stream_event_chain": True,
                    "native_dispatch_requested": True,
                    "native_dispatch_executed": False,
                    "native_dispatch_borrowed_stream_blocked_before_native_step": True,
                    "native_dispatch_borrowed_stream_policy_not_allowed": True,
                    "native_dispatch_runtime_blocked_reasons": ["native_update_training_executor_error"],
                    "native_dispatch_training_executor_last_error": (
                        "RuntimeError: borrowed_stream_policy_not_allowed"
                    ),
                    "native_dispatch_training_executor_blocked_reasons": [
                        "native_update_training_executor_error",
                        "borrowed_stream_policy_not_allowed",
                    ],
                    "gate_blocked_reasons": ["stream_lifetime_unbound"],
                    "readiness_blockers": ["stream_lifetime_unbound"],
                },
            }
        ],
    }
    summary = _summarize_matrix(payload)
    assert summary["native_dispatch_requested_borrowed_stream_cases"] == [
        "native_update_dispatch_ctx_sync_free_canary"
    ], summary
    assert summary["native_dispatch_borrowed_stream_executed_cases"] == [], summary
    assert summary["native_dispatch_borrowed_stream_blocked_cases"] == [
        "native_update_dispatch_ctx_sync_free_canary"
    ], summary
    assert summary["native_dispatch_borrowed_stream_policy_not_allowed_cases"] == [
        "native_update_dispatch_ctx_sync_free_canary"
    ], summary
    assert summary["native_dispatch_borrowed_stream_block_stage_by_case"] == {
        "native_update_dispatch_ctx_sync_free_canary": "training_executor"
    }, summary
    blockers = summary["native_dispatch_borrowed_stream_blockers_by_case"][
        "native_update_dispatch_ctx_sync_free_canary"
    ]
    assert "native_update_training_executor_error" in blockers, summary
    assert "borrowed_stream_policy_not_allowed" in blockers, summary
    assert "stream_lifetime_unbound" in blockers, summary


def main() -> int:
    test_case_summary_surfaces_training_executor_borrowed_stream_blocker()
    test_matrix_summary_groups_borrowed_stream_canary_blockers()
    print("turbocore_v5_borrowed_stream_canary_audit_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
