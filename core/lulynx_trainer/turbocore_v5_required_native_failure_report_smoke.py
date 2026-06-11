"""Smoke checks for V5-P14 required-native failure report preservation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lulynx_trainer.turbocore_update_benchmark_matrix import _summarize_matrix  # noqa: E402
from core.turbocore_flat_adamw_state import NativeAdamWRequiredError  # noqa: E402
from core.turbocore_native_update_dispatch_executor import (  # noqa: E402
    run_native_update_training_executor,
)
from core.turbocore_v5_borrowed_stream_canary_audit import (  # noqa: E402
    audit_training_executor_reports,
)


def test_training_executor_error_preserves_native_report() -> None:
    native_report = _native_report()

    def _executor(_request: dict) -> dict:
        raise NativeAdamWRequiredError("borrowed_stream_policy_not_allowed", native_report)

    report = run_native_update_training_executor(
        execution_plan={"execution_allowed": True},
        runtime_context={"native_update_training_dispatch_enabled": True},
        native_executor=_executor,
    )
    result = report["result"]
    assert report["ok"] is False, report
    assert result["native_report"]["reason"] == "borrowed_stream_policy_not_allowed", report
    assert result["borrowed_stream_policy"]["blocked_reasons"] == [
        "borrowed_stream_event_chain_not_verified"
    ], report
    assert result["borrowed_stream_launch_evidence"]["blocked_reasons"] == [
        "v5_p10_event_chain_verified_missing"
    ], report


def test_audit_groups_native_policy_and_launch_evidence_blockers() -> None:
    training_executor = {
        "attempted": True,
        "called": True,
        "ok": False,
        "reason": "native_update_training_executor_error",
        "native_step_executed": False,
        "result": {
            "error": "NativeAdamWRequiredError: borrowed_stream_policy_not_allowed",
            "native_report": _native_report(),
        },
        "blocked_reasons": ["native_update_training_executor_error"],
    }
    audit = audit_training_executor_reports([{"requested": True, "training_executor": training_executor}])
    assert audit["policy_not_allowed"] is True, audit
    assert audit["native_policy_allowed"] is False, audit
    assert audit["event_chain_verified"] is False, audit
    assert "borrowed_stream_event_chain_not_verified" in audit["blocked_reasons"], audit
    assert "v5_p10_event_chain_verified_missing" in audit["blocked_reasons"], audit


def test_matrix_summary_surfaces_p14_borrowed_stream_blockers() -> None:
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
                    "native_dispatch_borrowed_stream_native_policy_blocked_reasons": [
                        "borrowed_stream_event_chain_not_verified"
                    ],
                    "native_dispatch_borrowed_stream_launch_evidence_blocked_reasons": [
                        "v5_p10_event_chain_verified_missing"
                    ],
                    "native_dispatch_borrowed_stream_stream_guard_blocked_reasons": [
                        "v5_p10_event_chain_verified_missing"
                    ],
                    "native_dispatch_borrowed_stream_lease_blocked_reasons": [],
                    "native_dispatch_training_executor_blocked_reasons": [
                        "native_update_training_executor_error"
                    ],
                },
            }
        ],
    }
    summary = _summarize_matrix(payload)
    blockers = summary["native_dispatch_borrowed_stream_blockers_by_case"][
        "native_update_dispatch_ctx_sync_free_canary"
    ]
    assert "borrowed_stream_event_chain_not_verified" in blockers, summary
    assert "v5_p10_event_chain_verified_missing" in blockers, summary


def _native_report() -> dict:
    return {
        "schema_version": 1,
        "ok": False,
        "reason": "borrowed_stream_policy_not_allowed",
        "blocked_reasons": ["borrowed_stream_policy_not_allowed"],
        "borrowed_stream_policy": {
            "requested": True,
            "allowed": False,
            "stream_handle": 123456,
            "stream_handle_nonzero": True,
            "blocked_reasons": ["borrowed_stream_event_chain_not_verified"],
        },
        "borrowed_stream_launch_evidence": {
            "runtime_stream_guard_evidence_ready": False,
            "stream_handle_nonzero": True,
            "event_chain_verified": False,
            "stream_lifetime_bound": True,
            "blocked_reasons": ["v5_p10_event_chain_verified_missing"],
            "stream_guard_blocked_reasons": ["v5_p10_event_chain_verified_missing"],
            "lease_blocked_reasons": [],
        },
    }


def main() -> None:
    test_training_executor_error_preserves_native_report()
    test_audit_groups_native_policy_and_launch_evidence_blockers()
    test_matrix_summary_surfaces_p14_borrowed_stream_blockers()
    print(
        json.dumps(
            {
                "schema_version": 1,
                "probe": "turbocore_v5_required_native_failure_report_smoke",
                "ok": True,
                "default_training_path_enabled": False,
                "auto_rollout_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
