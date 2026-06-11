"""Smoke checks for V5 manual wider canary run review package."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_manual_wider_canary_explicit_run_manifest import (  # noqa: E402
    build_v5_manual_wider_canary_explicit_run_manifest,
)
from core.turbocore_v5_manual_wider_canary_run_audit_ingester import (  # noqa: E402
    build_v5_manual_wider_canary_run_audit,
)
from core.turbocore_v5_manual_wider_canary_run_review_package import (  # noqa: E402
    build_v5_manual_wider_canary_run_review_package,
)
from core.turbocore_v5_owner_review_evidence_package import (  # noqa: E402
    build_v5_owner_review_evidence_package,
)


def run_smoke() -> dict[str, Any]:
    owner_package = build_v5_owner_review_evidence_package(
        stability_gate=_stability_gate(),
        owner_review=_owner_review(),
    )
    manifest = build_v5_manual_wider_canary_explicit_run_manifest(
        owner_review_package=owner_package,
    )
    keep_audit = build_v5_manual_wider_canary_run_audit(
        explicit_run_manifest=manifest,
        run_result=_run_result(success=True),
    )
    ready = build_v5_manual_wider_canary_run_review_package(
        explicit_run_audit=keep_audit,
        explicit_run_manifest=manifest,
        owner_review_package=owner_package,
    )
    assert ready["ok"] is True, ready
    assert ready["run_review_package_ready"] is True, ready
    assert ready["ready_for_owner_rollout_review"] is True, ready
    assert ready["rollout_review_decision"] == "hold_for_owner_rollout_review", ready
    assert ready["default_behavior_changed"] is False, ready
    assert ready["default_training_path_enabled"] is False, ready
    assert ready["training_path_enabled"] is False, ready
    assert ready["default_rollout_allowed"] is False, ready
    assert ready["auto_rollout_allowed"] is False, ready
    assert ready["request_adapter_mapping_allowed"] is False, ready
    assert ready["request_fields_emitted"] is False, ready
    assert ready["post_review_request_fields"] == {}, ready

    missing_audit = build_v5_manual_wider_canary_run_audit(
        explicit_run_manifest=manifest,
        run_result=_run_result(
            success=True,
            missing_evidence=["checkpoint_resume_native_state_boundary"],
        ),
    )
    missing = build_v5_manual_wider_canary_run_review_package(
        explicit_run_audit=missing_audit,
        explicit_run_manifest=manifest,
    )
    assert missing["ok"] is False, missing
    assert missing["rollback_required"] is True, missing
    assert "v5_p23_explicit_run_audit_not_ready" in missing["blocked_reasons"], missing
    assert "missing:checkpoint_resume_native_state_boundary" in missing["blocked_reasons"], missing

    regression_audit = build_v5_manual_wider_canary_run_audit(
        explicit_run_manifest=manifest,
        run_result=_run_result(
            success=True,
            rollback_events=["performance_regression"],
            speedup=0.9,
        ),
    )
    regression = build_v5_manual_wider_canary_run_review_package(
        explicit_run_audit=regression_audit,
        explicit_run_manifest=manifest,
    )
    assert regression["ok"] is False, regression
    assert "rollback:performance_regression" in regression["blocked_reasons"], regression
    assert regression["default_rollout_allowed"] is False, regression

    non_finite_audit = build_v5_manual_wider_canary_run_audit(
        explicit_run_manifest=manifest,
        run_result=_run_result(
            success=True,
            rollback_events=["non_finite"],
            speedup=float("nan"),
        ),
    )
    non_finite = build_v5_manual_wider_canary_run_review_package(
        explicit_run_audit=non_finite_audit,
        explicit_run_manifest=manifest,
    )
    assert non_finite["ok"] is False, non_finite
    assert "rollback:non_finite" in non_finite["blocked_reasons"], non_finite

    open_history = build_v5_manual_wider_canary_run_review_package(
        explicit_run_audit=keep_audit,
        explicit_run_manifest=manifest,
        rollback_history={"events": [{"event": "native_error", "status": "open"}]},
    )
    assert open_history["ok"] is False, open_history
    assert "v5_p23_open_rollback_history" in open_history["blocked_reasons"], open_history

    missing_report = build_v5_manual_wider_canary_run_review_package(
        explicit_run_audit=build_v5_manual_wider_canary_run_audit(
            explicit_run_manifest=manifest,
            run_result=_run_result(
                success=True,
                missing_report_fields=["native_dispatch_owner_native_stream_lifetime_bound"],
            ),
        ),
        explicit_run_manifest=manifest,
    )
    assert missing_report["ok"] is False, missing_report
    assert "v5_p23_required_report_fields_missing" in missing_report["blocked_reasons"], missing_report

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_manual_wider_canary_run_review_package_smoke",
        "ok": True,
        "ready_decision": ready["rollout_review_decision"],
        "missing_decision": missing["rollout_review_decision"],
        "regression_decision": regression["rollout_review_decision"],
        "non_finite_decision": non_finite["rollout_review_decision"],
        "open_history_decision": open_history["rollout_review_decision"],
        "missing_report_decision": missing_report["rollout_review_decision"],
    }


def _stability_gate() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v5_replicate_stability_gate_v0",
        "stability_gate_ready": True,
        "run_count": 3,
        "ready_run_count": 3,
        "min_replicate_runs": 3,
        "aggregate": {
            "speedup_samples": [1.2151, 1.088, 1.196],
            "min_speedup": 1.088,
            "mean_speedup": 1.1664,
            "median_speedup": 1.196,
            "speedup_spread_ratio": 0.1063,
        },
        "blocked_reasons": [],
    }


def _owner_review() -> dict[str, Any]:
    return {
        "reviewer": "owner_fixture",
        "reviewed_at": "2026-06-01",
        "requested_scope": "manual_wider_canary",
        "approve_manual_wider_canary": True,
        "confirm_default_training_path_enabled": False,
        "confirm_training_path_enabled": False,
        "confirm_default_rollout_allowed": False,
        "confirm_auto_rollout_allowed": False,
        "acknowledge_runtime_synchronization": True,
        "rollback_policy": {
            "fallback_authoritative": True,
            "fallback_backend": "pytorch_adamw",
            "disable_for_run_on_native_error": True,
            "disable_for_run_on_state_sync_failure": True,
            "disable_for_run_on_checkpoint_resume_mismatch": True,
            "rollback_on_resume_mismatch": True,
            "rollback_on_performance_regression": True,
        },
    }


def _run_result(
    *,
    success: bool,
    rollback_events: list[str] | None = None,
    missing_evidence: list[str] | None = None,
    missing_report_fields: list[str] | None = None,
    speedup: float = 1.05,
) -> dict[str, Any]:
    evidence = {
        "native_dispatch_requested": True,
        "native_dispatch_executed": True,
        "native_dispatch_training_executor_timing_present": True,
        "native_dispatch_update_report_present": True,
        "native_dispatch_owner_native_report_present": True,
        "native_dispatch_probe_cache_retained": True,
        "native_dispatch_owner_native_runtime_synchronization": True,
        "native_dispatch_training_executor_last_error_empty": True,
        "fallback_state_sync_on_close_or_recovery": True,
        "checkpoint_resume_native_state_boundary": True,
    }
    for name in missing_evidence or []:
        evidence[str(name)] = False
    report_fields = {
        "native_dispatch_training_executor_elapsed_ms_mean": 19.0,
        "native_dispatch_update_executor_elapsed_ms_mean": 12.0,
        "native_dispatch_update_executor_grad_sync_ms_mean": 4.0,
        "native_dispatch_update_executor_copyback_ms_mean": 3.0,
        "native_dispatch_owner_native_runtime_stream_binding": "cuda_driver_default_stream_null_synchronized",
        "native_dispatch_owner_native_stream_lifetime_bound": True,
    }
    for name in missing_report_fields or []:
        report_fields.pop(str(name), None)
    return {
        "schema_version": 1,
        "result": "v5_manual_wider_canary_explicit_run_fixture",
        "success": bool(success),
        "evidence": evidence,
        "report_fields": report_fields,
        "rollback_events": list(rollback_events or []),
        "performance": {"representative_end_to_end_speedup": float(speedup)},
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
