"""Smoke checks for V5 manual wider canary run audit ingester."""

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
from core.turbocore_v5_checkpoint_resume_native_state_boundary_evidence import (  # noqa: E402
    build_v5_checkpoint_resume_native_state_boundary_evidence,
)
from core.turbocore_v5_owner_review_evidence_package import (  # noqa: E402
    build_v5_owner_review_evidence_package,
)


def run_smoke() -> dict[str, Any]:
    manifest = build_v5_manual_wider_canary_explicit_run_manifest(
        owner_review_package=build_v5_owner_review_evidence_package(
            stability_gate=_stability_gate(),
            owner_review=_owner_review(),
        )
    )
    missing = build_v5_manual_wider_canary_run_audit(explicit_run_manifest=manifest)
    assert missing["ok"] is False, missing
    assert "v5_p22_run_result_missing" in missing["blocked_reasons"], missing

    keep = build_v5_manual_wider_canary_run_audit(
        explicit_run_manifest=manifest,
        run_result=_run_result(success=True),
    )
    assert keep["ok"] is True, keep
    assert keep["decision"] == "keep_manual_wider_canary_evidence", keep
    assert keep["run_audit_ready"] is True, keep
    assert keep["keep_candidate_allowed"] is True, keep
    assert keep["rollback_required"] is False, keep
    assert keep["default_behavior_changed"] is False, keep
    assert keep["default_training_path_enabled"] is False, keep
    assert keep["training_path_enabled"] is False, keep
    assert keep["default_rollout_allowed"] is False, keep
    assert keep["auto_rollout_allowed"] is False, keep
    assert keep["blocked_reasons"] == [], keep

    denied_manifest = build_v5_manual_wider_canary_explicit_run_manifest(
        owner_review_package=build_v5_owner_review_evidence_package(
            stability_gate=_stability_gate(),
            owner_review=_owner_review(approve=False),
        )
    )
    explicit_denied = build_v5_manual_wider_canary_run_audit(
        explicit_run_manifest=denied_manifest,
        run_result=_run_result(success=True),
    )
    assert explicit_denied["ok"] is False, explicit_denied
    assert explicit_denied["keep_candidate_allowed"] is False, explicit_denied
    assert explicit_denied["rollback_required"] is True, explicit_denied
    assert explicit_denied["manifest_summary"]["manual_wider_canary_explicit_run_allowed"] is False, explicit_denied
    assert (
        "v5_p22_explicit_run_manifest_not_ready" in explicit_denied["blocked_reasons"]
    ), explicit_denied

    rollback = build_v5_manual_wider_canary_run_audit(
        explicit_run_manifest=manifest,
        run_result=_run_result(success=True, rollback_events=["native_error"]),
    )
    assert rollback["ok"] is False, rollback
    assert rollback["rollback_required"] is True, rollback
    assert "v5_p22_rollback_event_present" in rollback["blocked_reasons"], rollback

    performance_regression = build_v5_manual_wider_canary_run_audit(
        explicit_run_manifest=manifest,
        run_result=_run_result(
            success=True,
            rollback_events=["performance_regression"],
            speedup=0.91,
        ),
    )
    assert performance_regression["ok"] is False, performance_regression
    assert performance_regression["rollback_required"] is True, performance_regression
    assert performance_regression["default_rollout_allowed"] is False, performance_regression
    assert (
        "rollback:performance_regression" in performance_regression["blocked_reasons"]
    ), performance_regression

    non_finite = build_v5_manual_wider_canary_run_audit(
        explicit_run_manifest=manifest,
        run_result=_run_result(
            success=True,
            rollback_events=["non_finite"],
            speedup=float("nan"),
        ),
    )
    assert non_finite["ok"] is False, non_finite
    assert non_finite["rollback_required"] is True, non_finite
    assert "rollback:non_finite" in non_finite["blocked_reasons"], non_finite

    missing_evidence = build_v5_manual_wider_canary_run_audit(
        explicit_run_manifest=manifest,
        run_result=_run_result(
            success=True,
            missing_evidence=["checkpoint_resume_native_state_boundary"],
        ),
    )
    assert missing_evidence["ok"] is False, missing_evidence
    assert missing_evidence["rollback_required"] is True, missing_evidence
    assert (
        "v5_p22_required_runtime_evidence_missing" in missing_evidence["blocked_reasons"]
    ), missing_evidence
    assert (
        "missing:checkpoint_resume_native_state_boundary" in missing_evidence["blocked_reasons"]
    ), missing_evidence

    boundary = build_v5_checkpoint_resume_native_state_boundary_evidence(
        explicit_run_manifest=manifest,
        run_live_probe=True,
    )
    assert boundary["ok"] is True, boundary
    replayed = build_v5_manual_wider_canary_run_audit(
        explicit_run_manifest=manifest,
        run_result=_run_result(
            success=True,
            missing_evidence=["checkpoint_resume_native_state_boundary"],
        ),
        checkpoint_resume_boundary_evidence=boundary,
    )
    assert replayed["ok"] is True, replayed
    assert replayed["run_audit_ready"] is True, replayed
    assert replayed["blocked_reasons"] == [], replayed
    assert replayed["checkpoint_resume_boundary_evidence_summary"][
        "checkpoint_resume_native_state_boundary_ready"
    ] is True, replayed

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_manual_wider_canary_run_audit_ingester_smoke",
        "ok": True,
        "keep_decision": keep["decision"],
        "explicit_denied_decision": explicit_denied["decision"],
        "rollback_decision": rollback["decision"],
        "performance_regression_decision": performance_regression["decision"],
        "non_finite_decision": non_finite["decision"],
        "replayed_boundary_decision": replayed["decision"],
        "recommended_next_step": keep["recommended_next_step"],
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


def _owner_review(*, approve: bool = True) -> dict[str, Any]:
    return {
        "reviewer": "owner_fixture",
        "reviewed_at": "2026-06-01",
        "requested_scope": "manual_wider_canary",
        "approve_manual_wider_canary": bool(approve),
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
    return {
        "schema_version": 1,
        "result": "v5_manual_wider_canary_explicit_run_fixture",
        "success": bool(success),
        "evidence": evidence,
        "report_fields": {
            "native_dispatch_training_executor_elapsed_ms_mean": 19.0,
            "native_dispatch_update_executor_elapsed_ms_mean": 12.0,
            "native_dispatch_update_executor_grad_sync_ms_mean": 4.0,
            "native_dispatch_update_executor_copyback_ms_mean": 3.0,
            "native_dispatch_owner_native_runtime_stream_binding": "cuda_driver_default_stream_null_synchronized",
            "native_dispatch_owner_native_stream_lifetime_bound": True,
        },
        "rollback_events": list(rollback_events or []),
        "performance": {"representative_end_to_end_speedup": float(speedup)},
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
