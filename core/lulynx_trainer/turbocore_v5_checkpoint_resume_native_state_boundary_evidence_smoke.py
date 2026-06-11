"""Smoke checks for V5 checkpoint/resume native-state boundary evidence."""

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

from core.turbocore_v5_checkpoint_resume_native_state_boundary_evidence import (  # noqa: E402
    build_v5_checkpoint_resume_native_state_boundary_evidence,
)
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
    evidence = build_v5_checkpoint_resume_native_state_boundary_evidence(
        explicit_run_manifest=manifest,
        run_live_probe=True,
    )
    assert evidence["ok"] is True, evidence
    assert evidence["checkpoint_resume_native_state_boundary"] is True, evidence
    assert evidence["checkpoint_roundtrip_ok"] is True, evidence
    assert evidence["default_training_path_enabled"] is False, evidence
    assert evidence["default_rollout_allowed"] is False, evidence
    assert evidence["auto_rollout_allowed"] is False, evidence

    missing = build_v5_manual_wider_canary_run_audit(
        explicit_run_manifest=manifest,
        run_result=_run_result(success=True, missing_checkpoint_boundary=True),
    )
    assert missing["ok"] is False, missing
    assert "missing:checkpoint_resume_native_state_boundary" in missing["blocked_reasons"], missing

    replayed = build_v5_manual_wider_canary_run_audit(
        explicit_run_manifest=manifest,
        run_result=_run_result(success=True, missing_checkpoint_boundary=True),
        checkpoint_resume_boundary_evidence=evidence,
    )
    assert replayed["ok"] is True, replayed
    assert replayed["run_audit_ready"] is True, replayed
    assert replayed["keep_candidate_allowed"] is True, replayed

    review = build_v5_manual_wider_canary_run_review_package(
        explicit_run_audit=replayed,
        explicit_run_manifest=manifest,
        owner_review_package=owner_package,
    )
    assert review["ok"] is True, review
    assert review["ready_for_owner_rollout_review"] is True, review
    assert review["rollout_review_decision"] == "hold_for_owner_rollout_review", review
    assert review["request_adapter_mapping_allowed"] is False, review

    skipped = build_v5_checkpoint_resume_native_state_boundary_evidence(
        explicit_run_manifest=manifest,
        run_live_probe=False,
    )
    assert skipped["ok"] is False, skipped
    assert "v5_p24_checkpoint_metadata_integrated_missing" in skipped["blocked_reasons"], skipped

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_checkpoint_resume_native_state_boundary_evidence_smoke",
        "ok": True,
        "boundary_ready": evidence["checkpoint_resume_native_state_boundary"],
        "replayed_audit_decision": replayed["decision"],
        "review_decision": review["rollout_review_decision"],
        "skipped_decision": skipped["recommended_next_step"],
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


def _run_result(*, success: bool, missing_checkpoint_boundary: bool = False) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "result": "v5_manual_wider_canary_explicit_run_fixture",
        "success": bool(success),
        "evidence": {
            "native_dispatch_requested": True,
            "native_dispatch_executed": True,
            "native_dispatch_training_executor_timing_present": True,
            "native_dispatch_update_report_present": True,
            "native_dispatch_owner_native_report_present": True,
            "native_dispatch_probe_cache_retained": True,
            "native_dispatch_owner_native_runtime_synchronization": True,
            "native_dispatch_training_executor_last_error_empty": True,
            "fallback_state_sync_on_close_or_recovery": True,
            "checkpoint_resume_native_state_boundary": not bool(missing_checkpoint_boundary),
        },
        "report_fields": {
            "native_dispatch_training_executor_elapsed_ms_mean": 19.0,
            "native_dispatch_update_executor_elapsed_ms_mean": 12.0,
            "native_dispatch_update_executor_grad_sync_ms_mean": 4.0,
            "native_dispatch_update_executor_copyback_ms_mean": 3.0,
            "native_dispatch_owner_native_runtime_stream_binding": "cuda_driver_default_stream_null_synchronized",
            "native_dispatch_owner_native_stream_lifetime_bound": True,
        },
        "rollback_events": [],
        "performance": {"representative_end_to_end_speedup": 1.05},
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
