"""Smoke checks for V5 owner-review evidence package."""

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

from core.turbocore_v5_owner_review_evidence_package import (  # noqa: E402
    build_v5_owner_review_evidence_package,
)


def run_smoke() -> dict[str, Any]:
    missing_stability = build_v5_owner_review_evidence_package(stability_gate=_stability_gate(ok=False))
    assert missing_stability["ok"] is False, missing_stability
    assert missing_stability["ready_for_owner_review"] is False, missing_stability
    assert "v5_p6_stability_gate_not_ready" in missing_stability["blocked_reasons"], missing_stability
    assert missing_stability["manual_wider_canary_allowed"] is False, missing_stability
    assert missing_stability["default_rollout_allowed"] is False, missing_stability
    assert missing_stability["auto_rollout_allowed"] is False, missing_stability

    pending = build_v5_owner_review_evidence_package(stability_gate=_stability_gate(ok=True))
    assert pending["ok"] is True, pending
    assert pending["evidence_package_ready"] is True, pending
    assert pending["ready_for_owner_review"] is True, pending
    assert pending["promotion_review_ready"] is False, pending
    assert pending["promotion_decision"] == "hold_for_manual_owner_review", pending
    assert pending["owner_review_action_required"] is True, pending
    assert pending["manual_wider_canary_allowed"] is False, pending
    assert "v5_p6_owner_review_not_signed" in pending["approval_blockers"], pending
    assert pending["owner_review_template"]["approve_manual_wider_canary"] is False, pending
    assert pending["post_approval_request_fields"]["turbocoreNativeUpdateCanaryScope"] == "manual_wider_canary", pending

    pending_with_perf = build_v5_owner_review_evidence_package(
        stability_gate=_stability_gate(ok=True),
        performance_matrix=_performance_matrix(ok=True),
    )
    assert pending_with_perf["ok"] is True, pending_with_perf
    perf_summary = pending_with_perf["performance_matrix_summary"]
    assert perf_summary["present"] is True, pending_with_perf
    assert perf_summary["performance_gate_ready"] is True, pending_with_perf
    assert perf_summary["report_only_runtime_dispatch_off"] is True, pending_with_perf
    assert perf_summary["representative_end_to_end_speedup"] == 1.1223, pending_with_perf
    assert "performance_gate_ready" in {
        item["id"] for item in pending_with_perf["review_checklist"]
    }, pending_with_perf

    perf_blocked = build_v5_owner_review_evidence_package(
        stability_gate=_stability_gate(ok=True),
        performance_matrix=_performance_matrix(ok=False),
    )
    assert perf_blocked["ok"] is False, perf_blocked
    assert "v5_p19_performance_gate_not_ready" in perf_blocked["blocked_reasons"], perf_blocked

    adapter_blocked = build_v5_owner_review_evidence_package(
        stability_gate=_stability_gate(ok=True),
        config_adapter_scorecard={"config_adapter_ready": False, "blocked_reasons": ["fixture_adapter_blocked"]},
    )
    assert adapter_blocked["ok"] is False, adapter_blocked
    assert "v5_p6_config_adapter_not_ready" in adapter_blocked["blocked_reasons"], adapter_blocked
    assert "fixture_adapter_blocked" in adapter_blocked["blocked_reasons"], adapter_blocked

    approved = build_v5_owner_review_evidence_package(
        stability_gate=_stability_gate(ok=True),
        owner_review=_owner_review(approve=True),
    )
    assert approved["ok"] is True, approved
    assert approved["promotion_review_ready"] is True, approved
    assert approved["promotion_decision"] == "manual_wider_canary_review_ready", approved
    assert approved["owner_review_action_required"] is False, approved
    assert approved["manual_wider_canary_allowed"] is True, approved
    assert approved["approval_blockers"] == [], approved
    assert approved["default_training_path_enabled"] is False, approved
    assert approved["default_rollout_allowed"] is False, approved
    assert approved["auto_rollout_allowed"] is False, approved

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_owner_review_evidence_package_smoke",
        "ok": True,
        "pending_decision": pending["promotion_decision"],
        "approved_decision": approved["promotion_decision"],
        "recommended_next_step": pending["recommended_next_step"],
    }


def _stability_gate(*, ok: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v5_replicate_stability_gate_v0",
        "stability_gate_ready": ok,
        "run_count": 3 if ok else 1,
        "ready_run_count": 3 if ok else 1,
        "min_replicate_runs": 3,
        "aggregate": {
            "speedup_samples": [1.2151, 1.088, 1.196] if ok else [1.2151],
            "min_speedup": 1.088 if ok else 1.2151,
            "mean_speedup": 1.1664 if ok else 1.2151,
            "median_speedup": 1.196 if ok else 1.2151,
            "speedup_spread_ratio": 0.1063 if ok else 0.0,
        },
        "blocked_reasons": [] if ok else ["v5_p3_replicate_runs_too_few"],
    }


def _owner_review(*, approve: bool) -> dict[str, Any]:
    return {
        "reviewer": "owner",
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


def _performance_matrix(*, ok: bool) -> dict[str, Any]:
    blockers = [] if ok else ["optimizer_microbenchmark_missing"]
    return {
        "matrix_summary_path": "temp/turbocore_v5_p18/matrix_summary.json",
        "summary": {
            "native_update_performance_gate": {
                "ready": ok,
                "blocked_reasons": blockers,
            },
            "native_dispatch_ctx_sync_free_comparison": {
                "ctx_sync_free_case": "native_update_dispatch_ctx_sync_free_canary",
                "ctx_sync_free_speedup_vs_baseline": 1.1728 if ok else None,
                "ctx_sync_free_speedup_vs_context_sync_native": 1.045 if ok else None,
                "representative_candidate_ready": False,
            },
        },
        "native_update_performance_report": {
            "training_dispatch": False,
            "runtime_dispatch_allowed": False,
            "performance_gate": {
                "representative_performance_gate_ready": ok,
                "promotion_gate_ok": ok,
                "blocked_reasons": blockers,
                "evidence": {
                    "optimizer_microbenchmark": {
                        "present": ok,
                        "evidence_quality": "promotion_benchmark" if ok else "",
                        "best_speedup_vs_baseline": 22.3494 if ok else None,
                    },
                    "training_matrix": {
                        "native_case": "native_update_dispatch_promotion_perf",
                        "end_to_end_speedup": 1.1223 if ok else None,
                        "representative_steps": 20,
                    },
                },
            },
        },
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
