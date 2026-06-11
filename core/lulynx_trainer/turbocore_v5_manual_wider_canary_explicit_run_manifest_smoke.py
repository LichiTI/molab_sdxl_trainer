"""Smoke checks for V5 manual wider canary explicit-run manifest."""

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
from core.turbocore_v5_owner_review_evidence_package import (  # noqa: E402
    build_v5_owner_review_evidence_package,
)


def run_smoke() -> dict[str, Any]:
    pending_package = build_v5_owner_review_evidence_package(stability_gate=_stability_gate())
    signed_package = build_v5_owner_review_evidence_package(
        stability_gate=_stability_gate(),
        performance_matrix=_performance_matrix(),
        owner_review=_owner_review(),
    )

    pending = build_v5_manual_wider_canary_explicit_run_manifest(owner_review_package=pending_package)
    assert pending["ok"] is False, pending
    assert pending["manual_wider_canary_explicit_run_allowed"] is False, pending
    assert pending["route_decision"]["decision"] == "blocked_until_signed_owner_review", pending
    assert "v5_p21_signed_owner_review_package_missing" in pending["blocked_reasons"], pending

    ready = build_v5_manual_wider_canary_explicit_run_manifest(owner_review_package=signed_package)
    assert ready["ok"] is True, ready
    assert ready["explicit_run_manifest_ready"] is True, ready
    assert ready["manual_wider_canary_explicit_run_allowed"] is True, ready
    assert ready["default_rollout_allowed"] is False, ready
    assert ready["auto_rollout_allowed"] is False, ready
    assert ready["training_path_enabled"] is False, ready
    assert ready["manifest_request_fields"]["turbocoreNativeUpdateManualWiderCanaryReviewReady"] is True, ready
    assert "native_dispatch_executed" in ready["audit_skeleton"]["required_runtime_evidence"], ready

    observe = build_v5_manual_wider_canary_explicit_run_manifest(
        owner_review_package=signed_package,
        native_training_mode="observe",
    )
    assert observe["ok"] is True, observe
    assert observe["manual_wider_canary_explicit_run_allowed"] is False, observe
    assert observe["route_decision"]["decision"] == "observe_manifest_ready_but_no_training_dispatch", observe

    auto = build_v5_manual_wider_canary_explicit_run_manifest(
        owner_review_package=signed_package,
        native_training_mode="auto",
    )
    assert auto["ok"] is False, auto
    assert auto["route_decision"]["decision"] == "auto_blocked_for_v5", auto
    assert "v5_p21_auto_rollout_blocked" in auto["blocked_reasons"], auto

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_manual_wider_canary_explicit_run_manifest_smoke",
        "ok": True,
        "ready_decision": ready["route_decision"]["decision"],
        "observe_decision": observe["route_decision"]["decision"],
        "auto_decision": auto["route_decision"]["decision"],
        "recommended_next_step": ready["recommended_next_step"],
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


def _performance_matrix() -> dict[str, Any]:
    return {
        "summary": {
            "native_update_performance_gate": {"ready": True, "blocked_reasons": []},
            "native_dispatch_ctx_sync_free_comparison": {
                "ctx_sync_free_case": "native_update_dispatch_ctx_sync_free_canary",
                "ctx_sync_free_speedup_vs_context_sync_native": 1.045,
                "representative_candidate_ready": False,
            },
        },
        "native_update_performance_report": {
            "training_dispatch": False,
            "runtime_dispatch_allowed": False,
            "performance_gate": {
                "representative_performance_gate_ready": True,
                "promotion_gate_ok": True,
                "blocked_reasons": [],
                "evidence": {
                    "optimizer_microbenchmark": {
                        "present": True,
                        "evidence_quality": "promotion_benchmark",
                        "best_speedup_vs_baseline": 22.3494,
                    },
                    "training_matrix": {
                        "native_case": "native_update_dispatch_promotion_perf",
                        "end_to_end_speedup": 1.1223,
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
