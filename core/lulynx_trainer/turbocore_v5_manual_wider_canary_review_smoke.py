"""Smoke checks for V5 manual wider-canary review."""

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

from core.turbocore_v5_manual_wider_canary_review import (  # noqa: E402
    build_v5_manual_wider_canary_review,
)


def run_smoke() -> dict[str, Any]:
    missing_review = build_v5_manual_wider_canary_review(stability_gate=_stability_gate(ok=True))
    assert missing_review["ok"] is False, missing_review
    assert missing_review["promotion_decision"] == "hold_for_manual_owner_review", missing_review
    assert "v5_p4_manual_owner_review_present_missing" in missing_review["blocked_reasons"], missing_review
    assert missing_review["default_rollout_allowed"] is False, missing_review
    assert missing_review["auto_rollout_allowed"] is False, missing_review

    stability_blocked = build_v5_manual_wider_canary_review(
        stability_gate=_stability_gate(ok=False),
        owner_review=_owner_review(approve=True),
    )
    assert stability_blocked["ok"] is False, stability_blocked
    assert stability_blocked["promotion_decision"] == "hold_for_v5_p3_stability_gate", stability_blocked
    assert "v5_p4_p3_stability_gate_ready_missing" in stability_blocked["blocked_reasons"], stability_blocked
    assert "v5_p3_replicate_runs_too_few" in stability_blocked["blocked_reasons"], stability_blocked

    approved = build_v5_manual_wider_canary_review(
        stability_gate=_stability_gate(ok=True),
        owner_review=_owner_review(approve=True),
    )
    assert approved["ok"] is True, approved
    assert approved["promotion_review_ready"] is True, approved
    assert approved["promotion_decision"] == "manual_wider_canary_review_ready", approved
    assert approved["manual_wider_canary_allowed"] is True, approved
    assert approved["training_path_enabled"] is False, approved
    assert approved["default_training_path_enabled"] is False, approved
    assert approved["default_rollout_allowed"] is False, approved
    assert approved["auto_rollout_allowed"] is False, approved

    bad_scope = build_v5_manual_wider_canary_review(
        stability_gate=_stability_gate(ok=True),
        owner_review={**_owner_review(approve=True), "requested_scope": "auto"},
    )
    assert bad_scope["ok"] is False, bad_scope
    assert "v5_p4_scope_limited_to_manual_wider_canary_missing" in bad_scope["blocked_reasons"], bad_scope

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_manual_wider_canary_review_smoke",
        "ok": True,
        "missing_review_decision": missing_review["promotion_decision"],
        "stability_blocked_decision": stability_blocked["promotion_decision"],
        "approved_decision": approved["promotion_decision"],
        "recommended_next_step": approved["recommended_next_step"],
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


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
