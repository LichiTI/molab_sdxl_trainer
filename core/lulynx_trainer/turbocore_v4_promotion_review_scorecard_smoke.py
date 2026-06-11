"""Smoke checks for V4 promotion review scorecard."""

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

from core.turbocore_v4_promotion_review_scorecard import (  # noqa: E402
    build_v4_promotion_review_scorecard,
)


def run_smoke() -> dict[str, Any]:
    hold = build_v4_promotion_review_scorecard(p3_audit=_p3_fixture(real_ready=False))
    assert hold["ok"] is True, hold
    assert hold["promotion_review_ready"] is True, hold
    assert hold["manual_wider_canary_allowed"] is False, hold
    assert hold["promotion_decision"] == "hold_for_representative_benchmark", hold
    assert "real_benchmark_result_missing" in hold["promotion_hold_reasons"], hold
    assert hold["default_rollout_allowed"] is False, hold
    assert hold["auto_rollout_allowed"] is False, hold

    perf_hold = build_v4_promotion_review_scorecard(p3_audit=_p3_perf_blocked_fixture())
    assert perf_hold["ok"] is True, perf_hold
    assert perf_hold["manual_wider_canary_allowed"] is False, perf_hold
    assert perf_hold["promotion_decision"] == "hold_for_representative_performance_gate", perf_hold
    assert "real_benchmark_performance_gate_blocked" in perf_hold["promotion_hold_reasons"], perf_hold
    assert "end_to_end_speedup_below_threshold" in perf_hold["real_benchmark_performance_blockers"], perf_hold

    ready = build_v4_promotion_review_scorecard(p3_audit=_p3_fixture(real_ready=True))
    assert ready["ok"] is True, ready
    assert ready["manual_wider_canary_allowed"] is True, ready
    assert ready["promotion_decision"] == "manual_wider_canary_review_ready", ready
    assert ready["promotion_hold_reasons"] == [], ready
    assert ready["default_rollout_allowed"] is False, ready
    assert ready["auto_rollout_allowed"] is False, ready

    return {
        "schema_version": 1,
        "probe": "turbocore_v4_promotion_review_scorecard_smoke",
        "ok": True,
        "hold_decision": hold["promotion_decision"],
        "perf_hold_decision": perf_hold["promotion_decision"],
        "ready_decision": ready["promotion_decision"],
        "recommended_next_step": hold["recommended_next_step"],
    }


def _p3_fixture(*, real_ready: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "audit": "native_training_performance_v4_p3_audit_fixture",
        "milestone_completed": True,
        "real_benchmark_result_ready": real_ready,
        "sections": {
            "single_run_explicit_canary_policy": _policy(ok=True, explicit=True),
            "wider_canary_current_block": _policy(
                ok=real_ready,
                explicit=real_ready,
                larger=real_ready,
                blocked=[] if real_ready else ["v4_p3_real_benchmark_result_missing"],
            ),
            "auto_rollout_block": _policy(ok=False, explicit=False, blocked=["v4_p3_auto_rollout_blocked"]),
        },
    }


def _p3_perf_blocked_fixture() -> dict[str, Any]:
    payload = _p3_fixture(real_ready=False)
    payload.update(
        {
            "real_benchmark_input_present": True,
            "real_benchmark_executed": True,
            "real_benchmark_contract_ready": True,
            "real_benchmark_performance_gate_ready": False,
            "real_benchmark_status": "performance_gate_blocked",
            "real_benchmark_performance_blockers": ["end_to_end_speedup_below_threshold"],
        }
    )
    payload["sections"]["wider_canary_current_block"] = _policy(
        ok=False,
        explicit=False,
        blocked=[
            "v4_p3_real_benchmark_performance_gate_blocked",
            "end_to_end_speedup_below_threshold",
        ],
    )
    return payload


def _policy(
    *,
    ok: bool,
    explicit: bool,
    larger: bool = False,
    blocked: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "training_path_enabled": False,
        "default_training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "explicit_canary_allowed": explicit,
        "larger_manual_canary_allowed": larger,
        "blocked_reasons": list(blocked or []),
        "rollback_policy": {
            "fallback_authoritative": True,
            "fallback_backend": "pytorch_adamw",
            "disable_for_run_on_native_error": True,
            "disable_for_run_on_state_sync_failure": True,
            "disable_for_run_on_checkpoint_resume_mismatch": True,
            "rollback_on_resume_mismatch": True,
        },
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
