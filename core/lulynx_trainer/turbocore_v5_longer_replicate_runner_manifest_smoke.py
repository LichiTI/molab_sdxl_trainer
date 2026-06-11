"""Smoke checks for V5-P30 manual longer-replicate run manifest."""

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

from core.turbocore_v5_longer_replicate_evidence_collector import (  # noqa: E402
    build_v5_longer_replicate_evidence_bundle,
)
from core.turbocore_v5_longer_replicate_failure_history_gate import (  # noqa: E402
    build_v5_longer_replicate_failure_history_gate,
)
from core.turbocore_v5_longer_replicate_runner_manifest import (  # noqa: E402
    build_v5_longer_replicate_runner_manifest,
)
from core.turbocore_v5_next_stage_review_decision import (  # noqa: E402
    build_v5_next_stage_review_decision,
)
from core.turbocore_v5_owner_next_stage_package import (  # noqa: E402
    build_v5_owner_next_stage_package,
)


DEFAULT_OFF_FIELDS = (
    "default_training_path_enabled",
    "training_path_enabled",
    "default_rollout_allowed",
    "auto_rollout_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
)


def run_smoke() -> dict[str, Any]:
    p29_ready = _p29_package(ready=True)
    ready = build_v5_longer_replicate_runner_manifest(owner_next_stage_package=p29_ready)
    assert ready["ok"] is True, ready
    assert ready["run_manifest_ready"] is True, ready
    assert ready["explicit_run_plan_ready"] is True, ready
    assert ready["manual_run_required"] is True, ready
    assert ready["training_launch_allowed"] is False, ready
    assert ready["auto_launch_allowed"] is False, ready
    assert ready["runs_dispatched"] is False, ready
    _assert_default_off(ready)
    assert ready["post_manifest_request_fields"] == {}, ready
    assert ready["run_plan"]["run_count"] == 5, ready
    assert len(ready["run_plan"]["runs"]) == 5, ready
    assert ready["collector_followup"]["thresholds"]["min_runs"] == 5, ready
    assert "native_update_dispatch_promotion_perf" in ready["run_plan"]["cases"], ready

    missing = build_v5_longer_replicate_runner_manifest()
    _assert_blocked(missing, "p29")

    p29_blocked = build_v5_longer_replicate_runner_manifest(
        owner_next_stage_package=_p29_package(ready=False)
    )
    _assert_blocked(p29_blocked, "p29")

    default_violation = build_v5_longer_replicate_runner_manifest(
        owner_next_stage_package={**p29_ready, "default_training_path_enabled": True}
    )
    _assert_blocked(default_violation, "default")

    request_violation = build_v5_longer_replicate_runner_manifest(
        owner_next_stage_package={**p29_ready, "request_adapter_mapping_allowed": True}
    )
    _assert_blocked(request_violation, "request", "adapter")

    short_plan = build_v5_longer_replicate_runner_manifest(
        owner_next_stage_package=p29_ready,
        run_plan_options={"min_runs": 2, "steps": 128},
    )
    _assert_blocked(short_plan, "runs", "steps")

    auto_launch = build_v5_longer_replicate_runner_manifest(
        owner_next_stage_package=p29_ready,
        run_plan_options={"auto_launch": True},
    )
    _assert_blocked(auto_launch, "auto", "launch")

    missing_cases = build_v5_longer_replicate_runner_manifest(
        owner_next_stage_package=p29_ready,
        run_plan_options={"cases": ["baseline_phase"]},
    )
    _assert_blocked(missing_cases, "cases")

    rollback_unacked = build_v5_longer_replicate_runner_manifest(
        owner_next_stage_package=p29_ready,
        run_plan_options={"rollback_policy_acknowledged": False},
    )
    _assert_blocked(rollback_unacked, "rollback")

    custom = build_v5_longer_replicate_runner_manifest(
        owner_next_stage_package=p29_ready,
        run_plan_options={
            "plan_id": "custom_v5_p30_fixture",
            "min_runs": 6,
            "steps": 1024,
            "source_data": "sucai\\6_lulu",
            "seeds": [11, 12, 13, 14, 15, 16],
        },
    )
    assert custom["ok"] is True, custom
    assert custom["run_plan"]["run_count"] == 6, custom
    assert custom["run_plan"]["runs"][0]["seed"] == 11, custom
    assert custom["collector_followup"]["thresholds"]["min_representative_steps"] == 1024, custom
    _assert_default_off(custom)

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p30_longer_replicate_runner_manifest_smoke",
        "ok": True,
        "ready_decision": ready["decision"],
        "ready_run_count": ready["run_plan"]["run_count"],
        "missing_blocked_reasons": _blocked_reasons(missing),
        "p29_blocked_reasons": _blocked_reasons(p29_blocked),
        "default_violation_blocked_reasons": _blocked_reasons(default_violation),
        "request_violation_blocked_reasons": _blocked_reasons(request_violation),
        "short_plan_blocked_reasons": _blocked_reasons(short_plan),
        "auto_launch_blocked_reasons": _blocked_reasons(auto_launch),
        "missing_cases_blocked_reasons": _blocked_reasons(missing_cases),
        "rollback_unacked_blocked_reasons": _blocked_reasons(rollback_unacked),
        "custom_run_count": custom["run_plan"]["run_count"],
    }


def _p29_package(*, ready: bool) -> dict[str, Any]:
    p28 = build_v5_longer_replicate_evidence_bundle(
        run_payloads=_run_payloads(count=5 if ready else 2),
        min_runs=5,
        min_representative_steps=768,
        min_end_to_end_speedup=1.05,
    )
    p26 = build_v5_longer_replicate_failure_history_gate(
        owner_rollout_review_decision=_p25_decision(),
        longer_replicate_evidence=p28,
        failure_history=_empty_history("failure"),
        rollback_history=_empty_history("rollback"),
        min_longer_replicate_runs=5,
        min_end_to_end_speedup=1.05,
    )
    p27 = build_v5_next_stage_review_decision(
        p26_gate=p26,
        next_stage_review=_next_stage_review(approve=ready),
    )
    return build_v5_owner_next_stage_package(
        p28_evidence_bundle=p28,
        p26_gate=p26,
        p27_decision=p27,
    )


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in DEFAULT_OFF_FIELDS:
        assert report[field] is False, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    _assert_default_off(report)
    reasons = [reason.lower() for reason in _blocked_reasons(report)]
    assert reasons, report
    for fragment in fragments:
        assert any(fragment.lower() in reason for reason in reasons), report


def _blocked_reasons(report: dict[str, Any]) -> list[str]:
    value = report.get("blocked_reasons") or report.get("promotion_blockers") or []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _run_payloads(*, count: int) -> list[dict[str, Any]]:
    speedups = [1.10, 1.08, 1.07, 1.09, 1.11][:count]
    return [_run_payload(index=index, speedup=speedup) for index, speedup in enumerate(speedups, start=1)]


def _run_payload(*, index: int, speedup: float) -> dict[str, Any]:
    baseline_ms = 1000.0
    native_ms = baseline_ms / float(speedup)
    return {
        "schema_version": 1,
        "run_id": f"v5_p30_longer_replicate_fixture_{index:02d}",
        "status": "completed",
        "ok": True,
        "success": True,
        "ready": True,
        "steps_completed": 768,
        "representative_steps": 768,
        "end_to_end_speedup": float(speedup),
        "representative_end_to_end_speedup": float(speedup),
        "speedup_vs_baseline": float(speedup),
        "summary": {
            "success": True,
            "steps_completed": 768,
            "baseline_mean_step_ms": baseline_ms,
            "native_mean_step_ms": native_ms,
            "native_dispatch_executed": True,
            "checkpoint_resume_native_state_boundary": True,
            "rollback_events": [],
        },
        "rollback_events": [],
        "failure_events": [],
        "blocked_reasons": [],
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "post_gate_request_fields": {},
    }


def _p25_decision() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v5_owner_rollout_review_decision_v0",
        "gate": "v5_owner_rollout_review_decision",
        "ok": True,
        "decision_record_ready": True,
        "owner_rollout_review_recorded": True,
        "owner_rollout_review_signed": True,
        "approved_for_next_stage": True,
        "rejected_for_default_off_hold": False,
        "rollback_required": False,
        "rollout_decision": "owner_rollout_review_recorded_default_off",
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "blocked_reasons": [],
        "promotion_blockers": [],
    }


def _next_stage_review(*, approve: bool) -> dict[str, Any]:
    return {
        "reviewer": "v5_p30_fixture",
        "reviewed_at": "2026-06-01",
        "requested_scope": "manual_next_stage_review",
        "approve_next_stage": bool(approve),
        "approve_next_stage_review": bool(approve),
        "approve_next_stage_manual_experiment": bool(approve),
        "approve_keep_p26_evidence": bool(approve),
        "approve_keep_longer_replicate_failure_history_evidence": bool(approve),
        "approve_default_training": False,
        "approve_default_training_path_enabled": False,
        "approve_default_rollout": False,
        "approve_default_rollout_allowed": False,
        "approve_auto_rollout": False,
        "approve_auto_rollout_allowed": False,
        "approve_request_adapter_mapping_allowed": False,
        "approve_request_fields_emitted": False,
        "acknowledge_no_request_adapter_mapping": True,
        "acknowledge_p26_gate_ready": True,
        "acknowledge_runtime_evidence_complete": True,
        "acknowledge_manual_review_only": True,
        "acknowledge_longer_replicate_evidence": True,
        "acknowledge_failure_history_clear": True,
        "acknowledge_rollback_history_clear": True,
    }


def _empty_history(kind: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "history": f"v5_p30_empty_{kind}_history_fixture",
        "events": [],
        "open_failure_count": 0,
        "cooldown_active": False,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
