"""Smoke checks for V5-P32 P31 collector replay owner package."""

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

from core.turbocore_v5_longer_replicate_manual_run_audit import (  # noqa: E402
    build_v5_longer_replicate_manual_run_audit,
)
from core.turbocore_v5_p31_collector_replay_owner_package import (  # noqa: E402
    build_v5_p31_collector_replay_owner_package,
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
    p31_ready = _p31_audit()
    ready = build_v5_p31_collector_replay_owner_package(
        p31_manual_run_audit=p31_ready,
        owner_rollout_review_decision=_p25_decision(),
        next_stage_review=_next_stage_review(approve=True),
        failure_history=_empty_history("failure"),
        rollback_history=_empty_history("rollback"),
    )
    assert ready["ok"] is True, ready
    assert ready["p31_collector_replay_ready"] is True, ready
    assert ready["owner_next_stage_package_ready"] is True, ready
    assert ready["decision"] == "p31_collector_replay_owner_package_ready_default_off", ready
    assert ready["p29_owner_next_stage_package"]["package_ready"] is True, ready
    assert ready["p26_gate"]["ok"] is True, ready
    assert ready["p27_decision"]["ok"] is True, ready
    assert ready["training_launch_allowed"] is False, ready
    assert ready["runs_dispatched"] is False, ready
    _assert_default_off(ready)

    pending_review = build_v5_p31_collector_replay_owner_package(
        p31_manual_run_audit=p31_ready,
        owner_rollout_review_decision=_p25_decision(),
        failure_history=_empty_history("failure"),
        rollback_history=_empty_history("rollback"),
    )
    _assert_blocked(pending_review, "signed", "review")
    assert pending_review["ready_for_signed_next_stage_review"] is True, pending_review
    assert pending_review["decision"] == "p31_collector_replay_hold_for_signed_next_stage_review_default_off"

    missing_p25 = build_v5_p31_collector_replay_owner_package(
        p31_manual_run_audit=p31_ready,
        next_stage_review=_next_stage_review(approve=True),
        failure_history=_empty_history("failure"),
        rollback_history=_empty_history("rollback"),
    )
    _assert_blocked(missing_p25, "p26", "p25")

    p31_blocked = build_v5_p31_collector_replay_owner_package(
        p31_manual_run_audit=_p31_audit(rollback=True),
        owner_rollout_review_decision=_p25_decision(),
        next_stage_review=_next_stage_review(approve=True),
        failure_history=_empty_history("failure"),
        rollback_history=_empty_history("rollback"),
    )
    _assert_blocked(p31_blocked, "p31")

    collector_missing = build_v5_p31_collector_replay_owner_package(
        p31_manual_run_audit={**p31_ready, "collector_bundle": {}, "p28_collector_bundle": {}},
        owner_rollout_review_decision=_p25_decision(),
        next_stage_review=_next_stage_review(approve=True),
        failure_history=_empty_history("failure"),
        rollback_history=_empty_history("rollback"),
    )
    _assert_blocked(collector_missing, "collector", "bundle")

    rejected = build_v5_p31_collector_replay_owner_package(
        p31_manual_run_audit=p31_ready,
        owner_rollout_review_decision=_p25_decision(),
        next_stage_review=_next_stage_review(approve=False),
        failure_history=_empty_history("failure"),
        rollback_history=_empty_history("rollback"),
    )
    _assert_blocked(rejected, "p27")

    history_blocked = build_v5_p31_collector_replay_owner_package(
        p31_manual_run_audit=p31_ready,
        owner_rollout_review_decision=_p25_decision(),
        next_stage_review=_next_stage_review(approve=True),
        failure_history={"events": [{"event": "native_regression", "status": "open", "severity": "high"}]},
        rollback_history=_empty_history("rollback"),
    )
    _assert_blocked(history_blocked, "history")

    default_violation = build_v5_p31_collector_replay_owner_package(
        p31_manual_run_audit={**p31_ready, "default_rollout_allowed": True},
        owner_rollout_review_decision=_p25_decision(),
        next_stage_review=_next_stage_review(approve=True),
        failure_history=_empty_history("failure"),
        rollback_history=_empty_history("rollback"),
    )
    _assert_blocked(default_violation, "default")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p32_p31_collector_replay_owner_package_smoke",
        "ok": True,
        "ready": _summary(ready),
        "pending_review": _summary(pending_review),
        "missing_p25": _summary(missing_p25),
        "p31_blocked": _summary(p31_blocked),
        "collector_missing": _summary(collector_missing),
        "rejected": _summary(rejected),
        "history_blocked": _summary(history_blocked),
        "default_violation": _summary(default_violation),
    }


def _p31_audit(*, rollback: bool = False) -> dict[str, Any]:
    manifest = _p30_manifest()
    payloads = _run_payloads()
    if rollback:
        payloads[2] = {**payloads[2], "rollback_events": ["native_error"]}
        payloads[2]["summary"] = {**payloads[2]["summary"], "rollback_events": ["native_error"]}
    return build_v5_longer_replicate_manual_run_audit(
        runner_manifest=manifest,
        run_payloads=payloads,
    )


def _p30_manifest() -> dict[str, Any]:
    runs = []
    for index in range(1, 6):
        runs.append(
            {
                "run_id": f"v5_p32_manual_plan_run_{index:02d}",
                "matrix_summary_path": f"temp/v5_p32/run_{index:02d}/matrix_summary.json",
                "output_dir": f"temp/v5_p32/run_{index:02d}",
                "expected_cases": ["baseline_phase", "native_update_dispatch_promotion_perf"],
                "expected_steps": 768,
            }
        )
    return {
        "schema_version": 1,
        "ok": True,
        "run_manifest_ready": True,
        "explicit_run_plan_ready": True,
        "decision": "longer_replicate_runner_manifest_ready_default_off",
        "manual_run_required": True,
        "training_launch_allowed": False,
        "auto_launch_allowed": False,
        "runs_dispatched": False,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "post_manifest_request_fields": {},
        "collector_followup": {
            "thresholds": {
                "min_runs": 5,
                "min_representative_steps": 768,
                "min_end_to_end_speedup": 1.05,
                "max_speedup_spread_ratio": 0.30,
            }
        },
        "run_plan": {
            "run_count": 5,
            "required_thresholds": {
                "min_runs": 5,
                "min_representative_steps": 768,
                "min_end_to_end_speedup": 1.05,
                "max_speedup_spread_ratio": 0.30,
            },
            "runs": runs,
        },
        "blocked_reasons": [],
    }


def _run_payloads() -> list[dict[str, Any]]:
    speedups = [1.10, 1.08, 1.07, 1.09, 1.11]
    return [_run_payload(index=index, speedup=speedup) for index, speedup in enumerate(speedups, start=1)]


def _run_payload(*, index: int, speedup: float) -> dict[str, Any]:
    baseline_ms = 1000.0
    native_ms = baseline_ms / float(speedup)
    return {
        "schema_version": 1,
        "run_id": f"v5_p32_manual_plan_run_{index:02d}",
        "matrix_summary_path": f"temp/v5_p32/run_{index:02d}/matrix_summary.json",
        "status": "completed",
        "ok": True,
        "success": True,
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
        "cases": [
            {"case": {"name": "baseline_phase"}, "summary": {"native_dispatch_executed": False}},
            {
                "case": {"name": "native_update_dispatch_promotion_perf"},
                "summary": {"native_dispatch_executed": True},
            },
        ],
    }


def _p25_decision() -> dict[str, Any]:
    return {
        "schema_version": 1,
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
        "reviewer": "v5_p32_fixture",
        "reviewed_at": "2026-06-01",
        "requested_scope": "manual_next_stage_review",
        "approve_next_stage": bool(approve),
        "approve_next_stage_review": bool(approve),
        "approve_next_stage_manual_experiment": bool(approve),
        "approve_keep_p26_evidence": bool(approve),
        "approve_keep_longer_replicate_failure_history_evidence": bool(approve),
        "approve_default_training_path_enabled": False,
        "approve_default_rollout_allowed": False,
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
        "history": f"v5_p32_empty_{kind}_history_fixture",
        "events": [],
        "open_failure_count": 0,
        "cooldown_active": False,
    }


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


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok", False)),
        "decision": str(report.get("decision") or ""),
        "p31_collector_replay_ready": bool(report.get("p31_collector_replay_ready", False)),
        "owner_next_stage_package_ready": bool(report.get("owner_next_stage_package_ready", False)),
        "ready_for_signed_next_stage_review": bool(report.get("ready_for_signed_next_stage_review", False)),
        "blocked_reasons": _blocked_reasons(report),
    }


def _blocked_reasons(report: dict[str, Any]) -> list[str]:
    value = report.get("blocked_reasons") or report.get("promotion_blockers") or []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
