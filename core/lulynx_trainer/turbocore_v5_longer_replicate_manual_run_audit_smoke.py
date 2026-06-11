"""Smoke checks for V5-P31 manual longer-replicate run audit."""

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
from core.turbocore_v5_longer_replicate_manual_run_audit import (  # noqa: E402
    build_v5_longer_replicate_manual_run_audit,
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

DEFAULT_EXPECTED_CASES = ["baseline_phase", "native_update_dispatch_promotion_perf"]


def run_smoke() -> dict[str, Any]:
    manifest = _runner_manifest()
    run_payloads = _planned_run_payloads(manifest)
    ready = build_v5_longer_replicate_manual_run_audit(
        runner_manifest=manifest,
        run_payloads=run_payloads,
    )
    assert ready["ok"] is True, ready
    assert ready["manual_run_audit_ready"] is True, ready
    assert ready["collector_evidence_ready"] is True, ready
    assert ready["plan_match_summary"]["matched_run_count"] == 5, ready
    assert ready["collector_bundle"]["run_count"] == 5, ready
    assert ready["collector_invocation"]["thresholds"]["min_representative_steps"] == 768, ready
    assert len(ready["collector_ready_run_payloads"]) == 5, ready
    assert ready["decision"] == "longer_replicate_manual_run_audit_ready_default_off", ready
    assert ready["training_launch_allowed"] is False, ready
    assert ready["runs_dispatched"] is False, ready
    _assert_default_off(ready)

    collector_ready_only = build_v5_longer_replicate_manual_run_audit(
        runner_manifest=manifest,
        run_payloads=run_payloads,
        emit_collector_bundle=False,
    )
    assert collector_ready_only["ok"] is True, collector_ready_only
    assert collector_ready_only["p28_collector_bundle"] == {}, collector_ready_only

    missing_manifest = build_v5_longer_replicate_manual_run_audit(run_payloads=run_payloads)
    _assert_blocked(missing_manifest, "manifest")

    missing_results = build_v5_longer_replicate_manual_run_audit(runner_manifest=manifest)
    _assert_blocked(missing_results, "result")

    missing_one = build_v5_longer_replicate_manual_run_audit(
        runner_manifest=manifest,
        run_payloads=run_payloads[:-1],
    )
    _assert_blocked(missing_one, "missing")

    unexpected = build_v5_longer_replicate_manual_run_audit(
        runner_manifest=manifest,
        run_payloads=[*run_payloads, _run_payload(run_id="unexpected_run", speedup=1.07, steps=768)],
    )
    _assert_blocked(unexpected, "unexpected")

    duplicate = build_v5_longer_replicate_manual_run_audit(
        runner_manifest=manifest,
        run_payloads=[run_payloads[0], *run_payloads],
    )
    _assert_blocked(duplicate, "duplicate")

    identity_conflict = build_v5_longer_replicate_manual_run_audit(
        runner_manifest=manifest,
        run_payloads=[
            _run_payload(
                run_id=str(manifest["run_plan"]["runs"][0]["run_id"]),
                matrix_summary_path=str(manifest["run_plan"]["runs"][1]["matrix_summary_path"]),
                speedup=1.10,
                steps=768,
            ),
            *run_payloads[1:],
        ],
    )
    _assert_blocked(identity_conflict, "conflict")

    case_mismatch = build_v5_longer_replicate_manual_run_audit(
        runner_manifest=manifest,
        run_payloads=_replace_payload(run_payloads, 0, cases=["baseline_phase"]),
    )
    _assert_blocked(case_mismatch, "case")

    cases_missing = build_v5_longer_replicate_manual_run_audit(
        runner_manifest=manifest,
        run_payloads=_replace_payload(run_payloads, 0, cases=[]),
    )
    _assert_blocked(cases_missing, "case")

    rollback = build_v5_longer_replicate_manual_run_audit(
        runner_manifest=manifest,
        run_payloads=_replace_payload(run_payloads, 1, rollback_events=["native_error"]),
    )
    _assert_blocked(rollback, "rollback")

    slow = build_v5_longer_replicate_manual_run_audit(
        runner_manifest=manifest,
        run_payloads=_replace_payload(run_payloads, 2, speedup=1.01),
    )
    _assert_blocked(slow, "speedup")

    default_violation = build_v5_longer_replicate_manual_run_audit(
        runner_manifest=manifest,
        run_payloads=_replace_payload(run_payloads, 3, default_request_violation=True),
    )
    _assert_blocked(default_violation, "default", "request")

    manifest_launch_violation = build_v5_longer_replicate_manual_run_audit(
        runner_manifest={**manifest, "training_launch_allowed": True},
        run_payloads=run_payloads,
    )
    _assert_blocked(manifest_launch_violation, "training", "launch")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p31_longer_replicate_manual_run_audit_smoke",
        "ok": True,
        "ready_decision": ready["decision"],
        "matched_run_count": ready["plan_match_summary"]["matched_run_count"],
        "collector_ready": ready["collector_evidence_ready"],
        "collector_ready_only_embedded_bundle": bool(collector_ready_only["p28_collector_bundle"]),
        "missing_manifest_blocked_reasons": _blocked_reasons(missing_manifest),
        "missing_results_blocked_reasons": _blocked_reasons(missing_results),
        "missing_one_blocked_reasons": _blocked_reasons(missing_one),
        "unexpected_blocked_reasons": _blocked_reasons(unexpected),
        "duplicate_blocked_reasons": _blocked_reasons(duplicate),
        "identity_conflict_blocked_reasons": _blocked_reasons(identity_conflict),
        "case_mismatch_blocked_reasons": _blocked_reasons(case_mismatch),
        "cases_missing_blocked_reasons": _blocked_reasons(cases_missing),
        "rollback_blocked_reasons": _blocked_reasons(rollback),
        "slow_blocked_reasons": _blocked_reasons(slow),
        "default_violation_blocked_reasons": _blocked_reasons(default_violation),
        "manifest_launch_violation_blocked_reasons": _blocked_reasons(manifest_launch_violation),
    }


def _runner_manifest() -> dict[str, Any]:
    return build_v5_longer_replicate_runner_manifest(owner_next_stage_package=_p29_package())


def _p29_package() -> dict[str, Any]:
    p28 = build_v5_longer_replicate_evidence_bundle(
        run_payloads=_fixture_replicates(),
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
        next_stage_review=_next_stage_review(),
    )
    return build_v5_owner_next_stage_package(
        p28_evidence_bundle=p28,
        p26_gate=p26,
        p27_decision=p27,
    )


def _planned_run_payloads(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    runs = manifest["run_plan"]["runs"]
    speedups = [1.10, 1.08, 1.07, 1.09, 1.11]
    return [
        _run_payload(
            run_id=str(run["run_id"]),
            matrix_summary_path=str(run["matrix_summary_path"]),
            speedup=speedups[index],
            steps=int(run["expected_steps"]),
        )
        for index, run in enumerate(runs)
    ]


def _fixture_replicates() -> list[dict[str, Any]]:
    return [_run_payload(run_id=f"v5_p31_fixture_{index}", speedup=speedup, steps=768) for index, speedup in enumerate([1.10, 1.08, 1.07, 1.09, 1.11], start=1)]


def _run_payload(
    *,
    run_id: str,
    speedup: float,
    steps: int,
    matrix_summary_path: str = "",
    rollback_events: list[str] | None = None,
    default_request_violation: bool = False,
    cases: list[str] | None = None,
) -> dict[str, Any]:
    baseline_ms = 1000.0
    native_ms = baseline_ms / float(speedup)
    rollback_values = list(rollback_events or [])
    default_enabled = bool(default_request_violation)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "matrix_summary_path": matrix_summary_path,
        "status": "completed",
        "ok": True,
        "success": True,
        "steps_completed": int(steps),
        "representative_steps": int(steps),
        "end_to_end_speedup": float(speedup),
        "representative_end_to_end_speedup": float(speedup),
        "speedup_vs_baseline": float(speedup),
        "summary": {
            "success": True,
            "steps_completed": int(steps),
            "baseline_mean_step_ms": baseline_ms,
            "native_mean_step_ms": native_ms,
            "native_dispatch_executed": True,
            "checkpoint_resume_native_state_boundary": True,
            "rollback_events": rollback_values,
        },
        "rollback_events": rollback_values,
        "failure_events": [],
        "blocked_reasons": [],
        "default_behavior_changed": default_enabled,
        "default_training_path_enabled": default_enabled,
        "training_path_enabled": default_enabled,
        "default_rollout_allowed": default_enabled,
        "auto_rollout_allowed": default_enabled,
        "request_adapter_mapping_allowed": default_enabled,
        "request_fields_emitted": default_enabled,
        "post_gate_request_fields": {"unexpected": True} if default_enabled else {},
        "cases": [
            {
                "case": {"name": str(case)},
                "summary": {"native_dispatch_executed": case == "native_update_dispatch_promotion_perf"},
            }
            for case in (list(cases) if cases is not None else DEFAULT_EXPECTED_CASES)
        ],
    }


def _replace_payload(
    payloads: list[dict[str, Any]],
    index: int,
    **updates: Any,
) -> list[dict[str, Any]]:
    out = [dict(item) for item in payloads]
    target = dict(out[index])
    target.update(
        _run_payload(
            run_id=str(target["run_id"]),
            matrix_summary_path=str(target.get("matrix_summary_path") or ""),
            speedup=float(updates.get("speedup", target["representative_end_to_end_speedup"])),
            steps=int(updates.get("steps", target["representative_steps"])),
            rollback_events=updates.get("rollback_events"),
            default_request_violation=bool(updates.get("default_request_violation", False)),
            cases=updates.get("cases"),
        )
    )
    out[index] = target
    return out


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


def _next_stage_review() -> dict[str, Any]:
    return {
        "reviewer": "v5_p31_fixture",
        "reviewed_at": "2026-06-01",
        "requested_scope": "manual_next_stage_review",
        "approve_next_stage": True,
        "approve_next_stage_review": True,
        "approve_next_stage_manual_experiment": True,
        "approve_keep_p26_evidence": True,
        "approve_keep_longer_replicate_failure_history_evidence": True,
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
        "history": f"v5_p31_empty_{kind}_history_fixture",
        "events": [],
        "open_failure_count": 0,
        "cooldown_active": False,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
