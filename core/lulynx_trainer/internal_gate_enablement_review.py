"""Manual review package for staged internal-gate enablement.

This module is report-only. It summarizes whether the current batch1 evidence
is strong enough to start a human review of the internal orchestrator gate.
It does not enable the gate, start training, or authorize release claims.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_ENABLEMENT_REVIEW = "lulynx_internal_gate_enablement_review_v0"
_REVIEW_ACTION_ID = "review_lulynx_real_gpu_batch1_golden_before_internal_gate_enablement"
_REVIEW_ACTION_READY = "ready_for_internal_gate_enablement_review"


def build_lulynx_internal_gate_enablement_review(
    *,
    real_gpu_batch1_golden_evidence: Mapping[str, Any] | None = None,
    training_pipeline_refactor_readiness_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed package for manual internal-gate review."""

    golden = dict(_mapping(real_gpu_batch1_golden_evidence))
    readiness = dict(_mapping(training_pipeline_refactor_readiness_evidence))
    readiness_slice = _mapping(readiness.get("training_step_orchestrator_slice"))
    readiness_actions = _mapping(readiness.get("action_statuses"))
    golden_slice = _mapping(golden.get("training_step_orchestrator_slice"))
    deferred_release_probe_blockers = _string_list(readiness.get("blockers"))
    release_claim_leaks = _release_claim_leaks(golden=golden, readiness=readiness)

    checks = {
        "real_gpu_batch1_golden_passed": bool(golden.get("passed")),
        "refactor_readiness_ready": bool(readiness.get("ready_for_internal_orchestrator_gate")),
        "refactor_review_action_ready": (
            str(readiness_actions.get(_REVIEW_ACTION_ID) or "") == _REVIEW_ACTION_READY
        ),
        "internal_gate_still_disabled": (
            not bool(golden_slice.get("internal_gate_enabled"))
            and not bool(readiness.get("internal_orchestrator_gate_enabled"))
            and not bool(readiness_slice.get("internal_gate_enabled"))
        ),
        "behavior_equivalent_execution_path_not_enabled": not bool(
            readiness_slice.get("can_execute_behavior_equivalent_slice")
        ),
        "batch2_release_probe_still_blocked": not bool(readiness.get("ready_for_batch2_4_8_release_probe")),
        "release_claim_closed": not release_claim_leaks,
        "no_new_training_entrypoint_claim_preserved": (
            bool(golden.get("does_not_add_training_entrypoint"))
            and bool(readiness.get("does_not_add_training_entrypoint"))
        ),
    }
    blockers = _blockers_from_checks(checks)
    if not golden:
        blockers.append("real_gpu_batch1_golden_evidence_missing")
    if not readiness:
        blockers.append("training_pipeline_refactor_readiness_evidence_missing")
    blockers.extend(
        f"real_gpu_batch1_golden_evidence:{item}" for item in _string_list(golden.get("blockers"))
    )
    blockers.extend(f"release_claim_leak:{item}" for item in release_claim_leaks)
    blockers = _dedupe(blockers)
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_ENABLEMENT_REVIEW,
        "status": "ready_for_manual_internal_gate_review" if ready else "blocked",
        "passed": ready,
        "manual_review_required": True,
        "safe_to_auto_start": False,
        "internal_gate_enablement_allowed": False,
        "release_claim_allowed": False,
        "does_not_add_training_entrypoint": True,
        "does_not_start_gpu_work": True,
        "does_not_start_dataloader_iteration": True,
        "checks": checks,
        "blockers": blockers,
        "deferred_release_probe_blockers": deferred_release_probe_blockers,
        "real_gpu_batch1_golden_summary": _golden_summary(golden),
        "training_pipeline_refactor_readiness_summary": _readiness_summary(readiness),
        "review_template": _review_template(),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _golden_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "release_claim_allowed": bool(report.get("release_claim_allowed")),
        "blocker_count": len(_string_list(report.get("blockers"))),
        "multi_batch_promotion_gate_blockers": _string_list(report.get("multi_batch_promotion_gate_blockers")),
    }


def _readiness_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    actions = _mapping(report.get("action_statuses"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "ready_for_internal_orchestrator_gate": bool(report.get("ready_for_internal_orchestrator_gate")),
        "internal_orchestrator_gate_enabled": bool(report.get("internal_orchestrator_gate_enabled")),
        "ready_for_batch2_4_8_release_probe": bool(report.get("ready_for_batch2_4_8_release_probe")),
        "review_action_status": str(actions.get(_REVIEW_ACTION_ID) or ""),
        "release_claim_allowed": bool(report.get("release_claim_allowed")),
        "blocker_count": len(_string_list(report.get("blockers"))),
    }


def _review_template() -> dict[str, Any]:
    return {
        "scope": "training_step_orchestrator_internal_gate_enablement_review",
        "intent": "hold_for_more_evidence",
        "acknowledge_real_gpu_batch1_golden_reviewed": False,
        "acknowledge_refactor_readiness_reviewed": False,
        "acknowledge_internal_gate_still_disabled": False,
        "acknowledge_behavior_equivalent_execution_path_not_enabled": False,
        "acknowledge_batch2_4_8_release_probe_still_blocked": False,
        "acknowledge_no_new_training_entrypoint": False,
        "acknowledge_release_claim_stays_closed": False,
        "acknowledge_manual_review_only": False,
        "approve_internal_gate_enablement_for_limited_non_release_probe": False,
        "review_notes": "",
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "collect_manual_internal_gate_enablement_review_record",
            "keep_internal_gate_disabled_until_signed_review",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_internal_gate_enablement_review_inputs_before_manual_signoff"]
    if any("real_gpu_batch1_golden" in item for item in blockers):
        actions.append("refresh_real_gpu_batch1_golden_evidence")
    if any("training_pipeline_refactor_readiness_evidence" in item for item in blockers):
        actions.append("refresh_training_pipeline_refactor_readiness_evidence")
    if any("release_claim_leak" in item for item in blockers):
        actions.append("close_release_claim_leaks_before_internal_gate_review")
    return _dedupe(actions)


def _release_claim_leaks(*, golden: Mapping[str, Any], readiness: Mapping[str, Any]) -> list[str]:
    leaks: list[str] = []
    for name, payload in (
        ("real_gpu_batch1_golden_evidence", golden),
        ("training_pipeline_refactor_readiness_evidence", readiness),
    ):
        if bool(_mapping(payload).get("release_claim_allowed")):
            leaks.append(name)
    return leaks


def _blockers_from_checks(checks: Mapping[str, bool]) -> list[str]:
    return [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


__all__ = [
    "LULYNX_INTERNAL_GATE_ENABLEMENT_REVIEW",
    "build_lulynx_internal_gate_enablement_review",
]
