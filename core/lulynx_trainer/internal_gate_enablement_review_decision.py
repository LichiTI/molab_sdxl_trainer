"""Signed decision record for internal orchestrator gate enablement review.

This contract is report-only. It records a manual review decision after the
internal-gate review package is ready, while keeping the gate disabled and all
release claims closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_ENABLEMENT_REVIEW_DECISION = (
    "lulynx_internal_gate_enablement_review_decision_v0"
)
REVIEW_READY_STATUS = "ready_for_manual_internal_gate_review"
HOLD_DECISION = "internal_gate_enablement_review_hold_for_signed_review_default_off"
APPROVED_DECISION = "internal_gate_enablement_review_recorded_default_off"
REJECTED_DECISION = "internal_gate_enablement_review_rejected_default_off"


def build_lulynx_internal_gate_enablement_review_decision(
    *,
    internal_gate_enablement_review: Mapping[str, Any] | None = None,
    signed_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a signed review decision without enabling the internal gate."""

    review_package = dict(_mapping(internal_gate_enablement_review))
    review = dict(_mapping(signed_review))
    progress = _progress_gates(review_package, review)
    decision = _decision(progress, review)
    blockers = _blocked_reasons(progress, decision)
    ready = decision in {APPROVED_DECISION, REJECTED_DECISION} and not blockers
    approved = decision == APPROVED_DECISION and ready
    rejected = decision == REJECTED_DECISION and ready

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_ENABLEMENT_REVIEW_DECISION,
        "status": "decision_record_ready" if ready else "blocked",
        "ok": ready,
        "decision_record_ready": ready,
        "manual_review_required": True,
        "internal_gate_stays_disabled": True,
        "internal_gate_enablement_allowed": False,
        "approved_for_limited_non_release_probe_planning": approved,
        "rejected_for_default_off_hold": rejected,
        "decision": decision,
        "release_claim_allowed": False,
        "does_not_add_training_entrypoint": True,
        "does_not_start_gpu_work": True,
        "does_not_start_dataloader_iteration": True,
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "review_package_summary": _review_package_summary(review_package),
        "signed_review": _review_summary(review),
        "signed_review_template": _review_template(),
        "progress_gates": progress,
        "recommended_next_step": _recommended_next_step(decision),
        "notes": [
            "This decision record does not enable the internal gate.",
            "Approval only records that a limited non-release probe review path may be prepared later.",
            "Batch2/4/8 release probes and release claims remain blocked by separate evidence gates.",
        ],
    }


def _progress_gates(review_package: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    deferred = _string_list(review_package.get("deferred_release_probe_blockers"))
    return {
        "review_package_present": bool(review_package),
        "review_package_ready": (
            bool(review_package.get("passed"))
            and str(review_package.get("status") or "") == REVIEW_READY_STATUS
        ),
        "review_package_default_off": (
            not bool(review_package.get("internal_gate_enablement_allowed"))
            and not bool(review_package.get("release_claim_allowed"))
        ),
        "review_package_gate_still_disabled": bool(review_package.get("checks", {}).get("internal_gate_still_disabled")),
        "review_package_execution_path_not_enabled": bool(
            review_package.get("checks", {}).get("behavior_equivalent_execution_path_not_enabled")
        ),
        "review_package_batch2_release_probe_still_blocked": bool(
            review_package.get("checks", {}).get("batch2_release_probe_still_blocked")
        ),
        "review_package_deferred_probe_blockers_visible": bool(deferred),
        "signed_review_present": bool(review),
        "requested_scope_valid": str(review.get("requested_scope") or "") == "training_step_orchestrator_internal_gate_enablement_review",
        "manual_review_only_ack": bool(review.get("acknowledge_manual_review_only", False)),
        "default_off_ack": bool(review.get("acknowledge_internal_gate_stays_disabled", False)),
        "release_claim_closed_ack": bool(review.get("acknowledge_release_claim_stays_closed", False)),
        "review_package_ack": bool(review.get("acknowledge_review_package_ready", False)),
        "batch2_blocked_ack": bool(review.get("acknowledge_batch2_4_8_release_probe_still_blocked", False)),
        "limited_probe_only_ack": bool(review.get("acknowledge_approval_is_for_limited_non_release_probe_planning_only", False)),
        "gate_enable_not_requested": not bool(review.get("approve_turn_internal_gate_on_now", False)),
        "training_entrypoint_not_requested": not bool(review.get("approve_new_training_entrypoint", False)),
        "release_claim_not_requested": not bool(review.get("approve_release_claims", False)),
    }


def _decision(progress: Mapping[str, bool], review: Mapping[str, Any]) -> str:
    if not bool(progress.get("signed_review_present")):
        return HOLD_DECISION
    required = (
        "review_package_ready",
        "review_package_default_off",
        "review_package_gate_still_disabled",
        "review_package_execution_path_not_enabled",
        "review_package_batch2_release_probe_still_blocked",
        "requested_scope_valid",
        "manual_review_only_ack",
        "default_off_ack",
        "release_claim_closed_ack",
        "review_package_ack",
        "batch2_blocked_ack",
        "limited_probe_only_ack",
        "gate_enable_not_requested",
        "training_entrypoint_not_requested",
        "release_claim_not_requested",
    )
    if any(not bool(progress.get(name, False)) for name in required):
        return HOLD_DECISION
    if bool(review.get("approve_prepare_limited_non_release_probe_plan", False)):
        return APPROVED_DECISION
    return REJECTED_DECISION


def _blocked_reasons(progress: Mapping[str, bool], decision: str) -> list[str]:
    blockers: list[str] = []
    if not bool(progress.get("review_package_present")):
        blockers.append("internal_gate_enablement_review_missing")
    if not bool(progress.get("review_package_ready")):
        blockers.append("internal_gate_enablement_review_not_ready")
    if not bool(progress.get("review_package_default_off")):
        blockers.append("internal_gate_enablement_review_default_off_violation")
    if not bool(progress.get("review_package_gate_still_disabled")):
        blockers.append("internal_gate_enablement_review_gate_not_disabled")
    if not bool(progress.get("review_package_execution_path_not_enabled")):
        blockers.append("internal_gate_enablement_review_execution_path_enabled")
    if not bool(progress.get("review_package_batch2_release_probe_still_blocked")):
        blockers.append("internal_gate_enablement_review_batch2_release_probe_not_blocked")
    if decision == HOLD_DECISION and not bool(progress.get("signed_review_present")):
        blockers.append("signed_internal_gate_enablement_review_missing")
    if bool(progress.get("signed_review_present")):
        if not bool(progress.get("requested_scope_valid")):
            blockers.append("signed_internal_gate_enablement_review_scope_invalid")
        if not bool(progress.get("manual_review_only_ack")):
            blockers.append("signed_internal_gate_enablement_review_manual_review_ack_missing")
        if not bool(progress.get("default_off_ack")):
            blockers.append("signed_internal_gate_enablement_review_default_off_ack_missing")
        if not bool(progress.get("release_claim_closed_ack")):
            blockers.append("signed_internal_gate_enablement_review_release_claim_ack_missing")
        if not bool(progress.get("review_package_ack")):
            blockers.append("signed_internal_gate_enablement_review_package_ack_missing")
        if not bool(progress.get("batch2_blocked_ack")):
            blockers.append("signed_internal_gate_enablement_review_batch2_blocked_ack_missing")
        if not bool(progress.get("limited_probe_only_ack")):
            blockers.append("signed_internal_gate_enablement_review_limited_probe_ack_missing")
        if not bool(progress.get("gate_enable_not_requested")):
            blockers.append("signed_internal_gate_enablement_review_requests_gate_enablement")
        if not bool(progress.get("training_entrypoint_not_requested")):
            blockers.append("signed_internal_gate_enablement_review_requests_new_training_entrypoint")
        if not bool(progress.get("release_claim_not_requested")):
            blockers.append("signed_internal_gate_enablement_review_requests_release_claim")
    return _dedupe(blockers)


def _review_package_summary(review_package: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(review_package),
        "status": str(review_package.get("status") or ""),
        "passed": bool(review_package.get("passed")),
        "internal_gate_enablement_allowed": bool(review_package.get("internal_gate_enablement_allowed")),
        "release_claim_allowed": bool(review_package.get("release_claim_allowed")),
        "deferred_release_probe_blocker_count": len(
            _string_list(review_package.get("deferred_release_probe_blockers"))
        ),
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(review),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_prepare_limited_non_release_probe_plan": bool(
            review.get("approve_prepare_limited_non_release_probe_plan")
        ),
        "approve_turn_internal_gate_on_now": bool(review.get("approve_turn_internal_gate_on_now")),
        "approve_new_training_entrypoint": bool(review.get("approve_new_training_entrypoint")),
        "approve_release_claims": bool(review.get("approve_release_claims")),
        "reviewer": str(review.get("reviewer") or ""),
    }


def _review_template() -> dict[str, Any]:
    return {
        "requested_scope": "training_step_orchestrator_internal_gate_enablement_review",
        "reviewer": "",
        "acknowledge_review_package_ready": False,
        "acknowledge_internal_gate_stays_disabled": False,
        "acknowledge_release_claim_stays_closed": False,
        "acknowledge_batch2_4_8_release_probe_still_blocked": False,
        "acknowledge_manual_review_only": False,
        "acknowledge_approval_is_for_limited_non_release_probe_planning_only": False,
        "approve_prepare_limited_non_release_probe_plan": False,
        "approve_turn_internal_gate_on_now": False,
        "approve_new_training_entrypoint": False,
        "approve_release_claims": False,
        "review_notes": "",
    }


def _recommended_next_step(decision: str) -> str:
    if decision == APPROVED_DECISION:
        return "prepare_limited_non_release_probe_plan_while_keeping_internal_gate_disabled"
    if decision == REJECTED_DECISION:
        return "keep_internal_gate_default_off_and_continue_collecting_evidence"
    return "collect_signed_internal_gate_enablement_review"


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
    "APPROVED_DECISION",
    "HOLD_DECISION",
    "LULYNX_INTERNAL_GATE_ENABLEMENT_REVIEW_DECISION",
    "REJECTED_DECISION",
    "build_lulynx_internal_gate_enablement_review_decision",
]
