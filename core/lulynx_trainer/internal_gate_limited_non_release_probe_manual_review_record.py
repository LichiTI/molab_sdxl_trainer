"""Signed manual review record for a limited non-release probe.

This contract records a human decision after the manual review packet is ready.
It remains default-off and report-only: no gate enablement, no training start,
no new entrypoint, and no release claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_MANUAL_REVIEW_RECORD = (
    "lulynx_internal_gate_limited_non_release_probe_manual_review_record_v0"
)
READY_PACKET_STATUS = "ready_for_manual_batch1_non_release_probe_review_packet"
HOLD_DECISION = (
    "internal_gate_limited_non_release_probe_manual_review_hold_for_signed_review_default_off"
)
APPROVED_DECISION = (
    "internal_gate_limited_non_release_probe_manual_review_recorded_default_off"
)
REJECTED_DECISION = (
    "internal_gate_limited_non_release_probe_manual_review_rejected_default_off"
)
SCOPE = "training_step_orchestrator_internal_gate_limited_non_release_probe_manual_review_packet"


def build_lulynx_internal_gate_limited_non_release_probe_manual_review_record(
    *,
    internal_gate_limited_non_release_probe_manual_review_packet: Mapping[str, Any] | None = None,
    signed_manual_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a signed manual review without changing runtime behavior."""

    packet = dict(_mapping(internal_gate_limited_non_release_probe_manual_review_packet))
    review = dict(_mapping(signed_manual_review))
    progress = _progress_gates(packet=packet, review=review)
    decision = _decision(progress=progress, review=review)
    blockers = _blocked_reasons(packet=packet, progress=progress, decision=decision)
    ready = decision in {APPROVED_DECISION, REJECTED_DECISION} and not blockers
    approved = decision == APPROVED_DECISION and ready
    rejected = decision == REJECTED_DECISION and ready

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_MANUAL_REVIEW_RECORD,
        "status": "decision_record_ready" if ready else "blocked",
        "ok": ready,
        "decision_record_ready": ready,
        "manual_review_required": True,
        "internal_gate_stays_disabled": True,
        "internal_gate_enablement_allowed": False,
        "approved_for_followup_manual_probe_preparation": approved,
        "rejected_for_default_off_hold": rejected,
        "decision": decision,
        "release_claim_allowed": False,
        "does_not_add_training_entrypoint": True,
        "does_not_start_gpu_work": True,
        "does_not_start_dataloader_iteration": True,
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "manual_review_packet_summary": _packet_summary(packet),
        "signed_manual_review": _review_summary(review),
        "signed_manual_review_template": _review_template(),
        "progress_gates": progress,
        "recommended_next_step": _recommended_next_step(decision),
        "notes": [
            "This record does not enable the internal gate or start any probe.",
            "Approval only records that later manual probe preparation may continue under default-off boundaries.",
            "Batch2/4/8 release probes and release claims remain blocked by separate evidence gates.",
        ],
    }


def _progress_gates(*, packet: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    review_packet = _mapping(packet.get("review_packet"))
    forbidden = set(_string_list(review_packet.get("forbidden_approvals")))
    deferred = _string_list(review_packet.get("deferred_release_probe_blockers"))
    return {
        "manual_review_packet_present": bool(packet),
        "manual_review_packet_ready": bool(packet.get("passed"))
        and str(packet.get("status") or "") == READY_PACKET_STATUS,
        "manual_review_packet_default_off": (
            not bool(packet.get("internal_gate_enablement_allowed"))
            and not bool(packet.get("release_claim_allowed"))
        ),
        "manual_review_packet_batch2_still_blocked": "approve_batch2_4_8_release_probe" in forbidden,
        "manual_review_packet_gate_enablement_still_blocked": "turn_internal_gate_on_now" in forbidden,
        "manual_review_packet_training_start_still_blocked": "start_training_now" in forbidden,
        "manual_review_packet_release_claim_still_blocked": "approve_release_claim" in forbidden,
        "manual_review_packet_deferred_probe_blockers_visible": bool(deferred),
        "signed_manual_review_present": bool(review),
        "requested_scope_valid": str(review.get("requested_scope") or "") == SCOPE,
        "manual_review_only_ack": bool(review.get("acknowledge_manual_review_only", False)),
        "packet_ready_ack": bool(review.get("acknowledge_manual_review_packet_ready", False)),
        "default_off_ack": bool(review.get("acknowledge_internal_gate_stays_disabled", False)),
        "batch1_only_ack": bool(review.get("acknowledge_batch1_non_release_probe_only", False)),
        "batch2_blocked_ack": bool(review.get("acknowledge_batch2_4_8_release_probe_still_blocked", False)),
        "release_claim_closed_ack": bool(review.get("acknowledge_release_claim_stays_closed", False)),
        "gate_enable_not_requested": not bool(review.get("approve_turn_internal_gate_on_now", False)),
        "training_start_not_requested": not bool(review.get("approve_start_probe_now", False)),
        "training_entrypoint_not_requested": not bool(review.get("approve_new_training_entrypoint", False)),
        "batch2_release_probe_not_requested": not bool(review.get("approve_batch2_4_8_release_probe", False)),
        "release_claim_not_requested": not bool(review.get("approve_release_claims", False)),
    }


def _decision(*, progress: Mapping[str, bool], review: Mapping[str, Any]) -> str:
    if not bool(progress.get("signed_manual_review_present")):
        return HOLD_DECISION
    required = (
        "manual_review_packet_ready",
        "manual_review_packet_default_off",
        "manual_review_packet_batch2_still_blocked",
        "manual_review_packet_gate_enablement_still_blocked",
        "manual_review_packet_training_start_still_blocked",
        "manual_review_packet_release_claim_still_blocked",
        "requested_scope_valid",
        "manual_review_only_ack",
        "packet_ready_ack",
        "default_off_ack",
        "batch1_only_ack",
        "batch2_blocked_ack",
        "release_claim_closed_ack",
        "gate_enable_not_requested",
        "training_start_not_requested",
        "training_entrypoint_not_requested",
        "batch2_release_probe_not_requested",
        "release_claim_not_requested",
    )
    if any(not bool(progress.get(name, False)) for name in required):
        return HOLD_DECISION
    if bool(review.get("approve_followup_manual_probe_preparation", False)):
        return APPROVED_DECISION
    return REJECTED_DECISION


def _blocked_reasons(
    *,
    packet: Mapping[str, Any],
    progress: Mapping[str, bool],
    decision: str,
) -> list[str]:
    blockers: list[str] = []
    if not bool(progress.get("manual_review_packet_present")):
        blockers.append("internal_gate_limited_non_release_probe_manual_review_packet_missing")
    if not bool(progress.get("manual_review_packet_ready")):
        blockers.append("internal_gate_limited_non_release_probe_manual_review_packet_not_ready")
    if not bool(progress.get("manual_review_packet_default_off")):
        blockers.append("internal_gate_limited_non_release_probe_manual_review_packet_default_off_violation")
    if not bool(progress.get("manual_review_packet_batch2_still_blocked")):
        blockers.append("internal_gate_limited_non_release_probe_manual_review_packet_batch2_release_probe_not_blocked")
    if not bool(progress.get("manual_review_packet_gate_enablement_still_blocked")):
        blockers.append("internal_gate_limited_non_release_probe_manual_review_packet_gate_enablement_not_blocked")
    if not bool(progress.get("manual_review_packet_training_start_still_blocked")):
        blockers.append("internal_gate_limited_non_release_probe_manual_review_packet_training_start_not_blocked")
    if not bool(progress.get("manual_review_packet_release_claim_still_blocked")):
        blockers.append("internal_gate_limited_non_release_probe_manual_review_packet_release_claim_not_blocked")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_manual_review_packet:{item}"
        for item in _string_list(packet.get("blockers"))
    )
    if decision == HOLD_DECISION and not bool(progress.get("signed_manual_review_present")):
        blockers.append("signed_internal_gate_limited_non_release_probe_manual_review_missing")
    if bool(progress.get("signed_manual_review_present")):
        if not bool(progress.get("requested_scope_valid")):
            blockers.append("signed_internal_gate_limited_non_release_probe_manual_review_scope_invalid")
        if not bool(progress.get("manual_review_only_ack")):
            blockers.append("signed_internal_gate_limited_non_release_probe_manual_review_manual_only_ack_missing")
        if not bool(progress.get("packet_ready_ack")):
            blockers.append("signed_internal_gate_limited_non_release_probe_manual_review_packet_ack_missing")
        if not bool(progress.get("default_off_ack")):
            blockers.append("signed_internal_gate_limited_non_release_probe_manual_review_default_off_ack_missing")
        if not bool(progress.get("batch1_only_ack")):
            blockers.append("signed_internal_gate_limited_non_release_probe_manual_review_batch1_only_ack_missing")
        if not bool(progress.get("batch2_blocked_ack")):
            blockers.append("signed_internal_gate_limited_non_release_probe_manual_review_batch2_blocked_ack_missing")
        if not bool(progress.get("release_claim_closed_ack")):
            blockers.append("signed_internal_gate_limited_non_release_probe_manual_review_release_claim_ack_missing")
        if not bool(progress.get("gate_enable_not_requested")):
            blockers.append("signed_internal_gate_limited_non_release_probe_manual_review_requests_gate_enablement")
        if not bool(progress.get("training_start_not_requested")):
            blockers.append("signed_internal_gate_limited_non_release_probe_manual_review_requests_probe_start")
        if not bool(progress.get("training_entrypoint_not_requested")):
            blockers.append("signed_internal_gate_limited_non_release_probe_manual_review_requests_new_training_entrypoint")
        if not bool(progress.get("batch2_release_probe_not_requested")):
            blockers.append("signed_internal_gate_limited_non_release_probe_manual_review_requests_batch2_release_probe")
        if not bool(progress.get("release_claim_not_requested")):
            blockers.append("signed_internal_gate_limited_non_release_probe_manual_review_requests_release_claim")
    return _dedupe(blockers)


def _packet_summary(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(packet),
        "status": str(packet.get("status") or ""),
        "passed": bool(packet.get("passed")),
        "internal_gate_enablement_allowed": bool(packet.get("internal_gate_enablement_allowed")),
        "release_claim_allowed": bool(packet.get("release_claim_allowed")),
        "blocker_count": len(_string_list(packet.get("blockers"))),
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(review),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_followup_manual_probe_preparation": bool(
            review.get("approve_followup_manual_probe_preparation")
        ),
        "approve_turn_internal_gate_on_now": bool(review.get("approve_turn_internal_gate_on_now")),
        "approve_start_probe_now": bool(review.get("approve_start_probe_now")),
        "approve_new_training_entrypoint": bool(review.get("approve_new_training_entrypoint")),
        "approve_batch2_4_8_release_probe": bool(review.get("approve_batch2_4_8_release_probe")),
        "approve_release_claims": bool(review.get("approve_release_claims")),
        "reviewer": str(review.get("reviewer") or ""),
    }


def _review_template() -> dict[str, Any]:
    return {
        "requested_scope": SCOPE,
        "reviewer": "",
        "acknowledge_manual_review_packet_ready": False,
        "acknowledge_internal_gate_stays_disabled": False,
        "acknowledge_batch1_non_release_probe_only": False,
        "acknowledge_batch2_4_8_release_probe_still_blocked": False,
        "acknowledge_release_claim_stays_closed": False,
        "acknowledge_manual_review_only": False,
        "approve_followup_manual_probe_preparation": False,
        "approve_turn_internal_gate_on_now": False,
        "approve_start_probe_now": False,
        "approve_new_training_entrypoint": False,
        "approve_batch2_4_8_release_probe": False,
        "approve_release_claims": False,
        "review_notes": "",
    }


def _recommended_next_step(decision: str) -> str:
    if decision == APPROVED_DECISION:
        return "prepare_followup_manual_probe_material_while_keeping_internal_gate_disabled"
    if decision == REJECTED_DECISION:
        return "keep_internal_gate_default_off_and_continue_collecting_batch1_evidence"
    return "collect_signed_internal_gate_limited_non_release_probe_manual_review"


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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_MANUAL_REVIEW_RECORD",
    "REJECTED_DECISION",
    "SCOPE",
    "build_lulynx_internal_gate_limited_non_release_probe_manual_review_record",
]
