"""Manual review record for positive GPU-bubble candidates.

This is JSON-only bookkeeping. It can acknowledge that positive candidates
were reviewed for case-specific follow-up planning, but it never starts GPU
work, enables defaults, or opens GPU-bubble release claims.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


REPORT = "bubble_manual_review_positive_candidate_record_v0"
PACKET_REPORT = "bubble_manual_review_positive_candidate_packet_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
APPROVED_DECISION = "positive_candidates_review_recorded_non_release_followup"
HOLD_DECISION = "positive_candidates_review_hold_for_manual_record"
REJECTED_DECISION = "positive_candidates_review_rejected_non_release"
SCOPE = "gpu_bubble_positive_candidate_case_specific_non_release_review"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return list(value)


def _strings(value: Any) -> list[str]:
    return [str(item) for item in _list(value) if item is not None]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value if value is not None else default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def build_manual_review_positive_candidate_record(
    *,
    positive_candidate_packet: Mapping[str, Any] | None = None,
    release_claims: Mapping[str, Any] | None = None,
    readiness_next_actions: Mapping[str, Any] | None = None,
    signed_manual_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    packet = _mapping(positive_candidate_packet)
    claims = _mapping(release_claims)
    readiness = _mapping(readiness_next_actions)
    signed = _mapping(signed_manual_review)
    gates = _progress_gates(packet=packet, claims=claims, readiness=readiness, signed=signed)
    decision = _decision(gates=gates, signed=signed)
    blockers = _blockers(gates=gates, decision=decision)
    ready = decision in {APPROVED_DECISION, REJECTED_DECISION} and not blockers
    approved = ready and decision == APPROVED_DECISION
    candidate_ids = _strings(packet.get("candidate_action_ids"))

    return {
        "schema_version": 1,
        "report": REPORT,
        "roadmap": ROADMAP,
        "status": "decision_record_ready" if ready else "blocked",
        "ok": ready,
        "decision_record_ready": ready,
        "decision": decision,
        "approved_for_case_specific_followup_planning": approved,
        "rejected_for_non_release_hold": ready and decision == REJECTED_DECISION,
        "manual_review_required": True,
        "not_release_evidence": True,
        "release_claim_allowed": False,
        "publishable": False,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "diagnostic_only": True,
        "case_specific_only": True,
        "source_packet_status": str(packet.get("status") or ""),
        "source_release_readiness": str(claims.get("release_readiness") or ""),
        "source_readiness_status": str(readiness.get("artifact_status") or ""),
        "candidate_count": _safe_int(packet.get("candidate_count")),
        "candidate_action_ids": candidate_ids,
        "recorded_candidate_action_ids": candidate_ids if approved else [],
        "blocked_reasons": blockers,
        "packet_summary": _packet_summary(packet),
        "signed_manual_review": _signed_summary(signed),
        "signed_manual_review_template": _signed_template(candidate_ids),
        "progress_gates": gates,
        "allowed_followup_actions": [
            "prepare_case_specific_protected_followup_plan",
            "refresh_post_manual_evidence_rebuild_plan",
            "refresh_gpu_bubble_readiness_next_actions",
        ]
        if approved
        else [],
        "blocked_actions": [
            "promote_positive_candidate_record_as_release_evidence",
            "approve_release_claim_from_positive_candidate_record",
            "enable_batch2_by_default_from_positive_candidate_record",
            "auto_start_gpu_heavy_from_positive_candidate_record",
            "skip_sd15_or_natural_load_release_gates",
        ],
        "recommended_next_action": (
            "prepare_case_specific_protected_followup_plan_without_release_claim"
            if approved
            else "collect_signed_positive_candidate_manual_review_record"
        ),
        "notes": [
            "This record closes positive-candidate manual review bookkeeping only.",
            "GPU bubble release claims remain blocked by SD15 coverage and natural-load canary evidence.",
        ],
    }


def _progress_gates(
    *,
    packet: Mapping[str, Any],
    claims: Mapping[str, Any],
    readiness: Mapping[str, Any],
    signed: Mapping[str, Any],
) -> dict[str, bool]:
    candidate_ids = _strings(packet.get("candidate_action_ids"))
    signed_candidate_ids = _strings(signed.get("candidate_ids_reviewed"))
    return {
        "positive_candidate_packet_present": bool(packet),
        "positive_candidate_packet_identity_valid": str(packet.get("report") or "") == PACKET_REPORT,
        "positive_candidate_packet_ready": str(packet.get("status") or "") in {
            "positive_candidate_review_packet_ready",
            "no_positive_candidates_pending",
        },
        "positive_candidate_packet_release_closed": (
            not bool(packet.get("release_claim_allowed"))
            and not bool(packet.get("publishable"))
            and not bool(packet.get("safe_to_auto_start"))
        ),
        "global_release_claims_stay_blocked": (
            str(claims.get("release_readiness") or "") == "blocked_pending_evidence"
            and not bool(claims.get("release_claim_allowed"))
        ),
        "top_level_readiness_stays_blocked": (
            str(readiness.get("artifact_status") or "") == "blocked_pending_evidence"
            and not bool(readiness.get("release_claim_allowed"))
        ),
        "signed_manual_review_present": bool(signed),
        "requested_scope_valid": str(signed.get("requested_scope") or "") == SCOPE,
        "candidate_ids_match_packet": bool(candidate_ids)
        and sorted(candidate_ids) == sorted(signed_candidate_ids),
        "positive_packet_ack": bool(signed.get("acknowledge_positive_packet_only")),
        "case_specific_ack": bool(signed.get("acknowledge_case_specific_only")),
        "release_claim_closed_ack": bool(signed.get("acknowledge_no_release_claim")),
        "auto_gpu_start_closed_ack": bool(signed.get("acknowledge_no_auto_gpu_start")),
        "release_claim_not_requested": not bool(signed.get("approve_release_claim")),
        "gpu_start_not_requested": not bool(signed.get("approve_start_gpu_work_now")),
        "default_enable_not_requested": not bool(signed.get("approve_enable_batch2_by_default")),
    }


def _decision(*, gates: Mapping[str, bool], signed: Mapping[str, Any]) -> str:
    if not bool(gates.get("signed_manual_review_present")):
        return HOLD_DECISION
    required = (
        "positive_candidate_packet_identity_valid",
        "positive_candidate_packet_ready",
        "positive_candidate_packet_release_closed",
        "global_release_claims_stay_blocked",
        "top_level_readiness_stays_blocked",
        "requested_scope_valid",
        "candidate_ids_match_packet",
        "positive_packet_ack",
        "case_specific_ack",
        "release_claim_closed_ack",
        "auto_gpu_start_closed_ack",
        "release_claim_not_requested",
        "gpu_start_not_requested",
        "default_enable_not_requested",
    )
    if any(not bool(gates.get(name)) for name in required):
        return HOLD_DECISION
    if bool(signed.get("approve_case_specific_followup_records")):
        return APPROVED_DECISION
    return REJECTED_DECISION


def _blockers(*, gates: Mapping[str, bool], decision: str) -> list[str]:
    blockers = [name for name, passed in gates.items() if not passed]
    if decision == HOLD_DECISION:
        blockers.append("positive_candidate_manual_review_not_recorded")
    return _dedupe(blockers)


def _packet_summary(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(packet),
        "status": str(packet.get("status") or ""),
        "candidate_count": _safe_int(packet.get("candidate_count")),
        "candidate_action_ids": _strings(packet.get("candidate_action_ids"))[:20],
        "requires_case_specific_review_count": _safe_int(
            packet.get("requires_case_specific_review_count")
        ),
        "release_claim_allowed": bool(packet.get("release_claim_allowed")),
        "publishable": bool(packet.get("publishable")),
        "safe_to_auto_start": bool(packet.get("safe_to_auto_start")),
    }


def _signed_summary(signed: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(signed),
        "requested_scope": str(signed.get("requested_scope") or ""),
        "reviewer": str(signed.get("reviewer") or ""),
        "candidate_ids_reviewed": _strings(signed.get("candidate_ids_reviewed"))[:20],
        "approve_case_specific_followup_records": bool(
            signed.get("approve_case_specific_followup_records")
        ),
        "approve_start_gpu_work_now": bool(signed.get("approve_start_gpu_work_now")),
        "approve_enable_batch2_by_default": bool(signed.get("approve_enable_batch2_by_default")),
        "approve_release_claim": bool(signed.get("approve_release_claim")),
    }


def _signed_template(candidate_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "requested_scope": SCOPE,
        "roadmap": ROADMAP,
        "not_release_evidence": True,
        "reviewer": "",
        "candidate_ids_reviewed": list(candidate_ids),
        "acknowledge_positive_packet_only": False,
        "acknowledge_case_specific_only": False,
        "acknowledge_no_release_claim": False,
        "acknowledge_no_auto_gpu_start": False,
        "approve_case_specific_followup_records": False,
        "approve_start_gpu_work_now": False,
        "approve_enable_batch2_by_default": False,
        "approve_release_claim": False,
        "review_notes": "",
    }


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


__all__ = [
    "APPROVED_DECISION",
    "HOLD_DECISION",
    "PACKET_REPORT",
    "REJECTED_DECISION",
    "REPORT",
    "ROADMAP",
    "SCOPE",
    "build_manual_review_positive_candidate_record",
]
