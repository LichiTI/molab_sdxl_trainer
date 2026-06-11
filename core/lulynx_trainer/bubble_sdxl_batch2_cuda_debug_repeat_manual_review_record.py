"""Manual review record for SDXL CUDA debug repeat evidence.

This is a JSON-only decision record. It can acknowledge that an existing
case-specific diagnostic repeat is worth preparing for a protected follow-up
axis, but it never enables training, default batch2, or release claims.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


REPORT = "bubble_sdxl_batch2_cuda_debug_repeat_manual_review_record_v0"
PROMOTION_REVIEW_REPORT = "bubble_sdxl_batch2_cuda_debug_repeat_promotion_review_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
APPROVED_DECISION = "sdxl_cuda_debug_repeat_review_recorded_non_release_followup"
HOLD_DECISION = "sdxl_cuda_debug_repeat_review_hold_for_manual_record"
REJECTED_DECISION = "sdxl_cuda_debug_repeat_review_rejected_non_release"
SCOPE = "sdxl_cuda_debug_repeat_case_specific_non_release_followup_review"


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


def build_sdxl_batch2_cuda_debug_repeat_manual_review_record(
    *,
    promotion_review: Mapping[str, Any] | None = None,
    release_claims: Mapping[str, Any] | None = None,
    readiness_next_actions: Mapping[str, Any] | None = None,
    signed_manual_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    promotion = _mapping(promotion_review)
    claims = _mapping(release_claims)
    readiness = _mapping(readiness_next_actions)
    signed = _mapping(signed_manual_review)
    gates = _progress_gates(promotion=promotion, claims=claims, readiness=readiness, signed=signed)
    decision = _decision(gates=gates, signed=signed)
    blockers = _blockers(gates=gates, decision=decision)
    ready = decision in {APPROVED_DECISION, REJECTED_DECISION} and not blockers
    approved = ready and decision == APPROVED_DECISION

    return {
        "schema_version": 1,
        "report": REPORT,
        "roadmap": ROADMAP,
        "status": "decision_record_ready" if ready else "blocked",
        "ok": ready,
        "decision_record_ready": ready,
        "decision": decision,
        "approved_for_protected_followup_axis_preparation": approved,
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
        "source_promotion_review_status": str(promotion.get("status") or ""),
        "source_release_readiness": str(claims.get("release_readiness") or ""),
        "source_readiness_status": str(readiness.get("artifact_status") or ""),
        "blocked_reasons": blockers,
        "promotion_review_summary": _promotion_summary(promotion),
        "signed_manual_review": _signed_summary(signed),
        "signed_manual_review_template": _signed_template(),
        "progress_gates": gates,
        "allowed_followup_actions": [
            "prepare_protected_followup_axis",
            "refresh_post_manual_evidence_rebuild_plan",
            "refresh_gpu_bubble_readiness_next_actions",
        ]
        if approved
        else [],
        "blocked_actions": [
            "write_sdxl_batch2_release_gain_from_cuda_debug_repeat",
            "enable_sdxl_batch2_by_default",
            "use_cuda_debug_repeat_as_compile_anchor",
            "promote_microbatch_or_universal_gpu_utilization_claim",
            "skip_sd15_or_natural_load_release_gates",
        ],
        "recommended_next_action": (
            "prepare_protected_followup_axis_without_release_claim"
            if approved
            else "collect_signed_sdxl_cuda_debug_repeat_manual_review_record"
        ),
        "notes": [
            "This record only closes the manual review bookkeeping for existing diagnostic repeat evidence.",
            "GPU bubble release claims remain blocked by SD15 coverage and natural-load canary evidence.",
        ],
    }


def _progress_gates(
    *,
    promotion: Mapping[str, Any],
    claims: Mapping[str, Any],
    readiness: Mapping[str, Any],
    signed: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "promotion_review_present": bool(promotion),
        "promotion_review_identity_valid": str(promotion.get("report") or "") == PROMOTION_REVIEW_REPORT,
        "promotion_review_ready": bool(promotion.get("promotion_review_ready"))
        and str(promotion.get("status") or "") == "manual_promotion_review_ready",
        "promotion_review_has_no_blocked_gates": not _strings(promotion.get("blocked_gate_ids")),
        "promotion_review_release_closed": (
            not bool(promotion.get("release_claim_allowed"))
            and not bool(promotion.get("publishable"))
            and not bool(promotion.get("safe_to_auto_start"))
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
        "case_specific_ack": bool(signed.get("acknowledge_case_specific_only")),
        "diagnostic_only_ack": bool(signed.get("acknowledge_diagnostic_only")),
        "release_claim_closed_ack": bool(signed.get("acknowledge_release_claim_stays_closed")),
        "default_enable_not_requested": not bool(signed.get("approve_enable_batch2_by_default")),
        "release_claim_not_requested": not bool(signed.get("approve_release_claim")),
        "gpu_start_not_requested": not bool(signed.get("approve_start_gpu_work_now")),
    }


def _decision(*, gates: Mapping[str, bool], signed: Mapping[str, Any]) -> str:
    if not bool(gates.get("signed_manual_review_present")):
        return HOLD_DECISION
    required = (
        "promotion_review_identity_valid",
        "promotion_review_ready",
        "promotion_review_has_no_blocked_gates",
        "promotion_review_release_closed",
        "global_release_claims_stay_blocked",
        "top_level_readiness_stays_blocked",
        "requested_scope_valid",
        "case_specific_ack",
        "diagnostic_only_ack",
        "release_claim_closed_ack",
        "default_enable_not_requested",
        "release_claim_not_requested",
        "gpu_start_not_requested",
    )
    if any(not bool(gates.get(name)) for name in required):
        return HOLD_DECISION
    if bool(signed.get("approve_protected_followup_axis_preparation")):
        return APPROVED_DECISION
    return REJECTED_DECISION


def _blockers(*, gates: Mapping[str, bool], decision: str) -> list[str]:
    blockers = [name for name, passed in gates.items() if not passed]
    if decision == HOLD_DECISION:
        blockers.append("sdxl_cuda_debug_repeat_manual_review_not_recorded")
    return _dedupe(blockers)


def _promotion_summary(promotion: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(promotion.get("summary"))
    return {
        "present": bool(promotion),
        "status": str(promotion.get("status") or ""),
        "promotion_review_ready": bool(promotion.get("promotion_review_ready")),
        "blocked_gate_count": len(_strings(promotion.get("blocked_gate_ids"))),
        "comparison_count": _safe_int(summary.get("comparison_count")),
        "repeat_candidate_pass_count": _safe_int(summary.get("repeat_candidate_pass_count")),
        "fully_repeated_candidate_count": _safe_int(summary.get("fully_repeated_candidate_count")),
    }


def _signed_summary(signed: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(signed),
        "requested_scope": str(signed.get("requested_scope") or ""),
        "reviewer": str(signed.get("reviewer") or ""),
        "approve_protected_followup_axis_preparation": bool(
            signed.get("approve_protected_followup_axis_preparation")
        ),
        "approve_start_gpu_work_now": bool(signed.get("approve_start_gpu_work_now")),
        "approve_enable_batch2_by_default": bool(signed.get("approve_enable_batch2_by_default")),
        "approve_release_claim": bool(signed.get("approve_release_claim")),
    }


def _signed_template() -> dict[str, Any]:
    return {
        "requested_scope": SCOPE,
        "reviewer": "",
        "acknowledge_case_specific_only": False,
        "acknowledge_diagnostic_only": False,
        "acknowledge_release_claim_stays_closed": False,
        "approve_protected_followup_axis_preparation": False,
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
    "REJECTED_DECISION",
    "REPORT",
    "SCOPE",
    "build_sdxl_batch2_cuda_debug_repeat_manual_review_record",
]
