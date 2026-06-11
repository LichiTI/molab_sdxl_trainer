"""JSON-only closure record for neutral/design manual review rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


REPORT = "bubble_manual_review_neutral_design_record_v0"
READINESS_REPORT = "gpu_bubble_experiment_readiness_next_actions_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
TARGET_ACTION_ID = "review_sdxl_alternate_workload_shape_axes_after_batch2_loss_guard"
SCOPE = "gpu_bubble_neutral_design_non_release_review"
READY_DECISION = "neutral_design_review_recorded_non_release_closure"
HOLD_DECISION = "neutral_design_review_hold_for_manual_record"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return list(value)


def _strings(value: Any) -> list[str]:
    return [str(item) for item in _list(value) if item is not None]


def _action_by_id(readiness: Mapping[str, Any], action_id: str) -> Mapping[str, Any]:
    for raw in _list(readiness.get("next_actions")):
        action = _mapping(raw)
        if str(action.get("id") or "") == action_id:
            return action
    return {}


def build_manual_review_neutral_design_record(
    *,
    readiness_next_actions: Mapping[str, Any] | None = None,
    signed_manual_review: Mapping[str, Any] | None = None,
    target_action_id: str = TARGET_ACTION_ID,
) -> dict[str, Any]:
    readiness = _mapping(readiness_next_actions)
    signed = _mapping(signed_manual_review)
    action = _action_by_id(readiness, target_action_id)
    gates = _progress_gates(readiness=readiness, action=action, signed=signed)
    decision = _decision(gates=gates, signed=signed)
    blockers = _blockers(gates=gates, decision=decision)
    ready = decision == READY_DECISION and not blockers
    closed_ids = [target_action_id] if ready else []
    return {
        "schema_version": 1,
        "report": REPORT,
        "roadmap": ROADMAP,
        "status": "neutral_design_review_record_ready" if ready else "blocked",
        "ok": ready,
        "decision_record_ready": ready,
        "decision": decision,
        "target_action_id": target_action_id,
        "closed_action_ids": closed_ids,
        "manual_review_required": True,
        "not_release_evidence": True,
        "release_claim_allowed": False,
        "publishable": False,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "diagnostic_only": True,
        "case_specific_only": True,
        "source_readiness_report": str(readiness.get("report") or ""),
        "source_readiness_status": str(readiness.get("artifact_status") or ""),
        "source_release_readiness": str(readiness.get("release_readiness") or ""),
        "source_action_summary": _action_summary(action),
        "signed_manual_review": _signed_summary(signed),
        "signed_manual_review_template": _signed_template(target_action_id),
        "progress_gates": gates,
        "blocked_reasons": blockers,
        "allowed_followup_actions": [
            "refresh_gpu_bubble_readiness_next_actions",
            "refresh_gpu_bubble_terminal_self_check",
            "run_gpu_bubble_release_readiness_guard",
        ]
        if ready
        else [],
        "blocked_actions": [
            "promote_neutral_design_record_as_release_evidence",
            "approve_release_claim_from_neutral_design_record",
            "enable_batch2_by_default_from_neutral_design_record",
            "auto_start_gpu_heavy_from_neutral_design_record",
            "skip_sd15_or_natural_load_release_gates",
        ],
        "recommended_next_action": (
            "refresh_readiness_after_neutral_design_review_closure"
            if ready
            else "collect_signed_neutral_design_manual_review_record"
        ),
        "notes": [
            "This record closes neutral/design manual review bookkeeping only.",
            "GPU bubble release claims remain blocked by SD15 coverage and natural-load canary evidence.",
        ],
    }


def _progress_gates(
    *,
    readiness: Mapping[str, Any],
    action: Mapping[str, Any],
    signed: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "readiness_identity_valid": str(readiness.get("report") or "") == READINESS_REPORT,
        "top_level_readiness_stays_blocked": (
            str(readiness.get("artifact_status") or "") == "blocked_pending_evidence"
            and not bool(readiness.get("release_claim_allowed"))
        ),
        "target_action_present": bool(action),
        "target_action_manual_review_ready": str(action.get("readiness_state") or "") == "manual_review_ready",
        "target_action_non_gpu_current": not bool(action.get("requires_gpu_heavy_run")),
        "target_action_non_gpu_followup": not bool(action.get("followup_requires_gpu_heavy_run")),
        "target_action_release_closed": (
            not bool(action.get("safe_to_auto_start"))
            and not bool(action.get("release_claim_allowed_after_success"))
            and bool(action.get("not_release_evidence"))
        ),
        "signed_manual_review_present": bool(signed),
        "requested_scope_valid": str(signed.get("requested_scope") or "") == SCOPE,
        "target_action_ack": str(signed.get("target_action_id") or "") == str(action.get("id") or ""),
        "neutral_design_ack": bool(signed.get("acknowledge_neutral_design_closure_only")),
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
        "readiness_identity_valid",
        "top_level_readiness_stays_blocked",
        "target_action_present",
        "target_action_manual_review_ready",
        "target_action_non_gpu_current",
        "target_action_non_gpu_followup",
        "target_action_release_closed",
        "requested_scope_valid",
        "target_action_ack",
        "neutral_design_ack",
        "case_specific_ack",
        "release_claim_closed_ack",
        "auto_gpu_start_closed_ack",
        "release_claim_not_requested",
        "gpu_start_not_requested",
        "default_enable_not_requested",
    )
    if any(not bool(gates.get(name)) for name in required):
        return HOLD_DECISION
    if bool(signed.get("approve_neutral_design_closure")):
        return READY_DECISION
    return HOLD_DECISION


def _blockers(*, gates: Mapping[str, bool], decision: str) -> list[str]:
    blockers = [name for name, passed in gates.items() if not passed]
    if decision == HOLD_DECISION:
        blockers.append("neutral_design_manual_review_not_recorded")
    return _dedupe(blockers)


def _action_summary(action: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(action),
        "id": str(action.get("id") or ""),
        "readiness_state": str(action.get("readiness_state") or ""),
        "readiness_blocker_kind": str(action.get("readiness_blocker_kind") or ""),
        "manual_review_outcome_kind": str(action.get("manual_review_outcome_kind") or ""),
        "requires_gpu_heavy_run": bool(action.get("requires_gpu_heavy_run")),
        "followup_requires_gpu_heavy_run": bool(action.get("followup_requires_gpu_heavy_run")),
        "safe_to_auto_start": bool(action.get("safe_to_auto_start")),
        "release_claim_allowed_after_success": bool(action.get("release_claim_allowed_after_success")),
    }


def _signed_summary(signed: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(signed),
        "requested_scope": str(signed.get("requested_scope") or ""),
        "reviewer": str(signed.get("reviewer") or ""),
        "target_action_id": str(signed.get("target_action_id") or ""),
        "approve_neutral_design_closure": bool(signed.get("approve_neutral_design_closure")),
        "approve_start_gpu_work_now": bool(signed.get("approve_start_gpu_work_now")),
        "approve_enable_batch2_by_default": bool(signed.get("approve_enable_batch2_by_default")),
        "approve_release_claim": bool(signed.get("approve_release_claim")),
    }


def _signed_template(target_action_id: str) -> dict[str, Any]:
    return {
        "requested_scope": SCOPE,
        "target_action_id": target_action_id,
        "acknowledge_neutral_design_closure_only": True,
        "acknowledge_case_specific_only": True,
        "acknowledge_no_release_claim": True,
        "acknowledge_no_auto_gpu_start": True,
        "approve_neutral_design_closure": True,
        "approve_start_gpu_work_now": False,
        "approve_enable_batch2_by_default": False,
        "approve_release_claim": False,
        "roadmap": ROADMAP,
        "not_release_evidence": True,
    }


def _dedupe(values: Sequence[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            rows.append(value)
    return rows
