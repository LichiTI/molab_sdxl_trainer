"""JSON-only packet for positive GPU-bubble manual review candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


REPORT = "bubble_manual_review_positive_candidate_packet_v0"
TRIAGE_REPORT = "bubble_manual_review_triage_packet_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
POSITIVE_BUCKET = "positive_candidate_review"


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


def _review_rows(packet: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [_mapping(item) for item in _list(packet.get("review_rows")) if _mapping(item)]


def _positive_rows(packet: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in _review_rows(packet) if str(row.get("triage_bucket") or "") == POSITIVE_BUCKET]


def _candidate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    gates = _strings(row.get("acceptance_gates"))
    if not gates:
        gates = [
            "case_specific_manual_review_required",
            "repeat_stability_required_before_release_wording",
            "loss_stability_required_before_release_wording",
            "active_gpu_telemetry_required_before_release_wording",
            "release_claim_requires_sd15_and_natural_load_gates",
        ]
    return {
        "id": str(row.get("id") or ""),
        "roadmap": ROADMAP,
        "priority": _safe_int(row.get("priority"), 999),
        "family": str(row.get("family") or ""),
        "action_type": str(row.get("action_type") or ""),
        "source_status": str(row.get("status") or ""),
        "triage_bucket": POSITIVE_BUCKET,
        "candidate_status": "case_specific_review_required",
        "recommended_review_disposition": "case_specific_positive_candidate_review",
        "evidence_gap": str(row.get("evidence_gap") or ""),
        "evidence_paths": _strings(row.get("evidence_paths")),
        "acceptance_gates": gates,
        "rollback_or_block_rules": _strings(row.get("rollback_or_block_rules")),
        "requires_gpu_heavy_run": False,
        "followup_requires_gpu_heavy_run": bool(row.get("followup_requires_gpu_heavy_run")),
        "not_release_evidence": True,
        "safe_to_auto_start": False,
        "release_claim_allowed_after_success": False,
        "publishable_after_success": False,
    }


def _signed_review_template() -> dict[str, Any]:
    return {
        "requested_scope": "gpu_bubble_positive_candidate_case_specific_non_release",
        "roadmap": ROADMAP,
        "not_release_evidence": True,
        "reviewer": "",
        "candidate_ids_reviewed": [],
        "acknowledge_positive_packet_only": False,
        "acknowledge_no_release_claim": False,
        "acknowledge_no_auto_gpu_start": False,
        "approve_case_specific_followup_records": False,
        "approve_start_gpu_work_now": False,
        "approve_release_claim": False,
        "review_notes": "",
    }


def build_manual_review_positive_candidate_packet(
    *,
    manual_review_triage_packet: Mapping[str, Any] | None = None,
    manual_review_blocked_regression_closure_packet: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed packet for positive candidates requiring review."""

    triage = _mapping(manual_review_triage_packet)
    closure = _mapping(manual_review_blocked_regression_closure_packet)
    candidates = [_candidate_row(row) for row in _positive_rows(triage)]
    candidates = sorted(candidates, key=lambda item: (item["priority"], item["id"]))
    status = "positive_candidate_review_packet_ready" if candidates else "no_positive_candidates_pending"
    closure_closed_count = _safe_int(closure.get("closed_action_count"))
    return {
        "schema_version": 1,
        "report": REPORT,
        "roadmap": ROADMAP,
        "status": status,
        "ok": True,
        "not_release_evidence": True,
        "release_claim_allowed": False,
        "publishable": False,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "source_triage_report": str(triage.get("report") or ""),
        "source_triage_status": str(triage.get("status") or ""),
        "source_closure_report": str(closure.get("report") or ""),
        "source_closure_status": str(closure.get("status") or ""),
        "blocked_regression_closed_action_count": closure_closed_count,
        "candidate_count": len(candidates),
        "candidate_action_ids": [str(row.get("id") or "") for row in candidates],
        "requires_case_specific_review_count": len(candidates),
        "requires_gpu_heavy_run_count": sum(1 for row in candidates if row["requires_gpu_heavy_run"]),
        "followup_gpu_candidate_count": sum(1 for row in candidates if row["followup_requires_gpu_heavy_run"]),
        "candidate_rows": candidates,
        "signed_review_template": _signed_review_template(),
        "blocked_actions": [
            "auto_start_gpu_heavy_from_positive_candidate_packet",
            "promote_positive_candidate_packet_as_release_evidence",
            "approve_release_claim_from_positive_candidate_packet",
            "enable_batch2_by_default_from_positive_candidate_packet",
            "skip_sd15_or_natural_load_release_gates",
        ],
        "acceptance_gates": [
            "positive_candidate_packet_is_json_only",
            "case_specific_signed_review_required",
            "repeat_and_loss_and_telemetry_gates_required_before_promotion",
            "release_claim_requires_sd15_and_natural_load_gates",
            "no_batch2_default_enablement_from_positive_packet",
        ],
        "recommended_next_action": (
            "record_case_specific_positive_candidate_review_decisions"
            if candidates
            else "no_positive_candidate_review_needed"
        ),
        "notes": [
            "This packet isolates positive candidates for case-specific review.",
            "It is not a signed review record and cannot approve GPU execution or release claims.",
        ],
    }


__all__ = [
    "POSITIVE_BUCKET",
    "REPORT",
    "ROADMAP",
    "TRIAGE_REPORT",
    "build_manual_review_positive_candidate_packet",
]
