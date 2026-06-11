"""JSON-only triage packet for GPU-bubble manual review actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


REPORT = "bubble_manual_review_triage_packet_v0"
READINESS_REPORT = "gpu_bubble_experiment_readiness_next_actions_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"


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


def _review_actions(readiness: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    actions = [_mapping(item) for item in _list(readiness.get("next_actions"))]
    return [item for item in actions if str(item.get("readiness_state") or "") == "manual_review_ready"]


def _triage_bucket(action: Mapping[str, Any]) -> str:
    outcome = str(action.get("manual_review_outcome_kind") or "")
    if outcome in {
        "positive_candidate_review",
        "blocked_or_regression_review",
        "diagnostic_or_promotion_review",
        "followup_gpu_plan",
    }:
        return outcome
    return "manual_review"


def _recommended_review_disposition(bucket: str) -> str:
    if bucket == "positive_candidate_review":
        return "case_specific_positive_candidate_review"
    if bucket == "blocked_or_regression_review":
        return "record_blocked_or_regression_no_release_claim"
    if bucket == "diagnostic_or_promotion_review":
        return "diagnostic_or_promotion_review_without_release_claim"
    if bucket == "followup_gpu_plan":
        return "review_followup_gpu_plan_but_do_not_auto_start"
    return "manual_review_required"


def _row(action: Mapping[str, Any]) -> dict[str, Any]:
    bucket = _triage_bucket(action)
    return {
        "id": str(action.get("id") or ""),
        "roadmap": ROADMAP,
        "priority": _safe_int(action.get("priority"), 999),
        "family": str(action.get("family") or ""),
        "action_type": str(action.get("action_type") or ""),
        "status": str(action.get("status") or ""),
        "triage_bucket": bucket,
        "recommended_review_disposition": _recommended_review_disposition(bucket),
        "evidence_gap": str(action.get("evidence_gap") or ""),
        "evidence_paths": _strings(action.get("evidence_paths")),
        "acceptance_gates": _strings(action.get("acceptance_gates")),
        "rollback_or_block_rules": _strings(action.get("rollback_or_block_rules")),
        "requires_gpu_heavy_run": bool(action.get("requires_gpu_heavy_run")),
        "followup_requires_gpu_heavy_run": bool(action.get("followup_requires_gpu_heavy_run")),
        "not_release_evidence": True,
        "safe_to_auto_start": False,
        "release_claim_allowed_after_success": False,
        "diagnostic_only": bool(action.get("diagnostic_only")),
    }


def _bucket_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        bucket = str(row.get("triage_bucket") or "manual_review")
        counts[bucket] = counts.get(bucket, 0) + 1
    return dict(sorted(counts.items()))


def _bucket_ids(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("triage_bucket") or "manual_review"), []).append(str(row.get("id") or ""))
    return {key: ids[:20] for key, ids in sorted(grouped.items())}


def _signed_review_template() -> dict[str, Any]:
    return {
        "requested_scope": "gpu_bubble_manual_review_triage_non_release",
        "roadmap": ROADMAP,
        "not_release_evidence": True,
        "reviewer": "",
        "acknowledge_triage_packet_only": False,
        "acknowledge_no_release_claim": False,
        "acknowledge_no_auto_gpu_start": False,
        "approve_case_specific_review_records": False,
        "approve_start_gpu_work_now": False,
        "approve_release_claim": False,
        "review_notes": "",
    }


def build_manual_review_triage_packet(
    *,
    readiness_next_actions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed packet for manual review queue triage."""

    readiness = _mapping(readiness_next_actions)
    actions = sorted(_review_actions(readiness), key=lambda item: (_safe_int(item.get("priority"), 999), str(item.get("id") or "")))
    rows = [_row(action) for action in actions]
    review_only = [row for row in rows if not row["requires_gpu_heavy_run"] and not row["followup_requires_gpu_heavy_run"]]
    followup_gpu = [row for row in rows if row["followup_requires_gpu_heavy_run"]]
    current_gpu = [row for row in rows if row["requires_gpu_heavy_run"]]
    return {
        "schema_version": 1,
        "report": REPORT,
        "roadmap": ROADMAP,
        "status": "manual_review_triage_ready" if rows else "manual_review_queue_empty",
        "ok": True,
        "not_release_evidence": True,
        "release_claim_allowed": False,
        "publishable": False,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "source_readiness_report": str(readiness.get("report") or ""),
        "source_readiness_status": str(readiness.get("artifact_status") or ""),
        "source_release_readiness": str(readiness.get("release_readiness") or ""),
        "manual_review_ready_count": len(rows),
        "review_only_action_count": len(review_only),
        "followup_gpu_action_count": len(followup_gpu),
        "current_gpu_heavy_action_count": len(current_gpu),
        "triage_bucket_counts": _bucket_counts(rows),
        "triage_bucket_action_ids": _bucket_ids(rows),
        "review_rows": rows,
        "signed_review_template": _signed_review_template(),
        "blocked_actions": [
            "auto_start_gpu_heavy_from_triage_packet",
            "promote_triage_packet_as_release_evidence",
            "approve_release_claim_from_review_queue",
            "enable_batch2_by_default_from_review_queue",
            "skip_sd15_or_natural_load_release_gates",
        ],
        "acceptance_gates": [
            "triage_packet_is_json_only",
            "review_rows_require_separate_signed_records",
            "positive_candidates_remain_case_specific",
            "followup_gpu_plan_requires_separate_protected_runner",
            "release_claim_requires_sd15_and_natural_load_gates",
        ],
        "recommended_next_action": (
            "review_positive_candidates_then_record_case_specific_decisions"
            if rows
            else "no_manual_review_actions_pending"
        ),
        "notes": [
            "This packet sorts manual review actions; it is not a signed decision record.",
            "A positive bucket does not imply release readiness or default enablement.",
        ],
    }


__all__ = ["REPORT", "READINESS_REPORT", "ROADMAP", "build_manual_review_triage_packet"]
