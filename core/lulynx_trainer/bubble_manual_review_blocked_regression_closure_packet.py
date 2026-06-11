"""JSON-only closure packet for blocked/regression manual review rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


REPORT = "bubble_manual_review_blocked_regression_closure_packet_v0"
TRIAGE_REPORT = "bubble_manual_review_triage_packet_v0"
READINESS_REPORT = "gpu_bubble_experiment_readiness_next_actions_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
BLOCKED_BUCKET = "blocked_or_regression_review"


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


def _closure_rows(packet: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [_mapping(item) for item in _list(packet.get("closure_rows")) if _mapping(item)]


def _action_map(readiness: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for raw in _list(readiness.get("next_actions")):
        action = _mapping(raw)
        action_id = str(action.get("id") or "")
        if action_id:
            rows[action_id] = action
    return rows


def _row_from_triage(row: Mapping[str, Any], readiness_action: Mapping[str, Any] | None = None) -> dict[str, Any]:
    action = _mapping(readiness_action)
    evidence_gap = str(action.get("evidence_gap") or row.get("evidence_gap") or "")
    rollback_rules = _strings(action.get("rollback_or_block_rules")) or _strings(row.get("rollback_or_block_rules"))
    acceptance_gates = _strings(action.get("acceptance_gates")) or _strings(row.get("acceptance_gates"))
    return {
        "id": str(row.get("id") or action.get("id") or ""),
        "roadmap": ROADMAP,
        "priority": _safe_int(row.get("priority", action.get("priority")), 999),
        "family": str(row.get("family") or action.get("family") or ""),
        "action_type": str(row.get("action_type") or action.get("action_type") or ""),
        "source_status": str(row.get("status") or action.get("status") or ""),
        "triage_bucket": BLOCKED_BUCKET,
        "closure_status": "machine_fail_closed_recorded",
        "closed_as_blocked_or_regression": True,
        "closure_reason": evidence_gap or ", ".join(rollback_rules[:3]),
        "evidence_gap": evidence_gap,
        "evidence_paths": _strings(action.get("evidence_paths")) or _strings(row.get("evidence_paths")),
        "acceptance_gates": acceptance_gates,
        "rollback_or_block_rules": rollback_rules,
        "requires_gpu_heavy_run": False,
        "followup_requires_gpu_heavy_run": False,
        "not_release_evidence": True,
        "safe_to_auto_start": False,
        "release_claim_allowed_after_closure": False,
        "publishable_after_closure": False,
        "current_action_present": bool(action),
        "current_readiness_state": str(action.get("readiness_state") or ""),
        "current_readiness_blocker_kind": str(action.get("readiness_blocker_kind") or ""),
    }


def _row_from_previous(row: Mapping[str, Any], readiness_action: Mapping[str, Any] | None = None) -> dict[str, Any]:
    action = _mapping(readiness_action)
    carried = dict(row)
    carried["closed_as_blocked_or_regression"] = True
    carried["closure_status"] = str(carried.get("closure_status") or "machine_fail_closed_recorded")
    carried["triage_bucket"] = BLOCKED_BUCKET
    carried["roadmap"] = ROADMAP
    carried["requires_gpu_heavy_run"] = False
    carried["followup_requires_gpu_heavy_run"] = False
    carried["not_release_evidence"] = True
    carried["safe_to_auto_start"] = False
    carried["release_claim_allowed_after_closure"] = False
    carried["publishable_after_closure"] = False
    carried["current_action_present"] = bool(action) or bool(carried.get("current_action_present"))
    if action:
        carried["current_readiness_state"] = str(action.get("readiness_state") or "")
        carried["current_readiness_blocker_kind"] = str(action.get("readiness_blocker_kind") or "")
    return carried


def _remaining_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    remaining: list[dict[str, Any]] = []
    for raw in rows:
        row = _mapping(raw)
        bucket = str(row.get("triage_bucket") or "")
        if bucket == BLOCKED_BUCKET:
            continue
        remaining.append(
            {
                "id": str(row.get("id") or ""),
                "roadmap": ROADMAP,
                "priority": _safe_int(row.get("priority"), 999),
                "family": str(row.get("family") or ""),
                "action_type": str(row.get("action_type") or ""),
                "triage_bucket": bucket,
                "recommended_review_disposition": str(row.get("recommended_review_disposition") or ""),
                "requires_gpu_heavy_run": bool(row.get("requires_gpu_heavy_run")),
                "followup_requires_gpu_heavy_run": bool(row.get("followup_requires_gpu_heavy_run")),
                "not_release_evidence": True,
                "release_claim_allowed_after_success": False,
                "safe_to_auto_start": False,
            }
        )
    return sorted(remaining, key=lambda item: (item["priority"], item["id"]))


def _bucket_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        bucket = str(row.get("triage_bucket") or "manual_review")
        counts[bucket] = counts.get(bucket, 0) + 1
    return dict(sorted(counts.items()))


def _merge_closure_rows(
    *,
    triage_rows: Sequence[Mapping[str, Any]],
    previous_rows: Sequence[Mapping[str, Any]],
    readiness_actions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for raw in previous_rows:
        previous = _mapping(raw)
        row_id = str(previous.get("id") or "")
        if row_id and bool(previous.get("closed_as_blocked_or_regression")):
            rows[row_id] = _row_from_previous(previous, readiness_actions.get(row_id))
    for raw in triage_rows:
        triage = _mapping(raw)
        row_id = str(triage.get("id") or "")
        if row_id and str(triage.get("triage_bucket") or "") == BLOCKED_BUCKET:
            rows[row_id] = _row_from_triage(triage, readiness_actions.get(row_id))
    return sorted(rows.values(), key=lambda item: (_safe_int(item.get("priority"), 999), str(item.get("id") or "")))


def build_manual_review_blocked_regression_closure_packet(
    *,
    manual_review_triage_packet: Mapping[str, Any] | None = None,
    readiness_next_actions: Mapping[str, Any] | None = None,
    previous_closure_packet: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed closure packet for blocked/regression review rows."""

    triage = _mapping(manual_review_triage_packet)
    readiness = _mapping(readiness_next_actions)
    previous = _mapping(previous_closure_packet)
    triage_rows = _review_rows(triage)
    remaining = _remaining_rows(triage_rows)
    actions = _action_map(readiness)
    closure_rows = _merge_closure_rows(
        triage_rows=triage_rows,
        previous_rows=_closure_rows(previous),
        readiness_actions=actions,
    )
    status = "blocked_regression_closure_ready" if closure_rows else "no_blocked_regression_reviews_to_close"
    stale_count = sum(1 for row in closure_rows if not bool(row.get("current_action_present")))
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
        "source_readiness_report": str(readiness.get("report") or ""),
        "source_readiness_status": str(readiness.get("artifact_status") or ""),
        "source_release_readiness": str(readiness.get("release_readiness") or ""),
        "manual_review_ready_count_before_closure": _safe_int(
            triage.get("manual_review_ready_count"),
            len(triage_rows),
        ),
        "closed_action_count": len(closure_rows),
        "closed_current_action_count": len(closure_rows) - stale_count,
        "stale_closed_action_count": stale_count,
        "remaining_manual_review_count_after_closure": len(remaining),
        "remaining_review_only_action_count_after_closure": sum(
            1 for row in remaining if not row["requires_gpu_heavy_run"] and not row["followup_requires_gpu_heavy_run"]
        ),
        "remaining_followup_gpu_action_count_after_closure": sum(
            1 for row in remaining if row["followup_requires_gpu_heavy_run"]
        ),
        "remaining_current_gpu_heavy_action_count_after_closure": sum(
            1 for row in remaining if row["requires_gpu_heavy_run"]
        ),
        "closed_action_ids": [str(row.get("id") or "") for row in closure_rows],
        "remaining_manual_review_action_ids": [str(row.get("id") or "") for row in remaining],
        "remaining_triage_bucket_counts": _bucket_counts(remaining),
        "closure_rows": closure_rows,
        "remaining_review_rows": remaining,
        "blocked_actions": [
            "auto_start_gpu_heavy_from_blocked_regression_closure_packet",
            "promote_closed_regression_rows_as_release_evidence",
            "approve_positive_candidate_from_closure_packet",
            "enable_batch2_by_default_from_closed_rows",
            "skip_sd15_or_natural_load_release_gates",
        ],
        "acceptance_gates": [
            "closure_packet_is_json_only",
            "only_blocked_or_regression_rows_closed",
            "closed_rows_remain_no_release_claim",
            "positive_candidates_stay_manual_review",
            "diagnostic_and_followup_gpu_stay_manual_or_protected",
            "release_claim_requires_sd15_and_natural_load_gates",
        ],
        "recommended_next_action": (
            "review_remaining_positive_or_diagnostic_rows"
            if remaining
            else "refresh_gpu_bubble_readiness_after_blocked_regression_closure"
        ),
        "notes": [
            "This packet records blocked/regression review rows as machine fail-closed.",
            "It does not approve positive candidates, diagnostic promotion, GPU execution, or release claims.",
        ],
    }


__all__ = [
    "BLOCKED_BUCKET",
    "READINESS_REPORT",
    "REPORT",
    "ROADMAP",
    "TRIAGE_REPORT",
    "build_manual_review_blocked_regression_closure_packet",
]
