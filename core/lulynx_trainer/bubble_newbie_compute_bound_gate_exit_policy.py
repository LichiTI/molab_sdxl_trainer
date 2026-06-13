"""JSON-only Newbie compute-bound gate exit policy for GPU-bubble roadmap."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


REPORT = "bubble_newbie_compute_bound_gate_exit_policy_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
FAMILY = "newbie"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value if value is not None else default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _gap_ids(release_claims: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for gap in _sequence(release_claims.get("evidence_gaps")):
        gap_id = str(_mapping(gap).get("id") or "")
        if gap_id:
            ids.append(gap_id)
    return ids


def build_newbie_compute_bound_gate_exit_policy(
    *,
    semantics_review: Mapping[str, Any],
    blockskip_quality_review: Mapping[str, Any],
    release_claims: Mapping[str, Any],
) -> dict[str, Any]:
    """Define the fail-closed policy route after Newbie data-wait route exhaustion."""

    semantics_summary = _mapping(semantics_review.get("summary"))
    semantics_decision = _mapping(semantics_review.get("decision"))
    quality_summary = _mapping(blockskip_quality_review.get("summary"))
    release_gap_ids = _gap_ids(release_claims)
    review_ready = bool(semantics_review.get("review_ready"))
    natural_gate_exit = str(semantics_decision.get("natural_load_gate_exit") or "")
    compute_counts_as_natural = bool(semantics_decision.get("compute_reducer_counts_as_natural_load_evidence"))
    raw_cache_expansion = bool(semantics_decision.get("raw_cache_payload_expansion_recommended"))
    blockskip_throughput_ready = bool(quality_summary.get("throughput_repeat_ready"))
    blockskip_loss_ready = bool(quality_summary.get("loss_quality_ready"))
    release_claim_allowed = bool(release_claims.get("release_claim_allowed"))
    safe_to_auto_start = bool(release_claims.get("safe_to_auto_start"))

    policy_ready = (
        review_ready
        and natural_gate_exit == "not_satisfied"
        and not compute_counts_as_natural
        and not raw_cache_expansion
    )
    compute_bound_exception_allowed = bool(
        policy_ready
        and blockskip_throughput_ready
        and blockskip_loss_ready
        and "natural_load_canary_pending" not in release_gap_ids
        and release_claim_allowed
        and safe_to_auto_start
    )

    blockers: list[str] = []
    if not review_ready:
        blockers.append("semantics_review_not_ready")
    if natural_gate_exit != "not_satisfied":
        blockers.append("unexpected_natural_load_gate_exit")
    if compute_counts_as_natural:
        blockers.append("compute_reducer_misclassified_as_natural_load")
    if raw_cache_expansion:
        blockers.append("raw_cache_payload_expansion_still_recommended")
    if not blockskip_throughput_ready:
        blockers.append("blockskip_throughput_repeat_not_ready")
    if not blockskip_loss_ready:
        blockers.append("blockskip_quality_gate_not_ready")
    if "natural_load_canary_pending" in release_gap_ids:
        blockers.append("natural_load_canary_still_pending")
    if not release_claim_allowed:
        blockers.append("release_claim_still_closed")
    if not safe_to_auto_start:
        blockers.append("safe_to_auto_start_still_closed")

    status = (
        "newbie_compute_bound_gate_exit_policy_defined_blocked"
        if policy_ready
        else "newbie_compute_bound_gate_exit_policy_incomplete"
    )
    return {
        "report": REPORT,
        "schema_version": 1,
        "roadmap": ROADMAP,
        "family": FAMILY,
        "status": status,
        "policy_ready": policy_ready,
        "classification": "newbie_compute_bound_exception_policy_fail_closed",
        "summary": {
            "semantics_review_ready": review_ready,
            "analyzed_probe_count": _safe_int(semantics_summary.get("analyzed_probe_count")),
            "compute_bound_probe_count": _safe_int(semantics_summary.get("compute_bound_probe_count")),
            "low_data_wait_probe_count": _safe_int(semantics_summary.get("low_data_wait_probe_count")),
            "dataloader_rebuild_observed_count": _safe_int(
                semantics_summary.get("dataloader_rebuild_observed_count")
            ),
            "natural_candidate_count": _safe_int(semantics_summary.get("natural_candidate_count")),
            "blockskip_completed_seed_pair_count": _safe_int(
                blockskip_quality_review.get("completed_seed_pair_count")
            ),
            "blockskip_throughput_repeat_ready": blockskip_throughput_ready,
            "blockskip_loss_quality_ready": blockskip_loss_ready,
            "release_gap_ids": release_gap_ids,
            "blocker_count": len(blockers),
            "blockers": blockers,
        },
        "policy": {
            "natural_load_gate_exit_allowed": False,
            "compute_bound_exception_allowed": compute_bound_exception_allowed,
            "compute_reducer_counts_as_natural_load_evidence": False,
            "raw_cache_payload_expansion_recommended": False,
            "blockskip_counts_as_release_evidence": False,
            "blockskip_counts_as_quality_gated_compute_candidate": blockskip_throughput_ready,
            "release_claim_allowed_after_policy": False,
            "safe_to_auto_start_after_policy": False,
        },
        "exit_criteria": [
            {
                "id": "compute_bound_semantics_review_ready",
                "ready": review_ready,
                "required": True,
            },
            {
                "id": "quality_drift_or_render_review_ready",
                "ready": blockskip_loss_ready,
                "required": True,
            },
            {
                "id": "natural_load_release_gap_resolved_or_policy_exception_accepted",
                "ready": "natural_load_canary_pending" not in release_gap_ids,
                "required": True,
            },
            {
                "id": "release_claim_guard_explicitly_opened_by_downstream_policy",
                "ready": release_claim_allowed and safe_to_auto_start,
                "required": True,
            },
        ],
        "next_actions": [
            {
                "id": "run_blockskip_quality_drift_ab_or_render_review",
                "kind": "quality_gate_followup",
                "requires_gpu_heavy_run": True,
                "release_claim_allowed": False,
                "release_claim_allowed_after_success": False,
                "safe_to_auto_start": False,
            },
            {
                "id": "keep_newbie_natural_load_gate_fail_closed",
                "kind": "release_gate_policy",
                "requires_gpu_heavy_run": False,
                "release_claim_allowed": False,
                "release_claim_allowed_after_success": False,
                "safe_to_auto_start": False,
            },
        ],
        "release_claim": {
            "eligible": False,
            "scope": "not_eligible",
            "reason": "Newbie compute-bound policy is defined but blocked until quality and release gate criteria are satisfied",
        },
        "not_release_evidence": True,
        "release_claim_allowed": False,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "does_not_run_gpu_heavy": True,
    }


__all__ = [
    "FAMILY",
    "REPORT",
    "ROADMAP",
    "build_newbie_compute_bound_gate_exit_policy",
]
