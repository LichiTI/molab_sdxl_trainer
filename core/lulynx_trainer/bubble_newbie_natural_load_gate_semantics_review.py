"""JSON-only Newbie natural-load gate semantics review."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


REPORT = "bubble_newbie_natural_load_gate_semantics_review_v0"
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


def _blocked_families(natural_load_canary: Mapping[str, Any]) -> list[str]:
    blocked = [str(item) for item in _sequence(natural_load_canary.get("blocked_families")) if str(item)]
    missing = [str(item) for item in _sequence(natural_load_canary.get("missing_families")) if str(item)]
    merged: list[str] = []
    for family in blocked + missing:
        if family and family not in merged:
            merged.append(family)
    return merged


def _newbie_family_row(natural_load_canary: Mapping[str, Any]) -> Mapping[str, Any]:
    for row in _sequence(natural_load_canary.get("families")):
        row_map = _mapping(row)
        if str(row_map.get("family") or "") == FAMILY:
            return row_map
    return {}


def _release_gap_ids(release_claims: Mapping[str, Any]) -> list[str]:
    ids = []
    for gap in _sequence(release_claims.get("evidence_gaps")):
        gap_map = _mapping(gap)
        gap_id = str(gap_map.get("id") or "")
        if gap_id:
            ids.append(gap_id)
    return ids


def _action(
    action_id: str,
    *,
    kind: str,
    reason: str,
    requires_gpu_heavy_run: bool,
) -> dict[str, Any]:
    return {
        "id": action_id,
        "kind": kind,
        "reason": reason,
        "requires_gpu_heavy_run": requires_gpu_heavy_run,
        "release_claim_allowed": False,
        "release_claim_allowed_after_success": False,
        "safe_to_auto_start": False,
    }


def build_newbie_natural_load_gate_semantics_review(
    *,
    internal_phase_diagnosis: Mapping[str, Any],
    blockskip_quality_review: Mapping[str, Any],
    natural_load_canary: Mapping[str, Any],
    release_claims: Mapping[str, Any],
) -> dict[str, Any]:
    """Project current Newbie natural-load semantics without running training."""

    newbie_row = _newbie_family_row(natural_load_canary)
    blockskip_summary = _mapping(blockskip_quality_review.get("summary"))
    release_gap_ids = _release_gap_ids(release_claims)
    blocked_families = _blocked_families(natural_load_canary)
    analyzed_probe_count = _safe_int(internal_phase_diagnosis.get("analyzed_probe_count"))
    compute_bound_probe_count = _safe_int(internal_phase_diagnosis.get("compute_bound_probe_count"))
    low_data_wait_probe_count = _safe_int(internal_phase_diagnosis.get("low_data_wait_probe_count"))
    dataloader_rebuild_count = _safe_int(internal_phase_diagnosis.get("dataloader_rebuild_observed_count"))
    natural_candidate_count = _safe_int(internal_phase_diagnosis.get("natural_candidate_count"))
    blockskip_throughput_ready = bool(blockskip_summary.get("throughput_repeat_ready"))
    blockskip_loss_ready = bool(blockskip_summary.get("loss_quality_ready"))
    release_claim_allowed = bool(release_claims.get("release_claim_allowed"))
    safe_to_auto_start = bool(release_claims.get("safe_to_auto_start"))

    signals = {
        "newbie_is_blocked_family": FAMILY in blocked_families,
        "natural_load_canary_blocked": str(natural_load_canary.get("status") or "") != "ready",
        "release_claims_blocked": str(release_claims.get("release_readiness") or release_claims.get("status") or "")
        != "ready",
        "natural_load_gap_present": "natural_load_canary_pending" in release_gap_ids,
        "data_wait_route_exhausted": bool(internal_phase_diagnosis.get("data_wait_route_exhausted")),
        "all_analyzed_probes_compute_bound": analyzed_probe_count > 0 and compute_bound_probe_count == analyzed_probe_count,
        "all_analyzed_probes_low_data_wait": analyzed_probe_count > 0 and low_data_wait_probe_count == analyzed_probe_count,
        "no_dataloader_rebuild_observed": dataloader_rebuild_count == 0,
        "no_natural_candidate_observed": natural_candidate_count == 0,
        "blockskip_repeat_throughput_candidate": blockskip_throughput_ready,
        "blockskip_quality_gate_ready": blockskip_loss_ready,
        "blockskip_quality_gate_blocked": blockskip_throughput_ready and not blockskip_loss_ready,
    }

    blockers: list[str] = []
    if signals["newbie_is_blocked_family"]:
        blockers.append("newbie_natural_load_canary_still_blocked")
    if signals["data_wait_route_exhausted"]:
        blockers.append("newbie_data_wait_route_exhausted")
    if signals["all_analyzed_probes_compute_bound"] and signals["all_analyzed_probes_low_data_wait"]:
        blockers.append("newbie_probe_set_is_compute_bound_low_data_wait")
    if signals["no_dataloader_rebuild_observed"] and signals["no_natural_candidate_observed"]:
        blockers.append("no_newbie_natural_dataloader_wait_evidence")
    if signals["blockskip_repeat_throughput_candidate"]:
        blockers.append("compute_path_candidate_not_natural_load_evidence")
    if signals["blockskip_quality_gate_blocked"]:
        blockers.append("blockskip_quality_gate_not_ready")
    if signals["natural_load_gap_present"]:
        blockers.append("release_claims_still_blocked_by_natural_load_canary")
    if release_claim_allowed or safe_to_auto_start:
        blockers.append("unexpected_release_or_auto_start_open")

    review_ready = (
        signals["newbie_is_blocked_family"]
        and signals["data_wait_route_exhausted"]
        and signals["all_analyzed_probes_compute_bound"]
        and signals["no_dataloader_rebuild_observed"]
        and signals["no_natural_candidate_observed"]
    )
    status = (
        "newbie_natural_load_gate_semantics_reviewed_blocked"
        if review_ready
        else "newbie_natural_load_gate_semantics_evidence_incomplete"
    )
    decision = {
        "natural_load_gate_exit": "not_satisfied",
        "raw_cache_payload_expansion_recommended": False,
        "compute_reducer_counts_as_natural_load_evidence": False,
        "blockskip_release_claim_allowed": False,
        "quality_review_required_before_any_compute_reducer_claim": True,
        "gate_policy_review_required": True,
        "recommended_next_route": "newbie_compute_bound_exception_policy_or_quality_gated_compute_candidate",
    }
    return {
        "report": REPORT,
        "schema_version": 1,
        "roadmap": ROADMAP,
        "family": FAMILY,
        "status": status,
        "review_ready": review_ready,
        "classification": "newbie_compute_bound_not_natural_load_evidence",
        "summary": {
            "blocked_families": blocked_families,
            "newbie_family_status": str(newbie_row.get("status") or ""),
            "release_gap_ids": release_gap_ids,
            "analyzed_probe_count": analyzed_probe_count,
            "compute_bound_probe_count": compute_bound_probe_count,
            "low_data_wait_probe_count": low_data_wait_probe_count,
            "dataloader_rebuild_observed_count": dataloader_rebuild_count,
            "natural_candidate_count": natural_candidate_count,
            "blockskip_completed_seed_pair_count": _safe_int(
                blockskip_quality_review.get("completed_seed_pair_count")
            ),
            "blockskip_throughput_repeat_ready": blockskip_throughput_ready,
            "blockskip_loss_quality_ready": blockskip_loss_ready,
            "blocker_count": len(blockers),
            "blockers": blockers,
        },
        "signals": signals,
        "decision": decision,
        "next_actions": [
            _action(
                "define_newbie_compute_bound_gate_exit_policy",
                kind="gate_semantics_review",
                reason="Newbie evidence repeatedly stays compute-bound with low data_wait, so release wording needs a policy route separate from natural DataLoader wait claims",
                requires_gpu_heavy_run=False,
            ),
            _action(
                "run_blockskip_quality_drift_ab_or_render_review",
                kind="quality_gate_followup",
                reason="BlockSkip has repeat positive throughput but loss deltas are outside the review band and directionally inconsistent",
                requires_gpu_heavy_run=True,
            ),
            _action(
                "separate_newbie_cache_materialization_timing_boundary",
                kind="measurement_boundary_review",
                reason="Newbie raw/cache pressure is materialized before steady data_wait measurement and should not be chased by larger raw/cache payloads",
                requires_gpu_heavy_run=False,
            ),
        ],
        "release_claim": {
            "eligible": False,
            "scope": "not_eligible",
            "reason": "Newbie natural-load gate remains blocked and compute-path candidates are not natural DataLoader wait evidence",
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
    "build_newbie_natural_load_gate_semantics_review",
]
