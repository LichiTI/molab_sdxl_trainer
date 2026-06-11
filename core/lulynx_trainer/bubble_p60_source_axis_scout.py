"""Rank P60 real-material source axes without starting GPU runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


P60_SOURCE_AXIS_SCOUT_REPORT = "bubble_p60_source_axis_scout_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
CAPTION_COVERAGE_THRESHOLD = 0.875
MIN_CANDIDATE_RANK_SCORE = 4.0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_safe_float(value, float(default))))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _family_key(value: Any) -> str:
    family = str(value or "").strip().lower().replace("-", "_")
    return "newbie" if family in {"dit", "newbie_dit"} else family


def _norm_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve()).lower()
    except OSError:
        return text.lower()


def _source_axes(source_scan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if str(source_scan.get("mode") or "") == "sample_windows":
        return [_mapping(item) for item in source_scan.get("windows", []) if _mapping(item)]
    return [_mapping(item) for item in source_scan.get("candidates", []) if _mapping(item)]


def _candidate_rank_score(axis: Mapping[str, Any]) -> float:
    return _safe_float(axis.get("candidate_rank_score"), _safe_float(axis.get("pressure_score")))


def _family_readiness(axis: Mapping[str, Any], family: str) -> Mapping[str, Any]:
    for item in axis.get("family_readiness", []):
        mapped = _mapping(item)
        if _family_key(mapped.get("family")) == family:
            return mapped
    return {}


def _family_policies(followup_run_plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    policies: dict[str, Mapping[str, Any]] = {}
    for raw in followup_run_plan.get("family_policy", []):
        item = _mapping(raw)
        family = _family_key(item.get("family"))
        if family:
            policies[family] = item
    return policies


def _families(
    source_scan: Mapping[str, Any],
    followup_run_plan: Mapping[str, Any],
    natural_load_canary: Mapping[str, Any],
) -> list[str]:
    found = {_family_key(item) for item in _string_list(followup_run_plan.get("families"))}
    found.update(_family_key(item) for item in _string_list(natural_load_canary.get("blocked_families")))
    for axis in _source_axes(source_scan):
        found.update(_family_key(item) for item in axis.get("ready_families", []) if item)
        found.update(_family_key(item) for item in axis.get("blocked_families", []) if item)
    return sorted(family for family in found if family)


def _completed_attempts_from_evidence(completed_evidence: Any) -> list[dict[str, Any]]:
    payloads: list[Mapping[str, Any]] = []
    if isinstance(completed_evidence, Mapping):
        payloads = [completed_evidence]
    elif not isinstance(completed_evidence, (str, bytes)) and isinstance(completed_evidence, Sequence):
        payloads = [_mapping(item) for item in completed_evidence if _mapping(item)]

    axes: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for payload in payloads:
        for raw_case in payload.get("cases", []):
            case = _mapping(raw_case)
            fixture = _mapping(case.get("source_fixture"))
            family = _family_key(case.get("family") or fixture.get("family"))
            source = _norm_path(fixture.get("source_root") or fixture.get("root"))
            if not family or not source:
                continue
            sample_offset = _safe_int(fixture.get("sample_offset"))
            key = (family, source, sample_offset)
            if key in seen:
                continue
            seen.add(key)
            axes.append(
                {
                    "family": family,
                    "source_data": source,
                    "sample_offset": sample_offset,
                    "id": str(case.get("case_id") or "completed_real_material_canary"),
                    "profile": "completed_existing_evidence",
                    "diagnostic_only": False,
                    "completed_existing_evidence": True,
                    "source_manifest_sha1": str(fixture.get("source_manifest_sha1") or ""),
                }
            )
    return axes


def _attempted_axes(followup_run_plan: Mapping[str, Any], completed_evidence: Any = None) -> list[dict[str, Any]]:
    axes: list[dict[str, Any]] = []
    for raw in followup_run_plan.get("commands", []):
        item = _mapping(raw)
        family = _family_key(item.get("family"))
        source = _norm_path(item.get("source_data"))
        if not family or not source:
            continue
        axes.append(
            {
                "family": family,
                "source_data": source,
                "sample_offset": _safe_int(item.get("sample_offset")),
                "id": str(item.get("id") or ""),
                "profile": str(item.get("profile") or ""),
                "diagnostic_only": bool(item.get("diagnostic_only")),
                "completed_existing_evidence": False,
            }
        )
    axes.extend(_completed_attempts_from_evidence(completed_evidence))
    return axes


def _axis_attempts(axis: Mapping[str, Any], family: str, attempts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    root = _norm_path(axis.get("root"))
    offset = _safe_int(axis.get("sample_offset"))
    matches: list[dict[str, Any]] = []
    for attempt in attempts:
        if _family_key(attempt.get("family")) != family:
            continue
        if _safe_int(attempt.get("sample_offset")) != offset:
            continue
        if root and _norm_path(attempt.get("source_data")) != root:
            continue
        matches.append(dict(attempt))
    return matches


def _policy_signals(policy: Mapping[str, Any]) -> dict[str, Any]:
    hold = bool(policy.get("recent_failure_hold"))
    reasons = _string_list(policy.get("hold_reason_codes"))
    return {
        "recent_failure_hold": hold,
        "hold_reason_codes": reasons,
        "blocked_profiles": _string_list(policy.get("blocked_profiles")),
        "allowed_profiles": _string_list(policy.get("allowed_profiles")),
    }


def _axis_score(axis: Mapping[str, Any], *, ready: bool, attempted: bool, caption_ok: bool, rank_score_ok: bool) -> float:
    score = _candidate_rank_score(axis)
    if not ready:
        score *= 0.1
    if attempted:
        score *= 0.2
    if not caption_ok:
        score *= 0.8
    if not rank_score_ok:
        score *= 0.7
    return round(score, 6)


def _axis_recommendation(
    *,
    family: str,
    ready: bool,
    attempted: bool,
    caption_ok: bool,
    rank_score_ok: bool,
    policy: Mapping[str, Any],
) -> str:
    if not ready:
        return f"prepare_{family}_warm_cache_or_scan_another_axis"
    if attempted:
        return "search_different_source_axis_before_repeating_workers_prefetch"
    if not caption_ok:
        return "repair_caption_coverage_before_gpu_canary"
    if not rank_score_ok:
        return "scan_higher_pressure_source_axis_before_gpu_canary"
    if bool(policy.get("recent_failure_hold")):
        return "run_conservative_recheck_only_aggressive_held"
    return "run_release_relevant_conservative_recheck"


def _axis_brief(axis: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: axis.get(key)
        for key in (
            "axis_id",
            "source_data",
            "sample_offset",
            "score",
            "candidate_rank_score",
            "caption_sample_coverage",
            "blocked_reasons",
            "recommendation",
        )
        if key in axis
    }


def _family_axis_diagnostics(family_axes: Sequence[Mapping[str, Any]], top_candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ready_axes = [item for item in family_axes if bool(item.get("cache_ready"))]
    unattempted_ready_axes = [
        item for item in ready_axes if not bool(item.get("attempted_or_completed"))
    ]
    high_quality_ready_axes = [item for item in ready_axes if bool(item.get("quality_ok"))]
    unattempted_high_quality_ready_axes = [
        item for item in high_quality_ready_axes if not bool(item.get("attempted_or_completed"))
    ]
    planned_axes = [item for item in family_axes if bool(item.get("planned_followup_attempt"))]
    completed_axes = [item for item in family_axes if bool(item.get("completed_existing_evidence"))]
    completed_high_quality_axes = [item for item in completed_axes if bool(item.get("quality_ok"))]
    planned_high_quality_axes = [item for item in planned_axes if bool(item.get("quality_ok"))]
    low_quality_ready_axes = [
        item
        for item in unattempted_ready_axes
        if not bool(item.get("quality_ok"))
    ]

    reason_codes: set[str] = set()
    if not family_axes:
        reason_codes.add("source_axis_scan_empty")
    elif not ready_axes:
        reason_codes.add("no_ready_family_cache_axis")
    if not top_candidates and ready_axes and not unattempted_high_quality_ready_axes:
        reason_codes.add("no_unattempted_high_quality_ready_axis")
    if completed_high_quality_axes:
        reason_codes.add("completed_high_quality_axes")
    if planned_high_quality_axes:
        reason_codes.add("planned_high_quality_axes")
    if low_quality_ready_axes:
        reason_codes.add("remaining_ready_axes_low_quality")
    if any(not bool(item.get("caption_ok")) for item in low_quality_ready_axes):
        reason_codes.add("remaining_ready_axes_caption_coverage_low")
    if any(not bool(item.get("rank_score_ok")) for item in low_quality_ready_axes):
        reason_codes.add("remaining_ready_axes_rank_score_low")
    if ready_axes and len(ready_axes) == len([item for item in ready_axes if bool(item.get("attempted_or_completed"))]):
        reason_codes.add("all_ready_axes_already_attempted")

    if top_candidates:
        source_axis_state = "candidate_available"
    elif not family_axes:
        source_axis_state = "no_source_axes_found"
    elif not ready_axes:
        source_axis_state = "no_ready_source_axis"
    elif not unattempted_high_quality_ready_axes:
        source_axis_state = "exhausted_current_source_axis"
    else:
        source_axis_state = "needs_new_source_axis"

    return {
        "source_axis_state": source_axis_state,
        "source_axis_exhausted": source_axis_state == "exhausted_current_source_axis",
        "exhaustion_reason_codes": sorted(reason_codes) if not top_candidates else [],
        "ready_axis_count": len(ready_axes),
        "ready_unattempted_axis_count": len(unattempted_ready_axes),
        "high_quality_ready_axis_count": len(high_quality_ready_axes),
        "unattempted_high_quality_ready_axis_count": len(unattempted_high_quality_ready_axes),
        "planned_axis_count": len(planned_axes),
        "planned_high_quality_axis_count": len(planned_high_quality_axes),
        "completed_axis_count": len(completed_axes),
        "completed_high_quality_axis_count": len(completed_high_quality_axes),
        "low_quality_ready_axis_count": len(low_quality_ready_axes),
        "low_quality_ready_axes": [_axis_brief(item) for item in low_quality_ready_axes[:3]],
    }


def _rank_family_axes(
    source_scan: Mapping[str, Any],
    followup_run_plan: Mapping[str, Any],
    family: str,
    attempts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    policies = _family_policies(followup_run_plan)
    policy = policies.get(family, {})
    ranked: list[dict[str, Any]] = []
    for axis in _source_axes(source_scan):
        readiness = _family_readiness(axis, family)
        ready = bool(readiness.get("cache_ready")) or family in [_family_key(item) for item in axis.get("ready_families", [])]
        attempted_matches = _axis_attempts(axis, family, attempts)
        attempted = bool(attempted_matches)
        caption_coverage = _safe_float(axis.get("caption_sample_coverage"), 1.0)
        caption_ok = caption_coverage >= CAPTION_COVERAGE_THRESHOLD
        candidate_rank_score = _candidate_rank_score(axis)
        rank_score_ok = candidate_rank_score >= MIN_CANDIDATE_RANK_SCORE
        quality_ok = caption_ok and rank_score_ok
        planned_attempt = any(not bool(match.get("completed_existing_evidence")) for match in attempted_matches)
        completed_evidence = any(bool(match.get("completed_existing_evidence")) for match in attempted_matches)
        blocked_reasons = _string_list(readiness.get("blocked_reasons"))
        if completed_evidence:
            blocked_reasons.append("axis_already_has_completed_evidence")
        if planned_attempt:
            blocked_reasons.append("axis_already_in_followup_run_plan")
        if not caption_ok:
            blocked_reasons.append("caption_coverage_below_scout_threshold")
        if not rank_score_ok:
            blocked_reasons.append("candidate_rank_score_below_scout_threshold")
        state = "candidate" if ready and not attempted and quality_ok else "review_or_blocked"
        ranked.append(
            {
                "axis_id": f"{family}:{axis.get('source_manifest_sha1') or axis.get('root')}:{axis.get('sample_offset')}",
                "family": family,
                "state": state,
                "score": _axis_score(
                    axis,
                    ready=ready,
                    attempted=attempted,
                    caption_ok=caption_ok,
                    rank_score_ok=rank_score_ok,
                ),
                "source_data": str(axis.get("root") or ""),
                "sample_offset": _safe_int(axis.get("sample_offset")),
                "samples": _safe_int(axis.get("sample_image_count"), _safe_int(axis.get("samples_requested"), 0)),
                "source_manifest_sha1": str(axis.get("source_manifest_sha1") or ""),
                "pressure_score": _safe_float(axis.get("pressure_score")),
                "candidate_rank_score": candidate_rank_score,
                "caption_sample_coverage": round(caption_coverage, 6),
                "caption_ok": caption_ok,
                "rank_score_ok": rank_score_ok,
                "quality_ok": quality_ok,
                "cache_ready": ready,
                "cache_status": str(readiness.get("status") or "unknown"),
                "attempted_in_followup_plan": planned_attempt,
                "attempted_or_completed": attempted,
                "planned_followup_attempt": planned_attempt,
                "completed_existing_evidence": completed_evidence,
                "attempts": attempted_matches,
                "blocked_reasons": sorted(set(blocked_reasons)),
                "policy": _policy_signals(policy),
                "recommendation": _axis_recommendation(
                    family=family,
                    ready=ready,
                    attempted=attempted,
                    caption_ok=caption_ok,
                    rank_score_ok=rank_score_ok,
                    policy=policy,
                ),
                "image_stats": dict(_mapping(axis.get("image_stats"))),
            }
        )
    ranked.sort(key=lambda item: (item["state"] == "candidate", float(item["score"])), reverse=True)
    return ranked


def build_p60_source_axis_scout(
    source_scan: Mapping[str, Any],
    *,
    followup_run_plan: Mapping[str, Any] | None = None,
    natural_load_canary: Mapping[str, Any] | None = None,
    completed_evidence: Any = None,
    max_axes_per_family: int = 5,
) -> dict[str, Any]:
    """Combine source scans and P60 blocker policy into a non-GPU axis scout."""

    run_plan = _mapping(followup_run_plan)
    canary = _mapping(natural_load_canary)
    families = _families(source_scan, run_plan, canary)
    max_axes = max(_safe_int(max_axes_per_family, 5), 1)
    attempts = _attempted_axes(run_plan, completed_evidence)
    summaries: list[dict[str, Any]] = []
    ranked_axes: list[dict[str, Any]] = []
    for family in families:
        family_axes = _rank_family_axes(source_scan, run_plan, family, attempts)
        top_candidates = [item for item in family_axes if item["state"] == "candidate"]
        review_axes = [item for item in family_axes if item["state"] != "candidate"]
        best = top_candidates[0] if top_candidates else (family_axes[0] if family_axes else {})
        policy = _policy_signals(_family_policies(run_plan).get(family, {}))
        diagnostics = _family_axis_diagnostics(family_axes, top_candidates)
        summaries.append(
            {
                "family": family,
                "status": "candidate_available" if top_candidates else "needs_new_source_axis",
                **diagnostics,
                "candidate_count": len(top_candidates),
                "review_or_blocked_count": len(review_axes),
                "policy": policy,
                "top_axis": _axis_brief(best) if best else {},
                "next_action": best.get("recommendation") if best else f"scan_or_prepare_{family}_source_axis",
            }
        )
        ranked_axes.extend(family_axes[:max_axes])
    ranked_axes.sort(key=lambda item: (item["state"] == "candidate", float(item["score"])), reverse=True)
    return {
        "schema_version": 1,
        "report": P60_SOURCE_AXIS_SCOUT_REPORT,
        "roadmap": ROADMAP,
        "status": "source_axes_ranked" if ranked_axes else "no_source_axes_found",
        "source_scan_report": str(source_scan.get("report") or ""),
        "source_scan_mode": str(source_scan.get("mode") or "candidate_directories"),
        "source_axis_count": len(_source_axes(source_scan)),
        "scout_thresholds": {
            "caption_coverage_min": CAPTION_COVERAGE_THRESHOLD,
            "candidate_rank_score_min": MIN_CANDIDATE_RANK_SCORE,
        },
        "completed_evidence_axis_count": sum(1 for item in attempts if item.get("completed_existing_evidence")),
        "family_count": len(families),
        "families": families,
        "family_summaries": summaries,
        "source_axis_state_counts": {
            state: sum(1 for item in summaries if item.get("source_axis_state") == state)
            for state in sorted({str(item.get("source_axis_state") or "") for item in summaries})
            if state
        },
        "exhausted_family_count": sum(1 for item in summaries if item.get("source_axis_exhausted")),
        "ranked_axes": ranked_axes,
        "notes": [
            "This scout does not start GPU work.",
            "Aggressive worker/prefetch scaffolds remain blocked while recent_failure_hold is active.",
            "A candidate axis is not a release claim; release gates still require natural data-wait, throughput, loss, VRAM and action-boundary evidence.",
        ],
    }


__all__ = [
    "P60_SOURCE_AXIS_SCOUT_REPORT",
    "ROADMAP",
    "build_p60_source_axis_scout",
]
