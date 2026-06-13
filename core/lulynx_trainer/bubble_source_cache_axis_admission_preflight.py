"""Candidate source/cache-axis admission preflight for GPU bubble canaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .bubble_p60_source_axis_scout import (
    CAPTION_COVERAGE_THRESHOLD,
    MIN_CANDIDATE_RANK_SCORE,
)


SOURCE_CACHE_AXIS_ADMISSION_PREFLIGHT_REPORT = "bubble_source_cache_axis_admission_preflight_v0"
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


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


def _family_requirements(source_axis_requirement: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for raw in _list(source_axis_requirement.get("families")):
        item = _mapping(raw)
        family = _family_key(item.get("family"))
        if family:
            rows[family] = item
    return rows


def _blocked_canary_families(requirements: Mapping[str, Mapping[str, Any]]) -> set[str]:
    return {
        family
        for family, requirement in requirements.items()
        if bool(requirement.get("blocked_by_natural_load_canary"))
    }


def _family_is_not_current_gate(
    family: str,
    requirements: Mapping[str, Mapping[str, Any]],
) -> bool:
    requirement = requirements.get(family)
    if not requirement or "blocked_by_natural_load_canary" not in requirement:
        return False
    return not bool(requirement.get("blocked_by_natural_load_canary"))


def _ranked_axes(source_axis_scout: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [_mapping(item) for item in _list(source_axis_scout.get("ranked_axes")) if _mapping(item)]


def _scout_axis_by_family_root_offset(source_axis_scout: Mapping[str, Any]) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    rows: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for axis in _ranked_axes(source_axis_scout):
        key = (
            _family_key(axis.get("family")),
            _norm_path(axis.get("source_data")),
            _safe_int(axis.get("sample_offset")),
        )
        if key[0] and key[1] and key not in rows:
            rows[key] = axis
    return rows


def _warm_cache_inventory_axes(
    newbie_warm_cache_inventory: Mapping[str, Any],
    *,
    source_axis_scout: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    scout_by_axis = _scout_axis_by_family_root_offset(source_axis_scout)
    rows: list[Mapping[str, Any]] = []
    for raw in _list(newbie_warm_cache_inventory.get("axes")):
        axis = _mapping(raw)
        source_data = str(axis.get("source_data") or "")
        sample_offset = _safe_int(axis.get("sample_offset"))
        scout_axis = scout_by_axis.get(("newbie", _norm_path(source_data), sample_offset), {})
        manifest = str(axis.get("source_manifest_sha1") or "").strip() or str(
            scout_axis.get("source_manifest_sha1") or ""
        ).strip()
        if not source_data or sample_offset <= 0:
            continue
        completed_count = _safe_int(axis.get("completed_canary_command_count"))
        do_not_rerun = bool(axis.get("do_not_rerun_without_new_axis"))
        completed_or_stale = completed_count > 0 or do_not_rerun
        blocked_reasons = set(_strings(axis.get("blocked_reasons")))
        if str(newbie_warm_cache_inventory.get("status") or ""):
            blocked_reasons.add(str(newbie_warm_cache_inventory.get("status") or ""))
        if completed_or_stale:
            blocked_reasons.add("candidate_axis_already_attempted_or_completed")
        if do_not_rerun:
            blocked_reasons.add("do_not_rerun_without_new_axis")
        caption_coverage = _safe_float(
            scout_axis.get("caption_sample_coverage"),
            _safe_float(_mapping(axis.get("manifest")).get("caption_coverage")),
        )
        candidate_rank_score = _safe_float(scout_axis.get("candidate_rank_score"))
        caption_ok = caption_coverage >= CAPTION_COVERAGE_THRESHOLD
        rank_score_ok = bool(scout_axis) and candidate_rank_score >= MIN_CANDIDATE_RANK_SCORE
        if not scout_axis:
            blocked_reasons.add("candidate_rank_score_missing_from_scout")
        if not caption_ok:
            blocked_reasons.add("caption_coverage_below_scout_threshold")
        if not rank_score_ok:
            blocked_reasons.add("candidate_rank_score_below_scout_threshold")
        cache_ready = bool(axis.get("cache_ready"))
        rows.append(
            {
                "family": "newbie",
                "source_data": source_data,
                "prepared_source_data": str(axis.get("source_root") or ""),
                "source_data_original": source_data,
                "sample_offset": sample_offset,
                "source_manifest_sha1": manifest,
                "state": str(axis.get("status") or axis.get("axis_kind") or ""),
                "axis_kind": str(axis.get("axis_kind") or ""),
                "cache_ready": cache_ready,
                "quality_ok": bool(cache_ready and caption_ok and rank_score_ok),
                "caption_ok": caption_ok,
                "rank_score_ok": rank_score_ok,
                "caption_sample_coverage": caption_coverage,
                "candidate_rank_score": candidate_rank_score,
                "attempted_or_completed": completed_or_stale,
                "completed_existing_evidence": completed_count > 0,
                "attempted_in_followup_plan": False,
                "planned_followup_attempt": False,
                "blocked_reasons": sorted(blocked_reasons),
                "claimable": bool(axis.get("claimable")),
                "do_not_rerun_without_new_axis": do_not_rerun,
                "completed_canary_command_count": completed_count,
                "source_kind": "warm_cache_inventory_axis",
                "scout_axis_found": bool(scout_axis),
            }
        )
    return rows


def _warm_cache_inventory_candidate(
    newbie_warm_cache_inventory: Mapping[str, Any],
    *,
    source_axis_scout: Mapping[str, Any],
    allowed_families: set[str] | None = None,
    fresh_only: bool = False,
) -> Mapping[str, Any]:
    axes = _warm_cache_inventory_axes(
        newbie_warm_cache_inventory,
        source_axis_scout=source_axis_scout,
    )
    if allowed_families is not None:
        axes = [axis for axis in axes if _family_key(axis.get("family")) in allowed_families]
    completed = [
        axis for axis in axes
        if _safe_int(axis.get("completed_canary_command_count")) > 0
        or bool(axis.get("do_not_rerun_without_new_axis"))
    ]
    ready = [axis for axis in axes if bool(axis.get("cache_ready"))]
    fresh_ready = [axis for axis in ready if axis not in completed]
    if fresh_only:
        return fresh_ready[0] if fresh_ready else {}
    candidates = fresh_ready or ready or completed or axes
    return candidates[0] if candidates else {}


def _scout_candidate(
    source_axis_scout: Mapping[str, Any],
    *,
    allowed_families: set[str] | None = None,
) -> Mapping[str, Any]:
    ranked = [_mapping(item) for item in _list(source_axis_scout.get("ranked_axes")) if _mapping(item)]
    candidates = [
        axis
        for axis in ranked
        if str(axis.get("state") or "") == "candidate"
        and (allowed_families is None or _family_key(axis.get("family")) in allowed_families)
        and bool(axis.get("cache_ready"))
        and bool(axis.get("quality_ok"))
        and not bool(axis.get("attempted_or_completed"))
        and not bool(axis.get("completed_existing_evidence"))
        and not bool(axis.get("planned_followup_attempt"))
        and not bool(axis.get("attempted_in_followup_plan"))
    ]
    if not candidates:
        return {}
    candidates.sort(
        key=lambda axis: (
            _safe_float(axis.get("score")),
            _safe_float(axis.get("candidate_rank_score")),
            _safe_float(axis.get("caption_sample_coverage")),
        ),
        reverse=True,
    )
    return candidates[0]


def _scout_review_axis(
    source_axis_scout: Mapping[str, Any],
    *,
    allowed_families: set[str],
) -> Mapping[str, Any]:
    ranked = [_mapping(item) for item in _list(source_axis_scout.get("ranked_axes")) if _mapping(item)]
    candidates = [
        axis
        for axis in ranked
        if _family_key(axis.get("family")) in allowed_families
        and str(axis.get("state") or "") != "candidate"
        and str(axis.get("source_data") or "").strip()
        and not bool(axis.get("attempted_or_completed"))
        and not bool(axis.get("completed_existing_evidence"))
        and not bool(axis.get("planned_followup_attempt"))
        and not bool(axis.get("attempted_in_followup_plan"))
    ]
    if not candidates:
        return {}
    candidates.sort(
        key=lambda axis: (
            bool(axis.get("cache_ready")),
            bool(axis.get("quality_ok")),
            _safe_float(axis.get("score")),
            _safe_float(axis.get("candidate_rank_score")),
            _safe_float(axis.get("caption_sample_coverage")),
        ),
        reverse=True,
    )
    return candidates[0]


def _match_axis(
    source_axis_scout: Mapping[str, Any],
    *,
    family: str,
    candidate_root: str,
    sample_offset: int | None,
    source_manifest_sha1: str,
    newbie_warm_cache_inventory: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    root_key = _norm_path(candidate_root)
    manifest_key = str(source_manifest_sha1 or "").strip()
    axes = [
        *_warm_cache_inventory_axes(
            _mapping(newbie_warm_cache_inventory),
            source_axis_scout=source_axis_scout,
        ),
        *_ranked_axes(source_axis_scout),
    ]
    for axis in axes:
        if _family_key(axis.get("family")) != family:
            continue
        if root_key and _norm_path(axis.get("source_data")) != root_key:
            continue
        if sample_offset is not None and _safe_int(axis.get("sample_offset")) != sample_offset:
            continue
        if manifest_key and str(axis.get("source_manifest_sha1") or "").strip() != manifest_key:
            continue
        return axis
    return {}


def _classification(axis: Mapping[str, Any]) -> tuple[str, list[str]]:
    blockers = set(_strings(axis.get("blocked_reasons")))
    if not axis:
        return "external_input_required", ["candidate_axis_not_found_in_source_axis_scout"]
    if bool(axis.get("attempted_or_completed")) or bool(axis.get("completed_existing_evidence")):
        blockers.add("candidate_axis_already_attempted_or_completed")
    if bool(axis.get("attempted_in_followup_plan")) or bool(axis.get("planned_followup_attempt")):
        blockers.add("candidate_axis_already_in_followup_run_plan")
    if any(
        reason in blockers
        for reason in (
            "axis_already_has_completed_evidence",
            "axis_already_in_followup_run_plan",
            "candidate_axis_already_attempted_or_completed",
            "candidate_axis_already_in_followup_run_plan",
        )
    ):
        return "blocked_duplicate_axis", sorted(blockers)
    if not bool(axis.get("cache_ready")):
        blockers.add("family_cache_not_ready_for_candidate_axis")
        return "blocked_no_ready_cache", sorted(blockers)
    if not bool(axis.get("quality_ok")):
        if not bool(axis.get("caption_ok")):
            blockers.add("caption_coverage_below_scout_threshold")
        if not bool(axis.get("rank_score_ok")):
            blockers.add("candidate_rank_score_below_scout_threshold")
        return "blocked_low_quality", sorted(blockers)
    if str(axis.get("state") or "") == "candidate":
        return "admitted", sorted(blockers)
    return "manual_review_required", sorted(blockers)


def _new_axis_requirement_summary(
    *,
    status: str,
    axis: Mapping[str, Any],
    blockers: Sequence[str],
) -> dict[str, Any]:
    blocker_set = set(blockers)
    duplicate_or_stale = status == "blocked_duplicate_axis" or any(
        reason in blocker_set
        for reason in (
            "axis_already_has_completed_evidence",
            "axis_already_in_followup_run_plan",
            "candidate_axis_already_attempted_or_completed",
            "candidate_axis_already_in_followup_run_plan",
            "do_not_rerun_without_new_axis",
            "warm_cache_axis_completed_but_not_release_ready",
        )
    )
    current_axis_do_not_rerun = bool(axis.get("do_not_rerun_without_new_axis"))
    new_axis_required = duplicate_or_stale or current_axis_do_not_rerun
    if new_axis_required:
        next_action = "provide_distinct_unattempted_source_cache_axis"
    elif status == "blocked_low_quality":
        next_action = "repair_candidate_quality_or_rank_before_manual_canary"
    elif status == "blocked_no_ready_cache":
        next_action = "prepare_family_cache_for_candidate_axis"
    elif status == "admitted":
        next_action = "generate_protected_manual_canary_plan"
    else:
        next_action = "provide_or_repair_source_cache_axis"
    return {
        "new_axis_required": new_axis_required,
        "duplicate_or_stale_axis_blocked": duplicate_or_stale,
        "current_axis_do_not_rerun_without_new_axis": current_axis_do_not_rerun,
        "current_axis_completed_canary_command_count": _safe_int(
            axis.get("completed_canary_command_count")
        ),
        "reason_ids": sorted(
            reason
            for reason in blocker_set
            if reason
            in {
                "axis_already_has_completed_evidence",
                "axis_already_in_followup_run_plan",
                "candidate_axis_already_attempted_or_completed",
                "candidate_axis_already_in_followup_run_plan",
                "do_not_rerun_without_new_axis",
                "warm_cache_axis_completed_but_not_release_ready",
            }
        ),
        "required_identity_change_fields": [
            "source_data",
            "sample_offset",
            "source_manifest_sha1",
        ],
        "same_root_identity_change_fields": [
            "sample_offset",
            "source_manifest_sha1",
        ],
        "acceptance_requirements": [
            "new_axis_not_attempted_or_completed",
            f"caption_sample_coverage>={CAPTION_COVERAGE_THRESHOLD}",
            f"candidate_rank_score>={MIN_CANDIDATE_RANK_SCORE}",
            "family_cache_ready_for_candidate_axis",
        ],
        "next_action": next_action,
    }


def build_source_cache_axis_admission_preflight(
    *,
    source_axis_scout: Mapping[str, Any],
    source_axis_requirement: Mapping[str, Any] | None = None,
    newbie_warm_cache_inventory: Mapping[str, Any] | None = None,
    candidate_root: str | None = None,
    family: str | None = None,
    sample_offset: int | None = None,
    source_manifest_sha1: str | None = None,
) -> dict[str, Any]:
    """Classify one candidate source/cache axis without starting GPU work."""

    family_key = _family_key(family)
    root = str(candidate_root or "").strip()
    manifest = str(source_manifest_sha1 or "").strip()
    requirements = _family_requirements(_mapping(source_axis_requirement))
    candidate_source = "explicit_args" if root or family_key or sample_offset is not None or manifest else ""
    if not root and not family_key:
        allowed_families = _blocked_canary_families(requirements) or None
        scout_candidate = _scout_candidate(_mapping(source_axis_scout), allowed_families=allowed_families)
        if scout_candidate:
            family_key = _family_key(scout_candidate.get("family"))
            root = str(scout_candidate.get("source_data") or "")
            sample_offset = _safe_int(scout_candidate.get("sample_offset"))
            manifest = str(scout_candidate.get("source_manifest_sha1") or "").strip()
            candidate_source = "source_axis_scout"
        inventory_candidate = (
            _warm_cache_inventory_candidate(
                _mapping(newbie_warm_cache_inventory),
                source_axis_scout=_mapping(source_axis_scout),
                allowed_families=allowed_families,
                fresh_only=True,
            )
            if not scout_candidate
            else {}
        )
        if inventory_candidate:
            family_key = _family_key(inventory_candidate.get("family"))
            root = str(inventory_candidate.get("source_data") or "")
            sample_offset = _safe_int(inventory_candidate.get("sample_offset"))
            manifest = str(inventory_candidate.get("source_manifest_sha1") or "").strip()
            candidate_source = "newbie_warm_cache_inventory"
        scout_review_axis = (
            _scout_review_axis(_mapping(source_axis_scout), allowed_families=allowed_families)
            if allowed_families is not None and not scout_candidate and not inventory_candidate
            else {}
        )
        if scout_review_axis:
            family_key = _family_key(scout_review_axis.get("family"))
            root = str(scout_review_axis.get("source_data") or "")
            sample_offset = _safe_int(scout_review_axis.get("sample_offset"))
            manifest = str(scout_review_axis.get("source_manifest_sha1") or "").strip()
            candidate_source = "source_axis_scout_review"
        if not scout_candidate and not scout_review_axis and not inventory_candidate:
            inventory_candidate = _warm_cache_inventory_candidate(
                _mapping(newbie_warm_cache_inventory),
                source_axis_scout=_mapping(source_axis_scout),
                allowed_families=allowed_families,
            )
        if not scout_candidate and not scout_review_axis and inventory_candidate:
            family_key = _family_key(inventory_candidate.get("family"))
            root = str(inventory_candidate.get("source_data") or "")
            sample_offset = _safe_int(inventory_candidate.get("sample_offset"))
            manifest = str(inventory_candidate.get("source_manifest_sha1") or "").strip()
            candidate_source = "newbie_warm_cache_inventory"
    requirement = requirements.get(family_key, {})
    missing_inputs = []
    if not root:
        missing_inputs.append("candidate_root")
    if not family_key:
        missing_inputs.append("family")

    if missing_inputs:
        axis: Mapping[str, Any] = {}
        status = "external_input_required"
        blockers = [f"{item}_required" for item in missing_inputs]
    elif _family_is_not_current_gate(family_key, requirements):
        axis = _match_axis(
            _mapping(source_axis_scout),
            family=family_key,
            candidate_root=root,
            sample_offset=sample_offset,
            source_manifest_sha1=manifest,
            newbie_warm_cache_inventory=newbie_warm_cache_inventory,
        )
        status = "blocked_family_not_current_natural_load_gate"
        blockers = ["family_not_blocked_by_natural_load_canary"]
    else:
        axis = _match_axis(
            _mapping(source_axis_scout),
            family=family_key,
            candidate_root=root,
            sample_offset=sample_offset,
            source_manifest_sha1=manifest,
            newbie_warm_cache_inventory=newbie_warm_cache_inventory,
        )
        status, blockers = _classification(axis)

    admitted = status == "admitted"
    new_axis_requirement = _new_axis_requirement_summary(
        status=status,
        axis=_mapping(axis),
        blockers=blockers,
    )
    return {
        "schema_version": 1,
        "report": SOURCE_CACHE_AXIS_ADMISSION_PREFLIGHT_REPORT,
        "roadmap": ROADMAP,
        "status": status,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "release_claim_allowed": False,
        "publishable": False,
        "not_release_evidence": True,
        "admission_allows_protected_manual_gpu_plan": admitted,
        "candidate": {
            "family": family_key,
            "root": root,
            "sample_offset": sample_offset,
            "source_manifest_sha1": manifest,
            "candidate_source": candidate_source,
        },
        "matched_axis": {
            "found": bool(axis),
            "state": str(axis.get("state") or ""),
            "source_data": str(axis.get("source_data") or ""),
            "prepared_source_data": str(axis.get("prepared_source_data") or ""),
            "source_data_original": str(axis.get("source_data_original") or ""),
            "sample_offset": _safe_int(axis.get("sample_offset")) if axis else None,
            "source_manifest_sha1": str(axis.get("source_manifest_sha1") or ""),
            "cache_ready": bool(axis.get("cache_ready")),
            "quality_ok": bool(axis.get("quality_ok")),
            "caption_sample_coverage": _safe_float(axis.get("caption_sample_coverage")),
            "candidate_rank_score": _safe_float(axis.get("candidate_rank_score")),
            "attempted_or_completed": bool(axis.get("attempted_or_completed")),
            "completed_existing_evidence": bool(axis.get("completed_existing_evidence")),
            "attempted_in_followup_plan": bool(axis.get("attempted_in_followup_plan")),
            "source_kind": str(axis.get("source_kind") or "ranked_axis") if axis else "",
            "axis_kind": str(axis.get("axis_kind") or ""),
            "claimable": bool(axis.get("claimable")),
            "do_not_rerun_without_new_axis": bool(axis.get("do_not_rerun_without_new_axis")),
            "completed_canary_command_count": _safe_int(axis.get("completed_canary_command_count")),
        },
        "source_axis_requirement": {
            "requirement": str(requirement.get("requirement") or ""),
            "source_axis_state": str(requirement.get("source_axis_state") or ""),
            "do_not_rerun_current_axis": bool(requirement.get("do_not_rerun_current_axis")),
            "current_source_roots": _strings(requirement.get("current_source_roots")),
        },
        "new_axis_requirement_summary": new_axis_requirement,
        "blockers": blockers,
        "acceptance_gates": [
            f"caption_sample_coverage>={CAPTION_COVERAGE_THRESHOLD}",
            f"candidate_rank_score>={MIN_CANDIDATE_RANK_SCORE}",
            "family_cache_ready_for_candidate_axis",
            "candidate_axis_not_attempted_or_completed",
            "same_root_is_allowed_only_with_distinct_unattempted_offset_or_manifest",
            "manual_gpu_run_only_after_admission_review",
        ],
        "blocked_actions": [
            "auto_start_natural_load_canary_from_preflight",
            "promote_preflight_as_release_evidence",
            "rerun_completed_followup_out_dirs_without_new_axis",
        ],
        "next_action": (
            "generate_protected_manual_canary_plan"
            if admitted
            else str(new_axis_requirement.get("next_action") or "provide_or_repair_source_cache_axis")
        ),
        "notes": [
            "This preflight is JSON-only and does not start GPU work.",
            "Admission only permits protected manual plan generation; it is not release evidence.",
        ],
    }


__all__ = [
    "ROADMAP",
    "SOURCE_CACHE_AXIS_ADMISSION_PREFLIGHT_REPORT",
    "build_source_cache_axis_admission_preflight",
]
