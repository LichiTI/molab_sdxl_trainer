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


def _ranked_axes(source_axis_scout: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [_mapping(item) for item in _list(source_axis_scout.get("ranked_axes")) if _mapping(item)]


def _match_axis(
    source_axis_scout: Mapping[str, Any],
    *,
    family: str,
    candidate_root: str,
    sample_offset: int | None,
    source_manifest_sha1: str,
) -> Mapping[str, Any]:
    root_key = _norm_path(candidate_root)
    manifest_key = str(source_manifest_sha1 or "").strip()
    for axis in _ranked_axes(source_axis_scout):
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


def build_source_cache_axis_admission_preflight(
    *,
    source_axis_scout: Mapping[str, Any],
    source_axis_requirement: Mapping[str, Any] | None = None,
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
    else:
        axis = _match_axis(
            _mapping(source_axis_scout),
            family=family_key,
            candidate_root=root,
            sample_offset=sample_offset,
            source_manifest_sha1=manifest,
        )
        status, blockers = _classification(axis)

    admitted = status == "admitted"
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
        },
        "matched_axis": {
            "found": bool(axis),
            "state": str(axis.get("state") or ""),
            "source_data": str(axis.get("source_data") or ""),
            "sample_offset": _safe_int(axis.get("sample_offset")) if axis else None,
            "source_manifest_sha1": str(axis.get("source_manifest_sha1") or ""),
            "cache_ready": bool(axis.get("cache_ready")),
            "quality_ok": bool(axis.get("quality_ok")),
            "caption_sample_coverage": _safe_float(axis.get("caption_sample_coverage")),
            "candidate_rank_score": _safe_float(axis.get("candidate_rank_score")),
            "attempted_or_completed": bool(axis.get("attempted_or_completed")),
            "completed_existing_evidence": bool(axis.get("completed_existing_evidence")),
            "attempted_in_followup_plan": bool(axis.get("attempted_in_followup_plan")),
        },
        "source_axis_requirement": {
            "requirement": str(requirement.get("requirement") or ""),
            "source_axis_state": str(requirement.get("source_axis_state") or ""),
            "do_not_rerun_current_axis": bool(requirement.get("do_not_rerun_current_axis")),
            "current_source_roots": _strings(requirement.get("current_source_roots")),
        },
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
        "next_action": "generate_protected_manual_canary_plan" if admitted else "provide_or_repair_source_cache_axis",
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
