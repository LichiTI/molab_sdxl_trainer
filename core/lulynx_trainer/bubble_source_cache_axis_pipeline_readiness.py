"""JSON-only readiness for the GPU-bubble source/cache-axis artifact pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SOURCE_CACHE_AXIS_PIPELINE_READINESS_REPORT = "bubble_source_cache_axis_pipeline_readiness_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"

EXPECTED_REPORTS = {
    "source_scan": "bubble_real_material_source_scan_v0",
    "source_axis_scout": "bubble_p60_source_axis_scout_v0",
    "source_axis_requirement": "bubble_p60_source_axis_requirement_v0",
    "external_input_admission": "bubble_gpu_bubble_external_input_admission_v0",
    "source_cache_axis_admission_preflight": "bubble_source_cache_axis_admission_preflight_v0",
    "source_cache_axis_repair_plan": "bubble_source_cache_axis_repair_plan_v0",
    "source_cache_axis_manual_canary_plan": "bubble_source_cache_axis_manual_canary_plan_v0",
    "source_cache_axis_identity_registry": "bubble_source_cache_axis_identity_registry_v0",
    "readiness_next_actions": "gpu_bubble_experiment_readiness_next_actions_v0",
}


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


def _family_key(value: Any) -> str:
    family = str(value or "").strip().lower().replace("-", "_")
    return "newbie" if family in {"dit", "newbie_dit"} else family


def _artifact_path(artifact_paths: Mapping[str, Any], key: str) -> str:
    value = artifact_paths.get(key)
    if value is None:
        return ""
    return str(value)


def _artifact_modified_ns(artifact_paths: Mapping[str, Any], key: str) -> int:
    path = _artifact_path(artifact_paths, key)
    if not path:
        return 0
    try:
        return Path(path).stat().st_mtime_ns
    except OSError:
        return 0


def _stage(
    key: str,
    payload: Mapping[str, Any],
    artifact_paths: Mapping[str, Any],
    *,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected = EXPECTED_REPORTS[key]
    actual = str(payload.get("report") or "")
    exists = bool(payload)
    report_ok = actual == expected
    roadmap = str(payload.get("roadmap") or "")
    roadmap_ok = bool(not exists or roadmap == ROADMAP)
    if not exists:
        status = "missing"
    elif not report_ok:
        status = "report_mismatch"
    elif not roadmap_ok:
        status = "roadmap_mismatch"
    else:
        status = "ok"
    return {
        "id": key,
        "status": status,
        "path": _artifact_path(artifact_paths, key),
        "expected_report": expected,
        "actual_report": actual,
        "expected_roadmap": ROADMAP,
        "roadmap": roadmap,
        "exists": exists,
        "report_ok": report_ok,
        "roadmap_ok": roadmap_ok,
        "modified_ns": _artifact_modified_ns(artifact_paths, key),
        "summary": dict(summary or {}),
    }


def _source_scan_summary(source_scan: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(source_scan.get("mode") or "")
    axes = _list(source_scan.get("windows")) if mode == "sample_windows" else _list(source_scan.get("candidates"))
    ready_families: set[str] = set()
    blocked_families: set[str] = set()
    for raw in axes:
        axis = _mapping(raw)
        ready_families.update(_family_key(item) for item in _strings(axis.get("ready_families")) if item)
        blocked_families.update(_family_key(item) for item in _strings(axis.get("blocked_families")) if item)
    return {
        "mode": mode,
        "axis_count": len(axes),
        "source_image_count": _safe_int(source_scan.get("source_image_count")),
        "ready_families": sorted(item for item in ready_families if item),
        "blocked_families": sorted(item for item in blocked_families if item),
    }


def _scout_summary(source_axis_scout: Mapping[str, Any]) -> dict[str, Any]:
    family_summaries = [_mapping(item) for item in _list(source_axis_scout.get("family_summaries"))]
    return {
        "family_count": _safe_int(source_axis_scout.get("family_count"), len(family_summaries)),
        "ranked_axis_count": len(_list(source_axis_scout.get("ranked_axes"))),
        "completed_evidence_axis_count": _safe_int(source_axis_scout.get("completed_evidence_axis_count")),
        "exhausted_family_count": _safe_int(source_axis_scout.get("exhausted_family_count")),
        "candidate_available_family_count": sum(
            1 for item in family_summaries if _safe_int(item.get("candidate_count")) > 0
        ),
        "families": [_family_key(item.get("family")) for item in family_summaries if item.get("family")],
    }


def _requirement_summary(source_axis_requirement: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": str(source_axis_requirement.get("status") or ""),
        "family_count": _safe_int(source_axis_requirement.get("family_count")),
        "external_input_required_count": _safe_int(source_axis_requirement.get("external_input_required_count")),
        "candidate_available_family_count": _safe_int(source_axis_requirement.get("candidate_available_family_count")),
        "completed_existing_command_count": _safe_int(source_axis_requirement.get("completed_existing_command_count")),
    }


def _external_admission_summary(external_input_admission: Mapping[str, Any]) -> dict[str, Any]:
    source_axis = _mapping(external_input_admission.get("source_axis"))
    sd15 = _mapping(external_input_admission.get("sd15"))
    return {
        "status": str(external_input_admission.get("status") or ""),
        "external_input_required": bool(external_input_admission.get("external_input_required")),
        "sd15_status": str(sd15.get("status") or ""),
        "sd15_checkpoint_exists": bool(sd15.get("checkpoint_exists")),
        "source_axis_status": str(source_axis.get("status") or ""),
        "source_axis_external_input_required_count": _safe_int(source_axis.get("external_input_required_count")),
        "source_axis_candidate_available_family_count": _safe_int(source_axis.get("candidate_available_family_count")),
    }


def _preflight_summary(preflight: Mapping[str, Any]) -> dict[str, Any]:
    candidate = _mapping(preflight.get("candidate"))
    matched = _mapping(preflight.get("matched_axis"))
    return {
        "status": str(preflight.get("status") or ""),
        "admitted": bool(preflight.get("admission_allows_protected_manual_gpu_plan")),
        "candidate_family": str(candidate.get("family") or ""),
        "candidate_root": str(candidate.get("root") or ""),
        "matched_axis_found": bool(matched.get("found")),
        "blocker_count": len(_strings(preflight.get("blockers"))),
    }


def _manual_plan_summary(manual_plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": str(manual_plan.get("status") or ""),
        "preflight_admitted": bool(manual_plan.get("preflight_admitted")),
        "command_count": _safe_int(manual_plan.get("command_count")),
        "blocked_command_count": _safe_int(manual_plan.get("blocked_command_count")),
        "requires_gpu_if_executed": bool(manual_plan.get("requires_gpu_if_executed")),
        "blocker_count": len(_strings(manual_plan.get("blockers"))),
    }


def _repair_plan_summary(repair_plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": str(repair_plan.get("status") or ""),
        "command_count": _safe_int(repair_plan.get("command_count")),
        "blocked_command_count": _safe_int(repair_plan.get("blocked_command_count")),
        "repair_axis_count": _safe_int(repair_plan.get("repair_axis_count")),
        "requires_gpu_if_executed": bool(repair_plan.get("requires_gpu_if_executed")),
        "family_runner_missing": _strings(repair_plan.get("family_runner_missing")),
        "blocker_count": len(_strings(repair_plan.get("blockers"))),
    }


def _identity_registry_summary(identity_registry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": str(identity_registry.get("status") or ""),
        "row_count": _safe_int(identity_registry.get("row_count")),
        "full_axis_identity_row_count": _safe_int(identity_registry.get("full_axis_identity_row_count")),
        "duplicate_or_stale_axis_count": _safe_int(identity_registry.get("duplicate_or_stale_axis_count")),
        "fresh_axis_candidate_count": _safe_int(identity_registry.get("fresh_axis_candidate_count")),
        "unsafe_row_count": _safe_int(identity_registry.get("unsafe_row_count")),
        "fail_closed": bool(identity_registry.get("fail_closed")),
    }


def _axis_readiness_summary(
    *,
    req_summary: Mapping[str, Any],
    ext_summary: Mapping[str, Any],
    preflight_summary: Mapping[str, Any],
    manual_summary: Mapping[str, Any],
    blockers: Sequence[str],
    external_required: bool,
    preflight_admitted: bool,
    manual_plan_ready: bool,
) -> dict[str, Any]:
    preflight_status = str(preflight_summary.get("status") or "")
    blocker_set = set(str(item) for item in blockers if item)
    duplicate_blockers = {
        "axis_already_has_completed_evidence",
        "axis_already_in_followup_run_plan",
        "candidate_axis_already_attempted_or_completed",
        "candidate_axis_already_in_followup_run_plan",
    }
    duplicate_or_stale = preflight_status == "blocked_duplicate_axis" or bool(blocker_set & duplicate_blockers)
    cache_not_ready = preflight_status == "blocked_no_ready_cache" or (
        "family_cache_not_ready_for_candidate_axis" in blocker_set
    )
    waiting_external_input = external_required or preflight_status == "external_input_required"

    if manual_plan_ready:
        status = "manual_canary_plan_ready"
    elif preflight_admitted:
        status = "axis_preflight_admitted"
    elif duplicate_or_stale:
        status = "duplicate_or_stale_axis_blocked"
    elif cache_not_ready:
        status = "cache_axis_not_ready"
    elif waiting_external_input:
        status = "axis_waiting_external_input"
    else:
        status = "axis_manual_review_required"

    return {
        "status": status,
        "preflight_status": preflight_status,
        "manual_plan_status": str(manual_summary.get("status") or ""),
        "requirement_status": str(req_summary.get("status") or ""),
        "external_admission_status": str(ext_summary.get("status") or ""),
        "external_input_required": bool(external_required),
        "waiting_external_input": status == "axis_waiting_external_input",
        "duplicate_or_stale_axis_blocked": bool(duplicate_or_stale),
        "cache_axis_not_ready": bool(cache_not_ready),
        "preflight_admitted": bool(preflight_admitted),
        "manual_canary_plan_ready": bool(manual_plan_ready),
        "candidate_family": str(preflight_summary.get("candidate_family") or ""),
        "candidate_root": str(preflight_summary.get("candidate_root") or ""),
        "matched_axis_found": bool(preflight_summary.get("matched_axis_found")),
        "source_axis_external_input_required_count": _safe_int(
            ext_summary.get("source_axis_external_input_required_count")
        ),
        "source_axis_candidate_available_family_count": _safe_int(
            ext_summary.get("source_axis_candidate_available_family_count")
        ),
        "blockers": sorted(blocker_set),
    }


def _readiness_summary(readiness: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_status": str(readiness.get("artifact_status") or ""),
        "release_readiness": str(readiness.get("release_readiness") or ""),
        "evidence_gap_count": len(_list(readiness.get("evidence_gaps"))),
        "next_action_count": len(_list(readiness.get("next_actions"))),
    }


def _pipeline_status(
    *,
    missing_or_mismatch: bool,
    external_input_required: bool,
    preflight_admitted: bool,
    manual_plan_ready: bool,
) -> str:
    if missing_or_mismatch:
        return "pipeline_incomplete"
    if manual_plan_ready:
        return "manual_canary_plan_ready"
    if preflight_admitted:
        return "manual_canary_plan_required"
    if external_input_required:
        return "external_input_required"
    return "manual_review_required"


def _stage_freshness_summary(
    stages: Sequence[Mapping[str, Any]],
    *,
    preflight_admitted: bool,
    manual_plan_ready: bool,
) -> dict[str, Any]:
    stage_by_id = {str(stage.get("id") or ""): stage for stage in stages}
    anchor_ids = [
        "source_axis_requirement",
        "external_input_admission",
    ]
    dependent_ids = [
        "source_cache_axis_admission_preflight",
        "source_cache_axis_manual_canary_plan",
    ]
    anchor_rows = [
        stage_by_id[stage_id]
        for stage_id in anchor_ids
        if _safe_int(stage_by_id.get(stage_id, {}).get("modified_ns")) > 0
    ]
    anchor_ns = max((_safe_int(stage.get("modified_ns")) for stage in anchor_rows), default=0)
    stale_stage_ids = [
        stage_id
        for stage_id in dependent_ids
        if 0 < _safe_int(stage_by_id.get(stage_id, {}).get("modified_ns")) < anchor_ns
    ]
    stale_ready_stage_ids: list[str] = []
    if preflight_admitted and "source_cache_axis_admission_preflight" in stale_stage_ids:
        stale_ready_stage_ids.append("source_cache_axis_admission_preflight")
    if manual_plan_ready and "source_cache_axis_manual_canary_plan" in stale_stage_ids:
        stale_ready_stage_ids.append("source_cache_axis_manual_canary_plan")
    unknown_stage_ids = [
        stage_id
        for stage_id in [*anchor_ids, *dependent_ids]
        if stage_by_id.get(stage_id, {}).get("exists")
        and _safe_int(stage_by_id.get(stage_id, {}).get("modified_ns")) <= 0
    ]
    return {
        "artifact_role": "gpu_bubble_source_cache_axis_pipeline_stage_freshness_summary",
        "freshness_checked": bool(anchor_ns),
        "anchor_stage_ids": anchor_ids,
        "dependent_stage_ids": dependent_ids,
        "stale_stage_count": len(stale_stage_ids),
        "stale_stage_ids": stale_stage_ids,
        "stale_ready_stage_count": len(stale_ready_stage_ids),
        "stale_ready_stage_ids": stale_ready_stage_ids,
        "unknown_freshness_stage_count": len(unknown_stage_ids),
        "unknown_freshness_stage_ids": unknown_stage_ids,
        "ready_stage_freshness_ok": not stale_ready_stage_ids,
        "freshness_ok": bool(anchor_ns) and not stale_stage_ids,
        "fail_closed": not stale_ready_stage_ids,
        "not_release_evidence": True,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
    }


def build_source_cache_axis_pipeline_readiness(
    *,
    source_scan: Mapping[str, Any],
    source_axis_scout: Mapping[str, Any],
    source_axis_requirement: Mapping[str, Any],
    external_input_admission: Mapping[str, Any],
    source_cache_axis_admission_preflight: Mapping[str, Any],
    source_cache_axis_manual_canary_plan: Mapping[str, Any],
    source_cache_axis_repair_plan: Mapping[str, Any] | None = None,
    source_cache_axis_identity_registry: Mapping[str, Any] | None = None,
    readiness_next_actions: Mapping[str, Any] | None = None,
    artifact_paths: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a non-GPU readiness report for the source/cache-axis artifact chain."""

    paths = _mapping(artifact_paths)
    stages = [
        _stage("source_scan", _mapping(source_scan), paths, summary=_source_scan_summary(_mapping(source_scan))),
        _stage(
            "source_axis_scout",
            _mapping(source_axis_scout),
            paths,
            summary=_scout_summary(_mapping(source_axis_scout)),
        ),
        _stage(
            "source_axis_requirement",
            _mapping(source_axis_requirement),
            paths,
            summary=_requirement_summary(_mapping(source_axis_requirement)),
        ),
        _stage(
            "external_input_admission",
            _mapping(external_input_admission),
            paths,
            summary=_external_admission_summary(_mapping(external_input_admission)),
        ),
        _stage(
            "source_cache_axis_admission_preflight",
            _mapping(source_cache_axis_admission_preflight),
            paths,
            summary=_preflight_summary(_mapping(source_cache_axis_admission_preflight)),
        ),
        _stage(
            "source_cache_axis_manual_canary_plan",
            _mapping(source_cache_axis_manual_canary_plan),
            paths,
            summary=_manual_plan_summary(_mapping(source_cache_axis_manual_canary_plan)),
        ),
        _stage(
            "source_cache_axis_identity_registry",
            _mapping(source_cache_axis_identity_registry),
            paths,
            summary=_identity_registry_summary(_mapping(source_cache_axis_identity_registry)),
        ),
    ]
    if source_cache_axis_repair_plan is not None:
        stages.insert(
            5,
            _stage(
                "source_cache_axis_repair_plan",
                _mapping(source_cache_axis_repair_plan),
                paths,
                summary=_repair_plan_summary(_mapping(source_cache_axis_repair_plan)),
            ),
        )
    readiness = _mapping(readiness_next_actions)
    if readiness:
        stages.append(
            _stage(
                "readiness_next_actions",
                readiness,
                paths,
                summary=_readiness_summary(readiness),
            )
        )
    roadmap_mismatch_ids = [item["id"] for item in stages if item["exists"] and not item.get("roadmap_ok")]

    missing_or_mismatch = any(item["status"] != "ok" for item in stages)
    req_summary = _requirement_summary(_mapping(source_axis_requirement))
    ext_summary = _external_admission_summary(_mapping(external_input_admission))
    preflight_summary = _preflight_summary(_mapping(source_cache_axis_admission_preflight))
    manual_summary = _manual_plan_summary(_mapping(source_cache_axis_manual_canary_plan))
    external_required = bool(ext_summary["external_input_required"]) or bool(
        req_summary["external_input_required_count"]
    )
    preflight_admitted = bool(preflight_summary["admitted"])
    manual_plan_ready = manual_summary["status"] == "protected_manual_canary_plan_ready"
    stage_freshness = _stage_freshness_summary(
        stages,
        preflight_admitted=preflight_admitted,
        manual_plan_ready=manual_plan_ready,
    )
    stale_ready_stage_ids = _strings(stage_freshness.get("stale_ready_stage_ids"))
    freshness_blocks_ready = bool(stale_ready_stage_ids)
    missing_or_mismatch = missing_or_mismatch or freshness_blocks_ready
    status = _pipeline_status(
        missing_or_mismatch=missing_or_mismatch,
        external_input_required=external_required,
        preflight_admitted=preflight_admitted,
        manual_plan_ready=manual_plan_ready,
    )
    blockers = []
    if missing_or_mismatch:
        blockers.extend(f"{item['id']}_{item['status']}" for item in stages if item["status"] != "ok")
    if external_required:
        blockers.append("external_source_or_cache_axis_required")
    if ext_summary["sd15_status"] == "checkpoint_required":
        blockers.append("sd15_checkpoint_required")
    blockers.extend(f"{stage_id}_stale_ready_artifact" for stage_id in stale_ready_stage_ids)
    blockers.extend(_strings(source_cache_axis_admission_preflight.get("blockers")))
    blockers.extend(_strings(_mapping(source_cache_axis_repair_plan).get("blockers")))
    blockers.extend(_strings(source_cache_axis_manual_canary_plan.get("blockers")))
    sorted_blockers = sorted(set(blockers))
    axis_readiness = _axis_readiness_summary(
        req_summary=req_summary,
        ext_summary=ext_summary,
        preflight_summary=preflight_summary,
        manual_summary=manual_summary,
        blockers=sorted_blockers,
        external_required=external_required,
        preflight_admitted=preflight_admitted,
        manual_plan_ready=manual_plan_ready,
    )

    next_actions: list[dict[str, Any]] = []
    if missing_or_mismatch:
        next_actions.append(
            {
                "id": "refresh_source_cache_axis_pipeline_artifacts",
                "status": "artifact_refresh_required",
                "requires_gpu_if_executed": False,
                "safe_to_auto_start": False,
            }
        )
    if external_required:
        repair_command_count = _safe_int(_mapping(source_cache_axis_repair_plan).get("command_count"))
        next_actions.append(
            {
                "id": "review_source_cache_axis_repair_plan" if repair_command_count else "provide_new_source_or_cache_axis",
                "status": "external_input_required",
                "requires_gpu_if_executed": bool(repair_command_count),
                "safe_to_auto_start": False,
            }
        )
    if preflight_admitted and not manual_plan_ready:
        next_actions.append(
            {
                "id": "build_source_cache_axis_manual_canary_plan",
                "status": "manual_plan_required",
                "requires_gpu_if_executed": False,
                "safe_to_auto_start": False,
            }
        )
    if manual_plan_ready:
        next_actions.append(
            {
                "id": "review_protected_manual_canary_plan",
                "status": "manual_review_ready",
                "requires_gpu_if_executed": True,
                "safe_to_auto_start": False,
            }
        )

    return {
        "schema_version": 1,
        "report": SOURCE_CACHE_AXIS_PIPELINE_READINESS_REPORT,
        "roadmap": ROADMAP,
        "status": status,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "release_claim_allowed": False,
        "not_release_evidence": True,
        "pipeline_complete": not missing_or_mismatch,
        "external_input_required": external_required,
        "preflight_admitted": preflight_admitted,
        "manual_canary_plan_ready": manual_plan_ready,
        "stage_count": len(stages),
        "stage_ok_count": sum(1 for item in stages if item["status"] == "ok"),
        "stage_freshness": stage_freshness,
        "stage_roadmap_lineage": {
            "expected_roadmap": ROADMAP,
            "stage_count": len(stages),
            "roadmap_mismatch_count": len(roadmap_mismatch_ids),
            "roadmap_mismatch_stage_ids": roadmap_mismatch_ids,
            "lineage_ok": not roadmap_mismatch_ids,
            "safe_to_auto_start": False,
            "release_claim_allowed": False,
            "not_release_evidence": True,
        },
        "axis_readiness_status": axis_readiness["status"],
        "axis_readiness": axis_readiness,
        "stages": stages,
        "blockers": sorted_blockers,
        "next_actions": next_actions,
        "blocked_actions": [
            "auto_start_gpu_heavy_from_pipeline_readiness",
            "promote_pipeline_readiness_as_release_evidence",
            "skip_preflight_before_manual_canary_plan",
            "skip_release_claim_rebuild_after_manual_run",
        ],
        "acceptance_gates": [
            "all_pipeline_artifact_reports_match_expected_schema",
            "preflight_admitted_before_manual_canary_plan_ready",
            "manual_plan_requires_explicit_execution",
            "post_run_natural_load_and_release_claims_rebuild_required",
        ],
        "notes": [
            "This report is JSON-only and does not start GPU work.",
            "Pipeline completeness is not release evidence; it only proves the source/cache-axis gate chain is wired.",
        ],
    }


__all__ = [
    "ROADMAP",
    "SOURCE_CACHE_AXIS_PIPELINE_READINESS_REPORT",
    "build_source_cache_axis_pipeline_readiness",
]
