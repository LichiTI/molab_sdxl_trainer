"""External-input admission summary for GPU bubble release gaps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


EXTERNAL_INPUT_ADMISSION_REPORT = "bubble_gpu_bubble_external_input_admission_v0"
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


def _sd15_admission(sd15_readiness: Mapping[str, Any]) -> dict[str, Any]:
    prerequisites = _mapping(sd15_readiness.get("prerequisites"))
    output_status = _mapping(sd15_readiness.get("output_status"))
    checkpoint_exists = bool(prerequisites.get("sd15_checkpoint_exists"))
    evidence_ready = bool(sd15_readiness.get("evidence_ready"))
    execution_ready = bool(sd15_readiness.get("execution_ready"))
    coverage = _mapping(sd15_readiness.get("release_coverage"))
    blockers = _strings(sd15_readiness.get("blockers"))
    if bool(coverage.get("covered")):
        status = "release_gap_covered"
    elif evidence_ready:
        status = "evidence_available_pending_release_claim_refresh"
    elif execution_ready:
        status = "ready_for_manual_gpu_ab"
    elif not checkpoint_exists:
        status = "checkpoint_required"
    else:
        status = "blocked_missing_prerequisite"
    return {
        "family": "sd15",
        "release_case_id": str(sd15_readiness.get("release_case_id") or "sd15_lora_512"),
        "case_id": str(sd15_readiness.get("case_id") or "sd15_data_workers_smoke"),
        "status": status,
        "checkpoint_exists": checkpoint_exists,
        "checkpoint_path": str(prerequisites.get("sd15_checkpoint") or ""),
        "checkpoint_candidates": _strings(prerequisites.get("sd15_checkpoint_candidates")),
        "python_executable_exists": bool(prerequisites.get("python_executable_exists")),
        "source_data_exists": bool(prerequisites.get("source_data_exists")),
        "source_image_count": _safe_int(prerequisites.get("source_image_count")),
        "execution_ready": execution_ready,
        "evidence_ready": evidence_ready,
        "release_covered": bool(coverage.get("covered")),
        "blockers": blockers,
        "manual_execute_command": _strings(_mapping(sd15_readiness.get("commands")).get("execute_manual_gpu")),
        "expected_outputs": dict(_mapping(sd15_readiness.get("expected_outputs"))),
        "output_status": {
            "before_report_exists": bool(output_status.get("before_report_exists")),
            "after_report_exists": bool(output_status.get("after_report_exists")),
            "ab_evidence_exists": bool(output_status.get("ab_evidence_exists")),
        },
        "requires_external_input": not checkpoint_exists and not evidence_ready,
        "requires_gpu_if_executed": bool(sd15_readiness.get("requires_gpu_if_executed")),
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "blocked_actions": _strings(sd15_readiness.get("blocked_actions")),
        "acceptance_gates": _strings(sd15_readiness.get("acceptance_gates")),
    }


def _source_family_requirement(source_axis_requirement: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for raw in _list(source_axis_requirement.get("families")):
        item = _mapping(raw)
        family = _family_key(item.get("family"))
        if family:
            rows[family] = item
    return rows


def _candidate_roots(source_scan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if str(source_scan.get("mode") or "") == "sample_windows":
        return [_mapping(item) for item in _list(source_scan.get("windows")) if _mapping(item)]
    return [_mapping(item) for item in _list(source_scan.get("candidates")) if _mapping(item)]


def _scan_candidates_for_family(source_scan: Mapping[str, Any], family: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _candidate_roots(source_scan):
        readiness = {}
        for raw_ready in _list(item.get("family_readiness")):
            ready = _mapping(raw_ready)
            if _family_key(ready.get("family")) == family:
                readiness = ready
                break
        ready_families = {_family_key(value) for value in _strings(item.get("ready_families"))}
        cache_ready = bool(readiness.get("cache_ready")) or family in ready_families
        caption_coverage = _safe_float(item.get("caption_sample_coverage"), 0.0)
        rank_score = _safe_float(item.get("candidate_rank_score"), _safe_float(item.get("pressure_score")))
        quality_ok = caption_coverage >= 0.875 and rank_score >= 4.0
        if not cache_ready:
            continue
        rows.append(
            {
                "root": str(item.get("root") or ""),
                "sample_offset": _safe_int(item.get("sample_offset")),
                "sample_image_count": _safe_int(item.get("sample_image_count")),
                "source_manifest_sha1": str(item.get("source_manifest_sha1") or ""),
                "caption_sample_coverage": round(caption_coverage, 6),
                "candidate_rank_score": round(rank_score, 6),
                "quality_ok": quality_ok,
                "readiness_status": str(readiness.get("status") or ""),
                "blocked_reasons": _strings(readiness.get("blocked_reasons")),
            }
        )
    rows.sort(key=lambda item: (bool(item["quality_ok"]), float(item["candidate_rank_score"])), reverse=True)
    return rows


def _newbie_inventory_summary(newbie_inventory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": str(newbie_inventory.get("status") or ""),
        "selected_axis_kind": str(newbie_inventory.get("selected_axis_kind") or ""),
        "selected_axis_cache_ready": bool(newbie_inventory.get("selected_axis_cache_ready")),
        "selected_axis_completed_canary_command_count": _safe_int(
            newbie_inventory.get("selected_axis_completed_canary_command_count")
        ),
        "selected_axis_sample_count": _safe_int(newbie_inventory.get("selected_axis_sample_count")),
        "selected_axis_caption_coverage": _safe_float(newbie_inventory.get("selected_axis_caption_coverage")),
        "evidence_pack_indexed": bool(newbie_inventory.get("evidence_pack_indexed")),
        "release_claim_allowed": bool(newbie_inventory.get("release_claim_allowed")),
        "claimable": bool(newbie_inventory.get("claimable")),
    }


def _source_axis_admission(
    *,
    source_axis_requirement: Mapping[str, Any],
    source_scan: Mapping[str, Any],
    newbie_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    requirements = _source_family_requirement(source_axis_requirement)
    families: list[dict[str, Any]] = []
    for family in sorted(requirements):
        requirement = _mapping(requirements[family])
        candidates = _scan_candidates_for_family(source_scan, family)
        high_quality = [item for item in candidates if bool(item.get("quality_ok"))]
        requirement_kind = str(requirement.get("requirement") or "")
        requires_external_input = bool(requirement.get("requires_external_input"))
        if high_quality:
            status = "candidate_available_review_required"
        elif requires_external_input:
            status = "external_source_or_cache_required"
        else:
            status = "manual_review_required"
        row = {
            "family": family,
            "status": status,
            "requirement": requirement_kind,
            "source_axis_state": str(requirement.get("source_axis_state") or ""),
            "source_axis_exhausted": bool(requirement.get("source_axis_exhausted")),
            "requires_external_input": requires_external_input and not bool(high_quality),
            "candidate_count": len(candidates),
            "high_quality_candidate_count": len(high_quality),
            "top_candidates": high_quality[:3] if high_quality else candidates[:3],
            "do_not_rerun_current_axis": bool(requirement.get("do_not_rerun_current_axis")),
            "completed_command_ids": _strings(_mapping(requirement.get("run_readiness")).get("completed_command_ids")),
            "completed_out_dirs": _strings(_mapping(requirement.get("run_readiness")).get("completed_out_dirs")),
            "blocked_actions": _strings(requirement.get("blocked_actions")),
            "acceptance_gates": [
                "caption_sample_coverage>=0.875",
                "candidate_rank_score>=4.0",
                "family_cache_ready_for_target_family",
                "new_source_or_cache_axis_must_not_match_completed_out_dir_axis",
                "manual_gpu_run_only_after_admission_review",
            ],
            "safe_to_auto_start": False,
            "release_claim_allowed": False,
        }
        if family == "newbie":
            row["warm_cache_inventory"] = _newbie_inventory_summary(newbie_inventory)
        families.append(row)

    return {
        "status": "external_source_or_cache_required"
        if any(item["requires_external_input"] for item in families)
        else "candidate_available_review_required"
        if any(item["high_quality_candidate_count"] for item in families)
        else "manual_review_required",
        "source_scan_report": str(source_scan.get("report") or ""),
        "source_scan_candidate_count": len(_candidate_roots(source_scan)),
        "source_axis_requirement_report": str(source_axis_requirement.get("report") or ""),
        "family_count": len(families),
        "external_input_required_count": sum(1 for item in families if item["requires_external_input"]),
        "candidate_available_family_count": sum(1 for item in families if item["high_quality_candidate_count"]),
        "families": families,
    }


def build_external_input_admission(
    *,
    sd15_readiness: Mapping[str, Any],
    source_axis_requirement: Mapping[str, Any],
    source_scan: Mapping[str, Any],
    newbie_warm_cache_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-only external-input admission report."""

    sd15 = _sd15_admission(_mapping(sd15_readiness))
    source_axis = _source_axis_admission(
        source_axis_requirement=_mapping(source_axis_requirement),
        source_scan=_mapping(source_scan),
        newbie_inventory=_mapping(newbie_warm_cache_inventory),
    )
    external_input_required = bool(sd15.get("requires_external_input")) or bool(
        source_axis.get("external_input_required_count")
    )
    if external_input_required:
        status = "external_input_required"
    elif sd15.get("execution_ready") or source_axis.get("candidate_available_family_count"):
        status = "manual_review_ready"
    else:
        status = "no_admission_candidate"
    return {
        "schema_version": 1,
        "report": EXTERNAL_INPUT_ADMISSION_REPORT,
        "roadmap": ROADMAP,
        "status": status,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "release_claim_allowed": False,
        "publishable": False,
        "not_release_evidence": True,
        "external_input_required": external_input_required,
        "sd15": sd15,
        "source_axis": source_axis,
        "blocked_actions": [
            "auto_start_gpu_heavy_ab_matrix",
            "auto_start_natural_load_canary_on_new_axis",
            "rerun_completed_followup_out_dirs_without_new_axis",
            "promote_cache_or_admission_artifacts_as_release_claim",
        ],
        "next_actions": [
            {
                "id": "provide_sd15_checkpoint" if sd15.get("requires_external_input") else "review_sd15_manual_ab",
                "family": "sd15",
                "status": sd15["status"],
                "requires_external_input": bool(sd15.get("requires_external_input")),
                "requires_gpu_if_executed": bool(sd15.get("requires_gpu_if_executed")),
                "safe_to_auto_start": False,
            },
            {
                "id": "provide_new_source_or_cache_axis",
                "family": "multi",
                "status": source_axis["status"],
                "requires_external_input": bool(source_axis.get("external_input_required_count")),
                "requires_gpu_if_executed": False,
                "safe_to_auto_start": False,
            },
        ],
        "notes": [
            "This admission report is JSON-only and does not start GPU work.",
            "Ready admission only allows manual review or protected command generation; it is not release evidence.",
            "Release claims still require rebuilt evidence packs, natural-load gates, and release_claims coverage.",
        ],
    }


__all__ = ["EXTERNAL_INPUT_ADMISSION_REPORT", "ROADMAP", "build_external_input_admission"]
