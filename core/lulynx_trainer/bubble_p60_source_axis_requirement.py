"""Summarize P60 source-axis requirements without starting GPU work."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


P60_SOURCE_AXIS_REQUIREMENT_REPORT = "bubble_p60_source_axis_requirement_v0"
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


def _family_key(value: Any) -> str:
    family = str(value or "").strip().lower().replace("-", "_")
    return "newbie" if family in {"dit", "newbie_dit"} else family


def _run_readiness_by_family(run_readiness: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for raw in run_readiness.get("commands", []):
        item = _mapping(raw)
        family = _family_key(item.get("family"))
        if not family:
            continue
        row = rows.setdefault(
            family,
            {
                "manual_ready_command_ids": [],
                "diagnostic_command_ids": [],
                "completed_command_ids": [],
                "blocked_command_ids": [],
                "completed_out_dirs": [],
            },
        )
        status = str(item.get("status") or "")
        command_id = str(item.get("id") or "")
        if status == "manual_ready":
            row["manual_ready_command_ids"].append(command_id)
        elif status == "diagnostic_manual_ready":
            row["diagnostic_command_ids"].append(command_id)
        elif status == "completed_existing_evidence":
            row["completed_command_ids"].append(command_id)
            out_dir = str(item.get("out_dir") or "")
            if out_dir:
                row["completed_out_dirs"].append(out_dir)
        elif status == "blocked":
            row["blocked_command_ids"].append(command_id)
    return rows


def _current_source_roots(summary: Mapping[str, Any], ranked_axes: Sequence[Mapping[str, Any]], family: str) -> list[str]:
    roots = {
        str(_mapping(summary.get("top_axis")).get("source_data") or "").strip(),
        *(
            str(_mapping(axis).get("source_data") or "").strip()
            for axis in ranked_axes
            if _family_key(_mapping(axis).get("family")) == family
        ),
    }
    return sorted(root for root in roots if root)


def _requirement_kind(summary: Mapping[str, Any]) -> str:
    if _safe_int(summary.get("candidate_count")) > 0:
        return "candidate_available"
    state = str(summary.get("source_axis_state") or "")
    if state == "exhausted_current_source_axis":
        return "new_source_axis_required"
    if state == "no_ready_source_axis":
        return "warm_cache_or_new_source_axis_required"
    if state == "no_source_axes_found":
        return "source_scan_required"
    return "manual_review_required"


def _blocked_actions(family: str, requirement: str, completed_count: int) -> list[str]:
    actions: list[str] = []
    if requirement == "new_source_axis_required":
        actions.extend(
            [
                f"repeat_{family}_workers_prefetch_on_current_source_axis",
                f"run_{family}_followup_without_new_source_axis",
            ]
        )
    elif requirement == "warm_cache_or_new_source_axis_required":
        actions.extend(
            [
                f"run_{family}_canary_without_warm_cache",
                f"promote_{family}_natural_canary_without_cache_inventory",
            ]
        )
    elif requirement == "source_scan_required":
        actions.append(f"run_{family}_canary_without_source_scan")
    if completed_count:
        actions.append(f"rerun_{family}_completed_followup_out_dirs_without_new_axis")
    return actions


def build_p60_source_axis_requirement(
    source_axis_scout: Mapping[str, Any],
    *,
    run_readiness: Mapping[str, Any] | None = None,
    natural_load_canary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact requirement report from source-axis and run readiness evidence."""

    run_by_family = _run_readiness_by_family(_mapping(run_readiness))
    canary = _mapping(natural_load_canary)
    blocked_families = {_family_key(item) for item in _strings(canary.get("blocked_families"))}
    ranked_axes = [_mapping(item) for item in _list(source_axis_scout.get("ranked_axes")) if _mapping(item)]
    families: list[dict[str, Any]] = []
    for raw in source_axis_scout.get("family_summaries", []):
        summary = _mapping(raw)
        family = _family_key(summary.get("family"))
        if not family:
            continue
        run_state = run_by_family.get(family, {})
        requirement = _requirement_kind(summary)
        completed_count = len(_strings(run_state.get("completed_command_ids")))
        needs_external = requirement in {
            "new_source_axis_required",
            "warm_cache_or_new_source_axis_required",
            "source_scan_required",
        }
        families.append(
            {
                "family": family,
                "status": "candidate_available" if requirement == "candidate_available" else "external_input_required",
                "requirement": requirement,
                "blocked_by_natural_load_canary": family in blocked_families,
                "source_axis_state": str(summary.get("source_axis_state") or ""),
                "source_axis_exhausted": bool(summary.get("source_axis_exhausted")),
                "candidate_count": _safe_int(summary.get("candidate_count")),
                "ready_axis_count": _safe_int(summary.get("ready_axis_count")),
                "unattempted_high_quality_ready_axis_count": _safe_int(
                    summary.get("unattempted_high_quality_ready_axis_count")
                ),
                "completed_high_quality_axis_count": _safe_int(summary.get("completed_high_quality_axis_count")),
                "low_quality_ready_axis_count": _safe_int(summary.get("low_quality_ready_axis_count")),
                "exhaustion_reason_codes": _strings(summary.get("exhaustion_reason_codes")),
                "current_source_roots": _current_source_roots(summary, ranked_axes, family),
                "next_action": str(summary.get("next_action") or ""),
                "requires_external_input": needs_external,
                "requires_gpu_if_executed": False,
                "safe_to_auto_start": False,
                "release_claim_allowed_after_success": False,
                "do_not_rerun_current_axis": needs_external or completed_count > 0,
                "blocked_actions": _blocked_actions(family, requirement, completed_count),
                "run_readiness": {
                    "manual_ready_command_ids": _strings(run_state.get("manual_ready_command_ids")),
                    "diagnostic_command_ids": _strings(run_state.get("diagnostic_command_ids")),
                    "completed_command_ids": _strings(run_state.get("completed_command_ids")),
                    "blocked_command_ids": _strings(run_state.get("blocked_command_ids")),
                    "completed_out_dirs": _strings(run_state.get("completed_out_dirs")),
                },
            }
        )

    external_count = sum(1 for item in families if item["requires_external_input"])
    candidate_count = sum(1 for item in families if item["requirement"] == "candidate_available")
    completed_count = sum(len(item["run_readiness"]["completed_command_ids"]) for item in families)
    if external_count:
        status = "external_source_or_cache_required"
    elif candidate_count:
        status = "candidate_available_review_required"
    else:
        status = "no_source_axis_action"
    return {
        "schema_version": 1,
        "report": P60_SOURCE_AXIS_REQUIREMENT_REPORT,
        "roadmap": ROADMAP,
        "status": status,
        "source_axis_scout_report": str(source_axis_scout.get("report") or ""),
        "run_readiness_report": str(_mapping(run_readiness).get("report") or ""),
        "natural_load_canary_status": str(canary.get("status") or ""),
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "release_claim_allowed": False,
        "publishable": False,
        "not_release_evidence": True,
        "family_count": len(families),
        "external_input_required_count": external_count,
        "candidate_available_family_count": candidate_count,
        "exhausted_family_count": sum(1 for item in families if item["source_axis_exhausted"]),
        "no_ready_source_axis_family_count": sum(
            1 for item in families if item["source_axis_state"] == "no_ready_source_axis"
        ),
        "completed_existing_command_count": completed_count,
        "families": families,
        "notes": [
            "This report is JSON-only and does not start GPU work.",
            "Families that require external input should not repeat completed follow-up out dirs without a new source/cache axis.",
            "Release claims remain blocked until natural-load canary and release coverage gates are rebuilt and reviewed.",
        ],
    }


__all__ = [
    "P60_SOURCE_AXIS_REQUIREMENT_REPORT",
    "ROADMAP",
    "build_p60_source_axis_requirement",
]
