"""Source-axis scout command plans for Bubble Runtime follow-up runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _family_key(raw: Any) -> str:
    family = str(raw or "").strip().lower().replace("-", "_")
    return "newbie" if family == "dit" else family


def _path_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve()).lower()
    except OSError:
        return text.lower()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value if value is not None else default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _scout_family_flags(family: str) -> tuple[str, ...]:
    if family == "sdxl":
        return (
            "--sdxl-resolution",
            "1024",
            "--sdxl-samples",
            "8",
            "--sdxl-steps",
            "16",
            "--sdxl-warmup",
            "4",
            "--sdxl-tune-interval",
            "4",
            "--min-throughput-gain",
            "0.03",
        )
    if family == "anima":
        return (
            "--anima-resolution",
            "64",
            "--anima-samples",
            "8",
            "--anima-steps",
            "24",
            "--anima-warmup",
            "4",
            "--anima-tune-interval",
            "8",
            "--min-throughput-gain",
            "0.03",
        )
    return ()


def _scout_plan_id(family: str, sample_offset: int) -> str:
    resolution = "1024" if family == "sdxl" else "64"
    return f"{family}_scout_offset{sample_offset}_{resolution}_conservative"


def _guidance_tracks(family: str, source_axis_state: str, exhausted: bool) -> list[str]:
    if exhausted or source_axis_state == "exhausted_current_source_axis":
        tracks = [
            "avoid_repeating_workers_prefetch_on_current_axis",
            "scan_alternate_source_axis",
        ]
        if family == "sdxl":
            tracks.append("investigate_non_dataloader_bottleneck")
            tracks.append("check_compute_or_workload_underfill_axis")
        return tracks
    if source_axis_state == "no_ready_source_axis":
        return ["prepare_family_warm_cache", "scan_alternate_source_axis"]
    if source_axis_state == "no_source_axes_found":
        return ["scan_source_directories", "prepare_family_warm_cache"]
    return ["review_source_axis_blockers"]


def _guidance_status(source_axis_state: str, exhausted: bool) -> str:
    if exhausted or source_axis_state == "exhausted_current_source_axis":
        return "source_axis_exhausted"
    if source_axis_state == "no_ready_source_axis":
        return "source_axis_not_ready"
    if source_axis_state == "no_source_axes_found":
        return "source_axis_scan_missing"
    return "source_axis_needs_review"


def build_source_axis_scout_guidance(
    source_axis_scout: Mapping[str, Any],
    *,
    families: set[str],
) -> list[dict[str, Any]]:
    """Summarize scout blockers into non-GPU next actions."""

    guidance: list[dict[str, Any]] = []
    for raw in source_axis_scout.get("family_summaries", []):
        summary = _mapping(raw)
        family = _family_key(summary.get("family"))
        if family not in families:
            continue
        source_axis_state = str(summary.get("source_axis_state") or "")
        candidate_count = _safe_int(summary.get("candidate_count"))
        if candidate_count > 0 or source_axis_state == "candidate_available":
            continue
        exhausted = bool(summary.get("source_axis_exhausted"))
        tracks = _guidance_tracks(family, source_axis_state, exhausted)
        guidance.append(
            {
                "family": family,
                "status": _guidance_status(source_axis_state, exhausted),
                "source_axis_state": source_axis_state,
                "source_axis_exhausted": exhausted,
                "guidance_only": True,
                "gpu_command_generated": False,
                "next_action": str(summary.get("next_action") or f"scan_or_prepare_{family}_source_axis"),
                "recommended_tracks": tracks,
                "exhaustion_reason_codes": _string_list(summary.get("exhaustion_reason_codes")),
                "ready_axis_count": _safe_int(summary.get("ready_axis_count")),
                "completed_high_quality_axis_count": _safe_int(summary.get("completed_high_quality_axis_count")),
                "low_quality_ready_axis_count": _safe_int(summary.get("low_quality_ready_axis_count")),
                "top_axis": dict(_mapping(summary.get("top_axis"))),
            }
        )
    guidance.sort(key=lambda item: (str(item["family"]), str(item["status"])))
    return guidance


def build_source_axis_scout_plans(
    repo_root: Path,
    source_axis_scout: Mapping[str, Any],
    out_root: Path,
    *,
    families: set[str],
    existing_commands: Sequence[Mapping[str, Any]],
    real_material_command: Callable[..., list[str]],
) -> list[dict[str, Any]]:
    """Convert ready source-axis scout candidates into conservative dry-run commands."""

    existing_axes = {
        (
            _family_key(item.get("family")),
            _path_key(item.get("source_data")),
            int(item.get("sample_offset") or 0),
        )
        for item in existing_commands
        if item.get("family") and item.get("source_data") is not None
    }
    selected: set[str] = set()
    plans: list[dict[str, Any]] = []
    for raw in source_axis_scout.get("ranked_axes", []):
        axis = _mapping(raw)
        family = _family_key(axis.get("family"))
        if family not in families or family in selected:
            continue
        family_flags = _scout_family_flags(family)
        if not family_flags:
            continue
        if str(axis.get("state") or "") != "candidate" or not bool(axis.get("cache_ready")):
            continue
        recommendation = str(axis.get("recommendation") or "")
        if recommendation not in {
            "run_conservative_recheck_only_aggressive_held",
            "run_release_relevant_conservative_recheck",
        }:
            continue
        source_data = Path(str(axis.get("source_data") or ""))
        sample_offset = int(axis.get("sample_offset") or 0)
        axis_key = (family, _path_key(source_data), sample_offset)
        if axis_key in existing_axes:
            continue
        out_dir = out_root / f"real_material_canary_{family}_offset{sample_offset}_scout_p60_followup"
        plans.append(
            {
                "id": _scout_plan_id(family, sample_offset),
                "family": family,
                "priority": 5,
                "release_relevant": True,
                "diagnostic_only": False,
                "sample_offset": sample_offset,
                "source_data": str(source_data),
                "out_dir": str(out_dir),
                "rationale": (
                    f"Source-axis scout selected this ready {family.upper()} window as a conservative recheck while "
                    "aggressive worker/prefetch policy remains held."
                ),
                "source_axis_scout": {
                    "axis_id": str(axis.get("axis_id") or ""),
                    "recommendation": recommendation,
                    "score": axis.get("score"),
                    "source_manifest_sha1": str(axis.get("source_manifest_sha1") or ""),
                },
                "command": real_material_command(
                    repo_root,
                    family=family,
                    source_data=source_data,
                    sample_offset=sample_offset,
                    out_dir=out_dir,
                    flags=family_flags,
                ),
            }
        )
        selected.add(family)
    return plans


__all__ = ["build_source_axis_scout_guidance", "build_source_axis_scout_plans"]
