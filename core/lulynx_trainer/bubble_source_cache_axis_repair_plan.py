"""Manual-only repair plan for blocked source/cache-axis preflight rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .bubble_p60_source_axis_scout import CAPTION_COVERAGE_THRESHOLD, MIN_CANDIDATE_RANK_SCORE


SOURCE_CACHE_AXIS_REPAIR_PLAN_REPORT = "bubble_source_cache_axis_repair_plan_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
PREFLIGHT_REPORT = "bubble_source_cache_axis_admission_preflight_v0"


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


def _default_python_executable(repo_root: Path) -> Path:
    flashattention = repo_root / "backend" / "env" / "python-flashattention" / "python.exe"
    if flashattention.is_file():
        return flashattention
    return repo_root / "backend" / "env" / "python_launcher" / "python.exe"


def _norm_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve()).lower()
    except OSError:
        return text.lower()


def _axis_key(axis: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        _family_key(axis.get("family")),
        _norm_path(axis.get("source_data") or axis.get("root")),
        _safe_int(axis.get("sample_offset")),
        str(axis.get("source_manifest_sha1") or "").strip(),
    )


def _candidate_axes(
    source_axis_scout: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate = _mapping(preflight.get("candidate"))
    matched = _mapping(preflight.get("matched_axis"))
    if candidate:
        rows.append(
            {
                "family": _family_key(candidate.get("family")),
                "source_data": str(candidate.get("root") or matched.get("source_data") or ""),
                "sample_offset": _safe_int(candidate.get("sample_offset") or matched.get("sample_offset")),
                "source_manifest_sha1": str(
                    candidate.get("source_manifest_sha1") or matched.get("source_manifest_sha1") or ""
                ),
                "candidate_source": str(candidate.get("candidate_source") or "preflight"),
                "blocked_reasons": _strings(preflight.get("blockers")),
                "cache_ready": bool(matched.get("cache_ready")),
                "caption_sample_coverage": _safe_float(matched.get("caption_sample_coverage")),
                "candidate_rank_score": _safe_float(matched.get("candidate_rank_score")),
            }
        )
    for raw in _list(source_axis_scout.get("family_summaries")):
        top = _mapping(_mapping(raw).get("top_axis"))
        family = _family_key(_mapping(raw).get("family"))
        if not top or not family:
            continue
        rows.append(
            {
                "family": family,
                "source_data": str(top.get("source_data") or ""),
                "sample_offset": _safe_int(top.get("sample_offset")),
                "source_manifest_sha1": str(top.get("source_manifest_sha1") or ""),
                "candidate_source": "source_axis_scout_top_axis",
                "blocked_reasons": _strings(top.get("blocked_reasons")),
                "cache_ready": bool(top.get("cache_ready")),
                "caption_sample_coverage": _safe_float(top.get("caption_sample_coverage")),
                "candidate_rank_score": _safe_float(top.get("candidate_rank_score")),
            }
        )
    seen: set[tuple[str, str, int]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = _axis_key(row)
        short_key = (key[0], key[1], key[2])
        if not key[0] or not key[1] or short_key in seen:
            continue
        seen.add(short_key)
        deduped.append(row)
    return deduped


def _newbie_cache_command(
    *,
    repo_root: Path,
    python_executable: Path,
    source_data: str,
    sample_offset: int,
    out_root: Path,
) -> list[str]:
    return [
        str(python_executable),
        str(repo_root / "devtools" / "run_bubble_newbie_real_material_cache_prepare.py"),
        "--runner-mode",
        "runtime-cache-only",
        "--source-data",
        source_data,
        "--out-dir",
        str(out_root / f"newbie_real_material_cache_repair_offset{max(sample_offset, 0)}"),
        "--samples",
        "8",
        "--sample-offset",
        str(max(sample_offset, 0)),
        "--model-dir",
        str(repo_root / "models" / "newbie"),
        "--resolution",
        "64",
        "--device",
        "cuda",
    ]


def _repair_row(
    axis: Mapping[str, Any],
    *,
    repo_root: Path,
    python_executable: Path,
    out_root: Path,
) -> dict[str, Any]:
    family = _family_key(axis.get("family"))
    source_data = str(axis.get("source_data") or "")
    sample_offset = _safe_int(axis.get("sample_offset"))
    coverage = _safe_float(axis.get("caption_sample_coverage"))
    rank = _safe_float(axis.get("candidate_rank_score"))
    cache_ready = bool(axis.get("cache_ready"))
    blockers = set(_strings(axis.get("blocked_reasons")))
    caption_coverage_ok = coverage >= CAPTION_COVERAGE_THRESHOLD
    rank_score_ok = rank >= MIN_CANDIDATE_RANK_SCORE
    command: list[str] = []
    warm_cache_status = "cache_already_ready" if cache_ready else "manual_cache_prepare_required"
    if not cache_ready and family == "newbie":
        command = _newbie_cache_command(
            repo_root=repo_root,
            python_executable=python_executable,
            source_data=source_data,
            sample_offset=sample_offset,
            out_root=out_root,
        )
        warm_cache_status = "protected_newbie_runtime_cache_command_ready"
    elif not cache_ready:
        warm_cache_status = "family_cache_runner_missing"
        blockers.add(f"{family}_family_cache_runner_missing")
    caption_status = "caption_coverage_ok" if caption_coverage_ok else "caption_coverage_repair_required"
    if caption_coverage_ok and not rank_score_ok:
        caption_status = "caption_semantic_or_source_quality_review_required"
    return {
        "family": family,
        "source_data": source_data,
        "sample_offset": sample_offset,
        "source_manifest_sha1": str(axis.get("source_manifest_sha1") or ""),
        "candidate_source": str(axis.get("candidate_source") or ""),
        "cache_ready": cache_ready,
        "caption_sample_coverage": coverage,
        "caption_coverage_ok": caption_coverage_ok,
        "candidate_rank_score": rank,
        "rank_score_ok": rank_score_ok,
        "warm_cache_status": warm_cache_status,
        "caption_repair_status": caption_status,
        "manual_execute_command": command,
        "dry_run_command": [*command, "--dry-run"] if command else [],
        "requires_gpu_if_executed": bool(command),
        "manual_start_required": bool(command),
        "safe_to_auto_start": False,
        "release_claim_allowed_after_success": False,
        "post_run_refresh_commands": [
            "refresh_source_axis_scout",
            "refresh_source_axis_requirement",
            "refresh_source_cache_axis_admission_preflight",
            "refresh_source_cache_axis_repair_plan",
            "refresh_source_cache_axis_manual_canary_plan",
            "refresh_source_cache_axis_pipeline_readiness",
        ],
        "blockers": sorted(blockers),
    }


def build_source_cache_axis_repair_plan(
    *,
    source_axis_scout: Mapping[str, Any],
    source_cache_axis_admission_preflight: Mapping[str, Any],
    source_axis_requirement: Mapping[str, Any] | None = None,
    repo_root: Path,
    out_root: Path | None = None,
    python_executable: Path | None = None,
) -> dict[str, Any]:
    """Describe manual warm-cache/caption repair without starting GPU work."""

    repo = Path(repo_root)
    out_base = out_root or repo / "devtools" / "benchmark_evidence" / "bubble_runtime"
    python = python_executable or _default_python_executable(repo)
    preflight = _mapping(source_cache_axis_admission_preflight)
    axes = _candidate_axes(_mapping(source_axis_scout), preflight)
    repair_rows = [
        _repair_row(axis, repo_root=repo, python_executable=python, out_root=out_base)
        for axis in axes
    ]
    command_rows = [row for row in repair_rows if row["manual_execute_command"]]
    family_runner_missing = sorted(
        {
            str(row.get("family") or "")
            for row in repair_rows
            if str(row.get("warm_cache_status") or "") == "family_cache_runner_missing"
        }
    )
    status = (
        "protected_repair_commands_ready"
        if command_rows
        else "blocked_no_supported_warm_cache_command"
        if repair_rows
        else "blocked_no_repair_axis"
    )
    return {
        "schema_version": 1,
        "report": SOURCE_CACHE_AXIS_REPAIR_PLAN_REPORT,
        "roadmap": ROADMAP,
        "status": status,
        "preflight_report": str(preflight.get("report") or ""),
        "preflight_status": str(preflight.get("status") or ""),
        "source_axis_requirement_status": str(_mapping(source_axis_requirement).get("status") or ""),
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "requires_gpu_if_executed": bool(command_rows),
        "release_claim_allowed": False,
        "publishable": False,
        "not_release_evidence": True,
        "repair_axis_count": len(repair_rows),
        "command_count": len(command_rows),
        "blocked_command_count": len(repair_rows) - len(command_rows),
        "family_runner_missing": family_runner_missing,
        "commands": command_rows,
        "repair_axes": repair_rows,
        "blockers": sorted(
            {
                *(
                    ["source_cache_axis_preflight_missing_or_unrecognized"]
                    if str(preflight.get("report") or "") not in {"", PREFLIGHT_REPORT}
                    else []
                ),
                *(f"{family}_family_cache_runner_missing" for family in family_runner_missing),
                *(
                    ["candidate_rank_score_below_scout_threshold"]
                    if any(not bool(row.get("rank_score_ok")) for row in repair_rows)
                    else []
                ),
                *(
                    ["caption_coverage_repair_required"]
                    if any(
                        str(row.get("caption_repair_status") or "") == "caption_coverage_repair_required"
                        for row in repair_rows
                    )
                    else []
                ),
            }
        ),
        "blocked_actions": [
            "auto_start_repair_plan",
            "promote_repair_plan_as_release_evidence",
            "skip_preflight_after_cache_or_caption_repair",
        ],
        "acceptance_gates_after_manual_repair": [
            f"caption_sample_coverage>={CAPTION_COVERAGE_THRESHOLD}",
            f"candidate_rank_score>={MIN_CANDIDATE_RANK_SCORE}",
            "family_cache_ready_for_candidate_axis",
            "source_axis_scout_refreshed_after_manual_repair",
            "source_cache_axis_preflight_admitted_before_canary_plan",
        ],
        "notes": [
            "This plan is JSON-only and does not start GPU work.",
            "Commands are protected manual commands; dry-run only prints the command surface.",
        ],
    }


__all__ = [
    "ROADMAP",
    "SOURCE_CACHE_AXIS_REPAIR_PLAN_REPORT",
    "build_source_cache_axis_repair_plan",
]
