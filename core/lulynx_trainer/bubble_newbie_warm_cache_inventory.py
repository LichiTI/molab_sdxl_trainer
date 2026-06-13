"""Non-claimable Newbie warm-cache inventory for Bubble Runtime follow-ups."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


NEWBIE_WARM_CACHE_INVENTORY_REPORT = "bubble_newbie_warm_cache_inventory_v0"
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


def _path_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve()).lower()
    except OSError:
        return text.lower()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}


def _manifest_stats(path: Path) -> dict[str, Any]:
    manifest = _load_json(path)
    samples = [_mapping(item) for item in _list(manifest.get("samples")) if _mapping(item)]
    stems = [str(item.get("stem") or "") for item in samples if item.get("stem")]
    caption_count = sum(1 for item in samples if str(item.get("caption") or "").strip())
    cache_file_count = sum(len(_list(item.get("cache_files"))) for item in samples)
    sample_count = len(samples)
    return {
        "sample_count": sample_count,
        "stems": stems,
        "caption_count": caption_count,
        "caption_coverage": round(caption_count / sample_count, 6) if sample_count else 0.0,
        "cache_file_count": cache_file_count,
    }


def _metadata_sample_count(path: Path) -> int:
    metadata = _load_json(path)
    return len(_list(metadata.get("samples")))


def _candidate_path(root: Path, raw: Any, fallback: str) -> Path:
    text = str(raw or "").strip()
    if text:
        return Path(text)
    return root / fallback


def _extract_child_report(prepare_report: Mapping[str, Any], runner_report: Mapping[str, Any]) -> Mapping[str, Any]:
    if prepare_report:
        return prepare_report
    child = _mapping(runner_report.get("child_report"))
    return child if child else {}


def _newbie_readiness(report: Mapping[str, Any], runner_report: Mapping[str, Any]) -> Mapping[str, Any]:
    for candidate in (
        _mapping(report.get("postbuild_newbie_readiness")),
        _mapping(runner_report.get("post_run_newbie_readiness")),
        _mapping(_mapping(runner_report.get("child_report")).get("postbuild_newbie_readiness")),
    ):
        if candidate:
            return candidate
    return {}


def _cache_build(report: Mapping[str, Any], runner_report: Mapping[str, Any]) -> Mapping[str, Any]:
    for candidate in (
        _mapping(report.get("cache_build")),
        _mapping(_mapping(runner_report.get("child_report")).get("cache_build")),
    ):
        if candidate:
            return candidate
    return {}


def _source_fixture(report: Mapping[str, Any], runner_report: Mapping[str, Any]) -> Mapping[str, Any]:
    for candidate in (
        _mapping(report.get("source_fixture")),
        _mapping(_mapping(runner_report.get("child_report")).get("source_fixture")),
    ):
        if candidate:
            return candidate
    return {}


def _status_from_artifacts(
    report: Mapping[str, Any],
    runner_report: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> str:
    if bool(readiness.get("cache_ready")):
        return "cache_ready"
    for candidate in (report, runner_report):
        status = str(candidate.get("status") or "")
        if status:
            return status
    return "unknown"


def _axis_kind(root: Path, row: Mapping[str, Any]) -> str:
    name = root.name.lower()
    if str(row.get("runner_status") or "") == "dry_run":
        return "dry_run"
    if bool(row.get("timed_out")):
        return "timeout"
    if "probe" in name:
        return "probe_ready" if bool(row.get("cache_ready")) else "probe_incomplete"
    if bool(row.get("cache_ready")) and _safe_int(row.get("sample_count")) >= 8:
        return "full_ready"
    if bool(row.get("cache_ready")):
        return "small_ready"
    if str(row.get("status") or "") == "prepared_source_slice":
        return "prepared_source_only"
    return "incomplete"


def collect_newbie_warm_cache_artifacts(paths: Sequence[Path | str]) -> list[dict[str, Any]]:
    """Collect compact warm-cache inventory rows from cache prepare/runner dirs."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in paths:
        root = Path(raw_path)
        if root.is_file():
            root = root.parent
        key = _path_key(root)
        if not key or key in seen or not root.exists():
            continue
        seen.add(key)
        prepare_path = root / "newbie_cache_prepare_report.json"
        runner_path = root / "newbie_cache_runner_report.json"
        prepare_report = _load_json(prepare_path)
        runner_report = _load_json(runner_path)
        report = _extract_child_report(prepare_report, runner_report)
        readiness = _newbie_readiness(report, runner_report)
        cache_build = _cache_build(report, runner_report)
        fixture = _source_fixture(report, runner_report)
        source_root = str(
            readiness.get("source_root")
            or _mapping(report.get("postbuild_scan")).get("root")
            or _mapping(runner_report.get("post_run_scan")).get("root")
            or root / "source_data"
        )
        manifest_path = _candidate_path(root, cache_build.get("manifest_path"), "source_data/lulynx_cache_manifest_newbie.json")
        metadata_path = _candidate_path(root, cache_build.get("metadata_path"), "source_data/lulynx_cache_metadata_newbie.json")
        manifest_stats = _manifest_stats(manifest_path)
        row = {
            "root": str(root),
            "prepare_report_path": str(prepare_path),
            "prepare_report_exists": prepare_path.is_file(),
            "runner_report_path": str(runner_path),
            "runner_report_exists": runner_path.is_file(),
            "source_root": source_root,
            "source_root_key": _path_key(source_root),
            "source_data": str(report.get("source_data") or fixture.get("source_root") or fixture.get("root") or ""),
            "sample_offset": _safe_int(report.get("sample_offset"), _safe_int(fixture.get("sample_offset"))),
            "sample_count": _safe_int(readiness.get("sample_count"), _safe_int(cache_build.get("dataset_sample_count"))),
            "sample_cache_count": _safe_int(readiness.get("sample_cache_count")),
            "sample_cache_coverage": round(_safe_float(readiness.get("sample_cache_coverage")), 6),
            "cache_ready": bool(readiness.get("cache_ready")),
            "status": _status_from_artifacts(report, runner_report, readiness),
            "runner_status": str(runner_report.get("status") or ""),
            "runner_success": bool(runner_report.get("success")),
            "timed_out": bool(runner_report.get("timed_out")),
            "returncode": runner_report.get("returncode"),
            "cache_build": {
                "written": _safe_int(cache_build.get("written")),
                "skipped": _safe_int(cache_build.get("skipped")),
                "device": str(cache_build.get("device") or ""),
                "resolution": _safe_int(cache_build.get("resolution")),
                "skip_clip": bool(cache_build.get("skip_clip")),
                "metadata_fast_path": bool(cache_build.get("metadata_fast_path")),
                "first_latents_shape": _list(cache_build.get("first_latents_shape")),
                "first_hidden_shape": _list(cache_build.get("first_hidden_shape")),
                "first_pooled_shape": _list(cache_build.get("first_pooled_shape")),
            },
            "inventory_counts": dict(_mapping(readiness.get("inventory_counts"))),
            "manifest": {
                "path": str(manifest_path),
                "exists": manifest_path.is_file(),
                **manifest_stats,
            },
            "metadata": {
                "path": str(metadata_path),
                "exists": metadata_path.is_file(),
                "sample_count": _metadata_sample_count(metadata_path),
            },
            "evidence_pack_indexed": False,
            "release_claim_allowed": False,
            "claimable": False,
        }
        row["axis_kind"] = _axis_kind(root, row)
        rows.append(row)
    rows.sort(key=lambda item: (not bool(item.get("cache_ready")), str(item.get("root") or "")))
    return rows


def _run_readiness_commands(run_readiness: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        _mapping(item)
        for item in run_readiness.get("commands", [])
        if _family_key(_mapping(item).get("family")) == "newbie"
    ]


def _linked_commands(axis: Mapping[str, Any], commands: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    source_key = str(axis.get("source_root_key") or "")
    linked: list[dict[str, Any]] = []
    for command in commands:
        if _path_key(command.get("source_data")) != source_key:
            continue
        linked.append(
            {
                "id": str(command.get("id") or ""),
                "status": str(command.get("status") or ""),
                "out_dir": str(command.get("out_dir") or ""),
                "diagnostic_only": bool(command.get("diagnostic_only")),
                "release_relevant": bool(command.get("release_relevant")),
            }
        )
    return linked


def _newbie_canary_family(natural_load_canary: Mapping[str, Any]) -> Mapping[str, Any]:
    for raw in natural_load_canary.get("families", []):
        item = _mapping(raw)
        if _family_key(item.get("family")) == "newbie":
            return item
    return {}


def _current_source_root_keys(source_axis_requirement: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    for raw in _list(source_axis_requirement.get("families")):
        item = _mapping(raw)
        if _family_key(item.get("family")) != "newbie":
            continue
        keys.update(_path_key(root) for root in _strings(item.get("current_source_roots")))
    return {key for key in keys if key}


def _select_axis(
    rows: Sequence[Mapping[str, Any]],
    *,
    current_source_root_keys: set[str] | None = None,
) -> Mapping[str, Any]:
    ready = [item for item in rows if bool(item.get("cache_ready"))]
    if not ready:
        return rows[0] if rows else {}
    current_keys = current_source_root_keys or set()
    ready.sort(
        key=lambda item: (
            _path_key(item.get("source_data")) in current_keys,
            not bool(item.get("completed_canary_command_count")),
            not bool(item.get("do_not_rerun_without_new_axis")),
            str(item.get("axis_kind") or "") == "full_ready",
            "repair" in str(item.get("root") or "").lower(),
            _safe_float(item.get("sample_cache_coverage")),
            _safe_int(item.get("sample_count")),
        ),
        reverse=True,
    )
    return ready[0]


def _axis_kind_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    kinds = sorted({str(item.get("axis_kind") or "unknown") for item in rows})
    return {kind: sum(1 for item in rows if str(item.get("axis_kind") or "unknown") == kind) for kind in kinds}


def _runner_status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    statuses = sorted({str(item.get("runner_status") or "unknown") for item in rows})
    return {
        status: sum(1 for item in rows if str(item.get("runner_status") or "unknown") == status)
        for status in statuses
    }


def build_newbie_warm_cache_inventory(
    cache_axes: Sequence[Mapping[str, Any]],
    *,
    run_readiness: Mapping[str, Any] | None = None,
    natural_load_canary: Mapping[str, Any] | None = None,
    source_axis_requirement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-only, non-release inventory report for Newbie warm-cache axes."""

    axes = [dict(_mapping(item)) for item in cache_axes if _mapping(item)]
    commands = _run_readiness_commands(_mapping(run_readiness))
    linked_by_source = {str(axis.get("source_root_key") or ""): _linked_commands(axis, commands) for axis in axes}
    for axis in axes:
        axis["linked_canary_commands"] = linked_by_source.get(str(axis.get("source_root_key") or ""), [])
        axis["completed_canary_command_count"] = sum(
            1 for item in axis["linked_canary_commands"] if str(item.get("status") or "") == "completed_existing_evidence"
        )
        axis["do_not_rerun_without_new_axis"] = bool(axis["completed_canary_command_count"])

    current_source_root_keys = _current_source_root_keys(_mapping(source_axis_requirement))
    selected = _select_axis(axes, current_source_root_keys=current_source_root_keys)
    selected_key = str(selected.get("source_root_key") or "")
    canary_family = _newbie_canary_family(_mapping(natural_load_canary))
    blocker_summary = _mapping(_mapping(natural_load_canary).get("blocker_summary"))
    cache_blockers = _mapping(blocker_summary.get("cache_readiness"))
    source_requirement = {}
    for raw in _list(_mapping(source_axis_requirement).get("families")):
        item = _mapping(raw)
        if _family_key(item.get("family")) == "newbie":
            source_requirement = dict(item)
            break

    selected_ready = bool(selected.get("cache_ready"))
    selected_completed = bool(selected.get("completed_canary_command_count"))
    accepted_candidates = _safe_int(canary_family.get("accepted_candidate_count"))
    if selected_ready and selected_completed and accepted_candidates <= 0:
        status = "warm_cache_axis_completed_but_not_release_ready"
    elif selected_ready:
        status = "warm_cache_axis_ready"
    elif axes:
        status = "warm_cache_axis_incomplete"
    else:
        status = "warm_cache_axis_missing"

    return {
        "schema_version": 1,
        "report": NEWBIE_WARM_CACHE_INVENTORY_REPORT,
        "roadmap": ROADMAP,
        "status": status,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "release_claim_allowed": False,
        "publishable": False,
        "not_release_evidence": True,
        "claimable": False,
        "axis_count": len(axes),
        "ready_axis_count": sum(1 for item in axes if bool(item.get("cache_ready"))),
        "completed_canary_axis_count": sum(1 for item in axes if bool(item.get("completed_canary_command_count"))),
        "evidence_pack_indexed": False,
        "axis_kind_counts": _axis_kind_counts(axes),
        "runner_status_counts": _runner_status_counts(axes),
        "selected_axis_root": str(selected.get("source_root") or ""),
        "selected_axis_kind": str(selected.get("axis_kind") or ""),
        "source_data_original": str(selected.get("source_data") or ""),
        "prepared_source_data": str(selected.get("source_root") or ""),
        "selected_axis_repair_produced": "repair" in str(selected.get("root") or "").lower(),
        "selected_axis_current_source_root_match": _path_key(selected.get("source_data")) in current_source_root_keys,
        "selected_axis_sample_offset": _safe_int(selected.get("sample_offset")),
        "selected_axis_sample_count": _safe_int(selected.get("sample_count")),
        "selected_axis_stems": _strings(_mapping(selected.get("manifest")).get("stems")),
        "selected_axis_caption_count": _safe_int(_mapping(selected.get("manifest")).get("caption_count")),
        "selected_axis_caption_coverage": _safe_float(_mapping(selected.get("manifest")).get("caption_coverage")),
        "selected_axis_manifest_sample_count": _safe_int(_mapping(selected.get("manifest")).get("sample_count")),
        "selected_axis_metadata_sample_count": _safe_int(_mapping(selected.get("metadata")).get("sample_count")),
        "selected_axis_cache_file_count": _safe_int(_mapping(selected.get("manifest")).get("cache_file_count")),
        "selected_axis_cache_ready": selected_ready,
        "selected_axis_completed_canary_command_count": _safe_int(selected.get("completed_canary_command_count")),
        "natural_load_canary": {
            "status": str(_mapping(natural_load_canary).get("status") or ""),
            "newbie_status": str(canary_family.get("status") or ""),
            "accepted_candidate_count": accepted_candidates,
            "blocked_reasons": _strings(canary_family.get("blocked_reasons")),
            "blocking_categories": _strings(canary_family.get("blocking_categories")),
            "historical_cache_readiness_blocker_count": _safe_int(cache_blockers.get("count")),
            "historical_cache_readiness_reasons": _strings(cache_blockers.get("reasons")),
            "selected_axis_supersedes_cache_missing_blockers": selected_ready and _safe_int(cache_blockers.get("count")) > 0,
        },
        "source_axis_requirement": {
            "requirement": str(source_requirement.get("requirement") or ""),
            "source_axis_state": str(source_requirement.get("source_axis_state") or ""),
            "requires_external_input": bool(source_requirement.get("requires_external_input")),
            "do_not_rerun_current_axis": bool(source_requirement.get("do_not_rerun_current_axis")),
            "completed_command_ids": _strings(_mapping(source_requirement.get("run_readiness")).get("completed_command_ids")),
        },
        "blocked_actions": [
            "promote_newbie_warm_cache_inventory_as_release_claim",
            "treat_newbie_cache_ready_as_natural_load_canary_ready",
            "rerun_newbie_completed_followup_out_dirs_without_new_source_or_cache_axis",
        ],
        "next_action": (
            "prepare_newbie_warm_cache_or_new_source_axis"
            if status == "warm_cache_axis_completed_but_not_release_ready"
            else "inspect_newbie_warm_cache_axis"
        ),
        "axes": axes,
        "notes": [
            "This inventory is JSON-only and does not start GPU work.",
            "Warm-cache readiness is inventory evidence, not a release claim.",
            "A completed warm-cache canary still needs natural-load throughput, loss, VRAM and wording gates before release claims.",
        ],
    }


__all__ = [
    "NEWBIE_WARM_CACHE_INVENTORY_REPORT",
    "ROADMAP",
    "build_newbie_warm_cache_inventory",
    "collect_newbie_warm_cache_artifacts",
]
