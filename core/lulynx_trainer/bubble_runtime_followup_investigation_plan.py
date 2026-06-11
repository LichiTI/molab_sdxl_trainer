"""Non-GPU investigation planning for Bubble Runtime follow-up blockers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FOLLOWUP_INVESTIGATION_PLAN_REPORT = "bubble_runtime_followup_investigation_plan_v0"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value if value is not None else default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _family_key(raw: Any) -> str:
    family = str(raw or "").strip().lower().replace("-", "_")
    return "newbie" if family == "dit" else family


def _path_text(path: Path) -> str:
    return str(path)


def _path_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve()).lower()
    except OSError:
        return text.lower()


def _source_scan_roots(source_scan: Mapping[str, Any]) -> list[str]:
    if str(source_scan.get("mode") or "") == "sample_windows":
        root = str(source_scan.get("root") or "").strip()
        return [root] if root else []
    roots: list[str] = []
    for raw in source_scan.get("candidates", []):
        item = _mapping(raw)
        root = str(item.get("root") or "").strip()
        if root:
            roots.append(root)
    return sorted(set(roots), key=lambda item: item.lower())


def _source_axis_scan_state(source_scan: Mapping[str, Any], current_source: str) -> dict[str, Any]:
    roots = _source_scan_roots(source_scan)
    current = _path_key(current_source)
    alternates = [root for root in roots if _path_key(root) and _path_key(root) != current]
    return {
        "known_source_candidate_count": len(roots),
        "known_alternate_source_axis_count": len(alternates),
        "known_source_roots": roots[:8],
        "known_alternate_source_roots": alternates[:8],
    }


def _scan_command(repo_root: Path, roots: Sequence[Path], family: str, out_path: Path) -> list[str]:
    return [
        _path_text(repo_root / "backend" / "env" / "python_launcher" / "python.exe"),
        _path_text(repo_root / "devtools" / "scan_bubble_real_material_sources.py"),
        *[_path_text(root) for root in roots],
        "--family",
        family,
        "--samples",
        "8",
        "--max-depth",
        "2",
        "--min-images",
        "8",
        "--out",
        _path_text(out_path),
    ]


def _sdxl_non_dataloader_command(repo_root: Path, out_path: Path) -> list[str]:
    return [
        _path_text(repo_root / "backend" / "env" / "python_launcher" / "python.exe"),
        _path_text(repo_root / "devtools" / "build_bubble_sdxl_non_dataloader_investigation.py"),
        _path_text(repo_root / "devtools" / "benchmark_evidence" / "bubble_runtime"),
        "--out",
        _path_text(out_path),
    ]


def _newbie_cache_prepare_dry_run_command(
    repo_root: Path,
    *,
    source_data: Path,
    sample_offset: int,
    out_dir: Path,
) -> list[str]:
    return [
        _path_text(repo_root / "backend" / "env" / "python_launcher" / "python.exe"),
        _path_text(repo_root / "devtools" / "run_bubble_newbie_real_material_cache_prepare.py"),
        "--source-data",
        _path_text(source_data),
        "--out-dir",
        _path_text(out_dir),
        "--work-dir",
        _path_text(out_dir / "source_data"),
        "--samples",
        "8",
        "--sample-offset",
        str(max(int(sample_offset), 0)),
        "--model-dir",
        _path_text(repo_root / "models" / "newbie"),
        "--resolution",
        "64",
        "--device",
        "cuda",
        "--timeout-seconds",
        "900",
        "--dry-run",
    ]


def _base_item(
    guidance: Mapping[str, Any],
    *,
    track: str,
    category: str,
    priority: int,
    status: str,
) -> dict[str, Any]:
    family = _family_key(guidance.get("family"))
    return {
        "id": f"{family}_{track}",
        "family": family,
        "track": track,
        "category": category,
        "priority": int(priority),
        "status": status,
        "safe_to_auto_start": False,
        "manual_start_required": True,
        "requires_gpu_if_executed": False,
        "source_axis_state": str(guidance.get("source_axis_state") or ""),
        "source_axis_exhausted": bool(guidance.get("source_axis_exhausted")),
        "next_action": str(guidance.get("next_action") or ""),
        "reason_codes": _string_list(guidance.get("exhaustion_reason_codes")),
        "top_axis": dict(_mapping(guidance.get("top_axis"))),
    }


def _guardrail_item(guidance: Mapping[str, Any]) -> dict[str, Any]:
    item = _base_item(
        guidance,
        track="avoid_repeating_workers_prefetch_on_current_axis",
        category="guardrail",
        priority=5,
        status="guardrail_active",
    )
    item.update(
        {
            "manual_start_required": False,
            "blocked_by": [
                "current_source_axis_exhausted",
                "completed_high_quality_axes_failed_release_gate",
                "remaining_ready_axes_low_quality",
            ],
            "rationale": (
                "Do not repeat the same workers/prefetch canary on this source axis until a new "
                "source axis or a different bottleneck hypothesis is selected."
            ),
        }
    )
    return item


def _scan_item(
    repo_root: Path,
    guidance: Mapping[str, Any],
    *,
    roots: Sequence[Path],
    out_root: Path,
    source_scan: Mapping[str, Any],
) -> dict[str, Any]:
    family = _family_key(guidance.get("family"))
    top_axis = _mapping(guidance.get("top_axis"))
    scan_state = _source_axis_scan_state(source_scan, str(top_axis.get("source_data") or ""))
    status = "ready_non_gpu_scan"
    blocked_by: list[str] = []
    if scan_state["known_source_candidate_count"] and not scan_state["known_alternate_source_axis_count"]:
        status = "needs_external_source_axis_or_deeper_scan"
        blocked_by.append("no_alternate_source_axis_in_current_scan")

    out_path = out_root / f"real_material_source_scan_{family}_alternate_axes.json"
    item = _base_item(
        guidance,
        track="scan_alternate_source_axis",
        category="source_axis_search",
        priority=20,
        status=status,
    )
    item.update(
        {
            "manual_start_required": bool(blocked_by),
            "can_auto_run_without_gpu": not blocked_by,
            "blocked_by": blocked_by,
            "command": _scan_command(repo_root, roots, family, out_path),
            "expected_output": str(out_path),
            "source_scan_state": scan_state,
            "rationale": (
                "Scan real-material directories for a different ready source axis before spending GPU "
                "time on another workers/prefetch canary."
            ),
        }
    )
    return item


def _newbie_prepare_item(repo_root: Path, guidance: Mapping[str, Any], *, out_root: Path) -> dict[str, Any]:
    top_axis = _mapping(guidance.get("top_axis"))
    sample_offset = _safe_int(top_axis.get("sample_offset"), 32)
    source_data = Path(str(top_axis.get("source_data") or repo_root / "sucai" / "6_lulu"))
    out_dir = out_root / f"newbie_real_material_cache_runner_offset{sample_offset}_investigation"
    item = _base_item(
        guidance,
        track="prepare_family_warm_cache",
        category="warm_cache_prepare",
        priority=30,
        status="dry_run_ready_heavy_prepare_manual",
    )
    item.update(
        {
            "requires_gpu_if_executed": True,
            "dry_run_requires_gpu": False,
            "dry_run_command": _newbie_cache_prepare_dry_run_command(
                repo_root,
                source_data=source_data,
                sample_offset=sample_offset,
                out_dir=out_dir,
            ),
            "expected_output": str(out_dir / "newbie_cache_runner_report.json"),
            "rationale": (
                "Newbie has no ready pooled warm cache on this source axis; prepare must remain explicit "
                "and audited because the real run loads heavy model components."
            ),
        }
    )
    return item


def _non_dataloader_item(repo_root: Path, guidance: Mapping[str, Any], *, out_root: Path) -> dict[str, Any]:
    family = _family_key(guidance.get("family"))
    if family != "sdxl":
        return {}
    out_path = out_root / "sdxl_non_dataloader_investigation.json"
    item = _base_item(
        guidance,
        track="investigate_non_dataloader_bottleneck",
        category="bottleneck_investigation",
        priority=40,
        status="manual_review_required",
    )
    item.update(
        {
            "manual_start_required": False,
            "command": _sdxl_non_dataloader_command(repo_root, out_path),
            "expected_output": str(out_path),
            "required_evidence": [
                "phase_profile_dominant_bottleneck",
                "active_window_gpu_util",
                "steady_samples_per_second",
                "loss_stability",
                "batch_resolution_rank_token_shape",
            ],
            "recommended_probes": [
                "review_compute_bound_or_host_phase_windows",
                "check_profiler_sync_and_host_gap_share",
                "compare attention_or_compile_backend_guardrails",
            ],
            "rationale": (
                "The current source axis did not prove a natural DataLoader data-wait gain, so the next "
                "SDXL hypothesis should inspect compute, host, transfer or workload shape instead."
            ),
        }
    )
    return item


def _workload_item(guidance: Mapping[str, Any], *, out_root: Path) -> dict[str, Any]:
    family = _family_key(guidance.get("family"))
    if family != "sdxl":
        return {}
    out_path = out_root / "sdxl_non_dataloader_investigation.json"
    item = _base_item(
        guidance,
        track="check_compute_or_workload_underfill_axis",
        category="workload_shape_review",
        priority=45,
        status="manual_review_required",
    )
    item.update(
        {
            "manual_start_required": False,
            "expected_output": str(out_path),
            "depends_on": "sdxl_investigate_non_dataloader_bottleneck",
            "required_guardrails": [
                "throughput_first",
                "loss_stability_required",
                "vram_ratio_within_limit",
                "case_specific_claim_only",
            ],
            "candidate_axes": [
                "batch_or_microbatch_shape",
                "resolution_shape",
                "lora_rank_or_trainable_parameter_shape",
                "token_length_or_bucket_stability",
            ],
            "rationale": (
                "If SDXL is compute-bound or workload-underfilled, raising GPU util must be framed as "
                "workload shaping and must not override throughput/loss/VRAM gates."
            ),
        }
    )
    return item


def _items_for_guidance(
    repo_root: Path,
    guidance: Mapping[str, Any],
    *,
    roots: Sequence[Path],
    out_root: Path,
    source_scan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    tracks = _string_list(guidance.get("recommended_tracks"))
    items: list[dict[str, Any]] = []
    for track in tracks:
        if track == "avoid_repeating_workers_prefetch_on_current_axis":
            items.append(_guardrail_item(guidance))
        elif track == "scan_alternate_source_axis":
            items.append(_scan_item(repo_root, guidance, roots=roots, out_root=out_root, source_scan=source_scan))
        elif track == "prepare_family_warm_cache":
            items.append(_newbie_prepare_item(repo_root, guidance, out_root=out_root))
        elif track == "investigate_non_dataloader_bottleneck":
            item = _non_dataloader_item(repo_root, guidance, out_root=out_root)
            if item:
                items.append(item)
        elif track == "check_compute_or_workload_underfill_axis":
            item = _workload_item(guidance, out_root=out_root)
            if item:
                items.append(item)
    return items


def build_followup_investigation_plan(
    run_plan: Mapping[str, Any],
    *,
    repo_root: Path,
    source_roots: Sequence[Path] | None = None,
    out_root: Path | None = None,
    source_scan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a non-GPU queue from follow-up run-plan guidance."""

    repo = Path(repo_root)
    roots = list(source_roots or [repo / "sucai"])
    out_base = out_root or repo / "devtools" / "benchmark_evidence" / "bubble_runtime"
    scan = _mapping(source_scan)
    guidance = [_mapping(item) for item in run_plan.get("source_axis_scout_guidance", []) if _mapping(item)]
    items: list[dict[str, Any]] = []
    for entry in guidance:
        items.extend(_items_for_guidance(repo, entry, roots=roots, out_root=out_base, source_scan=scan))
    items.sort(key=lambda item: (int(item["priority"]), str(item["family"]), str(item["id"])))
    return {
        "schema_version": 1,
        "report": FOLLOWUP_INVESTIGATION_PLAN_REPORT,
        "status": "investigation_items_planned" if items else "no_followup_investigation_needed",
        "source_run_plan_report": str(run_plan.get("report") or ""),
        "source_run_plan_status": str(run_plan.get("status") or ""),
        "source_axis_scout_guidance_count": len(guidance),
        "item_count": len(items),
        "guardrail_item_count": sum(1 for item in items if item.get("category") == "guardrail"),
        "non_gpu_command_count": sum(1 for item in items if item.get("command")),
        "heavy_prepare_dry_run_count": sum(1 for item in items if item.get("dry_run_command")),
        "requires_gpu_if_executed_count": sum(1 for item in items if item.get("requires_gpu_if_executed")),
        "safe_to_auto_start": False,
        "items": items,
        "notes": [
            "This plan does not start GPU work.",
            "Guardrail items prevent repeated workers/prefetch canaries on exhausted source axes.",
            "Dry-run prepare commands are audit helpers; actual cache preparation remains explicit.",
        ],
    }


__all__ = ["FOLLOWUP_INVESTIGATION_PLAN_REPORT", "build_followup_investigation_plan"]
