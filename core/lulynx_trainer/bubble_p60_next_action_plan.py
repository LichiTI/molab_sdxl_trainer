"""Build the next P60 Bubble Runtime action plan from current evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from backend.core.lulynx_trainer.bubble_p60_sdxl_batch2_failure_actions import (
    build_cuda_debug_repeat_evidence_actions,
)
from backend.core.lulynx_trainer.bubble_p60_sdxl_probe_actions import build_sdxl_probe_actions


P60_NEXT_ACTION_PLAN_REPORT = "bubble_p60_next_action_plan_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"


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


def _family_key(value: Any) -> str:
    family = str(value or "").strip().lower().replace("-", "_")
    return "newbie" if family in {"dit", "newbie_dit"} else family


def _path_text(path: Path) -> str:
    return str(path)


def _command_index(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("id") or ""): _mapping(item)
        for item in plan.get("commands", [])
        if _mapping(item).get("id")
    }


def _investigation_item_index(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("id") or ""): _mapping(item)
        for item in plan.get("items", [])
        if _mapping(item).get("id")
    }


def _base_action(
    *,
    action_id: str,
    family: str,
    priority: int,
    action_type: str,
    status: str,
) -> dict[str, Any]:
    return {
        "id": action_id,
        "family": family,
        "priority": int(priority),
        "action_type": action_type,
        "status": status,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "manual_start_required": False,
        "requires_gpu_if_executed": False,
        "requires_external_input": False,
        "diagnostic_only": False,
        "release_relevant": False,
        "reasons": [],
        "warnings": [],
    }


def _manual_gpu_actions(run_readiness: Mapping[str, Any], run_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    commands = _command_index(run_plan)
    actions: list[dict[str, Any]] = []
    for raw in run_readiness.get("commands", []):
        item = _mapping(raw)
        status = str(item.get("status") or "")
        if status not in {"manual_ready", "diagnostic_manual_ready"}:
            continue
        command_id = str(item.get("id") or "")
        planned = _mapping(commands.get(command_id))
        diagnostic = status == "diagnostic_manual_ready" or bool(item.get("diagnostic_only"))
        action = _base_action(
            action_id=f"run_{command_id}",
            family=_family_key(item.get("family")),
            priority=50 if diagnostic else 10,
            action_type="diagnostic_gpu_probe" if diagnostic else "gpu_canary",
            status=status,
        )
        action.update(
            {
                "source_command_id": command_id,
                "manual_start_required": True,
                "requires_gpu_if_executed": True,
                "diagnostic_only": diagnostic,
                "release_relevant": bool(item.get("release_relevant")) and not diagnostic,
                "profile": str(item.get("profile") or planned.get("profile") or ""),
                "sample_offset": item.get("sample_offset"),
                "source_data": str(item.get("source_data") or planned.get("source_data") or ""),
                "out_dir": str(item.get("out_dir") or planned.get("out_dir") or ""),
                "command": list(planned.get("command") or []),
                "dry_run_command": list(planned.get("dry_run_command") or []),
                "warnings": _string_list(item.get("warnings")),
                "rationale": str(planned.get("rationale") or "Manual GPU canary is ready but must not auto-start."),
            }
        )
        if diagnostic:
            action["warnings"].append("diagnostic_only_not_release_evidence")
        actions.append(action)
    return actions


def _ready_non_gpu_actions(
    investigation_readiness: Mapping[str, Any],
    investigation_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    planned_items = _investigation_item_index(investigation_plan)
    actions: list[dict[str, Any]] = []
    for raw in investigation_readiness.get("items", []):
        item = _mapping(raw)
        if str(item.get("status") or "") != "ready_non_gpu_command":
            continue
        item_id = str(item.get("id") or "")
        planned = _mapping(planned_items.get(item_id))
        priority = _safe_int(planned.get("priority"), 20)
        action = _base_action(
            action_id=f"run_{item_id}",
            family=_family_key(item.get("family")),
            priority=priority,
            action_type="non_gpu_evidence_refresh",
            status="ready_non_gpu_command",
        )
        action.update(
            {
                "source_item_id": item_id,
                "command": list(planned.get("command") or []),
                "expected_output": str(item.get("expected_output") or planned.get("expected_output") or ""),
                "rationale": str(planned.get("rationale") or "Refresh non-GPU evidence before selecting more GPU work."),
            }
        )
        actions.append(action)
    return actions


def _guardrail_actions(investigation_readiness: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for raw in investigation_readiness.get("items", []):
        item = _mapping(raw)
        if str(item.get("status") or "") != "guardrail_active":
            continue
        item_id = str(item.get("id") or "")
        action = _base_action(
            action_id=f"guardrail_{item_id}",
            family=_family_key(item.get("family")),
            priority=5,
            action_type="guardrail",
            status="active",
        )
        action.update(
            {
                "source_item_id": item_id,
                "manual_start_required": False,
                "reasons": _string_list(item.get("reasons")),
                "rationale": "Keep this guardrail active while current evidence says repeating the same axis is wasteful.",
            }
        )
        actions.append(action)
    return actions


def _external_source_actions(
    investigation_plan: Mapping[str, Any],
    investigation_readiness: Mapping[str, Any],
    source_axis_scout: Mapping[str, Any],
) -> list[dict[str, Any]]:
    ready_by_id = {str(item.get("id") or ""): _mapping(item) for item in investigation_readiness.get("items", [])}
    actions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in investigation_plan.get("items", []):
        item = _mapping(raw)
        if str(item.get("category") or "") != "source_axis_search":
            continue
        blocked_by = _string_list(item.get("blocked_by"))
        scan_state = _mapping(item.get("source_scan_state"))
        no_alternate = "no_alternate_source_axis_in_current_scan" in blocked_by
        if not no_alternate and _safe_int(scan_state.get("known_alternate_source_axis_count")) > 0:
            continue
        family = _family_key(item.get("family"))
        key = (family, "source_axis")
        if key in seen:
            continue
        seen.add(key)
        item_id = str(item.get("id") or "")
        readiness = _mapping(ready_by_id.get(item_id))
        action = _base_action(
            action_id=f"provide_{family}_alternate_source_axis",
            family=family,
            priority=20,
            action_type="external_source_axis",
            status="external_input_required",
        )
        action.update(
            {
                "source_item_id": item_id,
                "requires_external_input": True,
                "blocked_by": blocked_by or _string_list(readiness.get("reasons")),
                "known_source_roots": _string_list(scan_state.get("known_source_roots")),
                "known_alternate_source_roots": _string_list(scan_state.get("known_alternate_source_roots")),
                "next_action": str(item.get("next_action") or "scan_alternate_source_axis"),
                "rationale": (
                    "Current scans do not contain a different ready source axis; add material, extend scan roots, "
                    "or prepare a cache-ready copy before more workers/prefetch canaries."
                ),
            }
        )
        actions.append(action)

    families_with_actions = {action["family"] for action in actions}
    for raw in source_axis_scout.get("family_summaries", []):
        summary = _mapping(raw)
        family = _family_key(summary.get("family"))
        if family in families_with_actions:
            continue
        state = str(summary.get("source_axis_state") or "")
        if state not in {"exhausted_current_source_axis", "no_ready_source_axis", "no_source_axes_found"}:
            continue
        action = _base_action(
            action_id=f"provide_{family}_source_axis",
            family=family,
            priority=25,
            action_type="external_source_axis",
            status="external_input_required",
        )
        action.update(
            {
                "requires_external_input": True,
                "source_axis_state": state,
                "source_axis_exhausted": bool(summary.get("source_axis_exhausted")),
                "exhaustion_reason_codes": _string_list(summary.get("exhaustion_reason_codes")),
                "next_action": str(summary.get("next_action") or f"scan_or_prepare_{family}_source_axis"),
                "top_axis": dict(_mapping(summary.get("top_axis"))),
                "rationale": "Source-axis scout has no release-ready new candidate for this family.",
            }
        )
        actions.append(action)
    return actions


def _refresh_needed_actions(
    source_axis_scout: Mapping[str, Any],
    run_readiness: Mapping[str, Any],
    *,
    repo_root: Path | None,
) -> list[dict[str, Any]]:
    if _safe_int(run_readiness.get("manual_ready_count")) > 0:
        return []
    candidate_families = [
        _family_key(item.get("family"))
        for item in source_axis_scout.get("family_summaries", [])
        if _safe_int(_mapping(item).get("candidate_count")) > 0
    ]
    if not candidate_families:
        return []
    repo = Path(repo_root) if repo_root is not None else None
    command = []
    if repo is not None:
        command = [
            _path_text(repo / "backend" / "env" / "python_launcher" / "python.exe"),
            _path_text(repo / "devtools" / "plan_bubble_runtime_followup_runs.py"),
        ]
    action = _base_action(
        action_id="refresh_followup_run_plan_from_source_axis_scout",
        family="multi",
        priority=15,
        action_type="non_gpu_planner_refresh",
        status="ready_non_gpu_command" if command else "manual_review_ready",
    )
    action.update(
        {
            "candidate_families": sorted(set(candidate_families)),
            "command": command,
            "rationale": "Source-axis scout has candidates, but run readiness has no manual-ready command; refresh the run plan/readiness.",
        }
    )
    return [action]


def _blocked_policy_actions(run_readiness: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for raw in run_readiness.get("aggressive_scaffolds", []):
        item = _mapping(raw)
        family = _family_key(item.get("family"))
        scaffold_id = str(item.get("id") or f"{family}_aggressive_scaffold")
        action = _base_action(
            action_id=f"blocked_{scaffold_id}",
            family=family,
            priority=90,
            action_type="blocked_aggressive_policy",
            status="blocked",
        )
        action.update(
            {
                "source_scaffold_id": scaffold_id,
                "reasons": _string_list(item.get("reasons")),
                "requires_gpu_if_executed": True,
                "rationale": "Aggressive workers/prefetch/persistent policy remains disabled until release gates clear.",
            }
        )
        actions.append(action)
    return actions


def _summary(actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = sorted({str(item.get("status") or "") for item in actions if item.get("status")})
    types = sorted({str(item.get("action_type") or "") for item in actions if item.get("action_type")})
    return {
        "action_count": len(actions),
        "ready_gpu_action_count": sum(1 for item in actions if item.get("status") == "manual_ready"),
        "diagnostic_gpu_action_count": sum(
            1 for item in actions if item.get("status") == "diagnostic_manual_ready"
        ),
        "ready_non_gpu_action_count": sum(1 for item in actions if item.get("status") == "ready_non_gpu_command"),
        "manual_review_action_count": sum(1 for item in actions if item.get("status") == "manual_review_ready"),
        "external_input_required_count": sum(1 for item in actions if item.get("requires_external_input")),
        "guardrail_action_count": sum(1 for item in actions if item.get("action_type") == "guardrail"),
        "blocked_action_count": sum(1 for item in actions if item.get("status") == "blocked"),
        "status_counts": {status: sum(1 for item in actions if item.get("status") == status) for status in statuses},
        "type_counts": {kind: sum(1 for item in actions if item.get("action_type") == kind) for kind in types},
    }


def _plan_status(summary: Mapping[str, Any]) -> str:
    if _safe_int(summary.get("ready_gpu_action_count")) > 0:
        return "manual_gpu_ready"
    if _safe_int(summary.get("ready_non_gpu_action_count")) > 0:
        return "non_gpu_action_ready"
    if _safe_int(summary.get("manual_review_action_count")) or _safe_int(summary.get("external_input_required_count")):
        return "manual_review_or_external_input_required"
    if _safe_int(summary.get("guardrail_action_count")):
        return "guardrail_only"
    return "no_next_actions"


def build_p60_next_action_plan(
    *,
    run_plan: Mapping[str, Any],
    run_readiness: Mapping[str, Any],
    investigation_plan: Mapping[str, Any] | None = None,
    investigation_readiness: Mapping[str, Any] | None = None,
    source_axis_scout: Mapping[str, Any] | None = None,
    sdxl_investigation: Mapping[str, Any] | None = None,
    sdxl_probe_plan: Mapping[str, Any] | None = None,
    sdxl_probe_command_plan: Mapping[str, Any] | None = None,
    sdxl_probe_runner_manifest: Mapping[str, Any] | None = None,
    sdxl_probe_results: Mapping[str, Any] | None = None,
    sdxl_workload_telemetry_evidence: Mapping[str, Any] | None = None,
    sdxl_workload_stability_manifest: Mapping[str, Any] | None = None,
    sdxl_workload_stability_evidence: Mapping[str, Any] | None = None,
    sdxl_workload_loss_guard_evidence: Mapping[str, Any] | None = None,
    sdxl_alternate_workload_shape_manifest: Mapping[str, Any] | None = None,
    sdxl_alternate_workload_shape_evidence: Mapping[str, Any] | None = None,
    sdxl_longer_window_batch2_repeat_manifest: Mapping[str, Any] | None = None,
    sdxl_longer_window_batch2_repeat_evidence: Mapping[str, Any] | None = None,
    sdxl_batch2_failure_diagnostic_manifest: Mapping[str, Any] | None = None,
    sdxl_batch2_failure_diagnostic_evidence: Mapping[str, Any] | None = None,
    sdxl_batch2_cuda_failure_debug_manifest: Mapping[str, Any] | None = None,
    sdxl_batch2_cuda_failure_debug_evidence: Mapping[str, Any] | None = None,
    sdxl_batch2_cuda_debug_repeat_evidence: Mapping[str, Any] | None = None,
    sdxl_compile_ab_manifest: Mapping[str, Any] | None = None,
    sdxl_compile_ab_evidence: Mapping[str, Any] | None = None,
    sdxl_compile_stable_anchor_manifest: Mapping[str, Any] | None = None,
    sdxl_compile_stable_anchor_evidence: Mapping[str, Any] | None = None,
    sdxl_compile_warm_cache_manifest: Mapping[str, Any] | None = None,
    sdxl_compile_warm_cache_evidence: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build a conservative action queue from P60 evidence reports."""

    investigation_plan = _mapping(investigation_plan)
    investigation_readiness = _mapping(investigation_readiness)
    source_axis_scout = _mapping(source_axis_scout)
    sdxl_investigation = _mapping(sdxl_investigation)
    sdxl_probe_plan = _mapping(sdxl_probe_plan)
    sdxl_probe_command_plan = _mapping(sdxl_probe_command_plan)
    sdxl_probe_runner_manifest = _mapping(sdxl_probe_runner_manifest)
    sdxl_probe_results = _mapping(sdxl_probe_results)
    sdxl_workload_telemetry_evidence = _mapping(sdxl_workload_telemetry_evidence)
    sdxl_workload_stability_manifest = _mapping(sdxl_workload_stability_manifest)
    sdxl_workload_stability_evidence = _mapping(sdxl_workload_stability_evidence)
    sdxl_workload_loss_guard_evidence = _mapping(sdxl_workload_loss_guard_evidence)
    sdxl_alternate_workload_shape_manifest = _mapping(sdxl_alternate_workload_shape_manifest)
    sdxl_alternate_workload_shape_evidence = _mapping(sdxl_alternate_workload_shape_evidence)
    sdxl_longer_window_batch2_repeat_manifest = _mapping(sdxl_longer_window_batch2_repeat_manifest)
    sdxl_longer_window_batch2_repeat_evidence = _mapping(sdxl_longer_window_batch2_repeat_evidence)
    sdxl_batch2_failure_diagnostic_manifest = _mapping(sdxl_batch2_failure_diagnostic_manifest)
    sdxl_batch2_failure_diagnostic_evidence = _mapping(sdxl_batch2_failure_diagnostic_evidence)
    sdxl_batch2_cuda_failure_debug_manifest = _mapping(sdxl_batch2_cuda_failure_debug_manifest)
    sdxl_batch2_cuda_failure_debug_evidence = _mapping(sdxl_batch2_cuda_failure_debug_evidence)
    sdxl_batch2_cuda_debug_repeat_evidence = _mapping(sdxl_batch2_cuda_debug_repeat_evidence)
    sdxl_compile_ab_manifest = _mapping(sdxl_compile_ab_manifest)
    sdxl_compile_ab_evidence = _mapping(sdxl_compile_ab_evidence)
    sdxl_compile_stable_anchor_manifest = _mapping(sdxl_compile_stable_anchor_manifest)
    sdxl_compile_stable_anchor_evidence = _mapping(sdxl_compile_stable_anchor_evidence)
    sdxl_compile_warm_cache_manifest = _mapping(sdxl_compile_warm_cache_manifest)
    sdxl_compile_warm_cache_evidence = _mapping(sdxl_compile_warm_cache_evidence)

    actions: list[dict[str, Any]] = []
    actions.extend(_manual_gpu_actions(_mapping(run_readiness), _mapping(run_plan)))
    actions.extend(_refresh_needed_actions(source_axis_scout, _mapping(run_readiness), repo_root=repo_root))
    actions.extend(_ready_non_gpu_actions(investigation_readiness, investigation_plan))
    actions.extend(_guardrail_actions(investigation_readiness))
    actions.extend(_external_source_actions(investigation_plan, investigation_readiness, source_axis_scout))
    actions.extend(
        build_sdxl_probe_actions(
            sdxl_investigation=sdxl_investigation,
            sdxl_probe_plan=sdxl_probe_plan,
            sdxl_probe_command_plan=sdxl_probe_command_plan,
            sdxl_probe_runner_manifest=sdxl_probe_runner_manifest,
            sdxl_probe_results=sdxl_probe_results,
            sdxl_workload_telemetry_evidence=sdxl_workload_telemetry_evidence,
            sdxl_workload_stability_manifest=sdxl_workload_stability_manifest,
            sdxl_workload_stability_evidence=sdxl_workload_stability_evidence,
            sdxl_workload_loss_guard_evidence=sdxl_workload_loss_guard_evidence,
            sdxl_alternate_workload_shape_manifest=sdxl_alternate_workload_shape_manifest,
            sdxl_alternate_workload_shape_evidence=sdxl_alternate_workload_shape_evidence,
            sdxl_longer_window_batch2_repeat_manifest=sdxl_longer_window_batch2_repeat_manifest,
            sdxl_longer_window_batch2_repeat_evidence=sdxl_longer_window_batch2_repeat_evidence,
            sdxl_batch2_failure_diagnostic_manifest=sdxl_batch2_failure_diagnostic_manifest,
            sdxl_batch2_failure_diagnostic_evidence=sdxl_batch2_failure_diagnostic_evidence,
            sdxl_batch2_cuda_failure_debug_manifest=sdxl_batch2_cuda_failure_debug_manifest,
            sdxl_batch2_cuda_failure_debug_evidence=sdxl_batch2_cuda_failure_debug_evidence,
            sdxl_compile_ab_manifest=sdxl_compile_ab_manifest,
            sdxl_compile_ab_evidence=sdxl_compile_ab_evidence,
            sdxl_compile_stable_anchor_manifest=sdxl_compile_stable_anchor_manifest,
            sdxl_compile_stable_anchor_evidence=sdxl_compile_stable_anchor_evidence,
            sdxl_compile_warm_cache_manifest=sdxl_compile_warm_cache_manifest,
            sdxl_compile_warm_cache_evidence=sdxl_compile_warm_cache_evidence,
            repo_root=repo_root,
        )
    )
    actions.extend(build_cuda_debug_repeat_evidence_actions(sdxl_batch2_cuda_debug_repeat_evidence))
    actions.extend(_blocked_policy_actions(_mapping(run_readiness)))

    seen: set[str] = set()
    unique_actions: list[dict[str, Any]] = []
    for action in sorted(actions, key=lambda item: (int(item.get("priority") or 999), str(item.get("id") or ""))):
        action_id = str(action.get("id") or "")
        if action_id in seen:
            continue
        seen.add(action_id)
        action["run_order"] = len(unique_actions) + 1
        unique_actions.append(action)

    summary = _summary(unique_actions)
    recommended = [
        str(item.get("id"))
        for item in unique_actions
        if item.get("status") not in {"blocked", "active"} and not item.get("diagnostic_only")
    ][:5]
    return {
        "schema_version": 1,
        "report": P60_NEXT_ACTION_PLAN_REPORT,
        "roadmap": ROADMAP,
        "status": _plan_status(summary),
        "source_run_plan_report": str(run_plan.get("report") or ""),
        "source_run_readiness_report": str(run_readiness.get("report") or ""),
        "source_investigation_plan_report": str(investigation_plan.get("report") or ""),
        "source_investigation_readiness_report": str(investigation_readiness.get("report") or ""),
        "source_axis_scout_report": str(source_axis_scout.get("report") or ""),
        "sdxl_investigation_report": str(sdxl_investigation.get("report") or ""),
        "sdxl_probe_plan_report": str(sdxl_probe_plan.get("report") or ""),
        "sdxl_probe_command_plan_report": str(sdxl_probe_command_plan.get("report") or ""),
        "sdxl_probe_runner_report": str(sdxl_probe_runner_manifest.get("report") or ""),
        "sdxl_probe_results_report": str(sdxl_probe_results.get("report") or ""),
        "sdxl_workload_telemetry_evidence_report": str(sdxl_workload_telemetry_evidence.get("report") or ""),
        "sdxl_workload_stability_manifest_report": str(sdxl_workload_stability_manifest.get("report") or ""),
        "sdxl_workload_stability_evidence_report": str(sdxl_workload_stability_evidence.get("report") or ""),
        "sdxl_workload_loss_guard_evidence_report": str(sdxl_workload_loss_guard_evidence.get("report") or ""),
        "sdxl_alternate_workload_shape_manifest_report": str(
            sdxl_alternate_workload_shape_manifest.get("report") or ""
        ),
        "sdxl_alternate_workload_shape_evidence_report": str(
            sdxl_alternate_workload_shape_evidence.get("report") or ""
        ),
        "sdxl_longer_window_batch2_repeat_manifest_report": str(
            sdxl_longer_window_batch2_repeat_manifest.get("report") or ""
        ),
        "sdxl_longer_window_batch2_repeat_evidence_report": str(
            sdxl_longer_window_batch2_repeat_evidence.get("report") or ""
        ),
        "sdxl_batch2_failure_diagnostic_manifest_report": str(
            sdxl_batch2_failure_diagnostic_manifest.get("report") or ""
        ),
        "sdxl_batch2_failure_diagnostic_evidence_report": str(
            sdxl_batch2_failure_diagnostic_evidence.get("report") or ""
        ),
        "sdxl_batch2_cuda_failure_debug_manifest_report": str(
            sdxl_batch2_cuda_failure_debug_manifest.get("report") or ""
        ),
        "sdxl_batch2_cuda_failure_debug_evidence_report": str(
            sdxl_batch2_cuda_failure_debug_evidence.get("report") or ""
        ),
        "sdxl_batch2_cuda_debug_repeat_evidence_report": str(
            sdxl_batch2_cuda_debug_repeat_evidence.get("report") or ""
        ),
        "sdxl_compile_ab_manifest_report": str(sdxl_compile_ab_manifest.get("report") or ""),
        "sdxl_compile_ab_evidence_report": str(sdxl_compile_ab_evidence.get("report") or ""),
        "sdxl_compile_stable_anchor_manifest_report": str(sdxl_compile_stable_anchor_manifest.get("report") or ""),
        "sdxl_compile_stable_anchor_evidence_report": str(sdxl_compile_stable_anchor_evidence.get("report") or ""),
        "sdxl_compile_warm_cache_manifest_report": str(sdxl_compile_warm_cache_manifest.get("report") or ""),
        "sdxl_compile_warm_cache_evidence_report": str(sdxl_compile_warm_cache_evidence.get("report") or ""),
        "not_release_evidence": True,
        "publishable": False,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "recommended_action_ids": recommended,
        "action_summary": summary,
        "actions": unique_actions,
        "notes": [
            "This plan does not start GPU work.",
            "Manual GPU actions are evidence collection steps, not release claims.",
            "Blocked aggressive policies must stay disabled until natural data-wait, throughput, loss, VRAM and action-boundary gates pass.",
        ],
    }


__all__ = ["P60_NEXT_ACTION_PLAN_REPORT", "ROADMAP", "build_p60_next_action_plan"]
