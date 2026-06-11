"""Readiness checks for Bubble Runtime follow-up investigation plans."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FOLLOWUP_INVESTIGATION_READINESS_REPORT = "bubble_runtime_followup_investigation_readiness_v0"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _path_exists(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and Path(text).exists()


def _read_json(value: Any) -> Mapping[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        path = Path(text)
        if not path.is_file():
            return {}
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, Mapping) else {}


def _command_has(command: Sequence[Any], value: str) -> bool:
    return value in [str(item) for item in command]


def _completed_output(item: Mapping[str, Any]) -> dict[str, Any]:
    expected = str(item.get("expected_output") or "").strip()
    exists = _path_exists(expected)
    return {
        "completed": exists,
        "expected_output": expected,
        "evidence_paths": [expected] if exists else [],
        "missing_paths": [] if exists else ([expected] if expected else ["expected_output_missing"]),
    }


def _inferred_prepare_report_path(expected_output: str, runner_report: Mapping[str, Any]) -> str:
    prepare_report = str(runner_report.get("prepare_report_path") or "").strip()
    if prepare_report:
        return prepare_report
    expected = str(expected_output or "").strip()
    if not expected:
        return ""
    path = Path(expected)
    if path.name == "newbie_cache_runner_report.json":
        return str(path.with_name("newbie_cache_prepare_report.json"))
    return ""


def _warm_cache_prepare_state(item: Mapping[str, Any], existing: Mapping[str, Any]) -> dict[str, Any]:
    expected = str(existing.get("expected_output") or item.get("expected_output") or "")
    runner_report = _read_json(expected)
    runner_status = str(runner_report.get("status") or "").strip()
    prepare_report_path = _inferred_prepare_report_path(expected, runner_report)
    prepare_report = _read_json(prepare_report_path)
    prepare_status = str(prepare_report.get("status") or runner_report.get("prepare_report_status") or "").strip()
    child_report = _mapping(runner_report.get("child_report"))
    child_status = str(child_report.get("status") or "").strip()
    actual_completed = any(status == "cache_ready" for status in (runner_status, prepare_status, child_status))
    dry_run_completed = bool(existing.get("completed")) and runner_status == "dry_run"
    actual_pending = bool(item.get("requires_gpu_if_executed")) and not actual_completed
    return {
        "runner_report_status": runner_status,
        "runner_report_path": expected,
        "dry_run_completed": dry_run_completed,
        "prepare_report_path": prepare_report_path,
        "prepare_report_exists": _path_exists(prepare_report_path),
        "prepare_report_status": prepare_status,
        "child_report_status": child_status,
        "actual_heavy_prepare_completed": actual_completed,
        "actual_heavy_prepare_pending": actual_pending,
    }


def _completed_family_canary_evidence(family: str, run_readiness: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not family:
        return []
    matches: list[dict[str, Any]] = []
    for raw in run_readiness.get("commands", []):
        item = _mapping(raw)
        if str(item.get("family") or "") != family:
            continue
        if str(item.get("status") or "") != "completed_existing_evidence":
            continue
        evidence = _mapping(item.get("existing_evidence"))
        paths = [str(path) for path in evidence.get("evidence_paths", []) if path]
        matches.append(
            {
                "id": str(item.get("id") or ""),
                "family": family,
                "profile": str(item.get("profile") or ""),
                "release_relevant": bool(item.get("release_relevant")),
                "diagnostic_only": bool(item.get("diagnostic_only")),
                "source_data": str(item.get("source_data") or ""),
                "out_dir": str(item.get("out_dir") or ""),
                "evidence_paths": paths,
            }
        )
    return matches


def _item_readiness(item: Mapping[str, Any], run_readiness: Mapping[str, Any]) -> dict[str, Any]:
    item_id = str(item.get("id") or "")
    category = str(item.get("category") or "")
    command = [str(part) for part in item.get("command", []) if part is not None]
    dry_run_command = [str(part) for part in item.get("dry_run_command", []) if part is not None]
    existing = _completed_output(item)
    reasons: list[str] = []
    warnings: list[str] = []
    prepare_state: dict[str, Any] = {}
    status = "manual_review_required"
    manual_start_required = bool(item.get("manual_start_required"))

    if category == "guardrail":
        status = "guardrail_active"
        manual_start_required = False
        reasons.extend(_string_list(item.get("blocked_by")))
    elif command:
        if not _command_has(command, "--out"):
            reasons.append("non_gpu_scan_command_missing_out_flag")
        if existing["completed"]:
            status = (
                "completed_existing_investigation_report"
                if category == "bottleneck_investigation"
                else "completed_existing_scan"
            )
            manual_start_required = False
            warnings.append("expected_output_already_exists")
        elif not reasons:
            status = "ready_non_gpu_command"
            manual_start_required = False
    elif dry_run_command:
        if dry_run_command[-1:] != ["--dry-run"]:
            reasons.append("missing_dry_run_flag")
        prepare_state = _warm_cache_prepare_state(item, existing)
        covered_canary_evidence = _completed_family_canary_evidence(str(item.get("family") or ""), run_readiness)
        if existing["completed"]:
            if prepare_state["actual_heavy_prepare_completed"]:
                status = "completed_existing_heavy_prepare"
                manual_start_required = False
            elif covered_canary_evidence:
                status = "covered_by_completed_canary_evidence"
                manual_start_required = False
                prepare_state["actual_heavy_prepare_pending"] = False
                prepare_state["covered_by_completed_canary_evidence"] = covered_canary_evidence
                warnings.append("actual_prepare_pending_but_equivalent_canary_evidence_exists")
            elif prepare_state["dry_run_completed"] and item.get("requires_gpu_if_executed"):
                status = "dry_run_completed_heavy_manual_pending"
                manual_start_required = True
                warnings.append("dry_run_report_only_actual_prepare_pending")
            elif item.get("requires_gpu_if_executed"):
                status = "heavy_prepare_report_not_ready_manual_pending"
                manual_start_required = True
                warnings.append("existing_prepare_report_not_cache_ready")
            else:
                status = "completed_existing_prepare_dry_run"
                manual_start_required = False
            warnings.append("expected_output_already_exists")
        elif not reasons:
            status = "dry_run_ready_heavy_manual" if item.get("requires_gpu_if_executed") else "dry_run_ready"
            manual_start_required = bool(item.get("requires_gpu_if_executed"))
    elif category in {"bottleneck_investigation", "workload_shape_review"}:
        if existing["completed"]:
            status = "completed_existing_investigation_report"
            warnings.append("expected_output_already_exists")
        else:
            status = "manual_review_required"
        manual_start_required = False
    else:
        reasons.append("unknown_investigation_item_category")
        status = "blocked"
        manual_start_required = False

    if reasons and category != "guardrail":
        status = "blocked"
        manual_start_required = False
    return {
        "id": item_id,
        "family": str(item.get("family") or ""),
        "category": category,
        "track": str(item.get("track") or ""),
        "status": status,
        "manual_start_required": manual_start_required,
        "safe_to_auto_start": False,
        "requires_gpu_if_executed": bool(item.get("requires_gpu_if_executed")),
        "reasons": reasons,
        "warnings": warnings,
        "expected_output": str(item.get("expected_output") or ""),
        "existing_output": existing,
        "dry_run_completed": bool(prepare_state.get("dry_run_completed")),
        "actual_heavy_prepare_completed": bool(prepare_state.get("actual_heavy_prepare_completed")),
        "actual_heavy_prepare_pending": bool(prepare_state.get("actual_heavy_prepare_pending")),
        "prepare_state": prepare_state,
    }


def build_followup_investigation_readiness(
    plan: Mapping[str, Any],
    *,
    run_readiness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify investigation-plan items without starting work."""

    items = [_mapping(item) for item in plan.get("items", []) if _mapping(item)]
    run_state = _mapping(run_readiness)
    readiness = [_item_readiness(item, run_state) for item in items]
    completed = [item for item in readiness if item["status"].startswith("completed_existing")]
    ready_non_gpu = [item for item in readiness if item["status"] == "ready_non_gpu_command"]
    heavy_manual_statuses = {
        "dry_run_ready_heavy_manual",
        "dry_run_completed_heavy_manual_pending",
        "heavy_prepare_report_not_ready_manual_pending",
    }
    heavy_manual = [item for item in readiness if item["status"] in heavy_manual_statuses]
    dry_run_completed = [item for item in readiness if item["dry_run_completed"]]
    actual_heavy_prepare_pending = [item for item in readiness if item["actual_heavy_prepare_pending"]]
    covered_by_canary = [item for item in readiness if item["status"] == "covered_by_completed_canary_evidence"]
    guardrails = [item for item in readiness if item["status"] == "guardrail_active"]
    review = [item for item in readiness if item["status"] == "manual_review_required"]
    blocked = [item for item in readiness if item["status"] == "blocked"]
    return {
        "schema_version": 1,
        "report": FOLLOWUP_INVESTIGATION_READINESS_REPORT,
        "status": "blocked" if blocked else "ready_or_completed_investigations",
        "source_investigation_plan_report": str(plan.get("report") or ""),
        "source_item_count": len(items),
        "completed_item_count": len(completed),
        "ready_non_gpu_command_count": len(ready_non_gpu),
        "heavy_manual_count": len(heavy_manual),
        "dry_run_completed_count": len(dry_run_completed),
        "actual_heavy_prepare_pending_count": len(actual_heavy_prepare_pending),
        "covered_by_completed_canary_evidence_count": len(covered_by_canary),
        "guardrail_active_count": len(guardrails),
        "manual_review_count": len(review),
        "blocked_item_count": len(blocked),
        "safe_to_auto_start": False,
        "completed_item_ids": [item["id"] for item in completed],
        "ready_non_gpu_command_ids": [item["id"] for item in ready_non_gpu],
        "heavy_manual_item_ids": [item["id"] for item in heavy_manual],
        "dry_run_completed_item_ids": [item["id"] for item in dry_run_completed],
        "actual_heavy_prepare_pending_item_ids": [item["id"] for item in actual_heavy_prepare_pending],
        "actual_heavy_manual_item_ids": [item["id"] for item in actual_heavy_prepare_pending],
        "covered_by_completed_canary_evidence_item_ids": [item["id"] for item in covered_by_canary],
        "guardrail_item_ids": [item["id"] for item in guardrails],
        "manual_review_item_ids": [item["id"] for item in review],
        "blocked_item_ids": [item["id"] for item in blocked],
        "items": readiness,
        "notes": [
            "This readiness report does not start GPU work.",
            "completed_existing_scan means the non-GPU source scan output already exists.",
            "dry_run_ready_heavy_manual means only a dry-run command is ready; actual heavy cache preparation remains explicit.",
            "dry_run_completed_heavy_manual_pending means the dry-run runner report exists but the actual cache build is still manual/pending.",
        ],
    }


__all__ = ["FOLLOWUP_INVESTIGATION_READINESS_REPORT", "build_followup_investigation_readiness"]
