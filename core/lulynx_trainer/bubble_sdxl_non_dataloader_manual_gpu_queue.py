"""Build a manual SDXL non-DataLoader GPU probe queue from P60 actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SDXL_NON_DATALOADER_MANUAL_GPU_QUEUE_REPORT = "bubble_sdxl_non_dataloader_manual_gpu_queue_v0"
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


def _summary_completion(expected_summaries: Sequence[str]) -> dict[str, Any]:
    completed: list[str] = []
    missing: list[str] = []
    for raw_path in expected_summaries:
        path = str(raw_path or "")
        if not path:
            continue
        if Path(path).is_file():
            completed.append(path)
        else:
            missing.append(path)

    if not expected_summaries:
        status = "expected_summaries_unknown"
    elif not completed:
        status = "pending_manual_gpu_commands"
    elif missing:
        status = "partially_completed"
    else:
        status = "completed_existing_summaries"

    return {
        "status": status,
        "expected_summary_count": len(expected_summaries),
        "completed_summary_count": len(completed),
        "missing_summary_count": len(missing),
        "completed_summaries": completed,
        "missing_summaries": missing,
    }


def _queue_item(action: Mapping[str, Any]) -> dict[str, Any]:
    expected_summaries = _string_list(action.get("expected_summaries"))
    completion = _summary_completion(expected_summaries)
    action_pending_count = _safe_int(action.get("pending_ready_command_count"))
    pending_count = completion["missing_summary_count"] if expected_summaries else action_pending_count
    return {
        "id": str(action.get("id") or ""),
        "kind": "manual_gpu_probe_group",
        "status": completion["status"],
        "family": "sdxl",
        "source_group_id": str(action.get("source_group_id") or ""),
        "run_order": _safe_int(action.get("run_order"), 999),
        "manual_start_required": True,
        "requires_gpu_if_executed": True,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "pending_ready_command_count": pending_count,
        "action_pending_ready_command_count": action_pending_count,
        "pending_command_ids": _string_list(action.get("pending_command_ids")),
        "expected_summaries": expected_summaries,
        "expected_summary_count": completion["expected_summary_count"],
        "completed_summary_count": completion["completed_summary_count"],
        "missing_summary_count": completion["missing_summary_count"],
        "completed_summaries": completion["completed_summaries"],
        "missing_summaries": completion["missing_summaries"],
        "command": _string_list(action.get("command")),
        "safety_checks": _string_list(action.get("safety_checks")),
        "rationale": str(action.get("rationale") or "Run this protected SDXL probe group after manual GPU authorization."),
    }


def _completed_queue_item(group_id: str, rows: Sequence[Mapping[str, Any]], *, run_order: int) -> dict[str, Any]:
    expected_summaries = [str(row.get("expected_summary") or "") for row in rows if row.get("expected_summary")]
    completion = _summary_completion(expected_summaries)
    return {
        "id": f"completed_{group_id}",
        "kind": "completed_gpu_probe_group",
        "status": completion["status"],
        "family": "sdxl",
        "source_group_id": group_id,
        "run_order": int(run_order),
        "manual_start_required": False,
        "requires_gpu_if_executed": False,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "pending_ready_command_count": completion["missing_summary_count"],
        "completed_command_ids": [str(row.get("command_id") or "") for row in rows if row.get("command_id")],
        "expected_summaries": expected_summaries,
        "expected_summary_count": completion["expected_summary_count"],
        "completed_summary_count": completion["completed_summary_count"],
        "missing_summary_count": completion["missing_summary_count"],
        "completed_summaries": completion["completed_summaries"],
        "missing_summaries": completion["missing_summaries"],
        "rationale": "All expected summaries for this SDXL probe group already exist; use results gate for promotion or rollback review.",
    }


def _refresh_item(
    action: Mapping[str, Any],
    *,
    run_order: int,
    missing_summary_count: int,
    completed_summary_count: int,
) -> dict[str, Any]:
    status = (
        "ready_to_refresh_existing_summaries"
        if completed_summary_count > 0 and missing_summary_count == 0
        else "waiting_for_manual_gpu_groups"
    )
    return {
        "id": str(action.get("id") or "refresh_sdxl_non_dataloader_probe_state_after_runner"),
        "kind": "non_gpu_state_refresh",
        "status": status,
        "family": "sdxl",
        "run_order": int(run_order),
        "manual_start_required": False,
        "requires_gpu_if_executed": False,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "refreshes": _string_list(action.get("refreshes")),
        "command": _string_list(action.get("command")),
        "safety_checks": _string_list(action.get("safety_checks")),
        "missing_summary_count_before_refresh": int(missing_summary_count),
        "completed_summary_count_before_refresh": int(completed_summary_count),
        "rationale": str(action.get("rationale") or "Refresh SDXL probe evidence after manual GPU commands finish."),
    }


def _top_status(*, gpu_count: int, completed_group_count: int, partial_group_count: int) -> str:
    if gpu_count <= 0:
        return "no_pending_manual_gpu_probe_groups"
    if completed_group_count == gpu_count:
        return "manual_gpu_queue_completed_refresh_ready"
    if completed_group_count or partial_group_count:
        return "manual_gpu_queue_partially_completed"
    return "manual_gpu_queue_ready"


def _completed_items_from_runner_manifest(runner_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(runner_manifest.get("report") or "") != "bubble_sdxl_non_dataloader_probe_command_runner_v0":
        return []
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for raw in runner_manifest.get("commands", []):
        row = _mapping(raw)
        if str(row.get("family") or "") != "sdxl":
            continue
        group_id = str(row.get("group_id") or "")
        if not group_id:
            continue
        by_group.setdefault(group_id, []).append(row)

    completed: list[dict[str, Any]] = []
    for group_id, rows in sorted(by_group.items()):
        item = _completed_queue_item(group_id, rows, run_order=0)
        if item.get("status") == "completed_existing_summaries":
            completed.append(item)
    return completed


def build_sdxl_non_dataloader_manual_gpu_queue(
    p60_plan: Mapping[str, Any],
    *,
    runner_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest-only queue for the currently pending SDXL probe groups."""

    protected = []
    refresh_action: Mapping[str, Any] = {}
    for raw in p60_plan.get("actions", []):
        action = _mapping(raw)
        action_type = str(action.get("action_type") or "")
        source_group_id = str(action.get("source_group_id") or "")
        if (
            action_type == "protected_manual_probe_runner"
            and str(action.get("family") or "") == "sdxl"
            and source_group_id
        ):
            protected.append(_queue_item(action))
        elif action_type == "non_gpu_probe_state_refresh" and str(action.get("family") or "") == "sdxl":
            refresh_action = action

    workload = [item for item in protected if "workload" in item.get("source_group_id", "")]
    others = [item for item in protected if item not in workload]
    ordered = workload + sorted(others, key=lambda item: (item["run_order"], item["id"]))
    pending_group_ids = {str(item.get("source_group_id") or "") for item in ordered}
    completed_items = [
        item
        for item in _completed_items_from_runner_manifest(_mapping(runner_manifest))
        if str(item.get("source_group_id") or "") not in pending_group_ids
    ]
    ordered.extend(completed_items)
    for index, item in enumerate(ordered, start=1):
        item["run_order"] = index

    expected_summary_count = sum(_safe_int(item.get("expected_summary_count")) for item in ordered)
    completed_summary_count = sum(_safe_int(item.get("completed_summary_count")) for item in ordered)
    missing_summary_count = sum(_safe_int(item.get("missing_summary_count")) for item in ordered)

    queue: list[dict[str, Any]] = list(ordered)
    if refresh_action:
        queue.append(
            _refresh_item(
                refresh_action,
                run_order=len(queue) + 1,
                missing_summary_count=missing_summary_count,
                completed_summary_count=completed_summary_count,
            )
        )

    manual_gpu_count = sum(1 for item in queue if item.get("kind") == "manual_gpu_probe_group")
    gpu_count = sum(1 for item in ordered if item.get("kind") in {"manual_gpu_probe_group", "completed_gpu_probe_group"})
    refresh_count = sum(1 for item in queue if item.get("kind") == "non_gpu_state_refresh")
    completed_item_count = sum(1 for item in queue if item.get("kind") == "completed_gpu_probe_group")
    pending_commands = sum(_safe_int(item.get("pending_ready_command_count")) for item in queue)
    completed_group_count = sum(1 for item in ordered if item.get("status") == "completed_existing_summaries")
    partial_group_count = sum(1 for item in ordered if item.get("status") == "partially_completed")
    pending_group_count = sum(1 for item in ordered if item.get("status") == "pending_manual_gpu_commands")
    return {
        "schema_version": 1,
        "report": SDXL_NON_DATALOADER_MANUAL_GPU_QUEUE_REPORT,
        "roadmap": ROADMAP,
        "status": _top_status(
            gpu_count=gpu_count,
            completed_group_count=completed_group_count,
            partial_group_count=partial_group_count,
        ),
        "ok": bool(manual_gpu_count),
        "manifest_only": True,
        "execute_requested": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "not_release_evidence": True,
        "publishable": False,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "source_p60_plan_report": str(p60_plan.get("report") or ""),
        "source_p60_plan_status": str(p60_plan.get("status") or ""),
        "summary": {
            "queue_item_count": len(queue),
            "manual_gpu_group_count": manual_gpu_count,
            "completed_gpu_probe_group_item_count": completed_item_count,
            "non_gpu_refresh_item_count": refresh_count,
            "pending_ready_command_count": pending_commands,
            "expected_summary_count": expected_summary_count,
            "completed_summary_count": completed_summary_count,
            "missing_summary_count": missing_summary_count,
            "completed_gpu_group_count": completed_group_count,
            "partially_completed_gpu_group_count": partial_group_count,
            "pending_manual_gpu_group_count": pending_group_count,
        },
        "queue": queue,
        "notes": [
            "This queue is manifest-only and does not execute GPU work.",
            "GPU probe groups must be run only after explicit manual authorization.",
            "Run the non-GPU refresh item after GPU summaries are written to update results and P60 state.",
        ],
    }


__all__ = [
    "ROADMAP",
    "SDXL_NON_DATALOADER_MANUAL_GPU_QUEUE_REPORT",
    "build_sdxl_non_dataloader_manual_gpu_queue",
]
