from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


CROSS_RUN_COOLDOWN_STATUSES = {
    "needs_more_evidence",
    "rollback_failed",
    "rolled_back",
}


def should_carry_bubble_closed_loop_action_history(config: Mapping[str, Any]) -> bool:
    source = dict(config or {})
    mode = (
        str(source.get("bubble_controller_mode") or source.get("bubbleControllerMode") or "")
        .strip()
        .lower()
        .replace("-", "_")
    )
    return mode == "auto_apply" or _boolish(source.get("bubble_controller_auto_apply")) or _boolish(
        source.get("bubbleControllerAutoApply")
    )


def attach_recent_bubble_closed_loop_action_history(
    config: dict[str, Any],
    *,
    runs_dir: Path,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Attach recent failed closed-loop actions as a backend cooldown fallback."""

    if not isinstance(config, dict):
        return []
    if _mapping_list(config.get("bubble_closed_loop_action_history")):
        return []
    if not should_carry_bubble_closed_loop_action_history(config):
        return []

    history = collect_recent_bubble_closed_loop_action_history(runs_dir=runs_dir, limit=limit)
    if not history:
        return []
    config["bubble_closed_loop_action_history"] = history
    if config.get("bubble_closed_loop_cross_run_cooldown_runs") in (None, ""):
        config["bubble_closed_loop_cross_run_cooldown_runs"] = 1
    return history


def collect_recent_bubble_closed_loop_action_history(
    *,
    runs_dir: Path,
    limit: int = 3,
) -> list[dict[str, Any]]:
    max_items = max(int(limit or 0), 0)
    if max_items <= 0:
        return []
    try:
        from backend.core.services.file_training_task_service import list_file_training_tasks
        from backend.core.telemetry_store import FileTelemetryReader

        tasks = list_file_training_tasks(FileTelemetryReader(runs_dir), limit=max_items * 8)
    except Exception:
        return []

    seen: set[str] = set()
    collected: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("task_id") or task.get("id") or "")
        metadata = _mapping(task.get("metadata"))
        state = _mapping(metadata.get("bubble_closed_loop_state"))
        actions = _actions_from_closed_loop_state(state, task_id)
        for action in reversed(actions):
            if str(action.get("status") or "") not in CROSS_RUN_COOLDOWN_STATUSES:
                continue
            key = _action_history_key(action)
            if not key or key in seen:
                continue
            seen.add(key)
            collected.append(action)
            if len(collected) >= max_items:
                return list(reversed(collected))
    return list(reversed(collected))


def _actions_from_closed_loop_state(state: Mapping[str, Any], source_task_id: str) -> list[dict[str, Any]]:
    source = _mapping(state)
    actions = [
        _compact_action_for_history(item, source_task_id)
        for item in _mapping_list(source.get("action_history"))
    ]
    latest = _compact_action_for_history(source.get("latest_action"), source_task_id)
    active = _compact_action_for_history(source.get("active_action"), source_task_id)
    if latest:
        actions.append(latest)
    if active:
        actions.append(active)
    return [item for item in actions if item]


def _compact_action_for_history(action: Any, source_task_id: str) -> dict[str, Any]:
    source = _mapping(action)
    status = str(source.get("status") or "").strip()
    action_id = str(source.get("action_id") or "").strip()
    domain = str(source.get("domain") or "").strip()
    action_kind = str(source.get("action_kind") or "").strip()
    if not status or not (action_id or domain or action_kind):
        return {}

    result: dict[str, Any] = {
        "action_id": action_id,
        "status": status,
        "domain": domain,
        "action_kind": action_kind,
        "applied_overlay": dict(_mapping(source.get("applied_overlay"))),
        "rollback_restore": dict(_mapping(source.get("rollback_restore"))),
    }
    if source_task_id:
        result["source_task_id"] = source_task_id
    for key in ("applied_step", "cooldown_until_step", "closed_step"):
        if source.get(key) is not None:
            result[key] = source.get(key)
    if source.get("rollback_applied_overlay") is not None:
        result["rollback_applied_overlay"] = dict(_mapping(source.get("rollback_applied_overlay")))
    evaluation = _compact_evaluation(source.get("evaluation"))
    if evaluation:
        result["evaluation"] = evaluation
    return result


def _compact_evaluation(evaluation: Any) -> dict[str, Any]:
    source = _mapping(evaluation)
    if not source:
        return {}
    result = {
        key: source[key]
        for key in (
            "steady_samples_per_second_gain_ratio",
            "steady_samples_per_second_gain_pct",
            "evaluated_step",
        )
        if key in source
    }
    for side in ("before", "after"):
        compact = {
            key: _mapping(source.get(side))[key]
            for key in (
                "steady_samples_per_second",
                "active_gpu_util_pct_mean",
                "host_gap_share",
                "final_loss",
            )
            if key in _mapping(source.get(side))
        }
        if compact:
            result[side] = compact
    return result


def _action_history_key(action: Mapping[str, Any]) -> str:
    action_id = str(action.get("action_id") or "").strip()
    if action_id:
        return action_id
    return "|".join(
        str(action.get(key) or "").strip()
        for key in ("domain", "action_kind", "status")
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _boolish(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}


__all__ = [
    "attach_recent_bubble_closed_loop_action_history",
    "collect_recent_bubble_closed_loop_action_history",
    "should_carry_bubble_closed_loop_action_history",
]
