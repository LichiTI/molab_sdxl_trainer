"""Failure/rollback history normalization for TurboCore review gates."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Mapping


CLOSED_STATUSES = {
    "cleared",
    "closed",
    "fixed",
    "ignored",
    "mitigated",
    "ok",
    "resolved",
    "restored",
    "success",
    "succeeded",
}
OPEN_STATUSES = {
    "active",
    "blocked",
    "cooldown",
    "degraded",
    "error",
    "failed",
    "failure",
    "open",
    "pending",
    "regression",
    "rollback_required",
    "unresolved",
}
HIGH_SEVERITIES = {"blocker", "critical", "fatal", "high", "p0", "p1", "sev0", "sev1"}


def summarize_v5_failure_history(value: Any, *, kind: str, now_ts: float | None = None) -> dict[str, Any]:
    """Normalize event-list or summary-count history into P26 blockers."""

    now = time.time() if now_ts is None else float(now_ts)
    items = _history_items(value)
    normalized = [_normalize_history_item(item, now_ts=now) for item in items]
    normalized = [item for item in normalized if item["event"] or item["status"]]
    open_items = [item for item in normalized if item["open"]]
    high_items = [item for item in normalized if item["high_severity"]]
    cooldown_items = [item for item in normalized if item["cooldown_active"]]
    blockers: list[str] = []
    if open_items:
        blockers.append(f"v5_p26_{kind}_history_open")
    if high_items:
        blockers.append(f"v5_p26_{kind}_history_high_severity")
    if cooldown_items:
        blockers.append(f"v5_p26_{kind}_history_cooldown_active")
    return {
        "present": bool(items),
        "event_count": len(normalized),
        "open_count": len(open_items),
        "high_severity_count": len(high_items),
        "cooldown_active_count": len(cooldown_items),
        "clear_for_p26": not blockers,
        "recent_events": normalized[-10:],
        "blocked_reasons": blockers,
    }


def _history_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if not isinstance(value, Mapping):
        return [value] if str(value or "") else []

    summary_items = _summary_marker_items(value)
    for key in (
        "events",
        "history",
        "records",
        "failures",
        "failure_events",
        "rollback_events",
        "recent_events",
        "items",
    ):
        items = value.get(key)
        if isinstance(items, list) and items:
            return [*items, *summary_items]
    if summary_items:
        return summary_items
    marker_keys = {"event", "type", "reason", "status", "severity", "cooldown_active", "cooldown_until", "open"}
    return [value] if any(key in value for key in marker_keys) else []


def _summary_marker_items(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if (
        _positive_count(value, "open_failure_count")
        or _positive_count(value, "open_rollback_count")
        or _positive_count(value, "open_count")
        or _truthy(value.get("has_open_failure"))
        or _truthy(value.get("has_open_rollback"))
        or _truthy(value.get("has_open"))
    ):
        items.append({"event": "history_summary_open", "status": "open"})
    if _positive_count(value, "high_severity_count"):
        items.append({"event": "history_summary_high_severity", "severity": "high", "status": "closed"})
    if _positive_count(value, "cooldown_active_count") or _truthy(value.get("cooldown_active")):
        items.append({"event": "history_summary_cooldown", "status": "cooldown", "cooldown_active": True})
    return items


def _normalize_history_item(value: Any, *, now_ts: float) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        event = str(value or "")
        return {
            "event": event,
            "status": "open" if event else "",
            "severity": "",
            "open": bool(event),
            "high_severity": False,
            "cooldown_active": False,
            "source": "",
        }
    event = str(value.get("event") or value.get("type") or value.get("reason") or value.get("name") or "")
    status = str(value.get("status") or value.get("state") or value.get("decision") or "").lower()
    severity = str(value.get("severity") or value.get("level") or value.get("priority") or "").lower()
    explicit_open = _bool_or_none(value.get("open"))
    cooldown_active = bool(value.get("cooldown_active", False)) or status == "cooldown" or _cooldown_active(value, now_ts)
    high = severity in HIGH_SEVERITIES or bool(value.get("high_severity", False))
    if explicit_open is not None:
        is_open = explicit_open
    elif status in CLOSED_STATUSES:
        is_open = False
    elif status in OPEN_STATUSES or cooldown_active:
        is_open = True
    else:
        is_open = bool(event or status) and not high
    return {
        "event": event or status,
        "status": status or ("open" if is_open else "closed"),
        "severity": severity,
        "open": bool(is_open),
        "high_severity": bool(high),
        "cooldown_active": bool(cooldown_active),
        "source": str(value.get("source") or value.get("source_task_id") or value.get("_source_path") or ""),
    }


def _cooldown_active(value: Mapping[str, Any], now_ts: float) -> bool:
    for key in ("cooldown_until", "cooldown_until_ts", "cooldown_until_iso"):
        raw = value.get(key)
        if raw in (None, ""):
            continue
        try:
            return float(raw) > now_ts
        except (TypeError, ValueError):
            pass
        try:
            text = str(raw).replace("Z", "+00:00")
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp() > now_ts
        except ValueError:
            continue
    return False


def _positive_count(value: Mapping[str, Any], key: str) -> bool:
    try:
        return int(value.get(key, 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _truthy(value: Any) -> bool:
    return value is True or str(value).lower() in {"1", "true", "yes", "open", "active"}


__all__ = ["summarize_v5_failure_history"]
