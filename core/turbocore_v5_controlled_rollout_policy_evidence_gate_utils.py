"""Shared helpers for the V5-P39 controlled rollout policy evidence gate."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def event_list(values: Sequence[Any] | None) -> list[str]:
    out: list[str] = []
    for index, value in enumerate(values or []):
        if isinstance(value, Mapping):
            if not history_event_active(value):
                continue
            text = str(value.get("reason") or value.get("event") or value.get("kind") or value.get("id") or f"event_{index}")
        else:
            text = str(value or "")
        if text:
            out.append(text)
    return dedupe(out)


def history_event_active(value: Mapping[str, Any]) -> bool:
    status = str(value.get("status") or "").strip().lower()
    severity = str(value.get("severity") or "").strip().lower()
    if status in {"closed", "resolved", "cleared", "clear", "ignored", "dismissed"}:
        return False
    return bool(
        value.get("open") is True
        or value.get("active") is True
        or value.get("cooldown") is True
        or value.get("cooldown_active") is True
        or value.get("rollback_required") is True
        or status in {"open", "active", "cooldown", "pending", "blocked"}
        or severity in {"high", "critical", "blocker", "fatal"}
    )


def history_summary(events: list[str]) -> dict[str, Any]:
    return {"clear": not events, "count": len(events), "events": events}


def history_clear(report: Mapping[str, Any], field: str) -> bool:
    summary = as_dict(report.get(field))
    if not summary:
        return True
    return bool(summary.get("clear", False) and not string_list(summary.get("events")))


def default_off_confirmed(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("default_training_path_enabled") is False
        and value.get("training_path_enabled") is False
        and value.get("default_rollout_allowed") is False
        and value.get("auto_rollout_allowed") is False
    )


def request_adapter_off(value: Mapping[str, Any]) -> bool:
    return bool(value.get("request_adapter_mapping_allowed") is False and value.get("request_fields_emitted") is False)


def digest(value: Mapping[str, Any]) -> str:
    return str(
        value.get("sha256")
        or value.get("policy_digest")
        or value.get("ledger_digest")
        or value.get("source_report_digest")
        or value.get("report_digest")
        or value.get("artifact_digest")
        or ""
    ).strip()


def source(value: Mapping[str, Any]) -> str:
    return str(value.get("source") or value.get("path") or "").strip()


def as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out
