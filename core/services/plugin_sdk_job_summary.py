"""Stable task metadata helpers for plugin SDK runner jobs."""

from __future__ import annotations

from typing import Any

from backend.core.contracts import RunResult


_TAIL_LIMIT = 5
_TEXT_LIMIT = 500


def build_plugin_sdk_job_summary(
    result: RunResult,
    *,
    runner_id: str,
    schema_id: str,
    approval_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a small UI-safe summary for JobManager metadata.

    The full RunResult remains available for audits, but Jobs panels should not
    need to parse the whole envelope just to show SDK progress and recent logs.
    """

    data = dict(result.data or {})
    logs = _records(data.get("sdk_logs"))
    progress = _records(data.get("sdk_progress"))
    sandbox_policy = data.get("sandbox_policy") if isinstance(data.get("sandbox_policy"), dict) else {}
    approval = dict(approval_snapshot or data.get("approval_snapshot") or {})
    last_progress = _last_progress(progress)
    return {
        "schema": "plugin-sdk-job-summary-v1",
        "runner_id": str(runner_id or ""),
        "schema_id": str(schema_id or result.request_id or ""),
        "status": str(result.status or ""),
        "ok": bool(result.ok),
        "message": _text(result.message),
        "log_count": len(logs),
        "progress_count": len(progress),
        "logs_tail": logs[-_TAIL_LIMIT:],
        "progress_tail": progress[-_TAIL_LIMIT:],
        "last_progress": last_progress,
        "artifact_count": len(result.artifacts or []),
        "issue_count": len(result.issues or []),
        "permission_source": str(approval.get("permission_source") or ""),
        "approved": bool(approval.get("approved")),
        "execution_mode": str(sandbox_policy.get("execution_mode") or ""),
        "trust_state": str(sandbox_policy.get("trust_state") or ""),
        "isolation_warning": str(sandbox_policy.get("isolation_warning") or ""),
    }


def _records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    records: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            records.append(_safe_record(item))
    return records


def _safe_record(record: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in record.items():
        name = str(key or "").strip()
        if not name:
            continue
        safe[name] = _safe_value(value)
    return safe


def _safe_value(value: Any) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, list):
        return [_safe_value(item) for item in value[:20]]
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in list(value.items())[:20]}
    return _text(str(value))


def _last_progress(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    item = records[-1]
    current = _number(item.get("current"))
    total = _number(item.get("total"))
    percent = None
    if current is not None and total is not None and total > 0:
        percent = max(0.0, min(100.0, current / total * 100.0))
    return {
        "current": current,
        "total": total,
        "percent": percent,
        "message": _text(item.get("message")),
    }


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _text(value: Any) -> str:
    text = str(value or "")
    return text[:_TEXT_LIMIT]
