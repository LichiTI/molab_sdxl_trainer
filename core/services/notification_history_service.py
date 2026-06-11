from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from heapq import heappop, heappush
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from backend.core.services.native_module_loader import native_with_entrypoints

NotificationType = Literal["success", "error", "warning", "info"]

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_HISTORY_PATH = _DATA_DIR / "notification_history.jsonl"
_MAX_TITLE = 240
_MAX_MESSAGE = 4000
_MAX_SOURCE = 120
_MAX_TARGET = 120
_MAX_RECORDS = 5000
_RETENTION_DAYS = {
    "7d": 7,
    "30d": 30,
    "90d": 90,
    "365d": 365,
}


@dataclass
class PersistentNotification:
    id: str
    type: NotificationType
    title: str
    timestamp: int
    read: bool = False
    message: str | None = None
    source: str | None = None
    target_page: str | None = None
    target_anchor: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "timestamp": self.timestamp,
            "read": self.read,
        }
        if self.message:
            data["message"] = self.message
        if self.source:
            data["source"] = self.source
        if self.target_page:
            data["target_page"] = self.target_page
        if self.target_anchor:
            data["target_anchor"] = self.target_anchor
        if self.metadata:
            data["metadata"] = self.metadata
        return data


class NotificationHistoryService:
    def __init__(self, path: Path = _HISTORY_PATH) -> None:
        self.path = path
        self._lock = RLock()

    def list(self, limit: int = 200) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 1000))
        with self._lock:
            records = self._read_recent(limit)
        records.sort(key=lambda item: _record_timestamp_ms(item) or 0, reverse=True)
        return records

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        notification = self._normalize(payload)
        record = notification.to_dict()
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._trim_if_needed()
        return record

    def mark_all_read(self) -> None:
        with self._lock:
            records = self._read_all()
            for record in records:
                record["read"] = True
            self._write_all(records)

    def clear(self) -> None:
        with self._lock:
            if self.path.exists():
                self.path.unlink()

    def cleanup_by_retention(self, retention: str) -> dict[str, Any]:
        policy = _normalize_retention(retention)
        with self._lock:
            valid_total, remaining = self._retained_records(policy)
            if valid_total != len(remaining):
                self._write_all(remaining)
            return {"retention": policy, "removed": max(0, valid_total - len(remaining)), "remaining": len(remaining)}

    def _read_recent(self, limit: int) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        native = native_notification_history_api("list_notification_history_recent")
        if native is not None:
            try:
                payload = native.list_notification_history_recent(str(self.path), int(limit))
                records = payload.get("notifications", []) if isinstance(payload, dict) else []
                if isinstance(records, list):
                    return [record for record in records if isinstance(record, dict)]
            except Exception:
                pass
        heap: list[tuple[int, int, dict[str, Any]]] = []
        seq = 0
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for record in _iter_valid_records(fh):
                    timestamp = _record_timestamp_ms(record) or 0
                    item = (timestamp, seq, record)
                    seq += 1
                    if len(heap) < limit:
                        heappush(heap, item)
                    elif item > heap[0]:
                        heappop(heap)
                        heappush(heap, item)
        except OSError:
            return []
        return [item[2] for item in heap]

    def _retained_records(self, policy: str) -> tuple[int, list[dict[str, Any]]]:
        from collections import deque

        if not self.path.is_file():
            return 0, []
        cutoff = None
        if policy not in {"on_close", "never"}:
            cutoff = int((time.time() - _RETENTION_DAYS[policy] * 86400) * 1000)
        native = native_notification_history_api("retain_notification_history_records")
        if native is not None:
            try:
                payload = native.retain_notification_history_records(str(self.path), int(cutoff if cutoff is not None else -1), int(_MAX_RECORDS))
                records = payload.get("records", []) if isinstance(payload, dict) else []
                valid_total = int(payload.get("valid_total", 0) or 0) if isinstance(payload, dict) else 0
                if isinstance(records, list):
                    return valid_total, [record for record in records if isinstance(record, dict)]
            except Exception:
                pass
        valid_total = 0
        kept = 0
        retained: deque[dict[str, Any]] = deque(maxlen=_MAX_RECORDS)
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for record in _iter_valid_records(fh):
                    valid_total += 1
                    if cutoff is not None and not _is_record_kept_after(record, cutoff):
                        continue
                    kept += 1
                    retained.append(record)
        except OSError:
            return 0, []
        return valid_total, list(retained)

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                records = list(_iter_valid_records(fh))
        except OSError:
            return []
        return records

    def _write_all(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        records = records[-_MAX_RECORDS:]
        with self.path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _trim_if_needed(self) -> None:
        if not self.path.is_file():
            return
        valid_total, remaining = self._retained_records("never")
        if valid_total > len(remaining):
            self._write_all(remaining)

    def _normalize(self, payload: dict[str, Any]) -> PersistentNotification:
        raw_type = str(payload.get("type") or "info")
        if raw_type not in {"success", "error", "warning", "info"}:
            raw_type = "info"
        title = _clean_text(payload.get("title"), _MAX_TITLE) or "Notification"
        message = _clean_text(payload.get("message"), _MAX_MESSAGE)
        source = _clean_text(payload.get("source"), _MAX_SOURCE)
        target_page = _clean_text(payload.get("target_page"), _MAX_TARGET)
        target_anchor = _clean_text(payload.get("target_anchor"), _MAX_TARGET)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        return PersistentNotification(
            id=_clean_text(payload.get("id"), 160) or f"notif-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}",
            type=raw_type,  # type: ignore[arg-type]
            title=title,
            message=message,
            timestamp=_normalize_timestamp(payload.get("timestamp")),
            read=bool(payload.get("read", False)),
            source=source,
            target_page=target_page,
            target_anchor=target_anchor,
            metadata=metadata,
        )


def _clean_text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:limit]


def native_notification_history_api(*entrypoints: str) -> Any:
    return native_with_entrypoints(*entrypoints)


def _normalize_timestamp(value: Any) -> int:
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        timestamp = int(time.time() * 1000)
    if timestamp < 10_000_000_000:
        timestamp *= 1000
    return timestamp


def _record_timestamp_ms(record: dict[str, Any]) -> int | None:
    try:
        timestamp = int(record.get("timestamp"))
    except (TypeError, ValueError):
        return None
    if timestamp < 10_000_000_000:
        timestamp *= 1000
    return timestamp


def _iter_valid_records(lines) -> Any:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("id") and item.get("title"):
            yield item


def _is_record_kept_after(record: dict[str, Any], cutoff: int) -> bool:
    timestamp = _record_timestamp_ms(record)
    return timestamp is None or timestamp >= cutoff


def _normalize_retention(retention: str) -> str:
    policy = str(retention or "30d").strip().lower()
    if policy in {"on_close", "never", *_RETENTION_DAYS.keys()}:
        return policy
    return "30d"


_service = NotificationHistoryService()


def get_notification_history_service() -> NotificationHistoryService:
    return _service
