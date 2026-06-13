"""Append-only JSONL audit trail.

Thread-safe append of structured event records to a JSONL file,
with a tail reader for recent events.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class AuditLog:
    """Append-only JSONL audit log for plugin events."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def append(
        self,
        *,
        event_type: str,
        level: str = "info",
        plugin_id: str = "",
        payload: dict | None = None,
    ) -> dict:
        """Append a single audit event and return the record."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": str(event_type or "").strip() or "unknown",
            "level": str(level or "").strip() or "info",
            "plugin_id": str(plugin_id or "").strip(),
            "payload": payload or {},
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False))
                fh.write("\n")
        return record

    def recent(self, limit: int = 200) -> list[dict]:
        """Read the most recent *limit* events from the audit log."""
        if limit <= 0 or not self._path.exists():
            return []
        with self._lock:
            with open(self._path, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        events: list[dict] = []
        for line in lines[-limit:]:
            try:
                parsed = json.loads(line.strip())
                if isinstance(parsed, dict):
                    events.append(parsed)
            except Exception:
                continue
        return events
