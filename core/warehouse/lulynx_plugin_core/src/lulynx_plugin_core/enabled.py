"""Plugin enabled/disabled state persistence.

Stores per-plugin enable/disable overrides in a JSON file, keyed by
plugin_id.  When no override exists, the manifest's default is used.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = "plugin-enabled-v1"


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class EnabledStore:
    """Thread-safe JSON-file plugin enabled-state store."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return []
        if isinstance(data, dict):
            return [r for r in data.get("records", []) if isinstance(r, dict)]
        return []

    def _save(self, records: list[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump({"schema": _SCHEMA, "records": records}, fh, indent=2)

    def resolve(self, plugin_id: str, *, default_enabled: bool) -> dict:
        """Resolve the effective enabled state for a plugin.

        Returns a dict with ``enabled``, ``has_override``, ``source``,
        and ``reason`` keys.
        """
        records = self._load()
        for r in records:
            if r.get("plugin_id") == plugin_id:
                return {
                    "enabled": _to_bool(r.get("enabled")),
                    "has_override": True,
                    "source": "user_override",
                    "reason": "enabled_by_user" if _to_bool(r.get("enabled")) else "disabled_by_user",
                }
        return {
            "enabled": default_enabled,
            "has_override": False,
            "source": "manifest_default",
            "reason": "enabled_by_manifest" if default_enabled else "disabled_by_manifest",
        }

    def set_override(self, plugin_id: str, *, enabled: bool, updated_by: str = "local-user") -> dict:
        """Set or update an enabled override for a plugin."""
        records = [r for r in self._load() if r.get("plugin_id") != plugin_id]
        record = {
            "plugin_id": plugin_id,
            "enabled": bool(enabled),
            "updated_by": updated_by,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        records.append(record)
        self._save(records)
        return record

    def clear_override(self, plugin_id: str) -> int:
        """Remove the override for a plugin.  Returns count of removed records."""
        records = self._load()
        kept: list[dict] = []
        removed = 0
        for r in records:
            if r.get("plugin_id") == plugin_id:
                removed += 1
            else:
                kept.append(r)
        self._save(kept)
        return removed
