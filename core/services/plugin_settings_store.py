"""Persistent settings JSON helpers for plugin runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PluginSettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read_all(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"plugins": {}}
        if not isinstance(data, dict):
            return {"plugins": {}}
        if not isinstance(data.get("plugins"), dict):
            data["plugins"] = {}
        return data

    def write_all(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def set_plugin_settings(self, plugin_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        data = self.read_all()
        data.setdefault("plugins", {})[plugin_id] = dict(settings or {})
        self.write_all(data)
        return data
