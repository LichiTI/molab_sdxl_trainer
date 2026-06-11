"""Filesystem readers for plugin manifest payloads.

These helpers intentionally treat manifests as data only. They must not import
or execute plugin code during discovery.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MANIFEST_FILENAMES = ("plugin_manifest.json", "manifest.json")


def read_raw_manifest_payload(plugin_root: Path, plugin_id: str) -> dict[str, Any]:
    """Read a manifest payload by declared plugin id or directory fallback."""

    plugin_id = str(plugin_id or "").strip()
    if not plugin_id or not plugin_root.is_dir():
        return {}
    for plugin_dir in plugin_root.iterdir():
        if not plugin_dir.is_dir():
            continue
        for manifest_name in MANIFEST_FILENAMES:
            manifest_path = plugin_dir / manifest_name
            if not manifest_path.is_file():
                continue
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict) and str(payload.get("id") or plugin_dir.name).strip() == plugin_id:
                return payload
    return {}


def iter_raw_manifest_payloads(plugin_root: Path) -> list[dict[str, Any]]:
    """Return all raw manifest payloads without importing plugin modules."""

    payloads: list[dict[str, Any]] = []
    if not plugin_root.is_dir():
        return payloads
    for plugin_dir in sorted(plugin_root.iterdir()):
        if not plugin_dir.is_dir():
            continue
        for manifest_name in MANIFEST_FILENAMES:
            manifest_path = plugin_dir / manifest_name
            if not manifest_path.is_file():
                continue
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                payload = dict(payload)
                payload.setdefault("id", plugin_dir.name)
                payloads.append(payload)
            break
    return payloads


def find_plugin_dir(plugin_root: Path, plugin_id: str) -> Path | None:
    """Find a plugin directory by manifest id or directory-name fallback."""

    plugin_id = str(plugin_id or "").strip()
    if not plugin_id or not plugin_root.is_dir():
        return None
    for plugin_dir in sorted(plugin_root.iterdir()):
        if not plugin_dir.is_dir():
            continue
        payload: dict[str, Any] = {}
        for manifest_name in MANIFEST_FILENAMES:
            manifest_path = plugin_dir / manifest_name
            if not manifest_path.is_file():
                continue
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                payload = loaded if isinstance(loaded, dict) else {}
            except Exception:
                payload = {}
            break
        if str(payload.get("id") or plugin_dir.name).strip() == plugin_id:
            return plugin_dir
    return None
