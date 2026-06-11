# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""History snapshot helpers for TagEditor caption sidecars."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

from core.services.tag_editor_common import (
    DatasetCaptionItem,
    TAGEDITOR_HISTORY_MANIFEST,
    native_tag_editor_history_api,
    relative_to,
)


def make_backup_name(operation: str = "") -> str:
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(operation or "").strip().lower()).strip("_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}__{suffix}" if suffix else timestamp


def backup_manifest_path(backup_root: Path) -> Path:
    return backup_root / TAGEDITOR_HISTORY_MANIFEST


def read_backup_manifest(backup_root: Path) -> Dict[str, Any]:
    manifest_path = backup_manifest_path(backup_root)
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def normalize_selected_paths(dataset_dir: Path, image_paths: Sequence[str]) -> set[str]:
    selected: set[str] = set()
    for raw_path in image_paths:
        text = str(raw_path or "").replace("\\", "/").strip()
        if not text:
            continue
        selected.add(text.lower())
        try:
            candidate = Path(raw_path)
            if candidate.is_absolute():
                selected.add(relative_to(candidate, dataset_dir).lower())
        except Exception:
            pass
    return selected


def create_backup_for_items(
    dataset_dir: Path,
    items: Sequence[DatasetCaptionItem],
    *,
    label: str = "",
    operation: str = "",
) -> str:
    backup_name = make_backup_name(operation)
    backup_root = dataset_dir / ".backups" / backup_name
    attempt = 1
    while backup_root.exists():
        backup_name = f"{make_backup_name(operation)}_{attempt}"
        backup_root = dataset_dir / ".backups" / backup_name
        attempt += 1
    backup_root.mkdir(parents=True, exist_ok=True)
    manifest_entries: List[Dict[str, Any]] = []
    for item in items:
        try:
            relative = item.caption_path.relative_to(dataset_dir)
        except ValueError:
            relative = Path(item.caption_path.name)
        relative_text = str(relative).replace("\\", "/")
        caption_existed = item.caption_path.exists()
        manifest_entries.append(
            {
                "image_relative_path": item.relative_path,
                "caption_relative_path": relative_text,
                "caption_existed": caption_existed,
            }
        )
        if caption_existed:
            destination = backup_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.caption_path, destination)
    backup_manifest_path(backup_root).write_text(
        json.dumps(
            {
                "kind": "tageditor",
                "created_at": datetime.now().isoformat(),
                "label": str(label or ""),
                "operation": str(operation or ""),
                "target_count": len(items),
                "entries": manifest_entries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return backup_name


def list_history_snapshots(dataset_dir: Path, *, limit: int = 50) -> Dict[str, Any]:
    backup_dir = dataset_dir / ".backups"
    if not backup_dir.is_dir():
        return {"backups": []}
    native_payload = list_history_snapshots_native(dataset_dir, limit=max(1, int(limit)))
    if native_payload is not None:
        return native_payload
    backups: List[Dict[str, Any]] = []
    for entry in sorted(backup_dir.iterdir(), reverse=True):
        if not entry.is_dir():
            continue
        manifest = read_backup_manifest(entry)
        file_count = sum(1 for f in entry.rglob("*") if f.is_file() and f.name != TAGEDITOR_HISTORY_MANIFEST)
        stat = entry.stat()
        backup_info = {
            "archive_name": entry.name,
            "name": entry.name,
            "file_count": file_count,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "kind": "tageditor" if manifest else "generic",
            "label": manifest.get("label", "") if manifest else "",
            "operation": manifest.get("operation", "") if manifest else "",
            "target_count": int(manifest.get("target_count", file_count) or file_count) if manifest else file_count,
        }
        backups.append(backup_info)
        if len(backups) >= max(1, int(limit)):
            break
    return {"backups": backups}


def list_history_snapshots_native(dataset_dir: Path, *, limit: int) -> Dict[str, Any] | None:
    native = native_tag_editor_history_api()
    if native is None:
        return None
    try:
        payload = native.list_tag_editor_history_snapshots(str(dataset_dir), int(limit))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    backups: List[Dict[str, Any]] = []
    for item in payload.get("backups", []) or []:
        if not isinstance(item, dict):
            continue
        entry = Path(str(item.get("path", "") or ""))
        if not entry.is_dir():
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        file_count = int(item.get("file_count", 0) or 0)
        name = str(item.get("name", "") or entry.name)
        kind = str(item.get("kind", "") or "generic")
        backups.append(
            {
                "archive_name": str(item.get("archive_name", "") or name),
                "name": name,
                "file_count": file_count,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "kind": "tageditor" if kind == "tageditor" else "generic",
                "label": str(item.get("label", "") or ""),
                "operation": str(item.get("operation", "") or ""),
                "target_count": int(item.get("target_count", file_count) or file_count),
            }
        )
    return {"backups": backups}


def restore_history_snapshot(
    dataset_dir: Path,
    *,
    backup_name: str,
    image_paths: Sequence[str] = (),
) -> Dict[str, Any]:
    backup_dir = dataset_dir / ".backups" / str(backup_name or "").strip()
    if not backup_dir.is_dir():
        raise FileNotFoundError(f"Backup not found: {backup_name}")
    manifest = read_backup_manifest(backup_dir)
    selected = normalize_selected_paths(dataset_dir, image_paths or [])
    restored = 0
    removed = 0
    if manifest:
        for entry in list(manifest.get("entries", []) or []):
            caption_rel = str(entry.get("caption_relative_path", "") or "").replace("\\", "/")
            image_rel = str(entry.get("image_relative_path", "") or "").replace("\\", "/")
            if selected and image_rel.lower() not in selected and caption_rel.lower() not in selected:
                continue
            destination = dataset_dir / Path(caption_rel)
            if bool(entry.get("caption_existed", False)):
                source = backup_dir / Path(caption_rel)
                if source.is_file():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    restored += 1
            elif destination.exists():
                destination.unlink()
                removed += 1
        return {"restored_count": restored, "removed_count": removed, "backup_name": str(backup_name)}
    for source in backup_dir.rglob("*"):
        if not source.is_file() or source.name == TAGEDITOR_HISTORY_MANIFEST:
            continue
        rel = source.relative_to(backup_dir)
        rel_key = str(rel).replace("\\", "/").lower()
        if selected and rel_key not in selected:
            continue
        destination = dataset_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        restored += 1
    return {"restored_count": restored, "removed_count": removed, "backup_name": str(backup_name)}
