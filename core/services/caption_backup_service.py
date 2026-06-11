"""Caption backup helpers shared by legacy compatibility routes."""

from __future__ import annotations

import shutil
from datetime import datetime
import os
from pathlib import Path
from typing import Any

from backend.core.services.native_module_loader import load_lulynx_native


def get_backup_dir(dataset_dir: Path) -> Path:
    return dataset_dir / ".backups"


def native_caption_backup_disabled() -> bool:
    return str(os.environ.get("LULYNX_DISABLE_NATIVE_CAPTION_BACKUP", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_native_caption_backup_api() -> Any:
    return load_lulynx_native()


load_native_caption_backup_api.cache_clear = load_lulynx_native.cache_clear  # type: ignore[attr-defined]


def native_caption_backup_api() -> Any:
    if native_caption_backup_disabled():
        return None
    native = load_native_caption_backup_api()
    required = (
        "scan_caption_backup_sources",
        "list_caption_backup_entries",
        "scan_caption_backup_restore_files",
    )
    if native is None or not all(hasattr(native, name) for name in required):
        return None
    return native


def create_caption_backup(
    dataset_dir: Path,
    *,
    snapshot_name: str = "",
    caption_extension: str = ".txt",
    recursive: bool = True,
) -> dict[str, Any]:
    caption_ext = str(caption_extension or ".txt")
    if not caption_ext.startswith("."):
        caption_ext = "." + caption_ext
    backup_dir = get_backup_dir(dataset_dir)
    name = str(snapshot_name or "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_sub = backup_dir / name
    backup_sub.mkdir(parents=True, exist_ok=True)
    count = 0
    for txt_file, rel_text in iter_caption_backup_sources(dataset_dir, caption_ext, recursive=recursive):
        rel = Path(rel_text)
        dest = backup_sub / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(txt_file, dest)
        count += 1
    return {"archive_name": name, "backup_name": name, "file_count": count}


def create_caption_backup_payload(params: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = _dataset_dir_from_params(params)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {dataset_dir}")
    return create_caption_backup(
        dataset_dir,
        snapshot_name=str(params.get("snapshot_name", "") or params.get("backup_name", "") or ""),
        caption_extension=str(params.get("caption_extension", ".txt") or ".txt"),
        recursive=bool(params.get("recursive", True)),
    )


def list_caption_backups(dataset_dir: Path) -> dict[str, Any]:
    backup_dir = get_backup_dir(dataset_dir)
    if not backup_dir.is_dir():
        return {"backups": []}
    native_payload = list_caption_backups_native(dataset_dir)
    if native_payload is not None:
        return native_payload
    backups = []
    for entry in sorted(backup_dir.iterdir(), reverse=True):
        if not entry.is_dir():
            continue
        file_count = sum(1 for file in entry.rglob("*") if file.is_file() and file.name != ".tageditor_manifest.json")
        stat = entry.stat()
        backups.append(
            {
                "archive_name": entry.name,
                "name": entry.name,
                "file_count": file_count,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            }
        )
    return {"backups": backups}


def list_caption_backups_payload(params: dict[str, Any]) -> dict[str, Any]:
    return list_caption_backups(_dataset_dir_from_params(params))


def restore_caption_backup(dataset_dir: Path, backup_name: str) -> dict[str, Any]:
    backup_sub = get_backup_dir(dataset_dir) / str(backup_name or "").strip()
    if not backup_sub.is_dir():
        raise FileNotFoundError(f"Backup not found: {backup_name}")
    count = 0
    for src_file, rel_text in iter_caption_backup_restore_files(backup_sub):
        rel = Path(rel_text)
        dest = dataset_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest)
        count += 1
    return {"restored_count": count}


def restore_caption_backup_payload(params: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = _dataset_dir_from_params(params)
    backup_name = str(params.get("backup_name", "") or params.get("archive_name", "") or params.get("name", "") or "")
    if not backup_name:
        raise ValueError("Missing path or backup_name")
    return restore_caption_backup(dataset_dir, backup_name)


def _dataset_dir_from_params(params: dict[str, Any]) -> Path:
    dir_path = str(params.get("path", "") or params.get("dir", "") or "")
    if not dir_path:
        raise ValueError("Missing path parameter")
    return Path(dir_path)


def iter_caption_backup_sources(dataset_dir: Path, caption_ext: str, *, recursive: bool) -> list[tuple[Path, str]]:
    native = native_caption_backup_api()
    if native is not None:
        try:
            payload = native.scan_caption_backup_sources(str(dataset_dir), str(caption_ext), bool(recursive))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            files = payload.get("files", [])
            records = [
                (Path(str(item.get("path", "") or "")), str(item.get("relative_path", "") or ""))
                for item in files
                if isinstance(item, dict) and str(item.get("path", "") or "") and str(item.get("relative_path", "") or "")
            ]
            if records or files == []:
                return records
    backup_dir = get_backup_dir(dataset_dir)
    glob_fn = dataset_dir.rglob if recursive else dataset_dir.glob
    records = []
    for txt_file in glob_fn(f"*{caption_ext}"):
        if not txt_file.is_file() or txt_file.name.startswith(".") or backup_dir in txt_file.parents:
            continue
        records.append((txt_file, str(txt_file.relative_to(dataset_dir)).replace("\\", "/")))
    return records


def list_caption_backups_native(dataset_dir: Path) -> dict[str, Any] | None:
    native = native_caption_backup_api()
    if native is None:
        return None
    try:
        payload = native.list_caption_backup_entries(str(dataset_dir))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    backups: list[dict[str, Any]] = []
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
        name = str(item.get("name", "") or entry.name)
        backups.append(
            {
                "archive_name": str(item.get("archive_name", "") or name),
                "name": name,
                "file_count": int(item.get("file_count", 0) or 0),
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            }
        )
    return {"backups": backups}


def iter_caption_backup_restore_files(backup_sub: Path) -> list[tuple[Path, str]]:
    native = native_caption_backup_api()
    if native is not None:
        try:
            payload = native.scan_caption_backup_restore_files(str(backup_sub))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            files = payload.get("files", [])
            records = [
                (Path(str(item.get("path", "") or "")), str(item.get("relative_path", "") or ""))
                for item in files
                if isinstance(item, dict) and str(item.get("path", "") or "") and str(item.get("relative_path", "") or "")
            ]
            if records or files == []:
                return records
    records = []
    for src_file in backup_sub.rglob("*"):
        if not src_file.is_file() or src_file.name == ".tageditor_manifest.json":
            continue
        records.append((src_file, str(src_file.relative_to(backup_sub)).replace("\\", "/")))
    return records


# Backwards-compatible private names used by older tests/patch points.
_load_native_caption_backup_api = load_native_caption_backup_api
_native_caption_backup_api = native_caption_backup_api
_native_caption_backup_disabled = native_caption_backup_disabled
