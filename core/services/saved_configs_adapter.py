"""Adapter helpers for legacy saved training config routes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend.core.services.native_module_loader import load_lulynx_native, native_with_entrypoints


def safe_saved_config_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", str(name or "")).strip()


def saved_configs_dir(project_root: Path) -> Path:
    """Return the saved_configs directory, creating it if needed."""

    return _ensure_configs_dir(project_root / "assets" / "ui_state" / "saved_configs")


def load_native_saved_configs_api() -> Any:
    return load_lulynx_native()


load_native_saved_configs_api.cache_clear = load_lulynx_native.cache_clear  # type: ignore[attr-defined]


def native_saved_configs_api() -> Any:
    return native_with_entrypoints("list_saved_config_files")


def list_saved_configs(configs_dir: Path) -> dict[str, Any]:
    root = _ensure_configs_dir(configs_dir)
    native_payload = list_saved_configs_native(root)
    if native_payload is not None:
        return native_payload
    configs = []
    for file_path in root.iterdir():
        if file_path.suffix == ".json":
            stat = file_path.stat()
            configs.append({"name": file_path.stem, "time": int(stat.st_mtime * 1000)})
    configs.sort(key=lambda item: item["time"], reverse=True)
    return {"configs": configs}


def list_saved_configs_native(configs_dir: Path) -> dict[str, Any] | None:
    native = native_saved_configs_api()
    if native is None:
        return None
    try:
        payload = native.list_saved_config_files(str(configs_dir))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    configs = []
    for item in payload.get("configs", []) or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "")
        if not name:
            continue
        configs.append({"name": name, "time": int(item.get("time", 0) or 0)})
    return {"configs": configs}


def save_saved_config_payload(configs_dir: Path, body: dict[str, Any]) -> dict[str, Any]:
    root = _ensure_configs_dir(configs_dir)
    name = str(body.get("name", "untitled") or "")
    if not name:
        raise ValueError("Missing config name")
    safe_name = safe_saved_config_name(name)
    file_path = root / f"{safe_name}.json"
    file_path.write_text(json.dumps(body.get("config", {}) or {}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"name": safe_name}


def load_saved_config_payload(configs_dir: Path, name: str) -> dict[str, Any]:
    if not name:
        raise ValueError("Missing config name")
    file_path = _ensure_configs_dir(configs_dir) / f"{safe_saved_config_name(name)}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Config '{name}' not found")
    return json.loads(file_path.read_text(encoding="utf-8"))


def delete_saved_config_payload(configs_dir: Path, name: str) -> None:
    if not name:
        raise ValueError("Missing config name")
    file_path = _ensure_configs_dir(configs_dir) / f"{safe_saved_config_name(name)}.json"
    if file_path.exists():
        file_path.unlink()


def rename_saved_config_payload(configs_dir: Path, body: dict[str, Any]) -> None:
    old_name = str(body.get("oldName", "") or "")
    new_name = str(body.get("newName", "") or "")
    if not old_name or not new_name:
        raise ValueError("Missing oldName or newName")
    root = _ensure_configs_dir(configs_dir)
    old_path = root / f"{safe_saved_config_name(old_name)}.json"
    new_path = root / f"{safe_saved_config_name(new_name)}.json"
    if not old_path.exists():
        raise FileNotFoundError(f"Config '{old_name}' not found")
    if new_path.exists():
        raise FileExistsError(f"Config '{new_name}' already exists")
    old_path.rename(new_path)


def load_saved_config_route_payload(configs_dir: Path, name: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return (payload, error_detail) for route-level compatibility mapping.

    Keeps the legacy behavior where JSON corruption raises a 500 detail in the
    route layer while other errors stay in the normal envelope path.
    """
    try:
        return load_saved_config_payload(configs_dir, name), None
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"配置文件损坏: {exc}"


def list_saved_configs_route_payload(project_root: Path) -> dict[str, Any]:
    return list_saved_configs(saved_configs_dir(project_root))


def save_saved_config_route_payload(project_root: Path, body: dict[str, Any]) -> dict[str, Any]:
    return save_saved_config_payload(saved_configs_dir(project_root), body)


def load_saved_config_from_project_route_payload(project_root: Path, name: str) -> tuple[dict[str, Any] | None, str | None]:
    return load_saved_config_route_payload(saved_configs_dir(project_root), name)


def delete_saved_config_route_payload(project_root: Path, name: str) -> None:
    delete_saved_config_payload(saved_configs_dir(project_root), name)


def rename_saved_config_route_payload(project_root: Path, body: dict[str, Any]) -> None:
    rename_saved_config_payload(saved_configs_dir(project_root), body)


def _ensure_configs_dir(configs_dir: Path) -> Path:
    configs_dir.mkdir(parents=True, exist_ok=True)
    return configs_dir


# Backwards-compatible private names used by older tests/patch points.
_load_native_saved_configs_api = load_native_saved_configs_api
_native_saved_configs_api = native_saved_configs_api
