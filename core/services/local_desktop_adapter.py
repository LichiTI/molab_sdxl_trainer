"""Local desktop integration helpers for compatibility routes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

OpenCommand = Callable[[Path], None]


class LocalDesktopError(RuntimeError):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def resolve_open_folder_target(raw_folder: str, *, project_root: Path) -> Path:
    value = str(raw_folder or "output").strip() or "output"
    target = Path(value).expanduser()
    if not target.is_absolute():
        target = project_root / target

    try:
        resolved = target.resolve()
    except OSError as exc:
        raise LocalDesktopError(f"路径无效: {exc}", "path.invalid") from exc

    normalized_raw = value.replace("\\", "/").strip("/").lower()
    if not resolved.exists() and normalized_raw in {"output", "logs", "output/sample", "sample"}:
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LocalDesktopError(f"无法创建目录: {exc}", "path.create_failed") from exc

    if not resolved.exists():
        raise LocalDesktopError(f"路径不存在: {resolved}", "path.missing")
    if not resolved.is_dir():
        resolved = resolved.parent
    return resolved


def open_folder_payload(
    body: dict[str, Any] | None,
    *,
    project_root: Path,
    opener: OpenCommand | None = None,
) -> dict[str, str]:
    raw_folder = str((body or {}).get("folder") or "output").strip() or "output"
    resolved = resolve_open_folder_target(raw_folder, project_root=project_root)
    try:
        (opener or open_folder_in_desktop)(resolved)
    except Exception as exc:
        raise LocalDesktopError(f"打开文件夹失败: {exc}", "path.open_failed") from exc
    return {"path": str(resolved)}


def open_folder_in_desktop(path: Path) -> None:
    if os.name == "nt":
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except Exception:
            subprocess.Popen(["explorer.exe", str(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
