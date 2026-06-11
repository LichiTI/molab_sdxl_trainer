# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Small helper for optional lulynx_native imports used by service adapters."""

from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


_DLL_DIRECTORY_HANDLES: list[Any] = []
_DLL_DIRECTORY_PATHS: set[str] = set()
_NATIVE_ARTIFACT_NAMES = (
    "lulynx_native.pyd",
    "lulynx_native.so",
    "lulynx_native.dylib",
    "lulynx_native.dll",
)


def discover_lulynx_native_artifact_dirs(extra_dirs: list[str | Path] | None = None) -> list[Path]:
    """Return candidate directories that contain a lulynx_native artifact."""

    candidates: list[Path] = []
    env_dir = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.extend(Path(item).expanduser() for item in (extra_dirs or []))
    repo_root = Path(__file__).resolve().parents[3]
    native_target = repo_root / "backend" / "native" / "target"
    candidates.extend(
        [
            native_target / "release",
            native_target / "debug",
            native_target / "release" / "deps",
            native_target / "debug" / "deps",
        ]
    )
    result: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        key = str(resolved).lower()
        if key in seen or not _has_lulynx_native_artifact(resolved):
            continue
        seen.add(key)
        result.append(resolved)
    return result


def ensure_lulynx_native_artifact_path() -> dict[str, Any]:
    """Inject known native artifact directories into sys.path and DLL search path."""

    artifact_dirs = discover_lulynx_native_artifact_dirs()
    inserted: list[str] = []
    for path in reversed(artifact_dirs):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
            inserted.append(text)
        _add_dll_directory_if_supported(path)
    return {
        "schema_version": 1,
        "loader": "lulynx_native_artifact_loader_v1",
        "artifact_dirs": [str(path) for path in artifact_dirs],
        "inserted_sys_paths": inserted,
    }


def probe_lulynx_native_loader() -> dict[str, Any]:
    """Return import diagnostics without raising when lulynx_native is absent."""

    path_report = ensure_lulynx_native_artifact_path()
    try:
        module = importlib.import_module("lulynx_native")
    except Exception as exc:
        return {
            **path_report,
            "importable": False,
            "origin": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        **path_report,
        "importable": True,
        "origin": str(getattr(module, "__file__", "") or ""),
    }


@lru_cache(maxsize=1)
def load_lulynx_native() -> Any:
    ensure_lulynx_native_artifact_path()
    try:
        return importlib.import_module("lulynx_native")
    except Exception:
        return None


def native_with_entrypoints(*entrypoints: str) -> Any:
    native = load_lulynx_native()
    if native is None:
        return None
    if any(not hasattr(native, name) for name in entrypoints):
        return None
    return native


def clear_lulynx_native_cache() -> None:
    load_lulynx_native.cache_clear()


def _has_lulynx_native_artifact(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / name).is_file() for name in _NATIVE_ARTIFACT_NAMES)


def _add_dll_directory_if_supported(path: Path) -> None:
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return
    key = str(path).lower()
    if key in _DLL_DIRECTORY_PATHS:
        return
    try:
        handle = add_dll_directory(str(path))
    except OSError:
        return
    _DLL_DIRECTORY_PATHS.add(key)
    _DLL_DIRECTORY_HANDLES.append(handle)
