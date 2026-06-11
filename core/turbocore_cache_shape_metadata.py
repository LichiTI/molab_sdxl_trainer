"""Native-first cached tensor shape metadata helpers."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


DISABLE_ENV = "LULYNX_DISABLE_NATIVE_CACHE_SHAPE_METADATA"
_NATIVE_API_CACHE: Any | None = None
_NATIVE_API_CACHE_KEY = ""


def _truthy_env(name: str) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on", "enable", "enabled"}


def _native_cache_key() -> str:
    raw = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    return os.path.abspath(os.path.expanduser(raw)) if raw else ""


def _inject_native_artifact_dir_from_env() -> None:
    raw = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    if not raw:
        return
    path = os.path.abspath(os.path.expanduser(raw))
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def _load_native_cache_shape_api() -> Any:
    global _NATIVE_API_CACHE, _NATIVE_API_CACHE_KEY
    if _truthy_env(DISABLE_ENV):
        raise RuntimeError("native_cache_shape_metadata_disabled_by_env")
    _inject_native_artifact_dir_from_env()
    cache_key = _native_cache_key()
    if _NATIVE_API_CACHE is not None and _NATIVE_API_CACHE_KEY == cache_key:
        return _NATIVE_API_CACHE
    if importlib.util.find_spec("lulynx_native") is None:
        raise RuntimeError("lulynx_native_not_importable")
    native = importlib.import_module("lulynx_native")
    if not callable(getattr(native, "scan_cache_shape_metadata", None)):
        raise RuntimeError("native_cache_shape_entrypoint_missing:scan_cache_shape_metadata")
    _NATIVE_API_CACHE = native
    _NATIVE_API_CACHE_KEY = cache_key
    return native


def scan_native_cache_shape_metadata(
    paths: Iterable[str | Path],
    *,
    max_tensors_per_file: int = 128,
    prefer_native: bool = True,
) -> Dict[str, Any]:
    resolved_paths = [str(Path(path)) for path in paths]
    if not prefer_native:
        raise RuntimeError("native_cache_shape_metadata_not_requested")
    native = _load_native_cache_shape_api()
    return dict(native.scan_cache_shape_metadata(resolved_paths, max(int(max_tensors_per_file), 1)))


def build_cache_shape_index(
    paths: Iterable[str | Path],
    *,
    max_tensors_per_file: int = 128,
    prefer_native: bool = True,
) -> Dict[str, Dict[str, Any]]:
    try:
        report = scan_native_cache_shape_metadata(
            paths,
            max_tensors_per_file=max_tensors_per_file,
            prefer_native=prefer_native,
        )
    except Exception:
        return {}
    index: Dict[str, Dict[str, Any]] = {}
    for record in report.get("records", []) if isinstance(report, dict) else []:
        if not isinstance(record, dict) or not bool(record.get("ok", False)):
            continue
        raw_path = str(record.get("path") or "")
        if not raw_path:
            continue
        path = Path(raw_path)
        keys = {str(path), path.as_posix(), path.name}
        try:
            keys.add(str(path.resolve()))
        except Exception:
            pass
        for key in keys:
            if key:
                index[key.replace("\\", "/")] = record
                index[key] = record
    return index


def shape_from_record(record: Dict[str, Any], key: str = "") -> tuple[int, ...] | None:
    shape: Any = None
    if key:
        shapes = record.get("shapes")
        if isinstance(shapes, dict):
            shape = shapes.get(key)
    if shape is None:
        shape = record.get("selected_latent_shape")
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        return None
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        return None


__all__ = [
    "DISABLE_ENV",
    "build_cache_shape_index",
    "scan_native_cache_shape_metadata",
    "shape_from_record",
]
