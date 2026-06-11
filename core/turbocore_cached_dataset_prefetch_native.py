"""Native API helpers for debug-only cached dataset prefetch probes."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


ENABLE_ENV = "LULYNX_ENABLE_NATIVE_CACHE_PREFETCH_SHADOW"
SEED_ENV = "LULYNX_NATIVE_CACHE_PREFETCH_SEED"
EPOCH_ENV = "LULYNX_NATIVE_CACHE_PREFETCH_EPOCH"
DISABLE_ENV = "LULYNX_DISABLE_NATIVE_CACHE_PREFETCH"
_NATIVE_API_CACHE: Any | None = None
_NATIVE_API_CACHE_KEY = ""


def truthy_env(name: str) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on", "enable", "enabled"}


def optional_env_int(name: str) -> int | None:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def payload_signature(value: Any) -> str:
    return hashlib.blake2b(stable_json(value).encode("utf-8"), digest_size=16).hexdigest()


def manifest_fingerprint(manifest: Dict[str, Any]) -> Dict[str, Any]:
    sample_count = 0
    file_count = 0
    existing_file_count = 0
    total_size = 0
    latest_mtime_ns = 0
    path_checksum = hashlib.blake2b(digest_size=16)
    samples = manifest.get("samples", []) if isinstance(manifest, dict) else []
    for sample in samples if isinstance(samples, list) else []:
        if not isinstance(sample, dict):
            continue
        sample_count += 1
        paths = sample.get("paths", [])
        for item in paths if isinstance(paths, list) else []:
            if not isinstance(item, dict):
                continue
            raw_path = str(item.get("path") or "")
            role = str(item.get("role") or "")
            if not raw_path:
                continue
            file_count += 1
            normalized = raw_path.replace("\\", "/")
            path_checksum.update(role.encode("utf-8", errors="ignore"))
            path_checksum.update(b"\0")
            path_checksum.update(normalized.encode("utf-8", errors="ignore"))
            path_checksum.update(b"\0")
            path = Path(raw_path)
            try:
                stat = path.stat()
            except OSError:
                continue
            if not path.is_file():
                continue
            existing_file_count += 1
            total_size += int(stat.st_size)
            mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
            latest_mtime_ns = max(latest_mtime_ns, mtime_ns)
            path_checksum.update(str(stat.st_size).encode("ascii"))
            path_checksum.update(b"\0")
            path_checksum.update(str(mtime_ns).encode("ascii"))
            path_checksum.update(b"\0")
    return {
        "schema_version": 1,
        "sample_count": sample_count,
        "file_count": file_count,
        "existing_file_count": existing_file_count,
        "missing_file_count": max(file_count - existing_file_count, 0),
        "total_size_bytes": total_size,
        "latest_mtime_ns": latest_mtime_ns,
        "checksum": path_checksum.hexdigest(),
        "training_path_enabled": False,
    }


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


def load_native_cache_prefetch_api() -> Any:
    global _NATIVE_API_CACHE, _NATIVE_API_CACHE_KEY
    if truthy_env(DISABLE_ENV):
        raise RuntimeError("native_cache_prefetch_disabled_by_env")
    _inject_native_artifact_dir_from_env()
    cache_key = _native_cache_key()
    if _NATIVE_API_CACHE is not None and _NATIVE_API_CACHE_KEY == cache_key:
        return _NATIVE_API_CACHE
    if importlib.util.find_spec("lulynx_native") is None:
        raise RuntimeError("lulynx_native_not_importable")
    native = importlib.import_module("lulynx_native")
    if not callable(getattr(native, "run_cache_prefetch_plan_probe", None)):
        raise RuntimeError("native_cache_prefetch_entrypoint_missing:run_cache_prefetch_plan_probe")
    _NATIVE_API_CACHE = native
    _NATIVE_API_CACHE_KEY = cache_key
    return native


def load_native_cache_prefetch_session_api() -> Any:
    native = load_native_cache_prefetch_api()
    required = (
        "create_cache_prefetch_session",
        "cache_prefetch_session_stats",
        "run_cache_prefetch_session_probe",
        "destroy_cache_prefetch_session",
    )
    missing = [name for name in required if not callable(getattr(native, name, None))]
    if missing:
        raise RuntimeError("native_cache_prefetch_session_entrypoints_missing:" + ",".join(missing))
    return native


def load_native_cache_prefetch_fast_session_api() -> Any:
    native = load_native_cache_prefetch_session_api()
    if not callable(getattr(native, "run_cache_prefetch_session_fast_probe", None)):
        raise RuntimeError("native_cache_prefetch_fast_entrypoint_missing:run_cache_prefetch_session_fast_probe")
    return native


__all__ = [
    "DISABLE_ENV",
    "ENABLE_ENV",
    "EPOCH_ENV",
    "SEED_ENV",
    "load_native_cache_prefetch_api",
    "load_native_cache_prefetch_fast_session_api",
    "load_native_cache_prefetch_session_api",
    "manifest_fingerprint",
    "optional_env_int",
    "payload_signature",
    "stable_json",
    "truthy_env",
]
