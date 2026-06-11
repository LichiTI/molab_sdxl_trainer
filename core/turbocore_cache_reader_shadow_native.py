"""Native module loading for cache reader shadow probes."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Any

from core.turbocore_cached_dataset_prefetch_native import truthy_env


ENABLE_ENV = "LULYNX_ENABLE_NATIVE_CACHE_READER_SHADOW"
DISABLE_ENV = "LULYNX_DISABLE_NATIVE_CACHE_READER_SHADOW"


_NATIVE_API_CACHE: Any | None = None
_NATIVE_API_CACHE_KEY = ""


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


def load_native_cache_reader_shadow_api() -> Any:
    global _NATIVE_API_CACHE, _NATIVE_API_CACHE_KEY
    if truthy_env(DISABLE_ENV):
        raise RuntimeError("native_cache_reader_shadow_disabled_by_env")
    _inject_native_artifact_dir_from_env()
    cache_key = _native_cache_key()
    if _NATIVE_API_CACHE is not None and _NATIVE_API_CACHE_KEY == cache_key:
        return _NATIVE_API_CACHE
    if importlib.util.find_spec("lulynx_native") is None:
        raise RuntimeError("lulynx_native_not_importable")
    native = importlib.import_module("lulynx_native")
    required = (
        "run_cache_reader_shadow_timing_probe",
        "create_cache_reader_shadow_session",
        "cache_reader_shadow_session_stats",
        "run_cache_reader_shadow_session_probe",
        "destroy_cache_reader_shadow_session",
        "cache_reader_shadow_header_cache_stats",
        "clear_cache_reader_shadow_header_cache",
        "cache_reader_shadow_payload_layout_cache_stats",
        "clear_cache_reader_shadow_payload_layout_cache",
        "run_cache_reader_shadow_payload_layout_probe",
        "run_cache_reader_shadow_payload_read_probe",
        "run_cache_reader_shadow_tensor_decode_contract_probe",
        "run_cache_reader_shadow_tensor_decode_parity_probe",
        "create_cache_reader_shadow_tensor_decode_session",
        "cache_reader_shadow_tensor_decode_session_stats",
        "run_cache_reader_shadow_tensor_decode_session_chunk",
        "run_cache_reader_shadow_tensor_decode_session_cpu_payload_chunk",
        "run_cache_reader_shadow_tensor_decode_session_batch_cpu_payload_chunk",
        "destroy_cache_reader_shadow_tensor_decode_session",
    )
    missing = [name for name in required if not callable(getattr(native, name, None))]
    if missing:
        raise RuntimeError("native_cache_reader_shadow_entrypoints_missing:" + ",".join(missing))
    _NATIVE_API_CACHE = native
    _NATIVE_API_CACHE_KEY = cache_key
    return native


def cache_reader_shadow_header_cache_stats() -> dict[str, Any]:
    native = load_native_cache_reader_shadow_api()
    return dict(native.cache_reader_shadow_header_cache_stats())


def clear_cache_reader_shadow_header_cache() -> dict[str, Any]:
    native = load_native_cache_reader_shadow_api()
    return dict(native.clear_cache_reader_shadow_header_cache())


def cache_reader_shadow_payload_layout_cache_stats() -> dict[str, Any]:
    native = load_native_cache_reader_shadow_api()
    return dict(native.cache_reader_shadow_payload_layout_cache_stats())


def clear_cache_reader_shadow_payload_layout_cache() -> dict[str, Any]:
    native = load_native_cache_reader_shadow_api()
    return dict(native.clear_cache_reader_shadow_payload_layout_cache())


__all__ = [
    "DISABLE_ENV",
    "ENABLE_ENV",
    "cache_reader_shadow_header_cache_stats",
    "cache_reader_shadow_payload_layout_cache_stats",
    "clear_cache_reader_shadow_header_cache",
    "clear_cache_reader_shadow_payload_layout_cache",
    "load_native_cache_reader_shadow_api",
]
