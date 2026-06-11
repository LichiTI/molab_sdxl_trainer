"""Timing probes for debug-only cache reader shadow."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from core.turbocore_cache_reader_shadow_manifest import build_cache_reader_shadow_manifest
from core.turbocore_cache_reader_shadow_native import ENABLE_ENV, load_native_cache_reader_shadow_api
from core.turbocore_cached_dataset_prefetch_native import stable_json, truthy_env


def _reader_record(index: int, role: str, path: Path, ok: bool, reason: str, bytes_read: int, start: float) -> Dict[str, Any]:
    return {
        "index": index,
        "role": role,
        "path": str(path),
        "suffix": path.suffix.lower(),
        "ok": ok,
        "exists": ok,
        "reason": reason,
        "bytes_read": bytes_read,
        "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
    }


def _python_cache_reader_shadow_timing(
    manifest: Dict[str, Any],
    *,
    max_files: int,
    max_bytes_per_file: int,
    buffer_size: int,
) -> Dict[str, Any]:
    paths = []
    for sample in manifest.get("samples", []) if isinstance(manifest, dict) else []:
        if not isinstance(sample, dict):
            continue
        for item in sample.get("paths", []) if isinstance(sample.get("paths"), list) else []:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "")
            if path:
                paths.append((str(item.get("role") or "cache"), path))
            if len(paths) >= max(int(max_files), 1):
                break
        if len(paths) >= max(int(max_files), 1):
            break
    start = time.perf_counter()
    read_limit = max(int(max_bytes_per_file), 1)
    chunk_size = max(min(int(buffer_size), read_limit), 4096)
    records = []
    format_counts: Dict[str, int] = {}
    role_counts: Dict[str, int] = {}
    total_bytes_read = 0
    missing_file_count = 0
    error_file_count = 0
    ok_file_count = 0
    for index, (role, raw_path) in enumerate(paths):
        file_start = time.perf_counter()
        path = Path(raw_path)
        suffix = path.suffix.lower()
        format_counts[suffix] = format_counts.get(suffix, 0) + 1
        role_counts[role] = role_counts.get(role, 0) + 1
        try:
            bytes_read = 0
            checksum = 0
            with path.open("rb") as handle:
                while bytes_read < read_limit:
                    payload = handle.read(min(chunk_size, read_limit - bytes_read))
                    if not payload:
                        break
                    for byte in payload:
                        checksum = ((checksum * 16_777_619) ^ int(byte)) & 0xFFFFFFFFFFFFFFFF
                    bytes_read += len(payload)
        except FileNotFoundError:
            missing_file_count += 1
            records.append(_reader_record(index, role, path, False, "file_not_found", 0, file_start))
            continue
        except OSError as exc:
            error_file_count += 1
            records.append(_reader_record(index, role, path, False, str(exc), 0, file_start))
            continue
        ok_file_count += 1
        total_bytes_read += bytes_read
        record = _reader_record(index, role, path, True, "", bytes_read, file_start)
        record["checksum"] = checksum
        records.append(record)
    return {
        "schema_version": 1,
        "probe": "turbocore_cache_reader_shadow_timing",
        "provider": "python_cache_reader_shadow_timing",
        "native_runtime": False,
        "ok": True,
        "file_count": len(paths),
        "ok_file_count": ok_file_count,
        "missing_file_count": missing_file_count,
        "error_file_count": error_file_count,
        "max_files": max(int(max_files), 1),
        "max_bytes_per_file": read_limit,
        "buffer_size": chunk_size,
        "total_bytes_read": total_bytes_read,
        "format_counts": format_counts,
        "role_counts": role_counts,
        "records": records,
        "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
        "debug_only": True,
        "shadow_run": True,
        "reads_file_bytes": True,
        "metadata_only": False,
        "parses_tensor_payloads": False,
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_path_enabled": False,
    }


def run_cache_reader_shadow_timing(
    dataset: Any,
    *,
    max_files: int = 16,
    max_bytes_per_file: int = 1_048_576,
    buffer_size: int = 65_536,
    prefer_native: bool = True,
) -> Dict[str, Any]:
    manifest = build_cache_reader_shadow_manifest(dataset, max_files=max_files)
    if not truthy_env(ENABLE_ENV):
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_timing",
            "provider": "native_cache_reader_shadow_policy",
            "ok": True,
            "skipped": True,
            "reason": "shadow_disabled_by_env",
            "enable_env": ENABLE_ENV,
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }
    native_error = ""
    if prefer_native:
        try:
            native = load_native_cache_reader_shadow_api()
            return dict(native.run_cache_reader_shadow_timing_probe(
                stable_json(manifest),
                max(int(max_files), 1),
                max(int(max_bytes_per_file), 1),
                max(int(buffer_size), 4096),
            ))
        except Exception as exc:
            native_error = f"{type(exc).__name__}: {exc}"
    report = _python_cache_reader_shadow_timing(
        manifest,
        max_files=max_files,
        max_bytes_per_file=max_bytes_per_file,
        buffer_size=buffer_size,
    )
    report["native_error"] = native_error
    return report


__all__ = ["_python_cache_reader_shadow_timing", "run_cache_reader_shadow_timing"]
