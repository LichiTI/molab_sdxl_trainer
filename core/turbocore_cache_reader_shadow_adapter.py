"""DataLoader attach helpers for debug-only cache reader shadow probes."""

from __future__ import annotations

from typing import Any

from core.turbocore_cache_reader_shadow_layout import NativeCacheReaderDecodeShadowSession
from core.turbocore_cache_reader_shadow_manifest import build_cache_reader_shadow_manifest
from core.turbocore_cache_reader_shadow_native import ENABLE_ENV
from core.turbocore_cache_reader_training_gate import maybe_attach_cache_reader_training_experimental_gate
from core.turbocore_cache_reader_shadow_session import run_cache_reader_shadow_header_session
from core.turbocore_cache_reader_shadow_timing import run_cache_reader_shadow_timing
from core.turbocore_cached_dataset_prefetch_native import optional_env_int, truthy_env


DECODE_SHADOW_BATCHES_ENV = "LULYNX_NATIVE_CACHE_READER_DECODE_SHADOW_BATCHES"
DECODE_SHADOW_MAX_BYTES_ENV = "LULYNX_NATIVE_CACHE_READER_DECODE_MAX_BYTES"


def _max_files_per_sample(dataset: Any) -> int:
    samples = list(getattr(dataset, "samples", []) or [])
    max_files = 1
    for sample in samples[:16]:
        paths = []
        if isinstance(sample, dict):
            paths = list(sample.get("paths", []) or [])
        elif hasattr(sample, "cache_path"):
            paths = [getattr(sample, "cache_path")]
        elif hasattr(sample, "latent_cache_path") or hasattr(sample, "text_cache_path"):
            paths = [getattr(sample, "latent_cache_path", None), getattr(sample, "text_cache_path", None)]
        max_files = max(max_files, len([path for path in paths if path]))
    return max_files


def _compact_chunk(chunk: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "chunk_index": index,
        "ok": bool(chunk.get("ok", False)),
        "cursor": int(chunk.get("cursor", 0) or 0),
        "next_cursor": int(chunk.get("next_cursor", 0) or 0),
        "tensor_decode_count": int(chunk.get("tensor_decode_count", 0) or 0),
        "data_payload_bytes_read": int(chunk.get("data_payload_bytes_read", 0) or 0),
        "chunk_complete": bool(chunk.get("chunk_complete", False)),
        "training_path_enabled": False,
    }


def run_cache_reader_decode_sidecar_adapter(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    prefetch_factor: int | None = None,
    max_shadow_batches: int | None = None,
    max_decode_payload_bytes: int | None = None,
) -> dict[str, Any]:
    resolved_batches = max(int(max_shadow_batches or optional_env_int(DECODE_SHADOW_BATCHES_ENV) or 2), 1)
    resolved_batch_size = max(int(batch_size), 1)
    resolved_max_bytes = max(int(max_decode_payload_bytes or optional_env_int(DECODE_SHADOW_MAX_BYTES_ENV) or 16 * 1024 * 1024), 1)
    max_files = resolved_batches * resolved_batch_size * _max_files_per_sample(dataset)
    manifest = build_cache_reader_shadow_manifest(dataset, max_files=max_files)
    chunks: list[dict[str, Any]] = []
    total_decoded = 0
    total_read = 0
    ok = True
    try:
        with NativeCacheReaderDecodeShadowSession(
            manifest,
            max_files=max_files,
            max_tensors_per_file=16,
            max_decode_payload_bytes=resolved_max_bytes,
            selected_only=True,
        ) as session:
            cursor = 0
            for chunk_index in range(resolved_batches):
                chunk = session.run_chunk(cursor=cursor, max_tensors=resolved_batch_size)
                compact = _compact_chunk(chunk, chunk_index)
                chunks.append(compact)
                ok = ok and bool(compact["ok"])
                total_decoded += int(compact["tensor_decode_count"])
                total_read += int(compact["data_payload_bytes_read"])
                cursor = int(compact["next_cursor"])
                if bool(compact["chunk_complete"]):
                    break
            stats = session.stats()
    except Exception as exc:
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_decode_sidecar_adapter",
            "provider": "python_cache_reader_decode_sidecar_fallback",
            "ok": False,
            "reason": "native_decode_sidecar_failed",
            "native_error": f"{type(exc).__name__}: {exc}",
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }
    return {
        "schema_version": 1,
        "probe": "turbocore_cache_reader_shadow_decode_sidecar_adapter",
        "provider": "native_cache_reader_decode_sidecar_session_adapter",
        "native_runtime": True,
        "ok": bool(ok),
        "debug_only": True,
        "shadow_run": True,
        "sidecar_only": True,
        "batch_size": resolved_batch_size,
        "drop_last": bool(drop_last),
        "shuffle": bool(shuffle),
        "worker_count": max(int(num_workers), 0),
        "prefetch_factor": None if prefetch_factor is None else max(int(prefetch_factor), 1),
        "planned_shadow_batches": resolved_batches,
        "chunk_count": len(chunks),
        "tensor_decode_count": total_decoded,
        "data_payload_bytes_read": total_read,
        "chunks": chunks,
        "session_summary": dict(stats.get("summary", {})) if isinstance(stats, dict) else {},
        "reads_tensor_payload_bytes": True,
        "parses_tensor_payloads": True,
        "decodes_tensor_payloads": True,
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_path_enabled": False,
    }


def maybe_attach_cache_reader_shadow_timing(
    dataloader: Any,
    dataset: Any,
    *,
    max_files: int = 16,
    max_bytes_per_file: int = 1_048_576,
    buffer_size: int = 65_536,
    batch_size: int = 1,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
) -> Any:
    attached = dataloader
    if truthy_env(ENABLE_ENV):
        report = run_cache_reader_shadow_timing(
            dataset,
            max_files=max_files,
            max_bytes_per_file=max_bytes_per_file,
            buffer_size=buffer_size,
            prefer_native=True,
        )
        for target in (attached, dataset):
            try:
                setattr(target, "native_cache_reader_shadow_timing", report)
            except Exception:
                pass
        try:
            session_report = run_cache_reader_shadow_header_session(dataset, persist_session=True)
            for target in (attached, dataset):
                try:
                    setattr(target, "native_cache_reader_shadow_session", session_report)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            decode_report = run_cache_reader_decode_sidecar_adapter(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )
            for target in (attached, dataset):
                try:
                    setattr(target, "native_cache_reader_decode_shadow_adapter", decode_report)
                except Exception:
                    pass
        except Exception:
            pass
    attached = maybe_attach_cache_reader_training_experimental_gate(
        attached,
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    try:
        from core.turbocore_cache_reader_dispatch_adapter import maybe_attach_cache_reader_dispatch_contract_boundary_shadow
    except Exception:
        return attached
    return maybe_attach_cache_reader_dispatch_contract_boundary_shadow(
        attached,
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


__all__ = [
    "DECODE_SHADOW_BATCHES_ENV",
    "DECODE_SHADOW_MAX_BYTES_ENV",
    "maybe_attach_cache_reader_shadow_timing",
    "run_cache_reader_decode_sidecar_adapter",
]
