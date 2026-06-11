"""TurboCore dataset staging planner prototype.

This module keeps dataset order planning behind the request/native boundary. It
does not build a DataLoader or activate training; it only returns compact batch
range/chunk metadata suitable for future native staging queues.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Any, Dict, List


NATIVE_DATA_STAGING_DISABLE_ENV = "LULYNX_DISABLE_NATIVE_DATA_STAGING"
_INDEX_PREVIEW = 16
_NATIVE_API_CACHE: Any | None = None
_NATIVE_API_CACHE_KEY = ""
_NATIVE_HANDLE_API_CACHE_KEY = ""


def _truthy_env(name: str) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on", "enable", "enabled"}


def _inject_native_artifact_dir_from_env() -> None:
    raw = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    if not raw:
        return
    path = os.path.abspath(os.path.expanduser(raw))
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def _native_cache_key() -> str:
    raw = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    return os.path.abspath(os.path.expanduser(raw)) if raw else ""


def _load_native_dataset_staging_api() -> Any:
    global _NATIVE_API_CACHE, _NATIVE_API_CACHE_KEY, _NATIVE_HANDLE_API_CACHE_KEY
    if _truthy_env(NATIVE_DATA_STAGING_DISABLE_ENV):
        raise RuntimeError("native_dataset_staging_disabled_by_env")
    _inject_native_artifact_dir_from_env()
    cache_key = _native_cache_key()
    if _NATIVE_API_CACHE is not None and _NATIVE_API_CACHE_KEY == cache_key:
        return _NATIVE_API_CACHE
    if importlib.util.find_spec("lulynx_native") is None:
        raise RuntimeError("lulynx_native_not_importable")
    native = importlib.import_module("lulynx_native")
    planner = getattr(native, "plan_dataset_staging", None)
    if not callable(planner):
        raise RuntimeError("native_dataset_staging_entrypoint_missing:plan_dataset_staging")
    _NATIVE_API_CACHE = native
    _NATIVE_API_CACHE_KEY = cache_key
    _NATIVE_HANDLE_API_CACHE_KEY = ""
    return native


def _load_native_dataset_staging_handle_api() -> Any:
    global _NATIVE_HANDLE_API_CACHE_KEY
    native = _load_native_dataset_staging_api()
    if _NATIVE_HANDLE_API_CACHE_KEY == _NATIVE_API_CACHE_KEY:
        return native
    required = (
        "create_dataset_staging_plan",
        "dataset_staging_plan_stats",
        "validate_dataset_staging_plan_sampler_order_parity",
        "consume_dataset_staging_chunk",
        "submit_dataset_staging_plan_to_pipeline",
        "run_dataset_staging_pipeline_bulk_probe",
        "run_dataset_staging_lazy_pipeline_bulk_probe",
        "run_dataset_staging_lazy_fast_pipeline_bulk_probe",
        "validate_dataset_sampler_order_parity",
        "run_dataset_shadow_lifecycle_probe",
        "destroy_dataset_staging_plan",
        "create_workspace_pool",
        "destroy_workspace_pool",
        "create_data_pipeline",
        "consume_and_release_counted_batches",
        "close_data_pipeline",
    )
    missing = [name for name in required if not callable(getattr(native, name, None))]
    if missing:
        raise RuntimeError("native_dataset_staging_handle_entrypoints_missing:" + ",".join(missing))
    _NATIVE_HANDLE_API_CACHE_KEY = _NATIVE_API_CACHE_KEY
    return native


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


class _SplitMix64:
    def __init__(self, seed: int) -> None:
        self.state = (int(seed) + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        return _splitmix64(self.state)

    def next_bounded(self, upper: int) -> int:
        if upper <= 1:
            return 0
        return self.next_u64() % int(upper)


def _batch_count(sample_count: int, batch_size: int, drop_last: bool) -> int:
    full, remainder = divmod(max(int(sample_count), 0), max(int(batch_size), 1))
    if drop_last or remainder == 0:
        return full
    return full + 1


def _build_chunks(
    *,
    batch_count: int,
    batch_size: int,
    covered_samples: int,
    chunk_size: int,
    shuffle: bool,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    cursor = 0
    while cursor < batch_count:
        span = min(max(int(chunk_size), 1), batch_count - cursor)
        start_sample = cursor * batch_size
        sample_span = min(max(covered_samples - start_sample, 0), span * batch_size)
        if shuffle:
            chunks.append(
                {
                    "start_batch": cursor,
                    "batch_count": span,
                    "sample_count": sample_span,
                    "order": "deterministic_shuffle",
                }
            )
        else:
            chunks.append(
                {
                    "start_batch": cursor,
                    "batch_count": span,
                    "start_sample": start_sample,
                    "sample_count": sample_span,
                    "order": "sequential_range",
                }
            )
        cursor += span
    return chunks


def _sequential_checksum(
    *,
    sample_count: int,
    batch_size: int,
    batch_count: int,
    covered_samples: int,
    drop_last: bool,
) -> int:
    mask = (1 << 64) - 1
    value = 0xD1B54A32D192ED03
    value ^= (sample_count * 0x9E3779B185EBCA87) & mask
    value ^= ((batch_size << 11) | (batch_size >> (64 - 11))) & mask
    value ^= ((batch_count << 23) | (batch_count >> (64 - 23))) & mask
    value ^= ((covered_samples << 37) | (covered_samples >> (64 - 37))) & mask
    if drop_last:
        value ^= 0xA24BAED4963EE407
    return _splitmix64(value & mask)


def _shuffled_preview_and_checksum(covered_samples: int, seed: int) -> tuple[List[int], int]:
    if covered_samples <= 0:
        return [], _splitmix64(int(seed) ^ 0x4F1BBCDCBFA53E0A)
    indices = list(range(covered_samples))
    rng = _SplitMix64(seed)
    for cursor in range(len(indices) - 1, 0, -1):
        swap_index = rng.next_bounded(cursor + 1)
        indices[cursor], indices[swap_index] = indices[swap_index], indices[cursor]

    mask = (1 << 64) - 1
    checksum = _splitmix64(int(seed) ^ covered_samples ^ 0x6A09E667F3BCC909)
    for position, index in enumerate(indices):
        mixed = _splitmix64(int(index) ^ ((position << 17) & mask))
        checksum = (((checksum << 7) | (checksum >> (64 - 7))) & mask) ^ mixed
        checksum = (checksum * 0x9E3779B185EBCA87) & mask
    return indices[:_INDEX_PREVIEW], checksum


def _python_plan_dataset_staging(
    *,
    sample_count: int,
    batch_size: int,
    drop_last: bool,
    shuffle: bool,
    seed: int,
    prefetch_depth: int,
    chunk_size: int,
) -> Dict[str, Any]:
    resolved_sample_count = max(int(sample_count), 0)
    resolved_batch_size = max(int(batch_size), 1)
    resolved_chunk_size = max(int(chunk_size), 1)
    batch_count = _batch_count(resolved_sample_count, resolved_batch_size, bool(drop_last))
    covered_samples = batch_count * resolved_batch_size if drop_last else resolved_sample_count
    dropped_samples = max(resolved_sample_count - covered_samples, 0)
    chunks = _build_chunks(
        batch_count=batch_count,
        batch_size=resolved_batch_size,
        covered_samples=covered_samples,
        chunk_size=resolved_chunk_size,
        shuffle=bool(shuffle),
    )
    if shuffle:
        preview, checksum = _shuffled_preview_and_checksum(covered_samples, int(seed))
        checksum_kind = "splitmix64_ordered_indices"
        native_index_materialized = covered_samples > 0
    else:
        preview = list(range(min(covered_samples, _INDEX_PREVIEW)))
        checksum = _sequential_checksum(
            sample_count=resolved_sample_count,
            batch_size=resolved_batch_size,
            batch_count=batch_count,
            covered_samples=covered_samples,
            drop_last=bool(drop_last),
        )
        checksum_kind = "sequential_range_config"
        native_index_materialized = False

    return {
        "schema_version": 1,
        "planner": "turbocore_dataset_staging",
        "provider": "python_dataset_staging",
        "native_runtime": False,
        "ok": True,
        "sample_count": resolved_sample_count,
        "batch_size": resolved_batch_size,
        "drop_last": bool(drop_last),
        "shuffle": bool(shuffle),
        "seed": int(seed),
        "prefetch_depth": max(int(prefetch_depth), 1),
        "chunk_size": resolved_chunk_size,
        "batch_count": batch_count,
        "covered_samples": covered_samples,
        "dropped_samples": dropped_samples,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "index_preview": preview,
        "index_checksum": checksum,
        "checksum_kind": checksum_kind,
        "indices_returned": False,
        "native_index_materialized": native_index_materialized,
        "training_path_enabled": False,
    }


def plan_dataset_staging(
    *,
    sample_count: int,
    batch_size: int,
    drop_last: bool = False,
    shuffle: bool = False,
    seed: int = 0,
    prefetch_depth: int = 2,
    chunk_size: int = 256,
    prefer_native: bool = True,
) -> Dict[str, Any]:
    """Return compact dataset staging metadata with native-first fallback."""

    native_error = ""
    native_skip_reason = ""
    native_candidate = bool(shuffle)
    if prefer_native and native_candidate:
        try:
            native = _load_native_dataset_staging_api()
            payload = native.plan_dataset_staging(
                max(int(sample_count), 0),
                max(int(batch_size), 1),
                bool(drop_last),
                bool(shuffle),
                max(int(seed), 0),
                max(int(prefetch_depth), 1),
                max(int(chunk_size), 1),
            )
            if isinstance(payload, dict) and bool(payload.get("ok", False)):
                return payload
            native_error = "native_dataset_staging_returned_unavailable"
        except Exception as exc:
            native_error = f"{type(exc).__name__}: {exc}"
    elif prefer_native:
        native_skip_reason = "native_dataset_staging_skipped:sequential_python_faster"

    payload = _python_plan_dataset_staging(
        sample_count=sample_count,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        prefetch_depth=prefetch_depth,
        chunk_size=chunk_size,
    )
    payload["native_attempted"] = bool(prefer_native)
    if native_skip_reason:
        payload["native_skip_reason"] = native_skip_reason
    if native_error:
        payload["native_fallback_reason"] = native_error
    return payload


def run_native_dataset_staging_handle_probe(
    *,
    sample_count: int,
    batch_size: int,
    drop_last: bool = False,
    shuffle: bool = True,
    seed: int = 0,
    prefetch_depth: int = 2,
    chunk_size: int = 256,
) -> Dict[str, Any]:
    """Exercise the native plan handle/cursor without returning all chunks."""

    native = _load_native_dataset_staging_handle_api()
    plan_id = int(native.create_dataset_staging_plan(
        max(int(sample_count), 0),
        max(int(batch_size), 1),
        bool(drop_last),
        bool(shuffle),
        max(int(seed), 0),
        max(int(prefetch_depth), 1),
        max(int(chunk_size), 1),
    ))
    if plan_id <= 0:
        raise RuntimeError("native_dataset_staging_plan_create_failed")
    try:
        stats = native.dataset_staging_plan_stats(plan_id)
        if not isinstance(stats, dict) or not bool(stats.get("ok", False)):
            raise RuntimeError("native_dataset_staging_plan_stats_invalid")
        emitted = 0
        chunks = 0
        preview: list[int] = []
        while True:
            chunk = native.consume_dataset_staging_chunk(plan_id, max(int(chunk_size), 1))
            if not isinstance(chunk, dict) or not bool(chunk.get("ok", False)):
                break
            emitted += int(chunk.get("batch_count", 0) or 0)
            chunks += 1
            if not preview:
                preview = [int(item) for item in chunk.get("index_preview", [])]
        final_stats = native.dataset_staging_plan_stats(plan_id)
        if not isinstance(final_stats, dict):
            final_stats = {}
        return {
            "schema_version": 1,
            "probe": "turbocore_dataset_staging_handle",
            "provider": "native_dataset_staging_handle",
            "native_runtime": True,
            "ok": emitted == int(stats.get("batch_count", 0) or 0),
            "plan_id": plan_id,
            "batch_count": int(stats.get("batch_count", 0) or 0),
            "emitted_batches": emitted,
            "chunk_count": chunks,
            "chunk_size": max(int(chunk_size), 1),
            "index_checksum": int(stats.get("index_checksum", 0) or 0),
            "index_preview": preview,
            "final_stats": final_stats,
            "training_path_enabled": False,
        }
    finally:
        native.destroy_dataset_staging_plan(plan_id)


def run_native_dataset_staging_pipeline_probe(
    *,
    sample_count: int,
    batch_size: int,
    drop_last: bool = False,
    shuffle: bool = True,
    seed: int = 0,
    prefetch_depth: int = 512,
    chunk_size: int = 256,
) -> Dict[str, Any]:
    """Submit native staging plan batches directly into the counted pipeline."""

    native = _load_native_dataset_staging_handle_api()
    chunk = max(int(chunk_size), 1)
    effective_prefetch = max(max(int(prefetch_depth), 1), chunk)
    plan_id = int(native.create_dataset_staging_plan(
        max(int(sample_count), 0),
        max(int(batch_size), 1),
        bool(drop_last),
        bool(shuffle),
        max(int(seed), 0),
        effective_prefetch,
        chunk,
    ))
    if plan_id <= 0:
        raise RuntimeError("native_dataset_staging_plan_create_failed")
    pool_id = int(native.create_workspace_pool(0))
    if pool_id <= 0:
        native.destroy_dataset_staging_plan(plan_id)
        raise RuntimeError("native_workspace_pool_create_failed")
    pipeline_id = int(native.create_data_pipeline(effective_prefetch, pool_id))
    if pipeline_id <= 0:
        native.destroy_dataset_staging_plan(plan_id)
        native.destroy_workspace_pool(pool_id)
        raise RuntimeError("native_data_pipeline_create_failed")

    try:
        stats = native.dataset_staging_plan_stats(plan_id)
        if not isinstance(stats, dict) or not bool(stats.get("ok", False)):
            raise RuntimeError("native_dataset_staging_plan_stats_invalid")
        target_batches = int(stats.get("batch_count", 0) or 0)
        submitted = 0
        consumed = 0
        iterations = 0
        while submitted < target_batches:
            written = int(native.submit_dataset_staging_plan_to_pipeline(plan_id, pipeline_id, chunk))
            if written == 0:
                drained = int(native.consume_and_release_counted_batches(pipeline_id, chunk))
                if drained == 0:
                    break
                consumed += drained
                continue
            submitted += written
            consumed += int(native.consume_and_release_counted_batches(pipeline_id, chunk))
            iterations += 1
        while True:
            drained = int(native.consume_and_release_counted_batches(pipeline_id, chunk))
            if not drained:
                break
            consumed += drained
        close_stats = native.close_data_pipeline(pipeline_id)
        pipeline_id = 0
        final_stats = native.dataset_staging_plan_stats(plan_id)
        return {
            "schema_version": 1,
            "probe": "turbocore_dataset_staging_pipeline",
            "provider": "native_dataset_staging_handle_pipeline",
            "native_runtime": True,
            "ok": submitted == target_batches and consumed == submitted,
            "batch_count": target_batches,
            "submitted_batches": submitted,
            "consumed_batches": consumed,
            "iterations": iterations,
            "chunk_size": chunk,
            "prefetch_depth": effective_prefetch,
            "index_checksum": int(stats.get("index_checksum", 0) or 0),
            "plan_stats": final_stats if isinstance(final_stats, dict) else {},
            "pipeline_close_stats": close_stats if isinstance(close_stats, dict) else {},
            "training_path_enabled": False,
        }
    finally:
        if pipeline_id:
            try:
                native.close_data_pipeline(pipeline_id)
            except Exception:
                pass
        native.destroy_dataset_staging_plan(plan_id)
        native.destroy_workspace_pool(pool_id)

def run_native_dataset_staging_bulk_pipeline_probe(
    *,
    sample_count: int,
    batch_size: int,
    drop_last: bool = False,
    shuffle: bool = True,
    seed: int = 0,
    prefetch_depth: int = 512,
    chunk_size: int = 256,
) -> Dict[str, Any]:
    """Run plan -> counted pipeline -> consume/release loop inside Rust."""

    native = _load_native_dataset_staging_handle_api()
    payload = native.run_dataset_staging_pipeline_bulk_probe(
        max(int(sample_count), 0),
        max(int(batch_size), 1),
        bool(drop_last),
        bool(shuffle),
        max(int(seed), 0),
        max(int(prefetch_depth), 1),
        max(int(chunk_size), 1),
    )
    if not isinstance(payload, dict) or not bool(payload.get("ok", False)):
        raise RuntimeError("native_dataset_staging_bulk_pipeline_probe_failed")
    return payload


def run_native_dataset_staging_lazy_bulk_pipeline_probe(
    *,
    sample_count: int,
    batch_size: int,
    drop_last: bool = False,
    seed: int = 0,
    prefetch_depth: int = 512,
    chunk_size: int = 256,
) -> Dict[str, Any]:
    """Run experimental lazy affine permutation bulk pipeline probe."""

    native = _load_native_dataset_staging_handle_api()
    payload = native.run_dataset_staging_lazy_pipeline_bulk_probe(
        max(int(sample_count), 0),
        max(int(batch_size), 1),
        bool(drop_last),
        max(int(seed), 0),
        max(int(prefetch_depth), 1),
        max(int(chunk_size), 1),
    )
    if not isinstance(payload, dict) or not bool(payload.get("ok", False)):
        raise RuntimeError("native_dataset_staging_lazy_bulk_pipeline_probe_failed")
    return payload


def run_native_dataset_staging_lazy_fast_bulk_pipeline_probe(
    *,
    sample_count: int,
    batch_size: int,
    drop_last: bool = False,
    seed: int = 0,
    prefetch_depth: int = 512,
    chunk_size: int = 256,
) -> Dict[str, Any]:
    """Run experimental lazy affine bulk pipeline probe with runtime-style summary."""

    native = _load_native_dataset_staging_handle_api()
    payload = native.run_dataset_staging_lazy_fast_pipeline_bulk_probe(
        max(int(sample_count), 0),
        max(int(batch_size), 1),
        bool(drop_last),
        max(int(seed), 0),
        max(int(prefetch_depth), 1),
        max(int(chunk_size), 1),
    )
    if not isinstance(payload, dict) or not bool(payload.get("ok", False)):
        raise RuntimeError("native_dataset_staging_lazy_fast_bulk_pipeline_probe_failed")
    return payload


__all__ = [
    "NATIVE_DATA_STAGING_DISABLE_ENV",
    "plan_dataset_staging",
    "run_native_dataset_staging_handle_probe",
    "run_native_dataset_staging_pipeline_probe",
    "run_native_dataset_staging_bulk_pipeline_probe",
    "run_native_dataset_staging_lazy_bulk_pipeline_probe",
    "run_native_dataset_staging_lazy_fast_bulk_pipeline_probe",
]
