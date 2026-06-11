"""Debug-only native cached dataset prefetch probes.

This module observes cached dataset/DataLoader boundaries without changing the
training path.  It builds a compact cache manifest and asks the native side for
metadata-only queue/order planning; tensor payload loading stays in Python.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from core.turbocore_cached_dataset_prefetch_manifest import build_cached_dataset_prefetch_manifest
from core.turbocore_cached_dataset_prefetch_native import (
    DISABLE_ENV,
    ENABLE_ENV,
    EPOCH_ENV,
    SEED_ENV,
    load_native_cache_prefetch_api,
    optional_env_int,
    stable_json,
    truthy_env,
)
from core.turbocore_cached_dataset_prefetch_policy import build_cached_dataset_prefetch_policy_details
from core.turbocore_cached_dataset_prefetch_session import (
    NativeCachedDatasetPrefetchSession,
    close_cached_dataset_prefetch_session,
    create_cached_dataset_prefetch_session,
    get_or_create_dataset_prefetch_session,
)


def _python_prefetch_plan_probe(
    manifest: Dict[str, Any],
    *,
    batch_size: int,
    drop_last: bool,
    shuffle: bool,
    seed: int,
    prefetch_depth: int,
    chunk_size: int,
    max_preview: int,
) -> Dict[str, Any]:
    from core.turbocore_dataset_staging import _python_plan_dataset_staging

    samples = list(manifest.get("samples", []) if isinstance(manifest.get("samples"), list) else [])
    plan = _python_plan_dataset_staging(
        sample_count=len(samples),
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        prefetch_depth=prefetch_depth,
        chunk_size=chunk_size,
    )
    index_preview = list(plan.get("index_preview", []))[: max(int(max_preview), 1)]
    format_counts: Dict[str, int] = {}
    file_count = 0
    missing_file_count = 0
    for sample in samples:
        for item in sample.get("paths", []) if isinstance(sample, dict) else []:
            path = str(item.get("path", "") or "") if isinstance(item, dict) else ""
            if not path:
                continue
            file_count += 1
            suffix = Path(path).suffix.lower()
            format_counts[suffix] = format_counts.get(suffix, 0) + 1
            if not Path(path).is_file():
                missing_file_count += 1
    return {
        "schema_version": 1,
        "probe": "turbocore_cached_dataset_prefetch_plan",
        "provider": "python_cached_dataset_prefetch_plan",
        "native_runtime": False,
        "ok": True,
        "family": str(manifest.get("family", "cached") or "cached"),
        "dataset_class": str(manifest.get("dataset_class", "") or ""),
        "manifest": {
            "sample_count": len(samples),
            "file_count": file_count,
            "missing_file_count": missing_file_count,
            "format_counts": format_counts,
        },
        "plan": plan,
        "queue": {
            "queue_capacity_batches": max(int(prefetch_depth), 1),
            "initial_submitted_batches": min(int(plan.get("batch_count", 0) or 0), max(int(prefetch_depth), 1)),
            "metadata_only": True,
        },
        "preview": {"ordered_indices": index_preview},
        "debug_only": True,
        "metadata_only": True,
        "reads_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_path_enabled": False,
    }


def build_cached_dataset_prefetch_adapter_policy(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    prefetch_factor: int | None = None,
    seed: int | None = None,
    epoch: int = 0,
) -> Dict[str, Any]:
    resolved_batch_size = max(int(batch_size), 1)
    resolved_workers = max(int(num_workers), 0)
    resolved_epoch = max(int(epoch or 0), 0)
    seed_provided = seed is not None
    base_seed = max(int(seed or 0), 0)
    effective_seed = base_seed + resolved_epoch
    policy_details = build_cached_dataset_prefetch_policy_details(
        dataset,
        shuffle=shuffle,
        seed_provided=seed_provided,
        num_workers=resolved_workers,
    )
    sampler_order_report = dict(policy_details.get("sampler_order_report", {}))
    live_equivalent = bool(sampler_order_report.get("live_equivalent", False))
    fallback_reasons = list(policy_details.get("fallback_reasons", []))
    fallback_reason = str(policy_details.get("fallback_reason", "") or "")
    return {
        "schema_version": 1,
        "probe": "turbocore_cached_dataset_prefetch_adapter_policy",
        "provider": "native_cached_dataset_prefetch_adapter_policy",
        "ok": True,
        "dataset_class": type(dataset).__name__,
        "sample_count": len(getattr(dataset, "samples", []) or []),
        "batch_size": resolved_batch_size,
        "drop_last": bool(drop_last),
        "shuffle": bool(shuffle),
        "seed": base_seed,
        "seed_provided": seed_provided,
        "epoch": resolved_epoch,
        "effective_seed": effective_seed,
        "epoch_reseed_policy": "base_seed_plus_epoch_v1",
        "cache_mmap": bool(getattr(dataset, "cache_mmap", False)),
        "cache_lazy": bool(getattr(dataset, "cache_lazy", False)),
        "file_handle_cache_size": max(int(getattr(dataset, "file_handle_cache_size", 0) or 0), 0),
        "policy_version": policy_details.get("policy_version", "p5g_cached_prefetch_policy_v1"),
        "native_shadow_decision": policy_details.get("native_shadow_decision", ""),
        "bucket_sampler_detected": bool(policy_details.get("bucket_sampler_detected", False)),
        "concept_geometry_sampler_detected": bool(policy_details.get("concept_geometry_sampler_detected", False)),
        "bucket_sampler_policy": policy_details.get("bucket_sampler_report", {}).get("policy", "flat_sampler_order"),
        "bucket_sampler_report": policy_details.get("bucket_sampler_report", {}),
        "concept_geometry_report": policy_details.get("concept_geometry_report", {}),
        "shape_metadata_report": policy_details.get("shape_metadata_report", {}),
        "sampler_order_report": sampler_order_report,
        "worker_count": resolved_workers,
        "worker_shard_policy": "main_process_sampler_order_only" if resolved_workers > 0 else "single_process_order",
        "worker_fetch_timing_equivalent": resolved_workers == 0,
        "prefetch_factor": None if prefetch_factor is None else max(int(prefetch_factor), 1),
        "native_shadow_supported": bool(policy_details.get("native_shadow_supported", False)),
        "shadow_order_scope": "live_equivalent" if live_equivalent else "diagnostic_reference_only",
        "fallback_reasons": fallback_reasons,
        "fallback_reason": fallback_reason,
        "debug_only": True,
        "shadow_run": False,
        "metadata_only": True,
        "reads_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_path_enabled": False,
    }


def run_cached_dataset_prefetch_shadow_adapter(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    prefetch_factor: int | None = None,
    seed: int | None = None,
    epoch: int = 0,
    chunk_size: int = 256,
    max_preview: int = 16,
    prefer_native: bool = True,
    prefer_session: bool = True,
    persist_session: bool = False,
) -> Dict[str, Any]:
    policy = build_cached_dataset_prefetch_adapter_policy(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
        epoch=epoch,
    )
    if not bool(policy.get("native_shadow_supported", False)):
        return {**policy, "probe": "turbocore_cached_dataset_prefetch_adapter", "ok": True, "skipped": True}
    manifest = build_cached_dataset_prefetch_manifest(dataset)
    resolved_prefetch = max((int(prefetch_factor or 2) * max(int(num_workers or 0), 1)), 1)
    start = time.perf_counter()
    native_error = ""
    if prefer_native and prefer_session:
        try:
            if persist_session:
                session, session_reused, requires_rebuild, fingerprint = get_or_create_dataset_prefetch_session(dataset)
                probe = session.run(
                    batch_size=batch_size,
                    drop_last=drop_last,
                    shuffle=shuffle,
                    seed=int(policy.get("effective_seed", 0) or 0),
                    prefetch_depth=resolved_prefetch,
                    chunk_size=chunk_size,
                    max_preview=max_preview,
                )
            else:
                with NativeCachedDatasetPrefetchSession(manifest) as session:
                    session_reused = False
                    requires_rebuild = False
                    fingerprint = session.fingerprint
                    probe = session.run(
                        batch_size=batch_size,
                        drop_last=drop_last,
                        shuffle=shuffle,
                        seed=int(policy.get("effective_seed", 0) or 0),
                        prefetch_depth=resolved_prefetch,
                        chunk_size=chunk_size,
                        max_preview=max_preview,
                        manifest=manifest,
                    )
        except Exception as exc:
            native_error = f"{type(exc).__name__}: {exc}"
        else:
            return {
                **policy,
                "probe": "turbocore_cached_dataset_prefetch_adapter",
                "provider": "native_cached_dataset_prefetch_persistent_session_adapter" if persist_session else "native_cached_dataset_prefetch_session_adapter",
                "ok": bool(probe.get("ok", False)),
                "skipped": False,
                "shadow_run": True,
                "persistent_session": bool(persist_session),
                "session_reused_by_adapter": bool(session_reused),
                "requires_rebuild": bool(requires_rebuild),
                "fingerprint": fingerprint,
                "native_error": "",
                "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
                "prefetch_probe": probe,
            }
    if prefer_native:
        try:
            native = load_native_cache_prefetch_api()
            probe = native.run_cache_prefetch_plan_probe(
                stable_json(manifest),
                int(batch_size),
                bool(drop_last),
                bool(shuffle),
                int(policy.get("effective_seed", 0) or 0),
                resolved_prefetch,
                max(int(chunk_size), 1),
                max(int(max_preview), 1),
            )
        except Exception as exc:
            native_error = f"{type(exc).__name__}: {exc}"
        else:
            return {
                **policy,
                "probe": "turbocore_cached_dataset_prefetch_adapter",
                "provider": "native_cached_dataset_prefetch_adapter",
                "ok": bool(probe.get("ok", False)),
                "skipped": False,
                "shadow_run": True,
                "native_error": "",
                "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
                "prefetch_probe": probe,
            }
    probe = _python_prefetch_plan_probe(
        manifest,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=int(policy.get("effective_seed", 0) or 0),
        prefetch_depth=resolved_prefetch,
        chunk_size=chunk_size,
        max_preview=max_preview,
    )
    return {
        **policy,
        "probe": "turbocore_cached_dataset_prefetch_adapter",
        "provider": "python_cached_dataset_prefetch_adapter_fallback",
        "ok": True,
        "skipped": False,
        "shadow_run": True,
        "native_error": native_error,
        "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
        "prefetch_probe": probe,
    }


def maybe_attach_cached_dataset_prefetch_shadow_adapter(
    dataloader: Any,
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    prefetch_factor: int | None = None,
) -> Any:
    if not truthy_env(ENABLE_ENV):
        return dataloader
    report = run_cached_dataset_prefetch_shadow_adapter(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        seed=optional_env_int(SEED_ENV),
        epoch=optional_env_int(EPOCH_ENV) or 0,
        persist_session=True,
    )
    for target in (dataloader, dataset):
        try:
            setattr(target, "native_cache_prefetch_shadow_adapter", report)
        except Exception:
            pass
    return dataloader


__all__ = [
    "DISABLE_ENV",
    "ENABLE_ENV",
    "EPOCH_ENV",
    "SEED_ENV",
    "NativeCachedDatasetPrefetchSession",
    "build_cached_dataset_prefetch_adapter_policy",
    "build_cached_dataset_prefetch_manifest",
    "close_cached_dataset_prefetch_session",
    "create_cached_dataset_prefetch_session",
    "maybe_attach_cached_dataset_prefetch_shadow_adapter",
    "run_cached_dataset_prefetch_shadow_adapter",
]
