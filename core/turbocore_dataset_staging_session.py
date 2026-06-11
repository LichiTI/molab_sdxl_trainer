"""Long-lived native dataset staging session probes."""

from __future__ import annotations

import json
from typing import Any, Dict

from core.turbocore_dataset_staging import _load_native_dataset_staging_handle_api


def create_native_dataset_staging_lazy_affine_session(
    *,
    sample_count: int,
    batch_size: int,
    drop_last: bool = False,
    seed: int = 0,
    prefetch_depth: int = 512,
    chunk_size: int = 256,
) -> int:
    """Create an experimental native lazy affine staging session."""

    native = _load_native_dataset_staging_handle_api()
    session_id = int(
        native.create_dataset_staging_lazy_affine_session(
            max(int(sample_count), 0),
            max(int(batch_size), 1),
            bool(drop_last),
            max(int(seed), 0),
            max(int(prefetch_depth), 1),
            max(int(chunk_size), 1),
        )
    )
    if session_id <= 0:
        raise RuntimeError("native_dataset_staging_lazy_affine_session_create_failed")
    return session_id


def native_dataset_staging_lazy_affine_session_stats(session_id: int) -> Dict[str, Any]:
    native = _load_native_dataset_staging_handle_api()
    payload = native.dataset_staging_lazy_affine_session_stats(max(int(session_id), 0))
    if not isinstance(payload, dict) or not bool(payload.get("ok", False)):
        raise RuntimeError("native_dataset_staging_lazy_affine_session_stats_failed")
    return payload


def run_native_dataset_staging_lazy_affine_session_epoch(session_id: int) -> Dict[str, Any]:
    native = _load_native_dataset_staging_handle_api()
    payload = native.run_dataset_staging_lazy_affine_session_epoch(max(int(session_id), 0))
    if not isinstance(payload, dict) or not bool(payload.get("ok", False)):
        raise RuntimeError("native_dataset_staging_lazy_affine_session_epoch_failed")
    return payload


def destroy_native_dataset_staging_lazy_affine_session(session_id: int) -> bool:
    native = _load_native_dataset_staging_handle_api()
    return bool(native.destroy_dataset_staging_lazy_affine_session(max(int(session_id), 0)))


def run_native_dataset_staging_lazy_affine_session_probe(
    *,
    sample_count: int,
    batch_size: int,
    drop_last: bool = False,
    seed: int = 0,
    prefetch_depth: int = 512,
    chunk_size: int = 256,
    epochs: int = 2,
) -> Dict[str, Any]:
    """Create a native session, run repeated epoch drains, and destroy it."""

    session_id = create_native_dataset_staging_lazy_affine_session(
        sample_count=sample_count,
        batch_size=batch_size,
        drop_last=drop_last,
        seed=seed,
        prefetch_depth=prefetch_depth,
        chunk_size=chunk_size,
    )
    try:
        stats = native_dataset_staging_lazy_affine_session_stats(session_id)
        epoch_payloads = [
            run_native_dataset_staging_lazy_affine_session_epoch(session_id)
            for _ in range(max(int(epochs), 1))
        ]
        last_epoch = epoch_payloads[-1]
        final_stats = native_dataset_staging_lazy_affine_session_stats(session_id)
        return {
            "schema_version": 1,
            "probe": "turbocore_dataset_staging_lazy_affine_session_probe",
            "provider": "native_dataset_staging_lazy_affine_session",
            "native_runtime": True,
            "ok": all(bool(item.get("ok", False)) for item in epoch_payloads),
            "session_id": session_id,
            "batch_count": int(stats.get("batch_count", 0) or 0),
            "epochs": len(epoch_payloads),
            "last_epoch": last_epoch,
            "initial_stats": stats,
            "final_stats": final_stats,
            "native_index_materialized": False,
            "long_lived_descriptor": True,
            "training_path_enabled": False,
        }
    finally:
        destroy_native_dataset_staging_lazy_affine_session(session_id)


def create_native_dataset_descriptor_session(
    manifest: Dict[str, Any],
    *,
    batch_size: int,
    drop_last: bool = False,
    prefetch_depth: int = 512,
    chunk_size: int = 256,
) -> int:
    """Create a native session that owns dataset sample descriptors."""

    native = _load_native_dataset_staging_handle_api()
    session_id = int(
        native.create_dataset_descriptor_session(
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
            max(int(batch_size), 1),
            bool(drop_last),
            max(int(prefetch_depth), 1),
            max(int(chunk_size), 1),
        )
    )
    if session_id <= 0:
        raise RuntimeError("native_dataset_descriptor_session_create_failed")
    return session_id


def native_dataset_descriptor_session_stats(session_id: int) -> Dict[str, Any]:
    native = _load_native_dataset_staging_handle_api()
    payload = native.dataset_descriptor_session_stats(max(int(session_id), 0))
    if not isinstance(payload, dict) or not bool(payload.get("ok", False)):
        raise RuntimeError("native_dataset_descriptor_session_stats_failed")
    return payload


def reset_native_dataset_descriptor_session_cursor(session_id: int) -> bool:
    native = _load_native_dataset_staging_handle_api()
    return bool(native.reset_dataset_descriptor_session_cursor(max(int(session_id), 0)))


def consume_native_dataset_descriptor_session_chunk(
    session_id: int,
    *,
    max_batches: int,
) -> Dict[str, Any]:
    native = _load_native_dataset_staging_handle_api()
    payload = native.consume_dataset_descriptor_session_chunk(
        max(int(session_id), 0),
        max(int(max_batches), 1),
    )
    if not isinstance(payload, dict) or not bool(payload.get("ok", False)):
        raise RuntimeError("native_dataset_descriptor_session_chunk_failed")
    return payload


def run_native_dataset_descriptor_session_worker_probe(
    session_id: int,
    *,
    worker_count: int = 2,
    queue_depth: int = 8,
    max_batches_per_submit: int = 256,
) -> Dict[str, Any]:
    native = _load_native_dataset_staging_handle_api()
    payload = native.run_dataset_descriptor_session_worker_probe(
        max(int(session_id), 0),
        max(int(worker_count), 1),
        max(int(queue_depth), 1),
        max(int(max_batches_per_submit), 1),
    )
    if not isinstance(payload, dict) or not bool(payload.get("ok", False)):
        raise RuntimeError("native_dataset_descriptor_session_worker_probe_failed")
    return payload


def validate_native_dataset_descriptor_session_parity(
    session_id: int,
    reference_manifest: Dict[str, Any],
    *,
    max_mismatches: int = 8,
) -> Dict[str, Any]:
    """Compare a native descriptor session with a Python-built reference manifest."""

    native = _load_native_dataset_staging_handle_api()
    payload = native.validate_dataset_descriptor_session_parity(
        max(int(session_id), 0),
        json.dumps(reference_manifest, ensure_ascii=False, separators=(",", ":")),
        max(int(max_mismatches), 1),
    )
    if not isinstance(payload, dict):
        raise RuntimeError("native_dataset_descriptor_session_parity_failed")
    return payload


def run_native_dataset_descriptor_session_epoch(session_id: int) -> Dict[str, Any]:
    native = _load_native_dataset_staging_handle_api()
    payload = native.run_dataset_descriptor_session_epoch(max(int(session_id), 0))
    if not isinstance(payload, dict) or not bool(payload.get("ok", False)):
        raise RuntimeError("native_dataset_descriptor_session_epoch_failed")
    return payload


def destroy_native_dataset_descriptor_session(session_id: int) -> bool:
    native = _load_native_dataset_staging_handle_api()
    return bool(native.destroy_dataset_descriptor_session(max(int(session_id), 0)))


def run_native_dataset_descriptor_session_probe(
    manifest: Dict[str, Any],
    *,
    batch_size: int,
    drop_last: bool = False,
    prefetch_depth: int = 512,
    chunk_size: int = 256,
    epochs: int = 2,
) -> Dict[str, Any]:
    """Create a descriptor-owning native session and run repeated epoch drains."""

    session_id = create_native_dataset_descriptor_session(
        manifest,
        batch_size=batch_size,
        drop_last=drop_last,
        prefetch_depth=prefetch_depth,
        chunk_size=chunk_size,
    )
    try:
        stats = native_dataset_descriptor_session_stats(session_id)
        first_chunk = consume_native_dataset_descriptor_session_chunk(
            session_id,
            max_batches=chunk_size,
        )
        reset_native_dataset_descriptor_session_cursor(session_id)
        epoch_payloads = [
            run_native_dataset_descriptor_session_epoch(session_id)
            for _ in range(max(int(epochs), 1))
        ]
        worker_probe = run_native_dataset_descriptor_session_worker_probe(
            session_id,
            worker_count=2,
            queue_depth=max(int(prefetch_depth), 1),
            max_batches_per_submit=max(int(chunk_size), 1),
        )
        parity_probe = validate_native_dataset_descriptor_session_parity(
            session_id,
            manifest,
            max_mismatches=4,
        )
        final_stats = native_dataset_descriptor_session_stats(session_id)
        return {
            "schema_version": 1,
            "probe": "turbocore_dataset_descriptor_session_probe",
            "provider": "native_dataset_descriptor_session",
            "native_runtime": True,
            "ok": all(bool(item.get("ok", False)) for item in epoch_payloads) and bool(parity_probe.get("ok", False)),
            "session_id": session_id,
            "descriptor_count": int(stats.get("descriptor_count", 0) or 0),
            "batch_count": int(stats.get("batch_count", 0) or 0),
            "epochs": len(epoch_payloads),
            "last_epoch": epoch_payloads[-1],
            "first_chunk": first_chunk,
            "worker_probe": worker_probe,
            "parity_probe": parity_probe,
            "initial_stats": stats,
            "final_stats": final_stats,
            "sample_descriptors_owned": True,
            "worker_results_owned": bool(worker_probe.get("worker_results_owned", False)),
            "descriptor_parity_ok": bool(parity_probe.get("ok", False)),
            "long_lived_descriptor": True,
            "training_path_enabled": False,
        }
    finally:
        destroy_native_dataset_descriptor_session(session_id)


__all__ = [
    "create_native_dataset_staging_lazy_affine_session",
    "native_dataset_staging_lazy_affine_session_stats",
    "run_native_dataset_staging_lazy_affine_session_epoch",
    "destroy_native_dataset_staging_lazy_affine_session",
    "run_native_dataset_staging_lazy_affine_session_probe",
    "create_native_dataset_descriptor_session",
    "native_dataset_descriptor_session_stats",
    "reset_native_dataset_descriptor_session_cursor",
    "consume_native_dataset_descriptor_session_chunk",
    "run_native_dataset_descriptor_session_worker_probe",
    "validate_native_dataset_descriptor_session_parity",
    "run_native_dataset_descriptor_session_epoch",
    "destroy_native_dataset_descriptor_session",
    "run_native_dataset_descriptor_session_probe",
]
