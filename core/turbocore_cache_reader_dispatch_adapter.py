"""DataLoader boundary adapter for cache reader dispatch contract shadows."""

from __future__ import annotations

from typing import Any

from core.turbocore_cache_reader_training_gate import (
    BATCH_CPU_PAYLOAD_BUFFER_BYTES_ENV,
    BATCH_DISPATCH_CONTRACT_ENV,
    DISPATCH_STRICT_FALLBACK_ENV,
    PARITY_BATCHES_ENV,
    PARITY_MAX_BYTES_ENV,
    run_cache_reader_training_experimental_gate,
)
from core.turbocore_cached_dataset_prefetch_native import optional_env_int, truthy_env
from core.turbocore_cache_reader_dispatch_eligibility import build_cache_reader_dispatch_eligibility_report


def run_cache_reader_dispatch_contract_boundary_shadow(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    prefetch_factor: int | None = None,
) -> dict[str, Any]:
    max_batch_bytes = max(int(optional_env_int(BATCH_CPU_PAYLOAD_BUFFER_BYTES_ENV) or 0), 0)
    eligibility = build_cache_reader_dispatch_eligibility_report(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        strict_fallback=truthy_env(DISPATCH_STRICT_FALLBACK_ENV),
    )
    if max_batch_bytes <= 0:
        return {
            "schema_version": 1,
            "provider": "native_cache_reader_dispatch_contract_boundary_shadow",
            "ok": False,
            "reason": "batch_cpu_payload_buffer_bytes_required",
            "dispatch_contract_ready": False,
            "dispatch_eligibility": eligibility,
            "native_dispatch_eligible": False,
            "native_dispatch_blockers": list(eligibility.get("native_dispatch_blockers", []) or []),
            "training_path_enabled": False,
        }
    report = run_cache_reader_training_experimental_gate(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        max_parity_batches=optional_env_int(PARITY_BATCHES_ENV),
        max_decode_payload_bytes=optional_env_int(PARITY_MAX_BYTES_ENV),
        max_batch_cpu_payload_buffer_bytes=max_batch_bytes,
        enable_batch_handoff_session_shadow=True,
        enable_batch_dispatch_contract_shadow=True,
    )
    contract = dict(report.get("batch_dispatch_contract", {}) or {})
    return {
        "schema_version": 1,
        "probe": "turbocore_cache_reader_dispatch_contract_boundary_shadow",
        "provider": "native_cache_reader_dispatch_contract_boundary_shadow",
        "ok": bool(report.get("ok", False)) and bool(contract.get("dispatch_contract_ready", False)),
        "debug_only": True,
        "shadow_run": True,
        "boundary": "dataloader_attach_metadata_only",
        "dataset_class": type(dataset).__name__,
        "batch_size": max(int(batch_size), 1),
        "shuffle": bool(shuffle),
        "drop_last": bool(drop_last),
        "worker_count": max(int(num_workers), 0),
        "prefetch_factor": None if prefetch_factor is None else max(int(prefetch_factor), 1),
        "dispatch_contract_ready": bool(contract.get("dispatch_contract_ready", False)),
        "would_allow_native_dispatch": False,
        "dispatch_eligibility": dict(contract.get("dispatch_eligibility", eligibility) or eligibility),
        "native_dispatch_eligible": False,
        "native_dispatch_blockers": [str(item) for item in list(contract.get("native_dispatch_blockers", []) or eligibility.get("native_dispatch_blockers", []) or [])],
        "fallback_to_python_batch": True,
        "batch_handle_count": int(contract.get("batch_handle_count", 0) or 0),
        "fallback_reasons": [str(item) for item in list(contract.get("fallback_reasons", []) or [])],
        "training_gate": report,
        "dispatch_contract": contract,
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_dispatch": False,
        "training_path_enabled": False,
    }


def maybe_attach_cache_reader_dispatch_contract_boundary_shadow(
    dataloader: Any,
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    prefetch_factor: int | None = None,
) -> Any:
    if not truthy_env(BATCH_DISPATCH_CONTRACT_ENV):
        return dataloader
    try:
        report = run_cache_reader_dispatch_contract_boundary_shadow(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
    except Exception as exc:
        report = {
            "schema_version": 1,
            "provider": "native_cache_reader_dispatch_contract_boundary_shadow",
            "ok": False,
            "reason": "dispatch_contract_boundary_shadow_failed",
            "native_error": f"{type(exc).__name__}: {exc}",
            "training_path_enabled": False,
        }
    for target in (dataloader, dataset):
        try:
            setattr(target, "native_cache_reader_batch_dispatch_contract_shadow", report)
        except Exception:
            pass
    return dataloader


__all__ = [
    "maybe_attach_cache_reader_dispatch_contract_boundary_shadow",
    "run_cache_reader_dispatch_contract_boundary_shadow",
]
