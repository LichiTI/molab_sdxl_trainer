"""Debug-only TurboCore dataset shadow adapter for DataLoader boundaries."""

from __future__ import annotations

import os
from typing import Any, Dict

from core.turbocore_dataset_descriptor_shadow import create_caption_dataset_shadow_session


ENABLE_ENV = "LULYNX_ENABLE_NATIVE_DATA_SHADOW_ADAPTER"
SEED_ENV = "LULYNX_NATIVE_DATA_SHADOW_SEED"
EPOCH_ENV = "LULYNX_NATIVE_DATA_SHADOW_EPOCH"


def _truthy_env(name: str) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on", "enable", "enabled"}


def _optional_env_int(name: str) -> int | None:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def build_caption_dataset_shadow_adapter_policy(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    seed: int | None = None,
    epoch: int = 0,
) -> Dict[str, Any]:
    """Describe the shadow parity scope for a CaptionDataset DataLoader."""

    resolved_batch_size = max(int(batch_size), 1)
    resolved_workers = max(int(num_workers), 0)
    resolved_epoch = max(int(epoch or 0), 0)
    seed_provided = seed is not None
    base_seed = max(int(seed or 0), 0)
    effective_seed = base_seed + resolved_epoch
    bucket_sampler = bool(getattr(dataset, "bucket_manager", None)) and resolved_batch_size > 1
    caption_bucket = int(getattr(dataset, "caption_length_bucket_size", 0) or 0)
    sampler_order_live_equivalent = (not bucket_sampler) and (not shuffle or seed_provided)
    shadow_order_scope = "live_equivalent" if sampler_order_live_equivalent else "diagnostic_reference_only"

    return {
        "schema_version": 1,
        "probe": "turbocore_dataset_shadow_adapter_policy",
        "provider": "native_dataset_shadow_adapter_policy",
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
        "bucket_sampler_detected": bucket_sampler,
        "bucket_sampler_policy": "python_bucket_batch_sampler_only" if bucket_sampler else "flat_sampler_order",
        "caption_length_bucket_size": caption_bucket,
        "sampler_order_live_equivalent": sampler_order_live_equivalent,
        "shadow_order_scope": shadow_order_scope,
        "worker_count": resolved_workers,
        "worker_shard_policy": "main_process_sampler_order_only" if resolved_workers > 0 else "single_process_order",
        "worker_fetch_timing_equivalent": resolved_workers == 0,
        "native_shadow_supported": not bucket_sampler,
        "fallback_reason": "bucket_sampler_order_parity_not_ready" if bucket_sampler else "",
        "debug_only": True,
        "shadow_run": False,
        "training_path_enabled": False,
    }


def run_caption_dataset_dataloader_shadow_adapter(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    seed: int | None = None,
    epoch: int = 0,
) -> Dict[str, Any]:
    """Run a debug-only native shadow check for a CaptionDataset/DataLoader boundary."""

    policy = build_caption_dataset_shadow_adapter_policy(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        seed=seed,
        epoch=epoch,
    )
    if not bool(policy.get("native_shadow_supported", False)):
        return {
            **policy,
            "probe": "turbocore_dataset_shadow_adapter",
            "provider": "native_dataset_shadow_adapter",
            "ok": True,
            "skipped": True,
            "shadow_run": False,
        }
    try:
        with create_caption_dataset_shadow_session(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=int(policy.get("effective_seed", 0) or 0),
            prefetch_depth=max(int(num_workers or 0), 1) * 256,
            chunk_size=256,
        ) as session:
            lifecycle = session.run_lifecycle(worker_count=max(int(num_workers or 0), 1))
    except Exception as exc:
        return {
            **policy,
            "probe": "turbocore_dataset_shadow_adapter",
            "provider": "native_dataset_shadow_adapter",
            "ok": False,
            "skipped": False,
            "shadow_run": False,
            "fallback_reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        **policy,
        "probe": "turbocore_dataset_shadow_adapter",
        "provider": "native_dataset_shadow_adapter",
        "ok": bool(lifecycle.get("ok", False)),
        "skipped": False,
        "shadow_run": True,
        "lifecycle": lifecycle,
    }


def maybe_attach_caption_dataset_shadow_adapter(
    dataloader: Any,
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
) -> Any:
    """Attach debug shadow metadata to a DataLoader when explicitly enabled."""

    if not _truthy_env(ENABLE_ENV):
        return dataloader
    report = run_caption_dataset_dataloader_shadow_adapter(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        seed=_optional_env_int(SEED_ENV),
        epoch=_optional_env_int(EPOCH_ENV) or 0,
    )
    try:
        setattr(dataloader, "native_dataset_shadow_adapter", report)
    except Exception:
        pass
    try:
        setattr(dataset, "native_dataset_shadow_adapter", report)
    except Exception:
        pass
    return dataloader


__all__ = [
    "ENABLE_ENV",
    "SEED_ENV",
    "EPOCH_ENV",
    "build_caption_dataset_shadow_adapter_policy",
    "maybe_attach_caption_dataset_shadow_adapter",
    "run_caption_dataset_dataloader_shadow_adapter",
]
