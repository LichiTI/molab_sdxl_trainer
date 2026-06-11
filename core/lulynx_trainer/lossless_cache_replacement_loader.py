"""Experimental replacement-style LXCS cache loader.

This is a P3 probe that builds Newbie cache batches directly from LXCS
prefetch payloads instead of running a normal DataLoader beside a shadow
reader.  It is intentionally opt-in and diagnostic-only; production training
must keep the existing dataset path until real wall-clock and GPU-idle A/B
evidence clears the roadmap gates.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
import queue
import random
import threading
import time
from typing import Any

try:
    from .lossless_cache_prefetch_queue import (
        LosslessCachePrefetchPayload,
        LosslessCachePrefetchQueueConfig,
        load_lossless_cache_prefetch_payload,
        prepare_lossless_cache_prefetch_sidecars,
    )
    from .lossless_tensor_block import DEFAULT_FAST_CACHE_CODECS
    from .newbie_cached_dataset import NewbieCachedDataset, NewbieCachedSample, newbie_cached_collate
except ImportError:  # pragma: no cover - direct script smoke loading
    from lossless_cache_prefetch_queue import (
        LosslessCachePrefetchPayload,
        LosslessCachePrefetchQueueConfig,
        load_lossless_cache_prefetch_payload,
        prepare_lossless_cache_prefetch_sidecars,
    )
    from lossless_tensor_block import DEFAULT_FAST_CACHE_CODECS
    from newbie_cached_dataset import NewbieCachedDataset, NewbieCachedSample, newbie_cached_collate


@dataclass(frozen=True)
class LosslessCacheReplacementLoaderConfig:
    batch_size: int = 1
    max_batches: int = 4
    prefetch_depth: int = 2
    sidecar_dir: str | None = None
    sidecar_suffix: str = ".lxcs"
    sidecar_strict: bool = False
    fallback_to_raw: bool = True
    sidecar_format: str = "lxcs"
    verify_sidecar: bool = True
    copy_arrays: bool = True
    prepare_sidecars: bool = True
    max_entries: int = 0
    handoff: str = "none"
    chunk_size: int = 1 << 20
    min_saving: float = 0.02
    collate_mode: str = "auto"
    consumer_delay_ms: float = 0.0
    shuffle: bool = False
    drop_last: bool = False
    seed: int = 42
    focus_sample_ids: tuple[str, ...] = ()


def _now() -> float:
    return time.perf_counter()


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _round(value: float) -> float:
    return round(float(value), 4)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    items = sorted(float(item) for item in values)
    if len(items) == 1:
        return items[0]
    rank = (len(items) - 1) * min(max(float(q), 0.0), 1.0)
    low = int(rank)
    high = min(low + 1, len(items) - 1)
    frac = rank - low
    return items[low] * (1.0 - frac) + items[high] * frac


def _timings(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "total_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "count": len(values),
        "total_ms": _round(sum(values)),
        "p50_ms": _round(_percentile(values, 0.50)),
        "p95_ms": _round(_percentile(values, 0.95)),
        "max_ms": _round(max(values)),
    }


def _select_array(arrays: dict[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in arrays:
            return arrays[key]
    return None


def _caption_prefix(dataset: NewbieCachedDataset, keys: list[str], path: Path) -> str:
    selector = getattr(dataset, "_select_caption_source_variant_prefix", None)
    if not callable(selector):
        return ""
    try:
        return str(selector(keys, path) or "")
    except Exception:
        return ""


def _prefixed_key(keys: set[str], prefix: str, key: str) -> str | None:
    candidate = f"{prefix}{key}" if prefix else key
    return candidate if candidate in keys else None


def _build_newbie_item_from_payload(
    dataset: NewbieCachedDataset,
    sample: NewbieCachedSample,
    payload: LosslessCachePrefetchPayload,
) -> dict[str, object]:
    import numpy as np
    import torch

    arrays = payload.arrays
    keys = list(arrays.keys())
    key_set = set(keys)
    prefix = _caption_prefix(dataset, keys, payload.path)

    latents = _select_array(arrays, ("latents", "latent", "model_input"))
    hidden_key = _prefixed_key(key_set, prefix, "encoder_hidden_states")
    hidden = arrays[hidden_key] if hidden_key else _select_array(
        arrays,
        ("encoder_hidden_states", "gemma_hidden_states", "prompt_embeds"),
    )
    if latents is None or hidden is None:
        raise ValueError(f"Newbie LXCS payload missing required arrays: {payload.path}")

    latents_t = torch.from_numpy(np.asarray(latents)).float()
    if latents_t.dim() == 4:
        latents_t = latents_t[0]
    if dataset.latent_crop_size > 0:
        crop = int(dataset.latent_crop_size)
        latents_t = latents_t[:, :crop, :crop].contiguous()

    hidden_t = torch.from_numpy(np.asarray(hidden)).float()
    if hidden_t.dim() == 3:
        hidden_t = hidden_t[0]
    if dataset.text_token_limit > 0:
        hidden_t = hidden_t[: int(dataset.text_token_limit)]

    item: dict[str, object] = {
        "latents": latents_t,
        "encoder_hidden_states": hidden_t,
        "captions": sample.stem,
        "sample_id": sample.stem,
    }

    pooled_key = _prefixed_key(key_set, prefix, "pooled_prompt_embeds")
    pooled = arrays[pooled_key] if pooled_key else _select_array(
        arrays,
        ("pooled_prompt_embeds", "clip_pooled_features", "text_embeds"),
    )
    if pooled is not None:
        pooled_t = torch.from_numpy(np.asarray(pooled)).float()
        if pooled_t.dim() == 2:
            pooled_t = pooled_t[0]
        item["pooled_prompt_embeds"] = pooled_t

    mask_key = _prefixed_key(key_set, prefix, "attention_mask")
    mask = arrays[mask_key] if mask_key else _select_array(
        arrays,
        ("attention_mask", "gemma_attention_mask", "attn_mask", "mask"),
    )
    if mask is not None:
        mask_t = torch.from_numpy(np.asarray(mask)).bool()
        if mask_t.dim() == 2:
            mask_t = mask_t[0]
        if dataset.text_token_limit > 0:
            mask_t = mask_t[: int(dataset.text_token_limit)]
        item["attention_mask"] = mask_t

    loss_mask = _select_array(arrays, ("loss_mask", "alpha_mask", "padding_mask"))
    if loss_mask is not None:
        loss_mask_t = torch.from_numpy(np.asarray(loss_mask)).float()
        if loss_mask_t.dim() == 3:
            loss_mask_t = loss_mask_t[0]
        if dataset.latent_crop_size > 0:
            crop = int(dataset.latent_crop_size)
            loss_mask_t = loss_mask_t[:crop, :crop].contiguous()
        item["loss_mask"] = loss_mask_t
    return item


def _sample_batches(
    samples: Sequence[NewbieCachedSample],
    *,
    batch_size: int,
    max_batches: int,
    shuffle: bool = False,
    drop_last: bool = False,
    seed: int = 42,
) -> list[list[NewbieCachedSample]]:
    resolved_batch = max(int(batch_size), 1)
    limit = max(int(max_batches), 1)
    ordered = list(samples)
    if shuffle:
        random.Random(int(seed)).shuffle(ordered)
    batches: list[list[NewbieCachedSample]] = []
    for start in range(0, len(ordered), resolved_batch):
        chunk = ordered[start : start + resolved_batch]
        if not chunk:
            continue
        if drop_last and len(chunk) < resolved_batch:
            continue
        batches.append(chunk)
        if len(batches) >= limit:
            break
    return batches


def _producer(
    paths: Sequence[Path],
    out_queue: "queue.Queue[LosslessCachePrefetchPayload | None]",
    config: LosslessCachePrefetchQueueConfig,
    metrics: dict[str, Any],
) -> None:
    full_wait = 0.0
    for index, path in enumerate(paths):
        payload = load_lossless_cache_prefetch_payload(path, index=index, config=config)
        started = _now()
        out_queue.put(payload)
        full_wait += _elapsed_ms(started)
    out_queue.put(None)
    metrics["queue_full_stall_ms"] = _round(full_wait)


def run_newbie_lossless_cache_replacement_loader(
    dataset: NewbieCachedDataset,
    *,
    config: LosslessCacheReplacementLoaderConfig | None = None,
    codecs: Iterable[str] = DEFAULT_FAST_CACHE_CODECS,
) -> dict[str, Any]:
    cfg = config or LosslessCacheReplacementLoaderConfig()
    batches = _sample_batches(
        list(dataset.samples),
        batch_size=max(int(cfg.batch_size), 1),
        max_batches=max(int(cfg.max_batches), 1),
        shuffle=bool(cfg.shuffle),
        drop_last=bool(cfg.drop_last),
        seed=int(cfg.seed),
    )
    flat_samples = [sample for batch in batches for sample in batch]
    flat_paths = [sample.cache_path for sample in flat_samples]

    prefetch_config = LosslessCachePrefetchQueueConfig(
        prefetch_depth=max(int(cfg.prefetch_depth), 1),
        sidecar_enabled=True,
        sidecar_strict=bool(cfg.sidecar_strict),
        fallback_to_raw=bool(cfg.fallback_to_raw),
        sidecar_format=str(cfg.sidecar_format or "lxcs"),
        sidecar_suffix=str(cfg.sidecar_suffix or ".lxcs"),
        sidecar_dir=cfg.sidecar_dir,
        verify_sidecar=bool(cfg.verify_sidecar),
        copy_arrays=bool(cfg.copy_arrays),
        max_entries=max(int(cfg.max_entries), 0),
        handoff=str(cfg.handoff or "none"),
    )
    if cfg.prepare_sidecars and flat_paths:
        prepare_report = prepare_lossless_cache_prefetch_sidecars(
            flat_paths,
            config=prefetch_config,
            chunk_size=max(int(cfg.chunk_size), 1),
            codecs=codecs,
            min_saving=float(cfg.min_saving),
        )
    else:
        prepare_report = {"ok": True, "skipped": True}

    out_queue: queue.Queue[LosslessCachePrefetchPayload | None] = queue.Queue(
        maxsize=max(int(cfg.prefetch_depth), 1)
    )
    worker_metrics: dict[str, Any] = {}
    worker = threading.Thread(
        target=_producer,
        args=(flat_paths, out_queue, prefetch_config, worker_metrics),
        name="lxcs-newbie-replacement-prefetch",
        daemon=True,
    )
    worker.start()

    payload_by_index: dict[int, LosslessCachePrefetchPayload] = {}
    decode_ms: list[float] = []
    handoff_ms: list[float] = []
    wait_ms: list[float] = []
    build_ms: list[float] = []
    collate_ms: list[float] = []
    batch_sizes: list[int] = []
    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    fallback_count = 0
    consumed = 0
    started = _now()

    for batch_index, batch_samples in enumerate(batches):
        items: list[dict[str, object]] = []
        for sample in batch_samples:
            wait_started = _now()
            payload = out_queue.get()
            wait_ms.append(_elapsed_ms(wait_started))
            if payload is None:
                errors.append(f"prefetch_finished_early_at_batch_{batch_index}")
                break
            consumed += 1
            payload_by_index[payload.index] = payload
            report = dict(payload.report)
            cases.append({key: value for key, value in report.items() if key != "entries"})
            decode_ms.append(float(report.get("decode_ms") or 0.0))
            handoff_ms.append(float(report.get("handoff_ms") or 0.0))
            if report.get("fallback_to_raw_cache"):
                fallback_count += 1
            if payload.error:
                errors.append(payload.error)
                continue
            build_started = _now()
            try:
                items.append(_build_newbie_item_from_payload(dataset, sample, payload))
            except Exception as exc:
                errors.append(f"item_build:{type(exc).__name__}: {exc}")
            build_ms.append(_elapsed_ms(build_started))
        if errors or not items:
            break
        collate_started = _now()
        try:
            batch = newbie_cached_collate(items, collate_mode=str(cfg.collate_mode or "auto"))
        except Exception as exc:
            errors.append(f"collate:{type(exc).__name__}: {exc}")
            break
        collate_ms.append(_elapsed_ms(collate_started))
        latents = batch.get("latents")
        batch_sizes.append(int(latents.shape[0]) if hasattr(latents, "shape") else len(items))

    worker.join(timeout=5.0)
    if worker.is_alive():
        errors.append("prefetch_worker_did_not_finish")

    return {
        "provider": "lxcs_newbie_replacement_loader_v1",
        "ok": not errors and consumed == len(flat_paths),
        "dataset_class": type(dataset).__name__,
        "sample_count": len(dataset.samples),
        "batch_size": max(int(cfg.batch_size), 1),
        "max_batches": max(int(cfg.max_batches), 1),
        "batch_count": len(batch_sizes),
        "batch_sizes": batch_sizes,
        "path_count": len(flat_paths),
        "consumed_path_count": consumed,
        "payload_index_count": len(payload_by_index),
        "fallback_count": fallback_count,
        "error_count": len(errors),
        "errors": errors,
        "prepare": prepare_report,
        "wall_ms": _round(_elapsed_ms(started)),
        "queue_empty_wait": _timings(wait_ms),
        "queue_full_stall_ms": worker_metrics.get("queue_full_stall_ms", 0.0),
        "decode": _timings(decode_ms),
        "handoff": _timings(handoff_ms),
        "item_build": _timings(build_ms),
        "collate": _timings(collate_ms),
        "cases": cases,
        "experimental_replacement_path": True,
        "real_dataloader_replacement": False,
        "duplicate_dataset_read": False,
        "reads_tensor_payloads": True,
        "training_path_enabled": False,
        "p3_replacement_loader_ready": not errors and consumed == len(flat_paths),
        "p3_prefetch_h2d_ready": False,
        "readiness_blockers": [
            "experimental_newbie_sequential_only",
            "not_integrated_with_real_training_dataloader",
            "no optimizer_forward_backward_wall_clock_ab",
            "gpu_idle_and_step_p95_not_verified",
        ],
    }


def iter_newbie_lossless_cache_replacement_batches(
    dataset: NewbieCachedDataset,
    *,
    config: LosslessCacheReplacementLoaderConfig | None = None,
    codecs: Iterable[str] = DEFAULT_FAST_CACHE_CODECS,
) -> Iterator[tuple[dict[str, object], dict[str, Any]]]:
    """Yield Newbie batches built directly from LXCS payloads.

    The background producer keeps decoding ahead while the caller performs
    compute between ``next()`` calls.  This is still an experimental probe API:
    it is sequential-only, Newbie-only, and not wired into the production
    DataLoader/runtime path.
    """

    cfg = config or LosslessCacheReplacementLoaderConfig()
    batches = _sample_batches(
        list(dataset.samples),
        batch_size=max(int(cfg.batch_size), 1),
        max_batches=max(int(cfg.max_batches), 1),
        shuffle=bool(cfg.shuffle),
        drop_last=bool(cfg.drop_last),
        seed=int(cfg.seed),
    )
    flat_paths = [sample.cache_path for batch in batches for sample in batch]
    prefetch_config = LosslessCachePrefetchQueueConfig(
        prefetch_depth=max(int(cfg.prefetch_depth), 1),
        sidecar_enabled=True,
        sidecar_strict=bool(cfg.sidecar_strict),
        fallback_to_raw=bool(cfg.fallback_to_raw),
        sidecar_format=str(cfg.sidecar_format or "lxcs"),
        sidecar_suffix=str(cfg.sidecar_suffix or ".lxcs"),
        sidecar_dir=cfg.sidecar_dir,
        verify_sidecar=bool(cfg.verify_sidecar),
        copy_arrays=bool(cfg.copy_arrays),
        max_entries=max(int(cfg.max_entries), 0),
        handoff=str(cfg.handoff or "none"),
    )
    if cfg.prepare_sidecars and flat_paths:
        prepare_report = prepare_lossless_cache_prefetch_sidecars(
            flat_paths,
            config=prefetch_config,
            chunk_size=max(int(cfg.chunk_size), 1),
            codecs=codecs,
            min_saving=float(cfg.min_saving),
        )
    else:
        prepare_report = {"ok": True, "skipped": True}

    out_queue: queue.Queue[LosslessCachePrefetchPayload | None] = queue.Queue(
        maxsize=max(int(cfg.prefetch_depth), 1)
    )
    worker_metrics: dict[str, Any] = {}
    worker = threading.Thread(
        target=_producer,
        args=(flat_paths, out_queue, prefetch_config, worker_metrics),
        name="lxcs-newbie-replacement-iterator",
        daemon=True,
    )
    worker.start()
    try:
        for batch_index, batch_samples in enumerate(batches):
            items: list[dict[str, object]] = []
            wait_ms: list[float] = []
            build_ms: list[float] = []
            decode_ms: list[float] = []
            handoff_ms: list[float] = []
            cases: list[dict[str, Any]] = []
            fallback_count = 0
            for sample in batch_samples:
                wait_started = _now()
                payload = out_queue.get()
                wait_ms.append(_elapsed_ms(wait_started))
                if payload is None:
                    raise RuntimeError(f"LXCS prefetch finished early at batch {batch_index}")
                report = dict(payload.report)
                cases.append({key: value for key, value in report.items() if key != "entries"})
                decode_ms.append(float(report.get("decode_ms") or 0.0))
                handoff_ms.append(float(report.get("handoff_ms") or 0.0))
                if report.get("fallback_to_raw_cache"):
                    fallback_count += 1
                if payload.error:
                    raise RuntimeError(payload.error)
                build_started = _now()
                items.append(_build_newbie_item_from_payload(dataset, sample, payload))
                build_ms.append(_elapsed_ms(build_started))
            collate_started = _now()
            batch = newbie_cached_collate(items, collate_mode=str(cfg.collate_mode or "auto"))
            collate_ms = _elapsed_ms(collate_started)
            yield batch, {
                "batch_index": batch_index,
                "sample_count": len(items),
                "sample_ids": [str(sample.stem) for sample in batch_samples],
                "prepare": prepare_report if batch_index == 0 else {"ok": True, "skipped": True},
                "queue_empty_wait": _timings(wait_ms),
                "decode": _timings(decode_ms),
                "handoff": _timings(handoff_ms),
                "item_build": _timings(build_ms),
                "collate_ms": _round(collate_ms),
                "fallback_count": fallback_count,
                "cases": cases,
                "experimental_replacement_path": True,
                "training_path_enabled": False,
            }
    finally:
        worker.join(timeout=5.0)


__all__ = [
    "LosslessCacheReplacementLoaderConfig",
    "iter_newbie_lossless_cache_replacement_batches",
    "run_newbie_lossless_cache_replacement_loader",
]
