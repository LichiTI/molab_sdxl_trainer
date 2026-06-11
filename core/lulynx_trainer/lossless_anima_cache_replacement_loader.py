"""Experimental Anima LXCS replacement-style cache loader.

This is a P3 probe only. It builds Anima cached batches from LXCS latent/text
payloads and compares them with the existing cached DataLoader in devtools.
It must not be used as a production training path until full trainer A/B gates
pass.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
import queue
import random
import threading
import time
from typing import Any
import warnings

try:
    from .anima_cached_dataset import (
        AnimaCachedDataset,
        AnimaCachedSample,
        _pad_latents_to_visual_tokens,
        anima_cached_collate,
    )
    from .lossless_cache_prefetch_queue import (
        LosslessCachePrefetchPayload,
        LosslessCachePrefetchQueueConfig,
        load_lossless_cache_prefetch_payload,
        prepare_lossless_cache_prefetch_sidecars,
    )
    from .lossless_cache_focus import focus_sample_sequence, sample_id_of
    from .lossless_cache_replacement_loader import LosslessCacheReplacementLoaderConfig
    from .lossless_tensor_block import DEFAULT_FAST_CACHE_CODECS
    from .lossless_tensor_layout_metadata import batch_tensor_layouts, mapping_tensor_layouts
except ImportError:  # pragma: no cover - direct script smoke loading
    from anima_cached_dataset import (  # type: ignore[no-redef]
        AnimaCachedDataset,
        AnimaCachedSample,
        _pad_latents_to_visual_tokens,
        anima_cached_collate,
    )
    from lossless_cache_prefetch_queue import (  # type: ignore[no-redef]
        LosslessCachePrefetchPayload,
        LosslessCachePrefetchQueueConfig,
        load_lossless_cache_prefetch_payload,
        prepare_lossless_cache_prefetch_sidecars,
    )
    from lossless_cache_focus import focus_sample_sequence, sample_id_of  # type: ignore[no-redef]
    from lossless_cache_replacement_loader import LosslessCacheReplacementLoaderConfig  # type: ignore[no-redef]
    from lossless_tensor_block import DEFAULT_FAST_CACHE_CODECS  # type: ignore[no-redef]
    from lossless_tensor_layout_metadata import batch_tensor_layouts, mapping_tensor_layouts  # type: ignore[no-redef]


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


def _sample_batches(
    samples: Sequence[AnimaCachedSample],
    *,
    batch_size: int,
    max_batches: int,
    shuffle: bool = False,
    drop_last: bool = False,
    seed: int = 42,
    focus_sample_ids: Iterable[Any] = (),
) -> tuple[list[list[AnimaCachedSample]], dict[str, Any]]:
    resolved_batch = max(int(batch_size), 1)
    limit = max(int(max_batches), 1)
    ordered = list(samples)
    if shuffle:
        random.Random(int(seed)).shuffle(ordered)
    ordered, focus_report = focus_sample_sequence(ordered, focus_sample_ids)
    batches: list[list[AnimaCachedSample]] = []
    for start in range(0, len(ordered), resolved_batch):
        chunk = ordered[start : start + resolved_batch]
        if not chunk:
            continue
        if drop_last and len(chunk) < resolved_batch:
            continue
        batches.append(chunk)
        if len(batches) >= limit:
            break
    focus_report = dict(focus_report)
    focus_report["first_sample_ids"] = [sample_id_of(sample) for sample in ordered[: min(len(ordered), 12)]]
    focus_report["first_batch_sample_ids"] = [
        sample_id_of(sample) for sample in (batches[0] if batches else [])
    ]
    return batches, focus_report


def _producer(
    paths: Sequence[Path],
    out_queue: "queue.Queue[LosslessCachePrefetchPayload | None]",
    config: LosslessCachePrefetchQueueConfig,
    metrics: dict[str, Any],
    stop_event: threading.Event | None = None,
) -> None:
    full_wait = 0.0
    for index, path in enumerate(paths):
        if stop_event is not None and stop_event.is_set():
            metrics["stopped_early"] = True
            metrics["queue_full_stall_ms"] = _round(full_wait)
            return
        payload = load_lossless_cache_prefetch_payload(path, index=index, config=config)
        started = _now()
        while True:
            if stop_event is not None and stop_event.is_set():
                metrics["stopped_early"] = True
                metrics["queue_full_stall_ms"] = _round(full_wait + _elapsed_ms(started))
                return
            try:
                out_queue.put(payload, timeout=0.05)
                break
            except queue.Full:
                continue
        full_wait += _elapsed_ms(started)
    if stop_event is None or not stop_event.is_set():
        out_queue.put(None)
    metrics["queue_full_stall_ms"] = _round(full_wait)


def _select_array(arrays: dict[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in arrays:
            return arrays[key]
    return None


def _latent_array(arrays: dict[str, Any]) -> Any:
    latent_keys = sorted(key for key in arrays if key.startswith("latents_"))
    if latent_keys:
        return arrays[latent_keys[0]]
    value = _select_array(arrays, ("latents", "latent", "model_input"))
    if value is None:
        raise ValueError("Anima LXCS latent payload missing latents_* array")
    return value


def _tensor_from_array(array: Any, *, dtype: str = "float"):
    import numpy as np
    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable.*",
            category=UserWarning,
        )
        tensor = torch.from_numpy(np.asarray(array))
    if dtype == "bool":
        return tensor.bool()
    if dtype == "long":
        return tensor.long()
    return tensor.float()


def _trim_tokens(tensor: Any, limit: int) -> Any:
    if max(int(limit), 0) <= 0:
        return tensor
    return tensor[: int(limit)]


def _build_anima_item_from_payloads(
    dataset: AnimaCachedDataset,
    sample: AnimaCachedSample,
    latent_payload: LosslessCachePrefetchPayload,
    text_payload: LosslessCachePrefetchPayload,
) -> dict[str, object]:
    latents = _tensor_from_array(_latent_array(latent_payload.arrays)).float()
    if latents.dim() == 4 and int(latents.shape[0]) == 1:
        latents = latents[0]
    if dataset.latent_crop_size > 0:
        crop = int(dataset.latent_crop_size)
        latents = latents[:, :crop, :crop].contiguous()
    if dataset.fixed_visual_tokens > 0:
        latents = _pad_latents_to_visual_tokens(latents, int(dataset.fixed_visual_tokens))

    prompt = _select_array(text_payload.arrays, ("prompt_embeds", "encoder_hidden_states"))
    if prompt is None:
        raise ValueError("Anima LXCS text payload missing prompt_embeds")
    hidden = _tensor_from_array(prompt).float()
    if hidden.dim() == 3 and int(hidden.shape[0]) == 1:
        hidden = hidden[0]
    hidden = _trim_tokens(hidden, dataset.text_token_limit)

    attention = _select_array(text_payload.arrays, ("attention_mask", "prompt_attention_mask"))
    if attention is None:
        import torch

        attention_mask = torch.ones((int(hidden.shape[0]),), dtype=torch.bool)
    else:
        attention_mask = _trim_tokens(_tensor_from_array(attention, dtype="bool"), dataset.text_token_limit)
        if attention_mask.dim() == 2 and int(attention_mask.shape[0]) == 1:
            attention_mask = attention_mask[0]

    item: dict[str, object] = {
        "latents": latents,
        "encoder_hidden_states": hidden,
        "attention_mask": attention_mask,
        "captions": sample.stem,
        "caption_weight": 1.0,
        "sample_id": sample.sample_id,
    }

    optional_specs = {
        "t5_input_ids": ("t5_input_ids", "input_ids"),
        "t5_attention_mask": ("t5_attention_mask", "t5_attn_mask"),
        "qwen3_hidden_states": ("qwen3_hidden_states",),
        "qwen3_attention_mask": ("qwen3_attention_mask",),
    }
    for out_key, keys in optional_specs.items():
        value = _select_array(text_payload.arrays, keys)
        if value is None:
            continue
        dtype = "long" if out_key.endswith("input_ids") else "bool" if out_key.endswith("attention_mask") else "float"
        tensor = _tensor_from_array(value, dtype=dtype)
        if tensor.dim() >= 2 and int(tensor.shape[0]) == 1:
            tensor = tensor[0]
        if out_key in {"qwen3_hidden_states", "qwen3_attention_mask"}:
            tensor = _trim_tokens(tensor, dataset.fixed_qwen3_tokens)
        elif out_key in {"t5_input_ids", "t5_attention_mask"}:
            tensor = _trim_tokens(tensor, dataset.fixed_t5_tokens)
        item[out_key] = tensor

    loss_mask = _select_array(latent_payload.arrays, ("loss_mask", "alpha_mask", "padding_mask"))
    if loss_mask is not None:
        mask_t = _tensor_from_array(loss_mask).float()
        if mask_t.dim() == 3 and int(mask_t.shape[0]) == 1:
            mask_t = mask_t[0]
        if dataset.latent_crop_size > 0:
            crop = int(dataset.latent_crop_size)
            mask_t = mask_t[:crop, :crop].contiguous()
        item["loss_mask"] = mask_t
    return item


def _prefetch_config(cfg: LosslessCacheReplacementLoaderConfig) -> LosslessCachePrefetchQueueConfig:
    return LosslessCachePrefetchQueueConfig(
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


def run_anima_lossless_cache_replacement_loader(
    dataset: AnimaCachedDataset,
    *,
    config: LosslessCacheReplacementLoaderConfig | None = None,
    codecs: Iterable[str] = DEFAULT_FAST_CACHE_CODECS,
) -> dict[str, Any]:
    cfg = config or LosslessCacheReplacementLoaderConfig()
    batches, focus_report = _sample_batches(
        list(dataset.samples),
        batch_size=max(int(cfg.batch_size), 1),
        max_batches=max(int(cfg.max_batches), 1),
        shuffle=bool(cfg.shuffle),
        drop_last=bool(cfg.drop_last),
        seed=int(cfg.seed),
        focus_sample_ids=cfg.focus_sample_ids,
    )
    flat_samples = [sample for batch in batches for sample in batch]
    flat_paths = [path for sample in flat_samples for path in (sample.latent_path, sample.text_path)]
    prefetch_config = _prefetch_config(cfg)

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
        name="lxcs-anima-replacement-prefetch",
        daemon=True,
    )
    worker.start()

    wait_ms: list[float] = []
    decode_ms: list[float] = []
    handoff_ms: list[float] = []
    build_ms: list[float] = []
    collate_ms: list[float] = []
    batch_sizes: list[int] = []
    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    fallback_count = 0
    consumed = 0
    started = _now()

    def next_payload(kind: str) -> LosslessCachePrefetchPayload | None:
        nonlocal consumed, fallback_count
        wait_started = _now()
        payload = out_queue.get()
        wait_ms.append(_elapsed_ms(wait_started))
        if payload is None:
            errors.append(f"prefetch_finished_early_before_{kind}")
            return None
        consumed += 1
        report = dict(payload.report)
        cases.append({key: value for key, value in report.items() if key != "entries"})
        decode_ms.append(float(report.get("decode_ms") or 0.0))
        handoff_ms.append(float(report.get("handoff_ms") or 0.0))
        if report.get("fallback_to_raw_cache"):
            fallback_count += 1
        if payload.error:
            errors.append(payload.error)
        return payload

    for batch_index, batch_samples in enumerate(batches):
        items: list[dict[str, object]] = []
        for sample in batch_samples:
            latent_payload = next_payload("latent")
            text_payload = next_payload("text")
            if latent_payload is None or text_payload is None:
                break
            build_started = _now()
            try:
                items.append(_build_anima_item_from_payloads(dataset, sample, latent_payload, text_payload))
            except Exception as exc:
                errors.append(f"item_build:{type(exc).__name__}: {exc}")
            build_ms.append(_elapsed_ms(build_started))
        if errors or not items:
            break
        collate_started = _now()
        try:
            batch = anima_cached_collate(
                items,
                fixed_text_tokens=int(dataset.fixed_text_tokens),
                collate_mode=str(cfg.collate_mode or "auto"),
            )
        except Exception as exc:
            errors.append(f"collate:{type(exc).__name__}: {exc}")
            break
        collate_ms.append(_elapsed_ms(collate_started))
        latents = batch.get("latents")
        batch_sizes.append(int(latents.shape[0]) if hasattr(latents, "shape") else len(items))
        if float(cfg.consumer_delay_ms or 0.0) > 0.0:
            time.sleep(float(cfg.consumer_delay_ms) / 1000.0)

    worker.join(timeout=5.0)
    if worker.is_alive():
        errors.append("prefetch_worker_did_not_finish")

    return {
        "provider": "lxcs_anima_replacement_loader_v1",
        "sidecar_format": str(prefetch_config.sidecar_format or "lxcs").lower(),
        "ok": not errors and consumed == len(flat_paths),
        "dataset_class": type(dataset).__name__,
        "sample_count": len(dataset.samples),
        "batch_size": max(int(cfg.batch_size), 1),
        "max_batches": max(int(cfg.max_batches), 1),
        "batch_count": len(batch_sizes),
        "batch_sizes": batch_sizes,
        "focus_sample_ids": list(cfg.focus_sample_ids),
        "focus_sample_report": focus_report,
        "path_count": len(flat_paths),
        "consumed_path_count": consumed,
        "fallback_count": fallback_count,
        "consumer_delay_ms": _round(float(cfg.consumer_delay_ms or 0.0)),
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
        "p3_anima_replacement_loader_ready": not errors and consumed == len(flat_paths),
        "readiness_blockers": [
            "anima_replacement_loader_probe_only",
            "no_optimizer_step_ab",
            "not_wired_into_trainer_runtime_request",
        ],
    }


def iter_anima_lossless_cache_replacement_batches(
    dataset: AnimaCachedDataset,
    *,
    config: LosslessCacheReplacementLoaderConfig | None = None,
    codecs: Iterable[str] = DEFAULT_FAST_CACHE_CODECS,
) -> Iterator[tuple[dict[str, object], dict[str, Any]]]:
    """Yield Anima batches built directly from LXCS/LXFS payloads.

    This is the batch iterator form of ``run_anima_lossless_cache_replacement_loader``.
    It exists for mini-step A/B probes where the caller performs real torch
    compute between batches, letting the background producer overlap the next
    cache decode. It is still diagnostic-only and not a production DataLoader.
    """

    cfg = config or LosslessCacheReplacementLoaderConfig()
    batches, focus_report = _sample_batches(
        list(dataset.samples),
        batch_size=max(int(cfg.batch_size), 1),
        max_batches=max(int(cfg.max_batches), 1),
        shuffle=bool(cfg.shuffle),
        drop_last=bool(cfg.drop_last),
        seed=int(cfg.seed),
        focus_sample_ids=cfg.focus_sample_ids,
    )
    flat_samples = [sample for batch in batches for sample in batch]
    flat_paths = [path for sample in flat_samples for path in (sample.latent_path, sample.text_path)]
    prefetch_config = _prefetch_config(cfg)

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
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_producer,
        args=(flat_paths, out_queue, prefetch_config, worker_metrics, stop_event),
        name="lxcs-anima-replacement-iterator",
        daemon=True,
    )
    worker.start()
    focus_sample_id_set = {str(value) for value in cfg.focus_sample_ids if str(value)}

    def next_payload(kind: str, batch_index: int, reports: dict[str, Any]) -> LosslessCachePrefetchPayload:
        wait_started = _now()
        payload = out_queue.get()
        reports["wait_ms"].append(_elapsed_ms(wait_started))
        if payload is None:
            raise RuntimeError(f"LXCS prefetch finished early before {kind} at batch {batch_index}")
        report = dict(payload.report)
        reports["cases"].append({key: value for key, value in report.items() if key != "entries"})
        reports["decode_ms"].append(float(report.get("decode_ms") or 0.0))
        reports["handoff_ms"].append(float(report.get("handoff_ms") or 0.0))
        if report.get("fallback_to_raw_cache"):
            reports["fallback_count"] += 1
        if payload.error:
            raise RuntimeError(payload.error)
        return payload

    try:
        for batch_index, batch_samples in enumerate(batches):
            items: list[dict[str, object]] = []
            reports: dict[str, Any] = {
                "wait_ms": [],
                "decode_ms": [],
                "handoff_ms": [],
                "build_ms": [],
                "cases": [],
                "target_tensor_layouts": [],
                "fallback_count": 0,
            }
            for sample in batch_samples:
                latent_payload = next_payload("latent", batch_index, reports)
                text_payload = next_payload("text", batch_index, reports)
                build_started = _now()
                item = _build_anima_item_from_payloads(dataset, sample, latent_payload, text_payload)
                if not focus_sample_id_set or str(sample.sample_id) in focus_sample_id_set:
                    reports["target_tensor_layouts"].extend(
                        mapping_tensor_layouts(
                            item,
                            sample_id=str(sample.sample_id),
                            payload_source="lxfs_sidecar",
                            copy_path="numpy_to_torch_from_replacement_payload",
                            array_source="replacement_payload_arrays",
                            cache_file=f"{sample.latent_path.name}|{sample.text_path.name}",
                        )
                    )
                items.append(item)
                reports["build_ms"].append(_elapsed_ms(build_started))

            collate_started = _now()
            batch = anima_cached_collate(
                items,
                fixed_text_tokens=int(dataset.fixed_text_tokens),
                collate_mode=str(cfg.collate_mode or "auto"),
            )
            collate_ms = _elapsed_ms(collate_started)
            yield batch, {
                "batch_index": batch_index,
                "sample_count": len(items),
                "sample_ids": [str(sample.sample_id) for sample in batch_samples],
                "focus_sample_ids": list(cfg.focus_sample_ids),
                "focus_sample_report": focus_report,
                "prepare": prepare_report if batch_index == 0 else {"ok": True, "skipped": True},
                "queue_empty_wait": _timings(reports["wait_ms"]),
                "decode": _timings(reports["decode_ms"]),
                "handoff": _timings(reports["handoff_ms"]),
                "item_build": _timings(reports["build_ms"]),
                "collate_ms": _round(collate_ms),
                "fallback_count": int(reports["fallback_count"]),
                "queue_full_stall_ms": worker_metrics.get("queue_full_stall_ms", 0.0),
                "cases": reports["cases"],
                "target_tensor_layouts": reports["target_tensor_layouts"],
                "batch_tensor_layouts": batch_tensor_layouts(
                    batch,
                    payload_source="lxfs_sidecar",
                    copy_path="collated_replacement_batch",
                ),
                "experimental_replacement_path": True,
                "training_path_enabled": False,
            }
    finally:
        stop_event.set()
        worker.join(timeout=1.0)


__all__ = [
    "iter_anima_lossless_cache_replacement_batches",
    "run_anima_lossless_cache_replacement_loader",
]
