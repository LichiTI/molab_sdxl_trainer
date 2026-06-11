"""Shadow DataLoader probe for the experimental LXCS prefetch path.

This is the next P3 step after the standalone queue probe: run a real
Dataset/DataLoader while a background LXCS prefetch queue follows the same
sequential sample order.  The probe reports dataloader wait, prefetch wait,
queue stalls, decode/handoff timing, and fallback/error counts.  It remains
diagnostic-only and does not replace dataset reads.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import queue
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
except ImportError:  # pragma: no cover - direct script smoke loading
    from lossless_cache_prefetch_queue import (
        LosslessCachePrefetchPayload,
        LosslessCachePrefetchQueueConfig,
        load_lossless_cache_prefetch_payload,
        prepare_lossless_cache_prefetch_sidecars,
    )
    from lossless_tensor_block import DEFAULT_FAST_CACHE_CODECS


DataLoaderFactory = Callable[[Any], Iterable[Any]]


@dataclass(frozen=True)
class LosslessCacheDataloaderShadowConfig:
    batch_size: int = 1
    max_batches: int = 4
    prefetch_depth: int = 2
    sidecar_dir: str | None = None
    sidecar_suffix: str = ".lxcs"
    prepare_sidecars: bool = True
    max_entries: int = 0
    handoff: str = "none"
    chunk_size: int = 1 << 20
    min_saving: float = 0.02


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


def cache_paths_for_sample(sample: Any) -> list[Path]:
    paths: list[Path] = []
    for name in ("cache_path", "latent_path", "text_path", "loss_mask_path"):
        value = getattr(sample, name, None)
        if not value:
            continue
        path = Path(value)
        if path.suffix.lower() in {".npz", ".npy"}:
            paths.append(path)
    return paths


def build_sequential_cache_path_batches(
    dataset: Any,
    *,
    batch_size: int,
    max_batches: int,
    drop_last: bool = False,
) -> list[list[Path]]:
    samples = list(getattr(dataset, "samples", []) or [])
    resolved_batch = max(int(batch_size), 1)
    limit_batches = max(int(max_batches), 1)
    batches: list[list[Path]] = []
    for start in range(0, len(samples), resolved_batch):
        chunk = samples[start : start + resolved_batch]
        if drop_last and len(chunk) < resolved_batch:
            break
        paths: list[Path] = []
        for sample in chunk:
            paths.extend(cache_paths_for_sample(sample))
        if paths:
            batches.append(paths)
        if len(batches) >= limit_batches:
            break
    return batches


def _batch_item_count(batch: Any) -> int:
    if isinstance(batch, dict):
        for key in ("sample_ids", "captions"):
            value = batch.get(key)
            if isinstance(value, list):
                return len(value)
        value = batch.get("latents")
        if hasattr(value, "shape") and len(value.shape) > 0:
            return int(value.shape[0])
    return 0


def _iterate_dataloader(dataloader: Iterable[Any], *, max_batches: int) -> dict[str, Any]:
    batch_wait_ms: list[float] = []
    batch_sizes: list[int] = []
    errors: list[str] = []
    started = _now()
    iterator = iter(dataloader)
    for _ in range(max(int(max_batches), 1)):
        item_started = _now()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            break
        batch_wait_ms.append(_elapsed_ms(item_started))
        batch_sizes.append(_batch_item_count(batch))
    return {
        "ok": not errors,
        "batch_count": len(batch_wait_ms),
        "batch_sizes": batch_sizes,
        "wall_ms": _round(_elapsed_ms(started)),
        "dataloader_wait": _timings(batch_wait_ms),
        "errors": errors,
    }


def _prefetch_worker(
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


def _consume_prefetch(
    out_queue: "queue.Queue[LosslessCachePrefetchPayload | None]",
    *,
    count: int,
) -> tuple[list[LosslessCachePrefetchPayload], float, bool]:
    payloads: list[LosslessCachePrefetchPayload] = []
    empty_wait = 0.0
    saw_done = False
    for _ in range(max(int(count), 0)):
        started = _now()
        payload = out_queue.get()
        empty_wait += _elapsed_ms(started)
        if payload is None:
            saw_done = True
            break
        payloads.append(payload)
    return payloads, empty_wait, saw_done


def run_lossless_cache_dataloader_shadow(
    dataset_factory: Callable[[], Any],
    dataloader_factory: DataLoaderFactory,
    *,
    config: LosslessCacheDataloaderShadowConfig | None = None,
    codecs: Iterable[str] = DEFAULT_FAST_CACHE_CODECS,
) -> dict[str, Any]:
    cfg = config or LosslessCacheDataloaderShadowConfig()
    baseline_dataset = dataset_factory()
    path_batches = build_sequential_cache_path_batches(
        baseline_dataset,
        batch_size=max(int(cfg.batch_size), 1),
        max_batches=max(int(cfg.max_batches), 1),
        drop_last=False,
    )
    flat_paths = [path for batch in path_batches for path in batch]
    if cfg.prepare_sidecars and flat_paths:
        prepare_report = prepare_lossless_cache_prefetch_sidecars(
            flat_paths,
            config=LosslessCachePrefetchQueueConfig(
                prefetch_depth=max(int(cfg.prefetch_depth), 1),
                sidecar_dir=cfg.sidecar_dir,
                sidecar_suffix=cfg.sidecar_suffix,
                max_entries=max(int(cfg.max_entries), 0),
                handoff=cfg.handoff,
            ),
            chunk_size=max(int(cfg.chunk_size), 1),
            codecs=codecs,
            min_saving=float(cfg.min_saving),
        )
    else:
        prepare_report = {"ok": True, "skipped": True}

    baseline = _iterate_dataloader(
        dataloader_factory(baseline_dataset),
        max_batches=max(int(cfg.max_batches), 1),
    )

    shadow_dataset = dataset_factory()
    shadow_loader = dataloader_factory(shadow_dataset)
    prefetch_config = LosslessCachePrefetchQueueConfig(
        prefetch_depth=max(int(cfg.prefetch_depth), 1),
        sidecar_dir=cfg.sidecar_dir,
        sidecar_suffix=cfg.sidecar_suffix,
        max_entries=max(int(cfg.max_entries), 0),
        handoff=cfg.handoff,
    )
    out_queue: queue.Queue[LosslessCachePrefetchPayload | None] = queue.Queue(maxsize=max(int(cfg.prefetch_depth), 1))
    worker_metrics: dict[str, Any] = {}
    worker = threading.Thread(
        target=_prefetch_worker,
        args=(flat_paths, out_queue, prefetch_config, worker_metrics),
        name="lxcs-dataloader-shadow-prefetch",
        daemon=True,
    )
    worker.start()

    batch_wait_ms: list[float] = []
    prefetch_wait_ms: list[float] = []
    decode_ms: list[float] = []
    handoff_ms: list[float] = []
    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    fallback_count = 0
    consumed_paths = 0
    started = _now()
    iterator = iter(shadow_loader)
    for batch_index, expected_paths in enumerate(path_batches):
        item_started = _now()
        try:
            next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            errors.append(f"dataloader:{type(exc).__name__}: {exc}")
            break
        batch_wait_ms.append(_elapsed_ms(item_started))
        payloads, wait_ms, saw_done = _consume_prefetch(out_queue, count=len(expected_paths))
        prefetch_wait_ms.append(wait_ms)
        if saw_done and len(payloads) < len(expected_paths):
            errors.append(f"prefetch_finished_early_at_batch_{batch_index}")
            break
        for payload in payloads:
            consumed_paths += 1
            report = dict(payload.report)
            cases.append({key: value for key, value in report.items() if key != "entries"})
            decode_ms.append(float(report.get("decode_ms") or 0.0))
            handoff_ms.append(float(report.get("handoff_ms") or 0.0))
            if report.get("fallback_to_raw_cache"):
                fallback_count += 1
            if payload.error:
                errors.append(payload.error)
    worker.join(timeout=5.0)

    shadow_wall = _elapsed_ms(started)
    baseline_wall = float(baseline.get("wall_ms") or 0.0)
    return {
        "provider": "lxcs_dataloader_shadow_v1",
        "ok": not errors and consumed_paths == len(flat_paths) and bool(baseline.get("ok", False)),
        "dataset_class": type(baseline_dataset).__name__,
        "sample_count": len(getattr(baseline_dataset, "samples", []) or []),
        "batch_size": max(int(cfg.batch_size), 1),
        "max_batches": max(int(cfg.max_batches), 1),
        "path_count": len(flat_paths),
        "consumed_path_count": consumed_paths,
        "fallback_count": fallback_count,
        "error_count": len(errors),
        "errors": errors,
        "prepare": prepare_report,
        "baseline": baseline,
        "shadow": {
            "batch_count": len(batch_wait_ms),
            "wall_ms": _round(shadow_wall),
            "dataloader_wait": _timings(batch_wait_ms),
            "prefetch_wait": _timings(prefetch_wait_ms),
            "queue_full_stall_ms": worker_metrics.get("queue_full_stall_ms", 0.0),
            "decode": _timings(decode_ms),
            "handoff": _timings(handoff_ms),
        },
        "shadow_vs_baseline_wall_ratio": _round(shadow_wall / baseline_wall) if baseline_wall > 0 else 0.0,
        "cases": cases,
        "real_dataloader_shadow": True,
        "prefetch_queue_training_path_enabled": False,
        "training_path_enabled": False,
        "shadow_only": True,
        "p3_dataloader_shadow_ready": not errors and consumed_paths == len(flat_paths),
        "p3_prefetch_h2d_ready": False,
        "readiness_blockers": [
            "dataloader_shadow_probe_only",
            "no optimizer_forward_backward_wall_clock_ab",
            "gpu_idle_and_step_p95_not_verified",
            "not_enabled_in_trainer_runtime",
        ],
    }


__all__ = [
    "LosslessCacheDataloaderShadowConfig",
    "build_sequential_cache_path_batches",
    "cache_paths_for_sample",
    "run_lossless_cache_dataloader_shadow",
]
