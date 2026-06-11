"""Mini training-step A/B probe for Anima LXCS/LXFS replacement batches.

This stays intentionally outside the trainer runtime path. It compares the
normal Anima cached DataLoader against the experimental replacement iterator
while both sides run the same small torch forward/backward/optimizer step.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

try:
    from .anima_cached_dataset import AnimaCachedDataset, create_anima_cached_dataloader
    from .lossless_anima_cache_replacement_loader import iter_anima_lossless_cache_replacement_batches
    from .lossless_cache_prefetch_queue import (
        LosslessCachePrefetchQueueConfig,
        prepare_lossless_cache_prefetch_sidecars,
    )
    from .lossless_cache_replacement_loader import LosslessCacheReplacementLoaderConfig
    from .lossless_cache_training_ab import _resolve_device, _round, _run_training_loop, _warm_training_runtime
    from .lossless_tensor_block import DEFAULT_FAST_CACHE_CODECS
except ImportError:  # pragma: no cover - direct script smoke loading
    from anima_cached_dataset import AnimaCachedDataset, create_anima_cached_dataloader  # type: ignore[no-redef]
    from lossless_anima_cache_replacement_loader import (  # type: ignore[no-redef]
        iter_anima_lossless_cache_replacement_batches,
    )
    from lossless_cache_prefetch_queue import (  # type: ignore[no-redef]
        LosslessCachePrefetchQueueConfig,
        prepare_lossless_cache_prefetch_sidecars,
    )
    from lossless_cache_replacement_loader import LosslessCacheReplacementLoaderConfig  # type: ignore[no-redef]
    from lossless_cache_training_ab import (  # type: ignore[no-redef]
        _resolve_device,
        _round,
        _run_training_loop,
        _warm_training_runtime,
    )
    from lossless_tensor_block import DEFAULT_FAST_CACHE_CODECS  # type: ignore[no-redef]


@dataclass(frozen=True)
class AnimaLosslessCacheTrainingAbConfig:
    batch_size: int = 1
    max_batches: int = 4
    prefetch_depth: int = 2
    sidecar_dir: str | None = None
    sidecar_format: str = "lxfs"
    sidecar_suffix: str = ".lxfs"
    sidecar_strict: bool = False
    fallback_to_raw: bool = True
    verify_sidecar: bool = True
    copy_arrays: bool = True
    prepare_sidecars: bool = True
    chunk_size: int = 1 << 20
    min_saving: float = 0.02
    handoff: str = "none"
    collate_mode: str = "auto"
    device: str = "auto"
    compute_repeat: int = 4
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = False
    sample_gpu_metrics: bool = False
    gpu_sample_interval_ms: int = 250


def _run_baseline_loop(
    dataset: AnimaCachedDataset,
    cfg: AnimaLosslessCacheTrainingAbConfig,
    device: str,
) -> dict[str, Any]:
    loader = create_anima_cached_dataloader(
        dataset,
        batch_size=max(int(cfg.batch_size), 1),
        shuffle=False,
        num_workers=max(int(cfg.num_workers), 0),
        persistent_workers=bool(cfg.persistent_workers),
        pin_memory=bool(cfg.pin_memory),
        prefetch_factor=max(int(cfg.prefetch_factor), 1),
        drop_last=False,
        collate_mode=str(cfg.collate_mode or "auto"),
    )
    return _run_training_loop(iter(loader), max_batches=max(int(cfg.max_batches), 1), device=device, cfg=cfg)


def _replacement_paths(dataset: AnimaCachedDataset, cfg: AnimaLosslessCacheTrainingAbConfig) -> list[Any]:
    limit = max(int(cfg.batch_size), 1) * max(int(cfg.max_batches), 1)
    samples = list(dataset.samples)[:limit]
    return [path for sample in samples for path in (sample.latent_path, sample.text_path)]


def _prepare_replacement_sidecars(
    dataset: AnimaCachedDataset,
    cfg: AnimaLosslessCacheTrainingAbConfig,
    codecs: Iterable[str],
) -> dict[str, Any]:
    if not bool(cfg.prepare_sidecars):
        return {"ok": True, "skipped": True}
    paths = _replacement_paths(dataset, cfg)
    if not paths:
        return {"ok": True, "skipped": True, "case_count": 0}
    return prepare_lossless_cache_prefetch_sidecars(
        paths,
        config=LosslessCachePrefetchQueueConfig(
            prefetch_depth=max(int(cfg.prefetch_depth), 1),
            sidecar_enabled=True,
            sidecar_strict=bool(cfg.sidecar_strict),
            fallback_to_raw=bool(cfg.fallback_to_raw),
            sidecar_format=str(cfg.sidecar_format or "lxfs"),
            sidecar_suffix=str(cfg.sidecar_suffix or ".lxfs"),
            sidecar_dir=cfg.sidecar_dir,
            verify_sidecar=bool(cfg.verify_sidecar),
            copy_arrays=bool(cfg.copy_arrays),
            handoff=str(cfg.handoff or "none"),
        ),
        chunk_size=max(int(cfg.chunk_size), 1),
        codecs=codecs,
        min_saving=float(cfg.min_saving),
    )


def _run_replacement_loop(
    dataset: AnimaCachedDataset,
    cfg: AnimaLosslessCacheTrainingAbConfig,
    device: str,
    codecs: Iterable[str],
) -> dict[str, Any]:
    sidecar_prepare = _prepare_replacement_sidecars(dataset, cfg, codecs)
    iterator = iter_anima_lossless_cache_replacement_batches(
        dataset,
        config=LosslessCacheReplacementLoaderConfig(
            batch_size=max(int(cfg.batch_size), 1),
            max_batches=max(int(cfg.max_batches), 1),
            prefetch_depth=max(int(cfg.prefetch_depth), 1),
            sidecar_dir=cfg.sidecar_dir,
            sidecar_format=str(cfg.sidecar_format or "lxfs"),
            sidecar_suffix=str(cfg.sidecar_suffix or ".lxfs"),
            sidecar_strict=bool(cfg.sidecar_strict),
            verify_sidecar=bool(cfg.verify_sidecar),
            copy_arrays=bool(cfg.copy_arrays),
            fallback_to_raw=bool(cfg.fallback_to_raw),
            prepare_sidecars=False,
            handoff=str(cfg.handoff or "none"),
            chunk_size=max(int(cfg.chunk_size), 1),
            min_saving=float(cfg.min_saving),
            collate_mode=str(cfg.collate_mode or "auto"),
        ),
        codecs=codecs,
    )
    report = _run_training_loop(iterator, max_batches=max(int(cfg.max_batches), 1), device=device, cfg=cfg)
    batch_reports = list(report.get("batch_reports") or [])
    report.update(
        {
            "selected_path": f"{str(cfg.sidecar_format or 'lxfs').lower()}_replacement",
            "fallback_count": sum(int(item.get("fallback_count") or 0) for item in batch_reports),
            "sidecar_prepare": sidecar_prepare,
            "bypassed": False,
            "training_path_enabled": False,
        }
    )
    return report


def run_anima_lossless_cache_training_ab(
    baseline_dataset: AnimaCachedDataset,
    replacement_dataset: AnimaCachedDataset,
    *,
    config: AnimaLosslessCacheTrainingAbConfig | None = None,
    codecs: Iterable[str] = DEFAULT_FAST_CACHE_CODECS,
) -> dict[str, Any]:
    cfg = config or AnimaLosslessCacheTrainingAbConfig()
    device = _resolve_device(cfg.device)
    _warm_training_runtime(device)
    baseline = _run_baseline_loop(baseline_dataset, cfg, device)
    replacement = _run_replacement_loop(replacement_dataset, cfg, device, codecs)
    baseline_wall = float(baseline.get("wall_ms") or 0.0)
    replacement_wall = float(replacement.get("wall_ms") or 0.0)
    blockers = [
        "mini_probe_not_real_anima_trainer_model",
        "replacement_path_not_enabled_in_runtime_request",
        "gpu_idle_and_full_step_p95_not_verified",
    ]
    if not bool(cfg.copy_arrays):
        blockers.append("no_copy_numpy_arrays_are_non_writable_probe_only")
    return {
        "provider": "lxfs_anima_training_step_ab_v1",
        "ok": bool(baseline.get("ok")) and bool(replacement.get("ok")),
        "device": device,
        "sample_count": min(len(baseline_dataset.samples), len(replacement_dataset.samples)),
        "batch_size": max(int(cfg.batch_size), 1),
        "max_batches": max(int(cfg.max_batches), 1),
        "compute_repeat": max(int(cfg.compute_repeat), 1),
        "sidecar_format": str(cfg.sidecar_format or "lxfs").lower(),
        "copy_arrays": bool(cfg.copy_arrays),
        "baseline": baseline,
        "replacement": replacement,
        "replacement_vs_baseline_wall_ratio": _round(replacement_wall / baseline_wall) if baseline_wall > 0 else 0.0,
        "selected_path": replacement.get("selected_path", ""),
        "mini_training_step_ab": True,
        "real_optimizer_step": True,
        "real_training_model": False,
        "training_path_enabled": False,
        "p3_anima_training_step_ab_probe_ready": bool(baseline.get("ok")) and bool(replacement.get("ok")),
        "readiness_blockers": blockers,
    }


__all__ = [
    "AnimaLosslessCacheTrainingAbConfig",
    "run_anima_lossless_cache_training_ab",
]
