"""Guarded Anima LXFS replacement DataLoader for explicit trainer A/B probes."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
import math
from typing import Any

try:
    from .anima_cached_dataset import AnimaCachedDataset
    from .lossless_anima_cache_replacement_loader import iter_anima_lossless_cache_replacement_batches
    from .lossless_cache_focus import parse_focus_sample_ids
    from .lossless_cache_replacement_loader import LosslessCacheReplacementLoaderConfig
    from .lossless_tensor_block import DEFAULT_FAST_CACHE_CODECS
except ImportError:  # pragma: no cover - direct script smoke loading
    from anima_cached_dataset import AnimaCachedDataset  # type: ignore[no-redef]
    from lossless_anima_cache_replacement_loader import (  # type: ignore[no-redef]
        iter_anima_lossless_cache_replacement_batches,
    )
    from lossless_cache_focus import parse_focus_sample_ids  # type: ignore[no-redef]
    from lossless_cache_replacement_loader import LosslessCacheReplacementLoaderConfig  # type: ignore[no-redef]
    from lossless_tensor_block import DEFAULT_FAST_CACHE_CODECS  # type: ignore[no-redef]


def parse_lossless_cache_codecs(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_FAST_CACHE_CODECS
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"fast", "fast-cache", "cache"}:
            return DEFAULT_FAST_CACHE_CODECS
        return tuple(item.strip().lower() for item in text.split(",") if item.strip())
    return tuple(str(item).strip().lower() for item in value if str(item).strip())


def _lossless_guard_metadata_report(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    source = dict(metadata)
    return {
        "lossless_guard_metadata_report_only": True,
        "lossless_guard_group_id": str(
            source.get("lossless_guard_group_id")
            or source.get("group_id")
            or "replacement_backward_tail"
        ),
        "lossless_guard_unit_kind": str(
            source.get("lossless_guard_unit_kind")
            or source.get("unit_kind")
            or "replacement_phase_guard_contract"
        ),
        "lossless_guard_runtime_activation_allowed": False,
        "lossless_guard_request_adapter_activation_allowed": False,
        "lossless_guard_runtime_default_change_allowed": False,
        "lossless_guard_training_path_enabled": False,
        "lossless_guard_resource_center_allowed": False,
        "lossless_guard_resource_center_candidate": False,
        "lossless_guard_default_enabled": False,
        "lossless_guard_product_ready": False,
        "lossless_guard_safe_to_auto_execute": False,
        "lossless_guard_requires_manual_heavy_validation": True,
        "lossless_guard_contract": str(
            source.get("lossless_guard_contract")
            or source.get("contract")
            or "lossless_guarded_variant_runtime_contract_v1"
        ),
    }


@dataclass(frozen=True)
class AnimaLosslessReplacementDataLoaderConfig:
    prefetch_depth: int = 2
    sidecar_dir: str | None = None
    sidecar_format: str = "lxfs"
    sidecar_suffix: str = ".lxfs"
    sidecar_strict: bool = False
    fallback_to_raw: bool = True
    prepare_sidecars: bool = False
    chunk_size: int = 256 * 1024
    min_saving: float = 0.0
    codecs: tuple[str, ...] = DEFAULT_FAST_CACHE_CODECS
    collate_mode: str = "auto"
    seed: int = 42
    focus_sample_ids: tuple[str, ...] = ()
    guard_metadata: Mapping[str, Any] | None = None


class AnimaLosslessReplacementDataLoader:
    """Small DataLoader-like wrapper around the experimental Anima iterator.

    It intentionally implements only the subset the trainer needs: ``dataset``,
    ``__iter__`` and ``__len__``. The feature is gated by an explicit trainer
    config flag and remains unsuitable for general product exposure.
    """

    def __init__(
        self,
        dataset: AnimaCachedDataset,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        config: AnimaLosslessReplacementDataLoaderConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(int(batch_size), 1)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.config = config or AnimaLosslessReplacementDataLoaderConfig()
        self.last_batch_reports: list[dict[str, Any]] = []

    def __len__(self) -> int:
        count = len(getattr(self.dataset, "samples", []) or [])
        if self.drop_last:
            return count // self.batch_size
        return int(math.ceil(count / float(self.batch_size))) if count else 0

    def __iter__(self) -> Iterator[dict[str, object]]:
        self.last_batch_reports = []
        cfg = self.config
        iterator = iter_anima_lossless_cache_replacement_batches(
            self.dataset,
            config=LosslessCacheReplacementLoaderConfig(
                batch_size=self.batch_size,
                max_batches=max(len(self), 1),
                prefetch_depth=max(int(cfg.prefetch_depth), 1),
                sidecar_dir=cfg.sidecar_dir,
                sidecar_format=str(cfg.sidecar_format or "lxfs"),
                sidecar_suffix=str(cfg.sidecar_suffix or ".lxfs"),
                sidecar_strict=bool(cfg.sidecar_strict),
                fallback_to_raw=bool(cfg.fallback_to_raw),
                prepare_sidecars=bool(cfg.prepare_sidecars),
                chunk_size=max(int(cfg.chunk_size), 1),
                min_saving=float(cfg.min_saving),
                collate_mode=str(cfg.collate_mode or "auto"),
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                seed=int(cfg.seed),
                focus_sample_ids=cfg.focus_sample_ids,
            ),
            codecs=cfg.codecs,
        )
        guard_metadata = _lossless_guard_metadata_report(cfg.guard_metadata)
        for batch, report in iterator:
            batch_report = dict(report)
            if guard_metadata:
                batch_report.update(guard_metadata)
            self.last_batch_reports.append(batch_report)
            yield batch


def create_anima_lossless_cache_replacement_dataloader(
    dataset: AnimaCachedDataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
    config: AnimaLosslessReplacementDataLoaderConfig | None = None,
) -> AnimaLosslessReplacementDataLoader:
    return AnimaLosslessReplacementDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        config=config,
    )


__all__ = [
    "AnimaLosslessReplacementDataLoader",
    "AnimaLosslessReplacementDataLoaderConfig",
    "create_anima_lossless_cache_replacement_dataloader",
    "_lossless_guard_metadata_report",
    "parse_focus_sample_ids",
    "parse_lossless_cache_codecs",
]
