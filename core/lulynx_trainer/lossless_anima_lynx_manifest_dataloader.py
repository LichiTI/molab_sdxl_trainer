# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Guarded Anima .lynx manifest DataLoader for explicit trainer A/B probes."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
import time
from pathlib import Path
from typing import Any

try:
    from .anima_cached_dataset import AnimaCachedDataset, AnimaCachedSample, anima_cached_collate
    from .lossless_anima_cache_replacement_loader import (
        _build_anima_item_from_payloads,
        _elapsed_ms,
        _sample_batches,
        _timings,
    )
    from .lossless_tensor_layout_metadata import batch_tensor_layouts, mapping_tensor_layouts
    from .lossless_cache_focus import parse_focus_sample_ids
    from .lossless_cache_prefetch_queue import LosslessCachePrefetchPayload
    from .lossless_cache_sidecar import LosslessCacheEntry, load_numpy_cache_entries
    from .lossless_cache_tensor_container import (
        GROUP_SUFFIX,
        MANIFEST_SUFFIX,
        write_tensor_group_container_entries_file,
        write_tensor_group_manifest_file,
    )
    from .lossless_cache_tensor_reader import TensorGroupManifestReader
    from .lossless_anima_cache_replacement_dataloader import _lossless_guard_metadata_report
except ImportError:  # pragma: no cover - direct script smoke loading
    from anima_cached_dataset import AnimaCachedDataset, AnimaCachedSample, anima_cached_collate  # type: ignore[no-redef]
    from lossless_anima_cache_replacement_loader import (  # type: ignore[no-redef]
        _build_anima_item_from_payloads,
        _elapsed_ms,
        _sample_batches,
        _timings,
    )
    from lossless_tensor_layout_metadata import batch_tensor_layouts, mapping_tensor_layouts  # type: ignore[no-redef]
    from lossless_cache_focus import parse_focus_sample_ids  # type: ignore[no-redef]
    from lossless_cache_prefetch_queue import LosslessCachePrefetchPayload  # type: ignore[no-redef]
    from lossless_cache_sidecar import LosslessCacheEntry, load_numpy_cache_entries  # type: ignore[no-redef]
    from lossless_cache_tensor_container import (  # type: ignore[no-redef]
        GROUP_SUFFIX,
        MANIFEST_SUFFIX,
        write_tensor_group_container_entries_file,
        write_tensor_group_manifest_file,
    )
    from lossless_cache_tensor_reader import TensorGroupManifestReader  # type: ignore[no-redef]
    from lossless_anima_cache_replacement_dataloader import _lossless_guard_metadata_report  # type: ignore[no-redef]


def _now() -> float:
    return time.perf_counter()


def _round(value: float) -> float:
    return round(float(value or 0.0), 4)


@dataclass(frozen=True)
class AnimaLynxManifestDataLoaderConfig:
    manifest_path: str | None = None
    container_dir: str | None = None
    shard_size: int = 16
    codecs: tuple[str, ...] = ("raw",)
    min_saving: float = 0.0
    prepare_manifest: bool = False
    copy_arrays: bool = True
    verify_crc32: bool = True
    collate_mode: str = "auto"
    seed: int = 42
    focus_sample_ids: tuple[str, ...] = ()
    guard_metadata: Mapping[str, Any] | None = None


def _manifest_path_for(root: Path) -> Path:
    return root / ("anima_lynx_manifest" + MANIFEST_SUFFIX)


def _shard_path_for(root: Path, index: int) -> Path:
    return root / f"anima_lynx_manifest_{index:05d}{GROUP_SUFFIX}"


def _prefixed_entries(path: Path, prefix: str) -> tuple[LosslessCacheEntry, ...]:
    entries: list[LosslessCacheEntry] = []
    for entry in load_numpy_cache_entries(path):
        metadata = dict(entry.metadata or {})
        metadata["anima_cache_role"] = prefix
        entries.append(
            LosslessCacheEntry(
                name=f"{prefix}:{entry.name}",
                data=entry.data,
                element_size=entry.element_size,
                metadata=metadata,
            )
        )
    return tuple(entries)


def _sample_group(sample: AnimaCachedSample) -> tuple[str, str, tuple[LosslessCacheEntry, ...]]:
    entries = list(_prefixed_entries(sample.latent_path, "latent"))
    entries.extend(_prefixed_entries(sample.text_path, "text"))
    source = f"{sample.latent_path}|{sample.text_path}"
    return str(sample.sample_id), source, tuple(entries)


def prepare_anima_lynx_manifest(
    samples: Sequence[AnimaCachedSample],
    *,
    container_dir: str | Path,
    shard_size: int = 16,
    codecs: Iterable[str] = ("raw",),
    min_saving: float = 0.0,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write native .lynx shards for Anima latent/text cache pairs."""

    root = Path(container_dir)
    root.mkdir(parents=True, exist_ok=True)
    resolved_manifest = Path(manifest_path) if manifest_path else _manifest_path_for(root)
    resolved_manifest.parent.mkdir(parents=True, exist_ok=True)
    size = max(int(shard_size or 1), 1)
    shard_paths: list[Path] = []
    shard_reports: list[dict[str, Any]] = []
    sample_list = list(samples)
    started = _now()
    for shard_index, start in enumerate(range(0, len(sample_list), size)):
        chunk = sample_list[start : start + size]
        groups = [_sample_group(sample) for sample in chunk]
        shard_path = _shard_path_for(root, shard_index)
        report = write_tensor_group_container_entries_file(
            groups,
            container_path=shard_path,
            codecs=tuple(codecs),
            min_saving=float(min_saving),
        )
        shard_paths.append(shard_path)
        shard_reports.append(report)
    manifest_report = write_tensor_group_manifest_file(shard_paths, manifest_path=resolved_manifest)
    return {
        "ok": bool(manifest_report.get("ok") and len(shard_paths) > 0),
        "provider": "anima_lynx_manifest_prepare_v1",
        "manifest_path": str(resolved_manifest),
        "container_dir": str(root),
        "shard_count": len(shard_paths),
        "sample_count": len(sample_list),
        "tensor_count": int(manifest_report.get("tensor_count") or 0),
        "total_raw_size": int(manifest_report.get("total_raw_size") or 0),
        "total_encoded_size": int(manifest_report.get("total_encoded_size") or 0),
        "codec_counts": manifest_report.get("codec_counts") or {},
        "atomic_publish": True,
        "npz_intermediate_required": False,
        "shard_reports": shard_reports,
        "wall_ms": _round(_elapsed_ms(started)),
    }


def _split_arrays(arrays: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    latent: dict[str, Any] = {}
    text: dict[str, Any] = {}
    text_aliases = {
        "attn_mask": "attention_mask",
        "t5_attn_mask": "t5_attention_mask",
    }
    for key, value in arrays.items():
        name = str(key)
        if name.startswith("latent:"):
            latent[name[len("latent:") :]] = value
        elif name.startswith("text:"):
            text_name = name[len("text:") :]
            text[text_aliases.get(text_name, text_name)] = value
    return latent, text


def _payload(sample: AnimaCachedSample, arrays: dict[str, Any], role: str) -> LosslessCachePrefetchPayload:
    path = sample.latent_path if role == "latent" else sample.text_path
    raw_bytes = sum(int(getattr(value, "nbytes", 0) or 0) for value in arrays.values())
    return LosslessCachePrefetchPayload(
        path=path,
        index=0,
        arrays=arrays,
        report={
            "source": str(path),
            "payload_source": "lynx_manifest",
            "decode_ms": 0.0,
            "handoff_ms": 0.0,
            "raw_bytes": raw_bytes,
            "fallback_to_raw_cache": False,
            "arrays": list(arrays.keys()),
        },
        error="",
    )


class AnimaLynxManifestDataLoader:
    """DataLoader-like wrapper for explicit .lynx manifest trainer A/B probes."""

    def __init__(
        self,
        dataset: AnimaCachedDataset,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        config: AnimaLynxManifestDataLoaderConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(int(batch_size), 1)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.config = config or AnimaLynxManifestDataLoaderConfig()
        self.last_batch_reports: list[dict[str, Any]] = []
        self._prepare_report: dict[str, Any] = {"ok": True, "skipped": True}
        self._manifest_path = self._resolve_manifest_path()
        self._active_reader: TensorGroupManifestReader | None = None

    def _close_active_reader(self) -> None:
        reader = self._active_reader
        self._active_reader = None
        if reader is None:
            return
        try:
            reader.close()
        except BufferError:
            # mmap-backed tensors may still be alive in the training step. Keep
            # the reader open instead of breaking the explicit no-copy probe.
            self._active_reader = reader

    def close(self) -> None:
        self._close_active_reader()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self._close_active_reader()
        except Exception:
            pass

    def _resolve_manifest_path(self) -> Path:
        if self.config.manifest_path:
            return Path(self.config.manifest_path)
        root = Path(self.config.container_dir or "temp/lossless_anima_lynx_manifest")
        return _manifest_path_for(root)

    def __len__(self) -> int:
        count = len(getattr(self.dataset, "samples", []) or [])
        if self.drop_last:
            return count // self.batch_size
        return int(math.ceil(count / float(self.batch_size))) if count else 0

    def _ensure_manifest(self) -> dict[str, Any]:
        if self._manifest_path.is_file() and not bool(self.config.prepare_manifest):
            return {"ok": True, "skipped": True, "manifest_path": str(self._manifest_path)}
        root = Path(self.config.container_dir or self._manifest_path.parent)
        return prepare_anima_lynx_manifest(
            list(getattr(self.dataset, "samples", []) or []),
            container_dir=root,
            shard_size=max(int(self.config.shard_size), 1),
            codecs=self.config.codecs,
            min_saving=float(self.config.min_saving),
            manifest_path=self._manifest_path,
        )

    def __iter__(self) -> Iterator[dict[str, object]]:
        self.last_batch_reports = []
        self._prepare_report = self._ensure_manifest()
        cfg = self.config
        batches, focus_report = _sample_batches(
            list(self.dataset.samples),
            batch_size=self.batch_size,
            max_batches=max(len(self), 1),
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            seed=int(cfg.seed),
            focus_sample_ids=cfg.focus_sample_ids,
        )
        if bool(cfg.copy_arrays):
            self._close_active_reader()
            reader = TensorGroupManifestReader(self._manifest_path, eager=True)
            close_reader_on_exit = True
            reader_lifetime = "iterator"
        else:
            if self._active_reader is None:
                self._active_reader = TensorGroupManifestReader(self._manifest_path, eager=True)
            reader = self._active_reader
            close_reader_on_exit = False
            reader_lifetime = "dataloader"
        focus_sample_id_set = {str(value) for value in cfg.focus_sample_ids if str(value)}
        guard_metadata = _lossless_guard_metadata_report(cfg.guard_metadata)
        try:
            for batch_index, batch_samples in enumerate(batches):
                sample_ids = [str(sample.sample_id) for sample in batch_samples]
                batch_plan = reader.plan_batch(sample_ids)
                read_started = _now()
                loaded = reader.load_samples(
                    sample_ids,
                    verify_crc32=bool(cfg.verify_crc32),
                    copy_arrays=bool(cfg.copy_arrays),
                )
                read_ms = _elapsed_ms(read_started)
                build_ms: list[float] = []
                items: list[dict[str, object]] = []
                cases: list[dict[str, Any]] = []
                target_tensor_layouts: list[dict[str, Any]] = []
                for sample in batch_samples:
                    arrays = loaded.get(str(sample.sample_id))
                    if arrays is None:
                        raise RuntimeError(f"missing .lynx manifest sample: {sample.sample_id}")
                    latent_arrays, text_arrays = _split_arrays(arrays)
                    build_started = _now()
                    item = _build_anima_item_from_payloads(
                        self.dataset,
                        sample,
                        _payload(sample, latent_arrays, "latent"),
                        _payload(sample, text_arrays, "text"),
                    )
                    build_ms.append(_elapsed_ms(build_started))
                    if not focus_sample_id_set or str(sample.sample_id) in focus_sample_id_set:
                        target_tensor_layouts.extend(
                            mapping_tensor_layouts(
                                item,
                                sample_id=str(sample.sample_id),
                                payload_source="lynx_manifest",
                                copy_path=(
                                    "copied_manifest_arrays"
                                    if bool(cfg.copy_arrays)
                                    else "mmap_manifest_tensor_view"
                                ),
                                array_source="lynx_manifest_arrays",
                                cache_file=str(self._manifest_path.name),
                            )
                        )
                    items.append(item)
                    cases.append(
                        {
                            "index": len(cases),
                            "source": str(sample.latent_path),
                            "payload_source": "lynx_manifest",
                            "decode_ms": 0.0,
                            "handoff_ms": 0.0,
                            "raw_bytes": sum(int(getattr(value, "nbytes", 0) or 0) for value in arrays.values()),
                            "arrays": list(arrays.keys()),
                        }
                    )
                collate_started = _now()
                batch = anima_cached_collate(
                    items,
                    fixed_text_tokens=int(self.dataset.fixed_text_tokens),
                    collate_mode=str(cfg.collate_mode or "auto"),
                )
                report = {
                    "batch_index": batch_index,
                    "sample_count": len(items),
                    "sample_ids": sample_ids,
                    "focus_sample_ids": list(cfg.focus_sample_ids),
                    "focus_sample_report": focus_report,
                    "prepare": self._prepare_report if batch_index == 0 else {"ok": True, "skipped": True},
                    "queue_empty_wait": _timings([read_ms]),
                    "decode": _timings([read_ms]),
                    "handoff": _timings([0.0]),
                    "item_build": _timings(build_ms),
                    "collate_ms": _round(_elapsed_ms(collate_started)),
                    "fallback_count": 0,
                    "queue_full_stall_ms": 0.0,
                    "cases": cases,
                    "target_tensor_layouts": target_tensor_layouts,
                    "batch_tensor_layouts": batch_tensor_layouts(
                        batch,
                        payload_source="lynx_manifest",
                        copy_path=(
                            "collated_copied_manifest_batch"
                            if bool(cfg.copy_arrays)
                            else "collated_mmap_manifest_batch"
                        ),
                    ),
                    "experimental_replacement_path": True,
                    "provider": "anima_lynx_manifest_dataloader_v1",
                    "training_path_enabled": False,
                    "copy_arrays": bool(cfg.copy_arrays),
                    "verify_crc32": bool(cfg.verify_crc32),
                    "persistent_mmap_reader": True,
                    "reader_lifetime": reader_lifetime,
                    "batch_plan": batch_plan,
                    "batch_plan_missing_sample_count": int(batch_plan.get("missing_sample_count") or 0),
                    "batch_plan_read_round_count": int(batch_plan.get("read_round_count") or 0),
                    "batch_plan_shard_count": int(batch_plan.get("shard_count") or 0),
                    "zero_copy_tensor_view_ready": not bool(cfg.copy_arrays),
                    "zero_copy_view_probe_only": not bool(cfg.copy_arrays),
                }
                if guard_metadata:
                    report.update(guard_metadata)
                self.last_batch_reports.append(report)
                yield batch
        finally:
            if close_reader_on_exit:
                reader.close()


def create_anima_lynx_manifest_dataloader(
    dataset: AnimaCachedDataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
    config: AnimaLynxManifestDataLoaderConfig | None = None,
) -> AnimaLynxManifestDataLoader:
    return AnimaLynxManifestDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        config=config,
    )


__all__ = [
    "AnimaLynxManifestDataLoader",
    "AnimaLynxManifestDataLoaderConfig",
    "create_anima_lynx_manifest_dataloader",
    "parse_focus_sample_ids",
    "prepare_anima_lynx_manifest",
]
