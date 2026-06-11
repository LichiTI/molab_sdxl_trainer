"""Compact manifest builder for cached dataset prefetch probes."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _path_record(role: str, path: Any) -> Dict[str, str]:
    return {"role": role, "path": str(path or "")}


def build_cached_dataset_prefetch_manifest(
    dataset: Any,
    *,
    max_samples: int | None = None,
) -> Dict[str, Any]:
    """Build a compact manifest from Newbie/Anima cached dataset samples."""

    samples: Iterable[Any] = getattr(dataset, "samples", []) or []
    dataset_class = type(dataset).__name__
    family = "anima" if dataset_class == "AnimaCachedDataset" else "newbie" if dataset_class == "NewbieCachedDataset" else "cached"
    limit = None if max_samples is None else max(int(max_samples), 0)
    records: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if limit is not None and index >= limit:
            break
        stem = str(getattr(sample, "stem", "") or f"sample_{index:08}")
        sample_id = str(getattr(sample, "sample_id", "") or stem)
        paths: List[Dict[str, str]] = []
        if hasattr(sample, "cache_path"):
            paths.append(_path_record("cache", getattr(sample, "cache_path", "")))
        if hasattr(sample, "latent_path"):
            paths.append(_path_record("latent", getattr(sample, "latent_path", "")))
        if hasattr(sample, "text_path"):
            paths.append(_path_record("text", getattr(sample, "text_path", "")))
        if getattr(sample, "loss_mask_path", None):
            paths.append(_path_record("loss_mask", getattr(sample, "loss_mask_path", "")))
        records.append({"index": index, "id": sample_id, "stem": stem, "paths": paths})
    return {
        "schema_version": 1,
        "family": family,
        "dataset_class": dataset_class,
        "data_dir": str(getattr(dataset, "data_dir", "") or ""),
        "cache_mmap": bool(getattr(dataset, "cache_mmap", False)),
        "cache_lazy": bool(getattr(dataset, "cache_lazy", False)),
        "file_handle_cache_size": max(int(getattr(dataset, "file_handle_cache_size", 0) or 0), 0),
        "samples": records,
    }


__all__ = ["build_cached_dataset_prefetch_manifest"]
