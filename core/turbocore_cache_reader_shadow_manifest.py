"""Manifest helpers for debug-only native cache reader shadow probes."""

from __future__ import annotations

from typing import Any, Dict

from core.turbocore_cached_dataset_prefetch_manifest import build_cached_dataset_prefetch_manifest


def build_cache_reader_shadow_manifest(dataset: Any, *, max_files: int = 64) -> Dict[str, Any]:
    manifest = build_cached_dataset_prefetch_manifest(dataset, max_samples=max(int(max_files), 1))
    remaining = max(int(max_files), 1)
    records = []
    for sample in manifest.get("samples", []) if isinstance(manifest, dict) else []:
        if not isinstance(sample, dict):
            continue
        paths = sample.get("paths", [])
        limited_paths = list(paths[:remaining]) if isinstance(paths, list) else []
        if not limited_paths:
            continue
        item = dict(sample)
        item["paths"] = limited_paths
        records.append(item)
        remaining -= len(limited_paths)
        if remaining <= 0:
            break
    manifest["samples"] = records
    manifest["reader_shadow_max_files"] = max(int(max_files), 1)
    return manifest


__all__ = ["build_cache_reader_shadow_manifest"]
