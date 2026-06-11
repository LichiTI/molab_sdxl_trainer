"""Shared helpers for image GGUF probe adapters."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..image_gguf_contracts import ImageGGUFCompatibility, ImageGGUFManifest, TensorInfo


def count_prefix_hits(tensors: dict[str, TensorInfo], prefixes: list[str]) -> dict[str, int]:
    return {
        prefix: sum(1 for key in tensors if key.startswith(prefix))
        for prefix in prefixes
    }


def missing_exact_keys(tensors: dict[str, TensorInfo], keys: list[str]) -> list[str]:
    return [key for key in keys if key not in tensors]


def missing_prefixes(tensors: dict[str, TensorInfo], prefixes: list[str]) -> list[str]:
    return [prefix for prefix in prefixes if not any(key.startswith(prefix) for key in tensors)]


def dtype_counts(tensors: dict[str, TensorInfo]) -> dict[str, int]:
    return dict(sorted(Counter(info.dtype for info in tensors.values()).items()))


def rank_counts(tensors: dict[str, TensorInfo]) -> dict[str, int]:
    return dict(sorted(Counter(str(info.rank) for info in tensors.values()).items()))


def shape_summary(tensors: dict[str, TensorInfo]) -> dict[str, Any]:
    total_numel = sum(info.numel for info in tensors.values())
    largest = sorted(tensors.values(), key=lambda item: item.numel, reverse=True)[:8]
    return {
        "total_numel": total_numel,
        "largest_tensors": [info.to_dict() for info in largest],
    }


def tensor_samples(tensors: dict[str, TensorInfo], *, limit: int = 20) -> list[dict[str, Any]]:
    return [tensors[key].to_dict() for key in sorted(tensors)[:limit]]


def build_probe_manifest(
    *,
    source_path: Path,
    adapter_id: str,
    component: str,
    family: str,
    tensors: dict[str, TensorInfo],
    required_tensors: list[str],
    required_prefixes: list[str],
    matched_tensors: int,
    unexpected_tensors_sample: list[str],
    notes: list[str] | None = None,
    warnings: list[str] | None = None,
) -> ImageGGUFManifest:
    missing_tensors = missing_exact_keys(tensors, required_tensors)
    missing_required_prefixes = missing_prefixes(tensors, required_prefixes)
    ok = not missing_tensors and not missing_required_prefixes and matched_tensors > 0
    return ImageGGUFManifest(
        schema_version=1,
        adapter_id=adapter_id,
        component=component,
        family=family,
        source_path=str(source_path),
        source_format="safetensors",
        compatibility=ImageGGUFCompatibility.PROBE_ONLY.value,
        ok=ok,
        tensor_count=len(tensors),
        matched_tensors=matched_tensors,
        missing_required_tensors=missing_tensors,
        missing_required_prefixes=missing_required_prefixes,
        unexpected_tensors_sample=unexpected_tensors_sample[:40],
        required_tensors=list(required_tensors),
        required_prefixes=list(required_prefixes),
        dtype_counts=dtype_counts(tensors),
        rank_counts=rank_counts(tensors),
        shape_summary=shape_summary(tensors),
        tensor_samples=tensor_samples(tensors),
        notes=list(notes or []),
        warnings=list(warnings or []),
    )


def build_prefix_probe_manifest(
    *,
    source_path: Path,
    adapter_id: str,
    component: str,
    family: str,
    tensors: dict[str, TensorInfo],
    required_tensors: list[str],
    required_prefixes: list[str],
    optional_prefixes: list[str] | None = None,
    notes: list[str] | None = None,
    warnings: list[str] | None = None,
) -> ImageGGUFManifest:
    optional = list(optional_prefixes or [])
    prefixes = list(required_prefixes) + optional
    prefix_hits = count_prefix_hits(tensors, prefixes)
    matched = sum(prefix_hits[prefix] for prefix in prefixes)
    expected_prefixes = tuple(prefixes)
    unexpected = [key for key in sorted(tensors) if not key.startswith(expected_prefixes)]
    missing_optional_warnings = [
        f"optional prefix not present: {prefix}"
        for prefix in optional
        if prefix_hits.get(prefix, 0) == 0
    ]
    return build_probe_manifest(
        source_path=source_path,
        adapter_id=adapter_id,
        component=component,
        family=family,
        tensors=tensors,
        required_tensors=list(required_tensors),
        required_prefixes=list(required_prefixes),
        matched_tensors=matched,
        unexpected_tensors_sample=unexpected,
        notes=notes,
        warnings=list(warnings or []) + missing_optional_warnings,
    )


__all__ = [
    "build_probe_manifest",
    "build_prefix_probe_manifest",
    "count_prefix_hits",
    "dtype_counts",
    "missing_exact_keys",
    "missing_prefixes",
    "rank_counts",
    "shape_summary",
    "tensor_samples",
]
