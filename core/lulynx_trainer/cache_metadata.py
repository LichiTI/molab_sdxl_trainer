# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Shape metadata sidecars for cache-first training datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


CACHE_METADATA_VERSION = 1


@dataclass(frozen=True)
class CacheMetadataWriteResult:
    metadata_path: Path
    sample_count: int


def metadata_path_for(root: str | Path, family: str) -> Path:
    return Path(root) / f"lulynx_cache_metadata_{_normalize_family(family)}.json"


def write_cache_metadata(root: str | Path, *, family: str) -> CacheMetadataWriteResult:
    root_path = Path(root)
    family = _normalize_family(family)
    samples = _discover_anima_metadata(root_path) if family == "anima" else _discover_newbie_metadata(root_path)
    return write_cache_metadata_records(root_path, family=family, samples=samples)


def write_cache_metadata_records(
    root: str | Path,
    *,
    family: str,
    samples: List[Dict[str, Any]],
) -> CacheMetadataWriteResult:
    """Write a cache metadata sidecar from already-known sample records."""

    root_path = Path(root)
    family = _normalize_family(family)
    metadata_path = metadata_path_for(root_path, family)
    payload = {
        "metadata_version": CACHE_METADATA_VERSION,
        "family": family,
        "root": str(root_path),
        "samples": list(samples),
    }
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return CacheMetadataWriteResult(metadata_path=metadata_path, sample_count=len(samples))


def _discover_anima_metadata(root: Path) -> List[Dict[str, Any]]:
    text_by_stem: Dict[str, Path] = {}
    for suffix in ("_anima_te.npz", "_anima_te.safetensors", "_anima_te.pt"):
        for path in root.rglob(f"*{suffix}"):
            stem = path.name[: -len(suffix)]
            text_by_stem.setdefault(stem, path)

    records: List[Dict[str, Any]] = []
    for stem, text_path in sorted(text_by_stem.items()):
        latent_candidates: List[Path] = []
        for ext in (".npz", ".safetensors", ".pt"):
            latent_candidates.extend(root.rglob(f"{stem}_*_anima{ext}"))
        for latent_path in sorted(set(latent_candidates), key=lambda path: _rel(root, path)):
            latent_key, latent_shape = _latent_shape(latent_path, preferred_prefix="latents_")
            if latent_shape is None:
                continue
            records.append(
                {
                    "stem": stem,
                    "latent_path": _rel(root, latent_path),
                    "text_path": _rel(root, text_path),
                    "latent_key": latent_key,
                    "latent_shape": list(latent_shape),
                }
            )
    return records


def _discover_newbie_metadata(root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for suffix in ("_newbie.npz", "_newbie.safetensors", "_newbie.pt"):
        for cache_path in sorted(root.rglob(f"*{suffix}"), key=lambda path: _rel(root, path)):
            stem = cache_path.name[: -len(suffix)]
            _, latent_shape = _latent_shape(cache_path, key="latents")
            if latent_shape is None:
                continue
            records.append(
                {
                    "stem": stem,
                    "cache_path": _rel(root, cache_path),
                    "latent_shape": list(latent_shape),
                }
            )
    return records


def _latent_shape(path: Path, *, key: str = "", preferred_prefix: str = "") -> tuple[str, Optional[tuple[int, ...]]]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(str(path), allow_pickle=False, mmap_mode="r") as data:
            selected = key or _select_key(data.files, preferred_prefix)
            if not selected:
                return "", None
            return selected, tuple(int(v) for v in data[selected].shape)
    if suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            selected = key or _select_key(list(handle.keys()), preferred_prefix)
            if not selected:
                return "", None
            return selected, tuple(int(v) for v in handle.get_tensor(selected).shape)
    if suffix in (".pt", ".pth"):
        data = torch.load(str(path), map_location="cpu", weights_only=True)
        if not isinstance(data, dict):
            return "", None
        selected = key or _select_key(list(data.keys()), preferred_prefix)
        value = data.get(selected) if selected else None
        shape = getattr(value, "shape", None)
        if shape is None:
            return "", None
        return selected, tuple(int(v) for v in shape)
    return "", None


def _select_key(keys: List[str], preferred_prefix: str) -> str:
    if preferred_prefix:
        for key in keys:
            if str(key).startswith(preferred_prefix):
                return str(key)
    return str(keys[0]) if keys else ""


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _normalize_family(family: str) -> str:
    normalized = str(family or "").strip().lower()
    if normalized not in {"anima", "newbie"}:
        raise ValueError(f"Unsupported cache metadata family: {family}")
    return normalized
