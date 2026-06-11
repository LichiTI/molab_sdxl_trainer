"""Shared readiness checks for Newbie cache-first artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


NEWBIE_CACHE_SUFFIXES = ("_newbie.npz", "_newbie.safetensors", "_newbie.pt")
NEWBIE_CACHE_PATTERNS = ("*_newbie.npz", "*_newbie.safetensors", "*_newbie.pt")
NEWBIE_POOLED_KEYS = ("pooled_prompt_embeds", "clip_pooled_features", "text_embeds")
NEWBIE_POOLED_DIM = 1024


def is_newbie_cache_file(path: Path) -> bool:
    return Path(path).name.endswith(NEWBIE_CACHE_SUFFIXES)


def find_newbie_cache_files(root: Path, *, recursive: bool = True) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        return []
    paths: set[Path] = set()
    globber = root.rglob if recursive else root.glob
    for pattern in NEWBIE_CACHE_PATTERNS:
        paths.update(path for path in globber(pattern) if path.is_file())
    return sorted(paths)


def _array_shape_from_cache(path: Path, keys: tuple[str, ...] = NEWBIE_POOLED_KEYS) -> tuple[str, tuple[int, ...]] | None:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        import numpy as np

        with np.load(str(path), allow_pickle=False, mmap_mode="r") as data:
            for key in keys:
                if key in data.files:
                    return key, tuple(int(dim) for dim in data[key].shape)
        return None
    if suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for key in keys:
                if key in available:
                    return key, tuple(int(dim) for dim in handle.get_tensor(key).shape)
        return None
    if suffix in {".pt", ".pth"}:
        import torch

        try:
            data = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(str(path), map_location="cpu")
        if isinstance(data, dict):
            for key in keys:
                value = data.get(key)
                if value is not None and hasattr(value, "shape"):
                    return key, tuple(int(dim) for dim in value.shape)
    return None


def newbie_cache_contract_for_files(files: Iterable[Path]) -> dict[str, Any]:
    cache_files = sorted(path for path in files if Path(path).is_file())
    if not cache_files:
        return {
            "ok": False,
            "reasons": ["newbie_cache_file_missing"],
            "pooled_shapes": [],
            "cache_file_count": 0,
        }

    pooled_shapes: list[list[int]] = []
    reasons: list[str] = []
    for path in cache_files:
        try:
            selected = _array_shape_from_cache(path)
        except Exception as exc:
            reasons.append(f"newbie_cache_contract_read_error:{type(exc).__name__}")
            continue
        if selected is None:
            reasons.append("newbie_cache_missing_pooled_prompt_embeds")
            continue
        key, shape = selected
        if key == "text_embeds" and not _is_valid_pooled_shape(shape):
            reasons.append("newbie_cache_missing_pooled_prompt_embeds")
            continue
        pooled_shapes.append(list(shape))
        if len(shape) not in (1, 2):
            reasons.append("newbie_cache_invalid_pooled_prompt_embeds_rank")
        elif len(shape) == 2 and int(shape[0]) != 1:
            reasons.append("newbie_cache_invalid_pooled_prompt_embeds_batch")
        elif int(shape[-1]) != NEWBIE_POOLED_DIM:
            reasons.append("newbie_cache_pooled_prompt_embeds_size_mismatch")

    return {
        "ok": not reasons,
        "reasons": sorted(set(reasons)),
        "pooled_shapes": pooled_shapes,
        "cache_file_count": len(cache_files),
    }


def _is_valid_pooled_shape(shape: tuple[int, ...]) -> bool:
    if len(shape) == 1:
        return int(shape[-1]) == NEWBIE_POOLED_DIM
    if len(shape) == 2:
        return int(shape[0]) == 1 and int(shape[-1]) == NEWBIE_POOLED_DIM
    return False


def newbie_cache_contract_for_root(root: Path, *, recursive: bool = True) -> dict[str, Any]:
    return newbie_cache_contract_for_files(find_newbie_cache_files(root, recursive=recursive))


__all__ = [
    "NEWBIE_CACHE_PATTERNS",
    "NEWBIE_CACHE_SUFFIXES",
    "NEWBIE_POOLED_DIM",
    "NEWBIE_POOLED_KEYS",
    "find_newbie_cache_files",
    "is_newbie_cache_file",
    "newbie_cache_contract_for_files",
    "newbie_cache_contract_for_root",
]
