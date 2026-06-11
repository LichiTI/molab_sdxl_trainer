"""Cache-first dataset for native Newbie training.

The production online cache builder is still a separate boundary.  This
dataset consumes explicit Newbie cache files and keeps the trainer path honest:
no cached tensors means no silent fallback to the SDXL image/text loop.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

try:
    from .caption_source_mix import normalize_caption_source_mix_config, select_caption_source
    from .lossless_cache_dataset_adapter import (
        LosslessCacheDatasetAdapterConfig,
        load_lossless_cache_arrays_for_dataset,
    )
except ImportError:  # pragma: no cover - direct script smoke loading
    from caption_source_mix import normalize_caption_source_mix_config, select_caption_source
    from lossless_cache_dataset_adapter import (
        LosslessCacheDatasetAdapterConfig,
        load_lossless_cache_arrays_for_dataset,
    )

logger = logging.getLogger(__name__)
NEWBIE_CACHE_METADATA_FILENAMES = ("lulynx_cache_metadata_newbie.json", "_metadata.json")
NEWBIE_LATENT_KEYS = ("latents", "latent", "model_input")
NEWBIE_HIDDEN_KEYS = ("encoder_hidden_states", "gemma_hidden_states", "prompt_embeds")
NEWBIE_POOLED_KEYS = ("pooled_prompt_embeds", "clip_pooled_features", "text_embeds")
NEWBIE_MASK_KEYS = ("attention_mask", "gemma_attention_mask", "attn_mask")
NEWBIE_LOSS_MASK_KEYS = ("loss_mask", "alpha_mask", "padding_mask")


def _native_cache_index_disabled() -> bool:
    return str(os.environ.get("LULYNX_DISABLE_NATIVE_CACHE_INDEX", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@lru_cache(maxsize=1)
def _load_native_cache_index_api() -> Any:
    artifact_dir = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    if artifact_dir and artifact_dir not in sys.path and Path(artifact_dir).is_dir():
        sys.path.insert(0, artifact_dir)
    try:
        return importlib.import_module("lulynx_native")
    except Exception:
        return None


def _native_cache_index_api() -> Any:
    if _native_cache_index_disabled():
        return None
    native = _load_native_cache_index_api()
    if not hasattr(native, "discover_newbie_cache_samples"):
        return None
    return native


@dataclass(frozen=True)
class NewbieCachedSample:
    stem: str
    cache_path: Path


@dataclass(frozen=True)
class NewbieCacheSchema:
    """Validation contract for ``*_newbie.npz`` training caches.

    Production Newbie caches are expected to contain 16-channel VAE latents,
    Gemma hidden states, a matching Gemma attention mask, and Jina CLIP pooled
    text features.  Tiny smokes can leave the optional dimensions at zero.
    """

    version: int = 2
    expected_latent_channels: int = 0
    expected_hidden_size: int = 0
    expected_pooled_size: int = 0
    require_schema_version: bool = True
    require_pooled_prompt_embeds: bool = True
    require_attention_mask: bool = True
    require_loss_mask: bool = False


@dataclass(frozen=True)
class NewbieCacheArrays:
    latents: np.ndarray
    encoder_hidden_states: np.ndarray
    pooled_prompt_embeds: Optional[np.ndarray]
    attention_mask: Optional[np.ndarray]
    loss_mask: Optional[np.ndarray] = None  # Alpha mask for masked loss (H, W) binary


class NewbieCachedDataset(Dataset):
    """Dataset backed by ``*_newbie.npz`` cache files."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        latent_crop_size: int = 0,
        text_token_limit: int = 0,
        schema: Optional[NewbieCacheSchema] = None,
        cache_mmap: bool = False,
        cache_lazy: bool = False,
        file_handle_cache_size: int = 128,
        lossless_cache_sidecar_enabled: bool = False,
        lossless_cache_sidecar_strict: bool = False,
        lossless_cache_sidecar_suffix: str = ".lxcs",
        # Caption Variants (validation only - not supported in cached mode)
        caption_variants_enabled: bool = False,
        caption_source_mix_enabled: bool = False,
        caption_source_nl_ratio: float = 65.0,
        caption_source_tag_ratio: float = 20.0,
        caption_source_trigger_only_ratio: float = 10.0,
        caption_source_empty_ratio: float = 5.0,
        caption_source_trigger_tokens: str = "",
    ):
        """Initialize Newbie cached dataset.

        Parameters
        ----------
        data_dir : str | Path
            Directory containing cache files.
        latent_crop_size : int
            Latent spatial crop size (0 = no crop).
        text_token_limit : int
            Text token limit for consumption (0 = no limit).
        schema : Optional[NewbieCacheSchema]
            Cache validation schema.
        cache_mmap : bool
            Use memory-mapped file loading for .npz files.
        cache_lazy : bool
            Use lazy loading (keep file handles open).
        file_handle_cache_size : int
            Maximum number of open file handles to cache.
        """
        self.data_dir = Path(data_dir)
        self.latent_crop_size = max(int(latent_crop_size or 0), 0)
        self.text_token_limit = max(int(text_token_limit or 0), 0)
        self.schema = schema or NewbieCacheSchema()
        self.cache_mmap = bool(cache_mmap)
        self.cache_lazy = bool(cache_lazy)
        self.file_handle_cache_size = max(int(file_handle_cache_size), 1)
        self.lossless_cache_sidecar_config = LosslessCacheDatasetAdapterConfig(
            enabled=bool(lossless_cache_sidecar_enabled),
            strict=bool(lossless_cache_sidecar_strict),
            sidecar_suffix=str(lossless_cache_sidecar_suffix or ".lxcs"),
        )
        self.lossless_cache_sidecar_last_report: Dict[str, Any] = {}
        self.caption_source_mix = normalize_caption_source_mix_config(
            enabled=caption_source_mix_enabled,
            nl_ratio=caption_source_nl_ratio,
            tag_ratio=caption_source_tag_ratio,
            trigger_only_ratio=caption_source_trigger_only_ratio,
            empty_ratio=caption_source_empty_ratio,
            trigger_tokens=caption_source_trigger_tokens,
        )
        self._caption_source_mix_missing_variant_warned = False

        # Caption Variants validation
        if caption_variants_enabled:
            logger.warning(
                "Caption variants are enabled but NewbieCachedDataset uses pre-computed text embeddings. "
                "Variants will be ignored. To use caption variants, train without cache or rebuild cache "
                "with each variant separately."
            )
        if caption_source_mix_enabled:
            logger.warning(
                "Structured Tag/NL caption mixing is enabled for NewbieCachedDataset. "
                "It will use caption_variant_* text caches when present; old caches fall back to the base text embedding."
            )

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Newbie cache data_dir does not exist: {self.data_dir}")
        self.samples = self._discover_samples()
        if not self.samples:
            raise ValueError(f"No Newbie cache samples found in {self.data_dir}")
        self._shape_cache: Dict[str, Optional[tuple]] = {}
        self._shape_profile: Dict[str, int] = {
            "shape_cache_hits": 0,
            "metadata_shape_hits": 0,
            "metadata_shape_misses": 0,
            "fallback_shape_loads": 0,
            "fallback_shape_failures": 0,
        }
        self._cache_metadata_path: Optional[Path] = None
        self._cache_metadata_error = ""
        self._bucket_build_profile: Dict[str, object] = {}
        # LRU cache for file handles when lazy loading is enabled.
        self._file_handle_cache: "OrderedDict[Path, object]" = OrderedDict()
        self._cache_metadata = self._load_cache_metadata()
        self._native_shape_metadata_index = self._build_native_shape_metadata_index()
        for sample in self.samples:
            validate_newbie_cache_file(sample.cache_path, self.schema)
        bucket_start = time.perf_counter()
        self._bucket_indices = self._build_bucket_indices()
        self._bucket_build_profile = {
            "enabled": True,
            "seconds": round(time.perf_counter() - bucket_start, 6),
            "bucket_count": len(self._bucket_indices or {}),
            "sample_count": len(self.samples),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        arrays = self._load_newbie_cache_arrays(sample.cache_path)

        latents_t = torch.from_numpy(arrays.latents).float()
        if latents_t.dim() == 4:
            latents_t = latents_t[0]
        if self.latent_crop_size > 0:
            crop = self.latent_crop_size
            latents_t = latents_t[:, :crop, :crop].contiguous()

        hidden_t = torch.from_numpy(arrays.encoder_hidden_states).float()
        if hidden_t.dim() == 3:
            hidden_t = hidden_t[0]
        if self.text_token_limit > 0:
            hidden_t = hidden_t[: self.text_token_limit]

        item: Dict[str, object] = {
            "latents": latents_t,
            "encoder_hidden_states": hidden_t,
            "captions": sample.stem,
            "sample_id": sample.stem,
        }
        if arrays.pooled_prompt_embeds is not None:
            pooled_t = torch.from_numpy(arrays.pooled_prompt_embeds).float()
            if pooled_t.dim() == 2:
                pooled_t = pooled_t[0]
            item["pooled_prompt_embeds"] = pooled_t
        if arrays.attention_mask is not None:
            mask_t = torch.from_numpy(arrays.attention_mask).bool()
            if mask_t.dim() == 2:
                mask_t = mask_t[0]
            if self.text_token_limit > 0:
                mask_t = mask_t[: self.text_token_limit]
            item["attention_mask"] = mask_t
        if arrays.loss_mask is not None:
            loss_mask_t = torch.from_numpy(arrays.loss_mask).float()
            if loss_mask_t.dim() == 3:
                loss_mask_t = loss_mask_t[0]
            if self.latent_crop_size > 0:
                crop = self.latent_crop_size
                loss_mask_t = loss_mask_t[:crop, :crop].contiguous()
            item["loss_mask"] = loss_mask_t
        return item

    def _array_shape_from_cache(self, cache_path: Path, keys: tuple[str, ...]) -> tuple:
        suffix = cache_path.suffix.lower()
        if suffix == ".npz":
            with np.load(str(cache_path), allow_pickle=False, mmap_mode="r") as data:
                key = _first_cache_key(list(data.files), keys)
                if key is None:
                    raise KeyError(keys)
                return tuple(data[key].shape)
        if suffix == ".safetensors":
            from safetensors import safe_open
            with safe_open(str(cache_path), framework="pt", device="cpu") as handle:
                key = _first_cache_key(list(handle.keys()), keys)
                if key is None:
                    raise KeyError(keys)
                return tuple(handle.get_tensor(key).shape)
        data = torch.load(str(cache_path), map_location="cpu", weights_only=True)
        key = _first_cache_key(list(data.keys()), keys) if isinstance(data, dict) else None
        if key is None:
            raise KeyError(keys)
        value = data[key]
        return tuple(value.shape)

    def _build_bucket_indices(self) -> Dict[str, List[int]]:
        buckets: Dict[str, List[int]] = {}
        for idx, sample in enumerate(self.samples):
            try:
                shape = self._latent_shape_for_sample(sample)
                if shape is None:
                    raise ValueError("missing latent shape")
                latent_h = int(shape[-2])
                latent_w = int(shape[-1])
                if self.latent_crop_size > 0:
                    latent_h = min(latent_h, self.latent_crop_size)
                    latent_w = min(latent_w, self.latent_crop_size)
                key = f"{latent_h}x{latent_w}"
            except Exception:
                key = "unknown"
            buckets.setdefault(key, []).append(idx)
        return buckets

    def get_bucket_indices(self) -> Optional[Dict[str, List[int]]]:
        return self._bucket_indices

    def get_token_bucket_summary(self) -> Dict[str, object]:
        buckets: Dict[str, Dict[str, object]] = {}
        for idx, sample in enumerate(self.samples):
            try:
                shape = self._latent_shape_for_sample(sample)
                if shape is None:
                    raise ValueError("missing latent shape")
                latent_h = int(shape[-2])
                latent_w = int(shape[-1])
                if self.latent_crop_size > 0:
                    latent_h = min(latent_h, self.latent_crop_size)
                    latent_w = min(latent_w, self.latent_crop_size)
                visual_tokens = latent_h * latent_w
                key = f"{visual_tokens}:{latent_h}x{latent_w}"
            except Exception:
                latent_h = latent_w = visual_tokens = 0
                key = "unknown"
            entry = buckets.setdefault(
                key,
                {
                    "sample_count": 0,
                    "visual_tokens": visual_tokens,
                    "latent_shape": [latent_h, latent_w],
                    "sample_indices": [],
                },
            )
            entry["sample_count"] = int(entry["sample_count"]) + 1
            if len(entry["sample_indices"]) < 8:
                entry["sample_indices"].append(idx)
        return {
            "family": "newbie",
            "mode": "no_pad",
            "bucket_count": len(buckets),
            "buckets": buckets,
        }

    def get_cache_metadata_summary(self) -> Dict[str, object]:
        """Return startup/cache metadata profile for manifest diagnostics."""
        return {
            "family": "newbie",
            "data_dir": str(self.data_dir),
            "sample_count": len(self.samples),
            "metadata_found": self._cache_metadata_path is not None,
            "metadata_path": str(self._cache_metadata_path or ""),
            "metadata_records": len(self._cache_metadata),
            "metadata_error": self._cache_metadata_error,
            "shape_cache_entries": len(self._shape_cache),
            "native_shape_metadata_records": len(getattr(self, "_native_shape_metadata_index", {}) or {}),
            **self._shape_profile,
            "bucket_build": dict(self._bucket_build_profile),
        }

    def _build_native_shape_metadata_index(self) -> Dict[str, Dict[str, Any]]:
        try:
            from core.turbocore_cache_shape_metadata import build_cache_shape_index
        except Exception:
            return {}
        return build_cache_shape_index((sample.cache_path for sample in self.samples))

    def _load_cache_metadata(self) -> Dict[str, Dict[str, Any]]:
        metadata_path = next((self.data_dir / name for name in NEWBIE_CACHE_METADATA_FILENAMES if (self.data_dir / name).is_file()), None)
        if metadata_path is None:
            return {}
        self._cache_metadata_path = metadata_path
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._cache_metadata_error = f"{type(exc).__name__}: {exc}"
            logger.debug("Newbie cache metadata ignored: %s", exc)
            return {}
        if not isinstance(payload, dict):
            self._cache_metadata_error = "metadata payload is not an object"
            return {}
        family = str(payload.get("family", "newbie") or "newbie").strip().lower()
        if family not in {"newbie", "qwen_image_edit", "native_newbie"}:
            self._cache_metadata_error = f"unsupported family: {family}"
            return {}
        samples = payload.get("samples", [])
        records = samples.values() if isinstance(samples, dict) else samples
        index: Dict[str, Dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            keys = [
                str(record.get("stem", "") or ""),
                str(record.get("cache_path", "") or ""),
                str(record.get("latent_path", "") or ""),
            ]
            for key in keys:
                normalized = key.replace("\\", "/").strip()
                if normalized:
                    index[normalized] = record
        if index:
            logger.info("Newbie cache metadata loaded: %s (%d records)", metadata_path.name, len(index))
        return index

    def _metadata_record_for_sample(self, sample: NewbieCachedSample) -> Optional[Dict[str, Any]]:
        rel_path = self._relative_metadata_path(sample.cache_path)
        for key in (rel_path, sample.cache_path.name, sample.stem):
            record = self._cache_metadata.get(key)
            if record is not None:
                return record
        return None

    def _native_shape_record_for_sample(self, sample: NewbieCachedSample) -> Optional[Dict[str, Any]]:
        rel_path = self._relative_metadata_path(sample.cache_path)
        for key in (str(sample.cache_path), sample.cache_path.as_posix(), rel_path, sample.cache_path.name):
            record = self._native_shape_metadata_index.get(key) if self._native_shape_metadata_index else None
            if record is not None:
                return record
        return None

    def _latent_shape_for_sample(self, sample: NewbieCachedSample) -> Optional[tuple]:
        cache_key = self._relative_metadata_path(sample.cache_path)
        if cache_key in self._shape_cache:
            self._shape_profile["shape_cache_hits"] += 1
            return self._shape_cache[cache_key]
        record = self._metadata_record_for_sample(sample)
        if record is not None:
            for key in ("latent_shape", "latents_shape", "shape"):
                shape = self._normalize_shape(record.get(key))
                if shape is not None:
                    self._shape_profile["metadata_shape_hits"] += 1
                    self._shape_cache[cache_key] = shape
                    return shape
        native_record = self._native_shape_record_for_sample(sample)
        if native_record is not None:
            try:
                from core.turbocore_cache_shape_metadata import shape_from_record
                shape = shape_from_record(native_record, "latents")
            except Exception:
                shape = None
            if shape is not None:
                self._shape_profile["metadata_shape_hits"] += 1
                self._shape_cache[cache_key] = shape
                return shape
        self._shape_profile["metadata_shape_misses"] += 1
        try:
            shape = self._array_shape_from_cache(sample.cache_path, NEWBIE_LATENT_KEYS)
            self._shape_profile["fallback_shape_loads"] += 1
            self._shape_cache[cache_key] = shape
            return shape
        except Exception:
            self._shape_profile["fallback_shape_failures"] += 1
            self._shape_cache[cache_key] = None
            raise

    def _relative_metadata_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.data_dir).as_posix()
        except ValueError:
            return path.name

    @staticmethod
    def _normalize_shape(value: Any) -> Optional[tuple]:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        try:
            return tuple(int(v) for v in value)
        except (TypeError, ValueError):
            return None

    def _load_newbie_cache_arrays(self, cache_path: Path) -> NewbieCacheArrays:
        """Load Newbie cache arrays with optional mmap/lazy loading."""
        sidecar_arrays, sidecar_report = load_lossless_cache_arrays_for_dataset(
            cache_path,
            config=self.lossless_cache_sidecar_config,
        )
        self.lossless_cache_sidecar_last_report = sidecar_report
        if sidecar_arrays is not None:
            data = sidecar_arrays
        # Check file handle cache for lazy loading
        elif self.cache_lazy and cache_path in self._file_handle_cache:
            self._file_handle_cache.move_to_end(cache_path)
            data = self._file_handle_cache[cache_path]
        else:
            suffix = cache_path.suffix.lower()
            if suffix == ".npz":
                if self.cache_mmap:
                    data = np.load(str(cache_path), allow_pickle=False, mmap_mode="r")
                else:
                    data = np.load(str(cache_path), allow_pickle=False)
            elif suffix == ".safetensors":
                from safetensors.torch import load_file
                data = load_file(str(cache_path))
            elif suffix in (".pt", ".pth"):
                data = torch.load(str(cache_path), map_location="cpu", weights_only=True)
            else:
                raise ValueError(f"Unsupported cache format: {suffix}")

            # Cache file handle if lazy loading is enabled
            if self.cache_lazy:
                self._update_file_handle_cache(cache_path, data)

        # Extract arrays
        keys = _cache_keys(data)
        prefix = self._select_caption_source_variant_prefix(keys, cache_path)
        latent_key = _first_cache_key(keys, NEWBIE_LATENT_KEYS)
        if latent_key is None:
            raise ValueError(f"Newbie cache {cache_path} missing required keys: {NEWBIE_LATENT_KEYS}")
        hidden_key = self._prefixed_cache_key_any(keys, prefix, NEWBIE_HIDDEN_KEYS)
        if hidden_key is None:
            raise ValueError(f"Newbie cache {cache_path} missing required keys: {NEWBIE_HIDDEN_KEYS}")
        pooled_key = self._prefixed_cache_key_any(keys, prefix, NEWBIE_POOLED_KEYS)
        mask_key = self._prefixed_cache_key_any(keys, prefix, NEWBIE_MASK_KEYS)
        loss_mask_key = _first_cache_key(keys, NEWBIE_LOSS_MASK_KEYS)
        latents = _cache_get_numpy(data, latent_key)
        encoder_hidden_states = _cache_get_numpy(data, hidden_key)
        pooled_prompt_embeds = _cache_get_numpy(data, pooled_key) if pooled_key else None
        attention_mask = _cache_get_numpy(data, mask_key) if mask_key else None
        loss_mask = _cache_get_numpy(data, loss_mask_key) if loss_mask_key else None

        return NewbieCacheArrays(
            latents=latents,
            encoder_hidden_states=encoder_hidden_states,
            pooled_prompt_embeds=pooled_prompt_embeds,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )

    def _select_caption_source_variant_prefix(self, keys: List[str], cache_path: Path) -> str:
        if not self.caption_source_mix.enabled:
            return ""
        available = {
            name
            for name in ("nl", "tag", "trigger_only", "empty")
            if f"caption_variant_{name}_encoder_hidden_states" in keys
        }
        if not available:
            if not self._caption_source_mix_missing_variant_warned:
                logger.warning(
                    "Newbie caption_source_mix enabled but no caption_variant_* keys found in %s; "
                    "rebuild caches to enable cache-first mixing.",
                    cache_path.name,
                )
                self._caption_source_mix_missing_variant_warned = True
            return ""
        source = select_caption_source(
            self.caption_source_mix,
            has_tags="tag" in available,
            has_nl="nl" in available,
            has_triggers="trigger_only" in available,
        )
        if source not in available:
            if source == "empty" and "empty" in available:
                return "caption_variant_empty_"
            return ""
        return f"caption_variant_{source}_"

    @staticmethod
    def _prefixed_cache_key(keys: List[str], prefix: str, key: str) -> Optional[str]:
        if prefix:
            candidate = f"{prefix}{key}"
            return candidate if candidate in keys else None
        return key if key in keys else None

    @staticmethod
    def _prefixed_cache_key_any(keys: List[str], prefix: str, aliases: tuple[str, ...]) -> Optional[str]:
        for key in aliases:
            candidate = f"{prefix}{key}" if prefix else key
            if candidate in keys:
                return candidate
        return None

    def _update_file_handle_cache(self, path: Path, data: object) -> None:
        """Update LRU file handle cache."""
        self._file_handle_cache[path] = data
        self._file_handle_cache.move_to_end(path)

        # Evict oldest if cache is full
        while len(self._file_handle_cache) > self.file_handle_cache_size:
            _, evicted = self._file_handle_cache.popitem(last=False)
            self._close_handle(evicted)

    @staticmethod
    def _close_handle(handle: object) -> None:
        if handle is None:
            return
        close_fn = getattr(handle, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    def close_file_handles(self) -> None:
        """Release any cached file handles. Safe to call multiple times."""
        for _, handle in list(self._file_handle_cache.items()):
            self._close_handle(handle)
        self._file_handle_cache.clear()

    def __del__(self):
        try:
            self.close_file_handles()
        except Exception:
            pass

    def _discover_samples(self) -> List[NewbieCachedSample]:
        native = _native_cache_index_api()
        if native is not None:
            try:
                records = native.discover_newbie_cache_samples(str(self.data_dir))
                samples = [
                    NewbieCachedSample(
                        stem=str(record.get("stem") or ""),
                        cache_path=Path(str(record.get("cache_path") or "")),
                    )
                    for record in records
                    if isinstance(record, dict) and str(record.get("stem") or "") and str(record.get("cache_path") or "")
                ]
                if samples:
                    return samples
            except Exception:
                logger.debug("Native Newbie cache index failed; falling back to Python", exc_info=True)

        samples: Dict[str, Path] = {}
        for suffix in ("_newbie.npz", "_newbie.safetensors", "_newbie.pt"):
            tag = suffix
            for path in self.data_dir.rglob(f"*{tag}"):
                stem = path.name[: -len(tag)]
                if stem not in samples:
                    samples[stem] = path
        return [
            NewbieCachedSample(stem=stem, cache_path=path)
            for stem, path in sorted(samples.items())
        ]


def _load_cache_file(path: Path) -> Dict[str, object]:
    """Load a cache file, dispatching on suffix (.npz / .safetensors / .pt)."""
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return np.load(str(path), allow_pickle=False)
    if suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    if suffix in (".pt", ".pth"):
        return torch.load(str(path), map_location="cpu", weights_only=True)
    raise ValueError(f"Unsupported cache format: {suffix}")


def _cache_keys(data: Dict[str, object]) -> List[str]:
    """Return available keys from any cache format (NpzFile or plain dict)."""
    if hasattr(data, "files"):
        return list(data.files)
    return list(data.keys())


def _first_cache_key(keys: List[str], aliases: tuple[str, ...]) -> Optional[str]:
    for key in aliases:
        if key in keys:
            return key
    return None


def _cache_get_numpy(data: Dict[str, object], key: str) -> np.ndarray:
    """Extract a numpy array from any cache format."""
    val = data[key]
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    return np.asarray(val)


def load_newbie_cache_arrays(
    cache_path: str | Path,
    schema: Optional[NewbieCacheSchema] = None,
) -> NewbieCacheArrays:
    """Load and validate one Newbie cache file."""

    path = Path(cache_path)
    resolved_schema = schema or NewbieCacheSchema()
    data = _load_cache_file(path)
    try:
        _validate_schema_version(path, data, resolved_schema)
        latents = _require_array(path, data, NEWBIE_LATENT_KEYS)
        hidden = _require_array(
            path,
            data,
            NEWBIE_HIDDEN_KEYS,
        )
        pooled = _optional_array(
            path,
            data,
            NEWBIE_POOLED_KEYS,
        )
        attention_mask = _optional_array(
            path,
            data,
            NEWBIE_MASK_KEYS,
        )

        loss_mask_arr = _optional_array(
            path,
            data,
            NEWBIE_LOSS_MASK_KEYS,
        )

        _validate_latents(path, latents, resolved_schema)
        _validate_hidden(path, hidden, resolved_schema)
        _validate_pooled(path, pooled, resolved_schema)
        _validate_attention_mask(path, attention_mask, hidden, resolved_schema)
        _validate_loss_mask(path, loss_mask_arr, latents, resolved_schema)

        return NewbieCacheArrays(
            latents=np.asarray(latents),
            encoder_hidden_states=np.asarray(hidden),
            pooled_prompt_embeds=np.asarray(pooled) if pooled is not None else None,
            attention_mask=np.asarray(attention_mask) if attention_mask is not None else None,
            loss_mask=np.asarray(loss_mask_arr) if loss_mask_arr is not None else None,
        )
    finally:
        if hasattr(data, "close"):
            data.close()


def validate_newbie_cache_file(
    cache_path: str | Path,
    schema: Optional[NewbieCacheSchema] = None,
) -> None:
    """Raise ``ValueError`` if a Newbie cache file cannot train safely."""

    load_newbie_cache_arrays(cache_path, schema)


def _validate_schema_version(
    path: Path,
    data: Dict[str, object],
    schema: NewbieCacheSchema,
) -> None:
    keys = _cache_keys(data)
    if "newbie_cache_schema_version" not in keys:
        if schema.require_schema_version:
            raise ValueError(
                f"Newbie cache missing newbie_cache_schema_version={schema.version}: {path}"
            )
        return
    raw = _cache_get_numpy(data, "newbie_cache_schema_version").reshape(-1)
    if raw.size != 1 or int(raw[0]) != schema.version:
        raise ValueError(
            f"Newbie cache schema version mismatch in {path}: "
            f"expected {schema.version}, got {raw.tolist()}"
        )


def _require_array(
    path: Path,
    data: Dict[str, object],
    keys: tuple[str, ...],
) -> np.ndarray:
    value = _optional_array(path, data, keys)
    if value is None:
        raise ValueError(f"Newbie cache {path} missing required keys: {keys}")
    return value


def _optional_array(
    path: Path,
    data: Dict[str, object],
    keys: tuple[str, ...],
) -> Optional[np.ndarray]:
    available = _cache_keys(data)
    for key in keys:
        if key in available:
            return _cache_get_numpy(data, key)
    return None


def _validate_latents(path: Path, latents: np.ndarray, schema: NewbieCacheSchema) -> None:
    if latents.ndim not in (3, 4):
        raise ValueError(
            f"Newbie latents must be CHW or 1CHW, got shape={latents.shape}: {path}"
        )
    if latents.ndim == 4 and latents.shape[0] != 1:
        raise ValueError(f"Newbie cache files must store one latent sample each: {path}")
    channels = int(latents.shape[-3])
    height = int(latents.shape[-2])
    width = int(latents.shape[-1])
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"Newbie latents must be non-empty, got shape={latents.shape}: {path}")
    if schema.expected_latent_channels and channels != schema.expected_latent_channels:
        raise ValueError(
            f"Newbie latent channels mismatch in {path}: "
            f"expected {schema.expected_latent_channels}, got {channels}"
        )
    _require_finite(path, "latents", latents)


def _validate_hidden(path: Path, hidden: np.ndarray, schema: NewbieCacheSchema) -> None:
    if hidden.ndim not in (2, 3):
        raise ValueError(
            f"Newbie encoder_hidden_states must be TD or 1TD, got shape={hidden.shape}: {path}"
        )
    if hidden.ndim == 3 and hidden.shape[0] != 1:
        raise ValueError(f"Newbie cache files must store one text sample each: {path}")
    token_count = int(hidden.shape[-2])
    hidden_size = int(hidden.shape[-1])
    if token_count <= 0 or hidden_size <= 0:
        raise ValueError(
            f"Newbie encoder_hidden_states must be non-empty, got shape={hidden.shape}: {path}"
        )
    if schema.expected_hidden_size and hidden_size != schema.expected_hidden_size:
        raise ValueError(
            f"Newbie hidden size mismatch in {path}: "
            f"expected {schema.expected_hidden_size}, got {hidden_size}"
        )
    _require_finite(path, "encoder_hidden_states", hidden)


def _validate_pooled(
    path: Path,
    pooled: Optional[np.ndarray],
    schema: NewbieCacheSchema,
) -> None:
    if pooled is None:
        if schema.require_pooled_prompt_embeds:
            raise ValueError(f"Newbie cache missing pooled CLIP features: {path}")
        return
    if pooled.ndim not in (1, 2):
        raise ValueError(
            f"Newbie pooled_prompt_embeds must be D or 1D, got shape={pooled.shape}: {path}"
        )
    if pooled.ndim == 2 and pooled.shape[0] != 1:
        raise ValueError(f"Newbie cache files must store one pooled feature each: {path}")
    pooled_size = int(pooled.shape[-1])
    if pooled_size <= 0:
        raise ValueError(f"Newbie pooled_prompt_embeds must be non-empty: {path}")
    if schema.expected_pooled_size and pooled_size != schema.expected_pooled_size:
        raise ValueError(
            f"Newbie pooled feature size mismatch in {path}: "
            f"expected {schema.expected_pooled_size}, got {pooled_size}"
        )
    _require_finite(path, "pooled_prompt_embeds", pooled)


def _validate_attention_mask(
    path: Path,
    attention_mask: Optional[np.ndarray],
    hidden: np.ndarray,
    schema: NewbieCacheSchema,
) -> None:
    if attention_mask is None:
        if schema.require_attention_mask:
            raise ValueError(f"Newbie cache missing Gemma attention mask: {path}")
        return
    if attention_mask.ndim not in (1, 2):
        raise ValueError(
            f"Newbie attention_mask must be T or 1T, got shape={attention_mask.shape}: {path}"
        )
    if attention_mask.ndim == 2 and attention_mask.shape[0] != 1:
        raise ValueError(f"Newbie cache files must store one attention mask each: {path}")
    token_count = int(hidden.shape[-2])
    mask_tokens = int(attention_mask.shape[-1])
    if mask_tokens != token_count:
        raise ValueError(
            f"Newbie attention_mask length mismatch in {path}: "
            f"expected {token_count}, got {mask_tokens}"
        )
    _require_finite(path, "attention_mask", attention_mask)


def _validate_loss_mask(
    path: Path,
    loss_mask: Optional[np.ndarray],
    latents: np.ndarray,
    schema: NewbieCacheSchema,
) -> None:
    if loss_mask is None:
        if schema.require_loss_mask:
            raise ValueError(f"Newbie cache missing loss_mask: {path}")
        return
    if loss_mask.ndim not in (2, 3):
        raise ValueError(
            f"Newbie loss_mask must be HW or 1HW, got shape={loss_mask.shape}: {path}"
        )
    if loss_mask.ndim == 3 and loss_mask.shape[0] != 1:
        raise ValueError(f"Newbie cache files must store one loss mask each: {path}")
    mask_h = int(loss_mask.shape[-2])
    mask_w = int(loss_mask.shape[-1])
    if mask_h <= 0 or mask_w <= 0:
        raise ValueError(f"Newbie loss_mask must be non-empty, got shape={loss_mask.shape}: {path}")
    latent_h = int(latents.shape[-2])
    latent_w = int(latents.shape[-1])
    if mask_h != latent_h or mask_w != latent_w:
        raise ValueError(
            f"Newbie loss_mask spatial mismatch in {path}: "
            f"mask {mask_h}x{mask_w}, latent {latent_h}x{latent_w}"
        )
    _require_finite(path, "loss_mask", loss_mask)


def _require_finite(path: Path, name: str, value: np.ndarray) -> None:
    if not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
        raise ValueError(f"Newbie cache key {name} must be numeric/bool, got {value.dtype}: {path}")
    if np.issubdtype(value.dtype, np.floating) and not bool(np.isfinite(value).all()):
        raise ValueError(f"Newbie cache key {name} contains NaN/Inf values: {path}")


def _normalize_cached_collate_mode(mode: str) -> str:
    normalized = str(mode or "auto").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "default": "auto",
        "fast": "pad_sequence",
        "pad": "pad_sequence",
        "torch": "pad_sequence",
        "manual": "legacy",
        "prealloc": "legacy",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"auto", "legacy", "pad_sequence"} else "auto"


def _pad_cached_hidden_and_mask(batch: List[Dict[str, object]]) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_items: List[torch.Tensor] = []
    mask_items: List[torch.Tensor] = []
    for item in batch:
        hidden = item["encoder_hidden_states"]  # type: ignore[assignment]
        hidden_tensor = hidden.to(dtype=torch.float32)  # type: ignore[union-attr]
        token_count = int(hidden_tensor.shape[0])
        hidden_items.append(hidden_tensor)
        source_mask = item.get("attention_mask")
        if isinstance(source_mask, torch.Tensor):
            mask_items.append(source_mask.to(dtype=torch.bool))
        else:
            mask_items.append(torch.ones((token_count,), dtype=torch.bool))
    return (
        pad_sequence(hidden_items, batch_first=True, padding_value=0.0),
        pad_sequence(mask_items, batch_first=True, padding_value=False),
    )


def newbie_cached_collate(batch: List[Dict[str, object]], collate_mode: str = "auto") -> Dict[str, object]:
    max_text_len = max(int(item["encoder_hidden_states"].shape[0]) for item in batch)  # type: ignore[index]
    text_dim = int(batch[0]["encoder_hidden_states"].shape[-1])  # type: ignore[index]
    resolved_mode = _normalize_cached_collate_mode(collate_mode)
    if resolved_mode in {"auto", "pad_sequence"}:
        hidden, mask = _pad_cached_hidden_and_mask(batch)
    else:
        hidden = torch.zeros((len(batch), max_text_len, text_dim), dtype=torch.float32)
        mask = torch.zeros((len(batch), max_text_len), dtype=torch.bool)
        for index, item in enumerate(batch):
            value = item["encoder_hidden_states"]  # type: ignore[assignment]
            token_count = int(value.shape[0])
            hidden[index, :token_count] = value
            source_mask = item.get("attention_mask")  # type: ignore[union-attr]
            if source_mask is None:
                mask[index, :token_count] = True
            else:
                mask[index, : source_mask.shape[0]] = source_mask

    result: Dict[str, object] = {
        "latents": torch.stack([item["latents"] for item in batch]),  # type: ignore[list-item]
        "encoder_hidden_states": hidden,
        "attention_mask": mask,
        "captions": [str(item.get("captions", "")) for item in batch],
        "sample_ids": [str(item.get("sample_id", "")) for item in batch],
    }
    pooled_items = [item.get("pooled_prompt_embeds") for item in batch]
    if all(isinstance(item, torch.Tensor) for item in pooled_items):
        result["pooled_prompt_embeds"] = torch.stack(pooled_items)  # type: ignore[arg-type]
    loss_mask_items = [item.get("loss_mask") for item in batch]
    if all(isinstance(item, torch.Tensor) for item in loss_mask_items):
        result["loss_masks"] = torch.stack(loss_mask_items)  # type: ignore[arg-type]
    return result


class _NewbieCachedCollator:
    """Pickle-safe collator for DataLoader worker processes."""

    def __init__(self, collate_mode: str = "auto"):
        self.collate_mode = _normalize_cached_collate_mode(collate_mode)

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, object]:
        return newbie_cached_collate(batch, collate_mode=self.collate_mode)


def _attach_native_cache_prefetch_shadow_adapter_if_enabled(
    dataloader: DataLoader,
    dataset: "NewbieCachedDataset",
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    prefetch_factor: Optional[int],
) -> DataLoader:
    try:
        from core.turbocore_cached_dataset_prefetch import maybe_attach_cached_dataset_prefetch_shadow_adapter
    except Exception:
        attached = dataloader
    else:
        attached = maybe_attach_cached_dataset_prefetch_shadow_adapter(
            dataloader,
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
    try:
        from core.turbocore_cache_reader_shadow import maybe_attach_cache_reader_shadow_timing
    except Exception:
        return attached
    return maybe_attach_cache_reader_shadow_timing(
        attached,
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


def create_newbie_cached_dataloader(
    dataset: NewbieCachedDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = 2,
    drop_last: bool = False,
    collate_mode: str = "auto",
) -> DataLoader:
    """Create DataLoader for NewbieCachedDataset with memory optimization support.

    Parameters
    ----------
    dataset : NewbieCachedDataset
        The cached dataset to load from.
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle samples.
    num_workers : int
        Number of worker processes for data loading.
    persistent_workers : bool
        Keep worker processes alive between epochs.
    pin_memory : bool
        Pin CPU memory for faster GPU transfer.
    prefetch_factor : Optional[int]
        Number of batches to prefetch per worker (None = default 2).
    """
    dl_kwargs: Dict[str, object] = {
        "num_workers": num_workers,
        "collate_fn": _NewbieCachedCollator(collate_mode),
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = prefetch_factor

    def rebuild_factory(descriptor):
        workers = max(int(descriptor.get("num_workers", num_workers) or 0), 0)
        return create_newbie_cached_dataloader(
            dataset,
            batch_size=max(int(descriptor.get("batch_size", batch_size) or 1), 1),
            shuffle=bool(descriptor.get("shuffle", shuffle)),
            num_workers=workers,
            persistent_workers=bool(descriptor.get("persistent_workers", persistent_workers)) and workers > 0,
            pin_memory=bool(descriptor.get("pin_memory", pin_memory)),
            prefetch_factor=None if descriptor.get("prefetch_factor") is None else max(int(descriptor.get("prefetch_factor") or 1), 1),
            drop_last=bool(descriptor.get("drop_last", drop_last)),
            collate_mode=collate_mode,
        )

    def attach(dataloader: DataLoader) -> DataLoader:
        attached = _attach_native_cache_prefetch_shadow_adapter_if_enabled(
            dataloader,
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        from .dataloader_rebuild_runtime import attach_dataloader_rebuild_descriptor
        from .multi_batch_contract import attach_dataloader_batching_contract
        from .training_data_pipeline_stage import attach_lulynx_dataloader_data_pipeline_report

        attached = attach_dataloader_rebuild_descriptor(
            attached,
            route="newbie_cached",
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            uses_batch_sampler=getattr(attached, "batch_sampler", None) is not None,
            rebuild_factory=rebuild_factory,
        )
        attached = attach_dataloader_batching_contract(
            attached,
            requested_physical_batch_size=batch_size,
        )
        return attach_lulynx_dataloader_data_pipeline_report(
            attached,
            requested_physical_batch_size=batch_size,
            route="newbie_cached",
            required_fields=("latents", "encoder_hidden_states", "captions"),
        )

    bucket_indices = dataset.get_bucket_indices()
    if bucket_indices and len(bucket_indices) > 1:
        from torch.utils.data import RandomSampler, SequentialSampler
        from .anima_cached_dataset import _ConcatBatchSampler, _IndexMappingBatchSampler

        bucket_samplers = []
        for indices in bucket_indices.values():
            sub_sampler = RandomSampler(indices) if shuffle else SequentialSampler(indices)
            bucket_samplers.append(_IndexMappingBatchSampler(indices, sub_sampler, batch_size, drop_last=drop_last))
        return attach(DataLoader(
            dataset,
            batch_sampler=_ConcatBatchSampler(bucket_samplers, shuffle=shuffle),
            **dl_kwargs,
        ))

    return attach(DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **dl_kwargs,
    ))
