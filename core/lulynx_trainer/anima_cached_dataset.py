"""Cache-first dataset for native Anima training.

The native Anima route can train from precomputed cache artifacts:

- ``<stem>_*_anima.npz``: Qwen Image VAE latents
- ``<stem>_anima_te.npz``: Qwen3/LLM-adapter text conditioning cache

This dataset intentionally avoids raw image/VAE/text-encoder work.  Online
cache building is a separate boundary.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
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
    from .dataset_discovery import assign_stable_sample_ids, discover_smart_subsets, resolve_caption_path
    from .caption_sidecar import json_caption_to_training_parts, json_caption_to_training_text
    from .caption_source_mix import normalize_caption_source_mix_config, select_caption_source
    from .lossless_cache_dataset_adapter import (
        LosslessCacheDatasetAdapterConfig,
        load_lossless_cache_arrays_for_dataset,
    )
except ImportError:  # pragma: no cover - direct script smoke loading
    from dataset_discovery import assign_stable_sample_ids, discover_smart_subsets, resolve_caption_path
    from caption_sidecar import json_caption_to_training_parts, json_caption_to_training_text
    from caption_source_mix import normalize_caption_source_mix_config, select_caption_source
    from lossless_cache_dataset_adapter import (
        LosslessCacheDatasetAdapterConfig,
        load_lossless_cache_arrays_for_dataset,
    )

logger = logging.getLogger(__name__)
ANIMA_CACHE_METADATA_FILENAMES = ("lulynx_cache_metadata_anima.json", "_metadata.json")


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
    if not hasattr(native, "discover_anima_cache_samples"):
        return None
    return native


@dataclass(frozen=True)
class AnimaCacheSchema:
    """Validation contract for ``*_anima.npz`` / ``*_anima_te.npz`` cache pairs.

    Production Anima caches contain VAE latents (with optional per-resolution
    keys) and Qwen3/T5 text conditioning.  Tiny smokes can leave optional
    dimensions at zero.
    """

    version: int = 1
    require_loss_mask: bool = False


@dataclass(frozen=True)
class AnimaCachedSample:
    stem: str
    sample_id: str
    latent_path: Path
    text_path: Path
    caption_path: Optional[Path]
    loss_mask_path: Optional[Path] = None


@dataclass(frozen=True)
class ConceptGeometryEntry:
    """Per-sample geometry metadata for Concept Geometry curriculum sampling."""

    stage: str = "mid"
    density: float = 1.0
    radius: float = 0.0
    curriculum_score: float = 0.5
    loss_weight: float = 1.0
    concept_group: str = ""
    concept_path: tuple[str, ...] = ()
    path_depth: int = 0
    geometry_version: int = 1
    backend_requested: str = ""
    backend_resolved: str = ""
    feature_sources: tuple[str, ...] = ()
    fallback_reasons: tuple[str, ...] = ()
    tag_buckets: Dict[str, tuple[str, ...]] = field(default_factory=dict)
    source_density: Dict[str, float] = field(default_factory=dict)
    neighbor_ids: tuple[str, ...] = ()
    sibling_ids: tuple[str, ...] = ()
    conflict_score: float = 0.0

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ConceptGeometryEntry":
        stage = str(payload.get("stage", "mid") or "mid").strip().lower()
        if stage not in {"core", "mid", "edge"}:
            stage = "mid"
        concept_path_raw = payload.get("concept_path", ())
        if isinstance(concept_path_raw, (list, tuple)):
            concept_path = tuple(str(item).strip() for item in concept_path_raw if str(item).strip())
        elif concept_path_raw:
            concept_path = tuple(
                part.strip()
                for part in str(concept_path_raw).replace("::", ">").replace("/", ">").split(">")
                if part.strip()
            )
        else:
            concept_path = ()
        raw_sources = payload.get("feature_sources", ())
        if isinstance(raw_sources, dict):
            feature_sources = tuple(str(key) for key in raw_sources.keys())
        elif isinstance(raw_sources, (list, tuple, set)):
            feature_sources = tuple(str(item) for item in raw_sources)
        else:
            feature_sources = ()
        raw_fallbacks = payload.get("fallback_reasons", ())
        fallback_reasons = tuple(str(item) for item in raw_fallbacks) if isinstance(raw_fallbacks, (list, tuple)) else ()
        raw_buckets = payload.get("tag_buckets", {})
        tag_buckets = {
            str(key): tuple(str(item) for item in value)
            for key, value in raw_buckets.items()
            if isinstance(value, (list, tuple))
        } if isinstance(raw_buckets, dict) else {}
        raw_source_density = payload.get("source_density", {})
        source_density = {
            str(key): max(float(value or 0.0), 0.0)
            for key, value in raw_source_density.items()
        } if isinstance(raw_source_density, dict) else {}
        neighbor_ids = tuple(str(item) for item in payload.get("neighbor_ids", ()) if str(item).strip()) if isinstance(payload.get("neighbor_ids", ()), (list, tuple)) else ()
        sibling_ids = tuple(str(item) for item in payload.get("sibling_ids", ()) if str(item).strip()) if isinstance(payload.get("sibling_ids", ()), (list, tuple)) else ()
        return cls(
            stage=stage,
            density=max(float(payload.get("density", 1.0) or 1.0), 0.0),
            radius=max(float(payload.get("radius", 0.0) or 0.0), 0.0),
            curriculum_score=float(payload.get("curriculum_score", 0.5) or 0.5),
            loss_weight=max(float(payload.get("loss_weight", 1.0) or 1.0), 1e-6),
            concept_group=str(payload.get("concept_group", "") or "").strip(),
            concept_path=concept_path,
            path_depth=max(int(payload.get("path_depth", len(concept_path)) or len(concept_path)), 0),
            geometry_version=max(int(payload.get("geometry_version", 1) or 1), 1),
            backend_requested=str(payload.get("backend_requested", "") or "").strip(),
            backend_resolved=str(payload.get("backend_resolved", "") or "").strip(),
            feature_sources=feature_sources,
            fallback_reasons=fallback_reasons,
            tag_buckets=tag_buckets,
            source_density=source_density,
            neighbor_ids=neighbor_ids,
            sibling_ids=sibling_ids,
            conflict_score=max(float(payload.get("conflict_score", 0.0) or 0.0), 0.0),
        )


class AnimaCachedDataset(Dataset):
    """Dataset backed by Anima latent/text cache files."""

    def __init__(
        self,
        data_dir: str | Path,
        latent_crop_size: int = 0,
        text_token_limit: int = 0,
        fixed_text_tokens: int = 0,
        fixed_visual_tokens: int = 0,
        fixed_qwen3_tokens: int = 0,
        fixed_t5_tokens: int = 0,
        caption_extension: str = ".txt",
        shuffle_caption: bool = False,
        shuffle_caption_tags_only: bool = False,
        keep_tokens: int = 0,
        keep_tokens_separator: str = "",
        weighted_captions: bool = False,
        schema: Optional[AnimaCacheSchema] = None,
        enable_bucket: bool = False,
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
        # Concept Geometry MVP (geometry-aware curriculum)
        concept_geometry_enabled: bool = False,
        concept_geometry_path: str = "",
        concept_geometry_sampler_mode: str = "density_curriculum",
        concept_geometry_loss_weighting: bool = False,
        concept_geometry_density_power: float = 1.0,
        concept_geometry_seed: int = 42,
        concept_geometry_total_epochs: int = 1,
        concept_geometry_total_steps: int = 0,
        benchmark_data_wait_stall_ms: float = 0.0,
        **legacy_geometry_kwargs: Any,
    ):
        """Initialize Anima cached dataset.

        Parameters
        ----------
        data_dir : str | Path
            Directory containing cache files.
        latent_crop_size : int
            Latent spatial crop size (0 = no crop).
        text_token_limit : int
            Text token limit for consumption (0 = no limit).
        fixed_text_tokens : int
            Fixed text token count for static shape.
        fixed_visual_tokens : int
            Fixed visual token count for static shape.
        fixed_qwen3_tokens : int
            Fixed Qwen3 token count for static shape.
        fixed_t5_tokens : int
            Fixed T5 token count for static shape.
        caption_extension : str
            Caption file extension.
        weighted_captions : bool
            Enable weighted caption parsing.
        schema : Optional[AnimaCacheSchema]
            Cache validation schema.
        enable_bucket : bool
            Enable resolution-based bucketing.
        cache_mmap : bool
            Use memory-mapped file loading for .npz files.
        cache_lazy : bool
            Use lazy loading for .safetensors files.
        file_handle_cache_size : int
            Maximum number of open file handles to cache.
        """
        if legacy_geometry_kwargs:
            concept_geometry_enabled = legacy_geometry_kwargs.pop("h_lora_enabled", concept_geometry_enabled)
            concept_geometry_path = legacy_geometry_kwargs.pop("h_lora_geometry_path", concept_geometry_path)
            concept_geometry_sampler_mode = legacy_geometry_kwargs.pop("h_lora_sampler_mode", concept_geometry_sampler_mode)
            concept_geometry_loss_weighting = legacy_geometry_kwargs.pop("h_lora_loss_weighting", concept_geometry_loss_weighting)
            concept_geometry_density_power = legacy_geometry_kwargs.pop("h_lora_density_power", concept_geometry_density_power)
            concept_geometry_seed = legacy_geometry_kwargs.pop("h_lora_seed", concept_geometry_seed)
            concept_geometry_total_epochs = legacy_geometry_kwargs.pop("h_lora_total_epochs", concept_geometry_total_epochs)
            concept_geometry_total_steps = legacy_geometry_kwargs.pop("h_lora_total_steps", concept_geometry_total_steps)
            if legacy_geometry_kwargs:
                unknown = ", ".join(sorted(legacy_geometry_kwargs))
                raise TypeError(f"Unexpected AnimaCachedDataset keyword argument(s): {unknown}")

        self.data_dir = Path(data_dir)
        self.latent_crop_size = max(int(latent_crop_size or 0), 0)
        self.text_token_limit = max(int(text_token_limit or 0), 0)
        self.fixed_text_tokens = max(int(fixed_text_tokens or 0), 0)
        self.fixed_visual_tokens = max(int(fixed_visual_tokens or 0), 0)
        self.fixed_qwen3_tokens = max(int(fixed_qwen3_tokens or 0), 0)
        self.fixed_t5_tokens = max(int(fixed_t5_tokens or 0), 0)
        self.caption_extension = (
            str(caption_extension or ".txt")
            if str(caption_extension or ".txt").startswith(".")
            else f".{caption_extension}"
        )
        self.shuffle_caption = bool(shuffle_caption)
        self.shuffle_caption_tags_only = bool(shuffle_caption_tags_only)
        self.keep_tokens = max(int(keep_tokens or 0), 0)
        self.keep_tokens_separator = str(keep_tokens_separator or "")
        self.weighted_captions = bool(weighted_captions)
        self.schema = schema or AnimaCacheSchema()
        self.enable_bucket = bool(enable_bucket)
        self.cache_mmap = bool(cache_mmap)
        self.cache_lazy = bool(cache_lazy)
        self.file_handle_cache_size = max(int(file_handle_cache_size), 1)
        self.lossless_cache_sidecar_config = LosslessCacheDatasetAdapterConfig(
            enabled=bool(lossless_cache_sidecar_enabled),
            strict=bool(lossless_cache_sidecar_strict),
            sidecar_suffix=str(lossless_cache_sidecar_suffix or ".lxcs"),
        )
        self.lossless_cache_sidecar_last_report: Dict[str, Any] = {}
        self._current_epoch = 0
        self._current_step = 0
        self.concept_geometry_enabled = bool(concept_geometry_enabled)
        self.concept_geometry_loss_weighting = bool(concept_geometry_loss_weighting)
        self.concept_geometry_sampler_mode = self._normalize_concept_geometry_sampler_mode(concept_geometry_sampler_mode)
        self.concept_geometry_density_power = max(float(concept_geometry_density_power or 1.0), 0.0)
        self.concept_geometry_seed = int(concept_geometry_seed or 42)
        self.concept_geometry_total_epochs = max(int(concept_geometry_total_epochs or 1), 1)
        self.concept_geometry_total_steps = max(int(concept_geometry_total_steps or 0), 0)
        self.concept_geometry_path = self._resolve_concept_geometry_path(concept_geometry_path)
        self._concept_geometry_attached_count = 0
        self._concept_geometry_stage_counts: Dict[str, int] = {"core": 0, "mid": 0, "edge": 0}
        self._concept_geometry_concept_groups: set[str] = set()
        self._concept_geometry_structure_scale = 1.0
        self._benchmark_data_wait_stall_seconds = max(float(benchmark_data_wait_stall_ms or 0.0), 0.0) / 1000.0
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
                "Caption variants are enabled but AnimaCachedDataset uses pre-computed text embeddings. "
                "Variants will be ignored. To use caption variants, train without cache or rebuild cache "
                "with each variant separately."
            )
        if caption_source_mix_enabled:
            logger.warning(
                "Structured Tag/NL caption mixing is enabled for AnimaCachedDataset. "
                "It will use caption_variant_* text caches when present; old caches fall back to the base text embedding."
            )

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Anima cache data_dir does not exist: {self.data_dir}")
        self.samples = self._discover_samples()
        if not self.samples:
            raise ValueError(f"No paired Anima cache samples found in {self.data_dir}")
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
        self._concept_geometry = self._load_concept_geometry()
        bucket_start = time.perf_counter()
        self._bucket_indices = self._build_bucket_indices() if self.enable_bucket else None
        self._bucket_build_profile = {
            "enabled": self.enable_bucket,
            "seconds": round(time.perf_counter() - bucket_start, 6) if self.enable_bucket else 0.0,
            "bucket_count": len(self._bucket_indices or {}),
            "sample_count": len(self.samples),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        if self._benchmark_data_wait_stall_seconds > 0.0:
            time.sleep(self._benchmark_data_wait_stall_seconds)
        sample = self.samples[index]
        latents, cached_loss_mask = self._load_latents(sample.latent_path)
        text = self._load_text(sample.text_path)
        caption, caption_weight = self._load_caption(sample)
        item: Dict[str, object] = {
            "latents": latents,
            "encoder_hidden_states": text["encoder_hidden_states"],
            "attention_mask": text.get("attention_mask"),
            "t5_input_ids": text.get("t5_input_ids"),
            "t5_attention_mask": text.get("t5_attention_mask"),
            "captions": caption or sample.stem,
            "caption_weight": caption_weight,
            "sample_id": sample.sample_id,
        }
        geometry = self._get_concept_geometry(sample)
        if geometry is not None:
            item["geometry_stage"] = geometry.stage
            item["geometry_density"] = geometry.density
            item["geometry_radius"] = geometry.radius
            item["geometry_concept_group"] = geometry.concept_group
            item["geometry_path_depth"] = geometry.path_depth
            if self.concept_geometry_loss_weighting:
                item["geometry_weight"] = self._compute_concept_geometry_loss_weight(geometry)
        if "qwen3_hidden_states" in text:
            item["qwen3_hidden_states"] = text["qwen3_hidden_states"]
        if "qwen3_attention_mask" in text:
            item["qwen3_attention_mask"] = text["qwen3_attention_mask"]
        if cached_loss_mask is not None:
            item["loss_mask"] = cached_loss_mask
        elif sample.loss_mask_path is not None:
            loss_mask = self._load_loss_mask(sample.loss_mask_path, latents)
            if loss_mask is not None:
                item["loss_mask"] = loss_mask
        if self.schema.require_loss_mask and "loss_mask" not in item:
            raise ValueError(f"Anima cache missing loss_mask: {sample.latent_path}")
        return item

    def _discover_samples(self) -> List[AnimaCachedSample]:
        native = _native_cache_index_api()
        if native is not None:
            try:
                subsets = discover_smart_subsets(self.data_dir)
                records = native.discover_anima_cache_samples([str(subset.root) for subset in subsets])
                discovered_native = [
                    (
                        str(record.get("stem") or ""),
                        Path(str(record.get("root_path") or "")),
                        Path(str(record.get("text_path") or "")),
                        Path(str(record.get("latent_path") or "")),
                    )
                    for record in records
                    if isinstance(record, dict)
                    and str(record.get("stem") or "")
                    and str(record.get("root_path") or "")
                    and str(record.get("text_path") or "")
                    and str(record.get("latent_path") or "")
                ]
                if discovered_native:
                    id_map = assign_stable_sample_ids([(stem, root) for stem, root, _text, _latent in discovered_native], self.data_dir)
                    return [
                        AnimaCachedSample(
                            stem=stem,
                            sample_id=id_map[(stem, root)],
                            latent_path=latent_path,
                            text_path=text_path,
                            caption_path=self._resolve_caption_path(root, stem),
                            loss_mask_path=self._resolve_loss_mask_path(root, stem),
                        )
                        for stem, root, text_path, latent_path in sorted(discovered_native, key=lambda item: (str(item[1]), item[0]))
                    ]
            except Exception:
                logger.debug("Native Anima cache index failed; falling back to Python", exc_info=True)

        discovered: List[tuple[str, Path, Path, Path]] = []
        for subset in discover_smart_subsets(self.data_dir):
            text_by_stem: Dict[str, Path] = {}
            for suffix in ("_anima_te.npz", "_anima_te.safetensors", "_anima_te.pt"):
                for path in subset.root.glob(f"*{suffix}"):
                    stem = path.name[: -len(suffix)]
                    if stem not in text_by_stem:
                        text_by_stem[stem] = path

            for stem, text_path in sorted(text_by_stem.items()):
                latent_candidates = []
                for ext in (".npz", ".safetensors", ".pt"):
                    latent_candidates.extend(subset.root.glob(f"{stem}_*_anima{ext}"))
                latent_candidates.sort(key=lambda p: p.stat().st_mtime)
                if not latent_candidates:
                    continue
                discovered.append((stem, subset.root, text_path, latent_candidates[0]))

        id_map = assign_stable_sample_ids([(stem, root) for stem, root, _text, _latent in discovered], self.data_dir)
        samples: List[AnimaCachedSample] = []
        for stem, root, text_path, latent_path in sorted(discovered, key=lambda item: (str(item[1]), item[0])):
            samples.append(
                AnimaCachedSample(
                    stem=stem,
                    sample_id=id_map[(stem, root)],
                    latent_path=latent_path,
                    text_path=text_path,
                    caption_path=self._resolve_caption_path(root, stem),
                    loss_mask_path=self._resolve_loss_mask_path(root, stem),
                )
            )
        return samples

    def _resolve_caption_path(self, root: Path, stem: str) -> Optional[Path]:
        return resolve_caption_path(root, stem, self.caption_extension)

    def _resolve_loss_mask_path(self, root: Path, stem: str) -> Optional[Path]:
        for suffix in ("_mask.png", "_mask.jpg", "_mask.jpeg", "_alpha.png"):
            candidate = root / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def _normalize_concept_geometry_sampler_mode(self, mode: str) -> str:
        normalized = str(mode or "density_curriculum").strip().lower().replace("-", "_")
        if normalized in {"curriculum", "density", "density_curriculum", "concept_batch"}:
            return normalized
        return "density_curriculum"

    def _resolve_concept_geometry_path(self, value: str) -> Path:
        raw = str(value or "").strip()
        if not raw:
            modern = self.data_dir / "concept_geometry.json"
            return modern if modern.is_file() else self.data_dir / "h_lora_geometry.json"
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.data_dir / candidate
        return candidate

    def _load_concept_geometry(self) -> Dict[str, ConceptGeometryEntry]:
        if not self.concept_geometry_enabled:
            return {}
        if not self.concept_geometry_path.is_file():
            logger.warning(
                "[concept-geometry] metadata not found at %s; sampler will be disabled.",
                self.concept_geometry_path,
            )
            self.concept_geometry_enabled = False
            return {}
        try:
            payload = json.loads(self.concept_geometry_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "[concept-geometry] failed to load geometry metadata from %s: %s; sampler will be disabled.",
                self.concept_geometry_path,
                exc,
            )
            self.concept_geometry_enabled = False
            return {}

        samples_payload = payload.get("samples", payload) if isinstance(payload, dict) else {}
        geometry: Dict[str, ConceptGeometryEntry] = {}
        if isinstance(samples_payload, dict):
            for stem, sample_payload in samples_payload.items():
                if isinstance(sample_payload, dict):
                    geometry[str(stem)] = ConceptGeometryEntry.from_payload(sample_payload)
        if not geometry:
            logger.warning(
                "[concept-geometry] metadata at %s had no usable samples; sampler will be disabled.",
                self.concept_geometry_path,
            )
            self.concept_geometry_enabled = False
            return {}

        stage_counts = {"core": 0, "mid": 0, "edge": 0}
        attached = 0
        concept_groups: set[str] = set()
        for sample in self.samples:
            entry = geometry.get(sample.sample_id) or geometry.get(sample.stem)
            if entry is None:
                continue
            attached += 1
            stage_counts[entry.stage] = stage_counts.get(entry.stage, 0) + 1
            if entry.concept_group:
                concept_groups.add(entry.concept_group)
        self._concept_geometry_attached_count = attached
        self._concept_geometry_stage_counts = dict(stage_counts)
        self._concept_geometry_concept_groups = set(concept_groups)
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        meta_version = int(meta.get("geometry_version", 1) or 1) if isinstance(meta, dict) else 1
        feature_sources = meta.get("feature_sources", []) if isinstance(meta, dict) else []
        fallback_reasons = meta.get("fallback_reasons", []) if isinstance(meta, dict) else []
        self._concept_geometry_structure_scale = self._compute_concept_geometry_structure_scale(
            attached=attached,
            stage_counts=stage_counts,
            concept_groups=concept_groups,
        )
        logger.info(
            "[concept-geometry] loaded from %s: version=%s backend=%s sources=%s attached=%s/%s "
            "core=%s mid=%s edge=%s groups=%s structure_scale=%.2f mode=%s",
            self.concept_geometry_path,
            meta_version,
            meta.get("backend_resolved", "") if isinstance(meta, dict) else "",
            ",".join(str(item) for item in feature_sources) if isinstance(feature_sources, list) else str(feature_sources),
            attached,
            len(self.samples),
            stage_counts.get("core", 0),
            stage_counts.get("mid", 0),
            stage_counts.get("edge", 0),
            len(concept_groups),
            self._concept_geometry_structure_scale,
            self.concept_geometry_sampler_mode,
        )
        for reason in fallback_reasons if isinstance(fallback_reasons, list) else []:
            logger.info("[concept-geometry-fallback] %s", reason)
        return geometry

    def _get_concept_geometry(self, sample: AnimaCachedSample) -> Optional[ConceptGeometryEntry]:
        if not self.concept_geometry_enabled:
            return None
        return self._concept_geometry.get(sample.sample_id) or self._concept_geometry.get(sample.stem)

    def is_concept_geometry_enabled(self) -> bool:
        return self.concept_geometry_enabled and bool(self._concept_geometry)

    def is_h_lora_enabled(self) -> bool:
        """Legacy alias for callers that have not migrated yet."""
        return self.is_concept_geometry_enabled()

    def set_current_epoch(self, epoch: int) -> None:
        self._current_epoch = max(int(epoch), 0)

    def set_global_step(self, step: int) -> None:
        self._current_step = max(int(step or 0), 0)

    def set_concept_geometry_total_steps(self, total_steps: int) -> None:
        self.concept_geometry_total_steps = max(int(total_steps or 0), 0)

    def set_h_lora_total_steps(self, total_steps: int) -> None:
        """Legacy alias for callers that have not migrated yet."""
        self.set_concept_geometry_total_steps(total_steps)

    def _concept_geometry_progress(self) -> float:
        if self.concept_geometry_total_steps > 1:
            return min(max(self._current_step / max(self.concept_geometry_total_steps - 1, 1), 0.0), 1.0)
        if self.concept_geometry_total_epochs <= 1:
            return 0.0
        return min(max(self._current_epoch / max(self.concept_geometry_total_epochs - 1, 1), 0.0), 1.0)

    def _compute_concept_geometry_structure_scale(
        self,
        *,
        attached: int,
        stage_counts: Dict[str, int],
        concept_groups: set[str],
    ) -> float:
        if attached <= 0:
            return 1.0
        populated_stages = sum(1 for value in stage_counts.values() if int(value or 0) > 0)
        group_count = len(concept_groups)
        scale = 1.0
        if group_count <= 1:
            scale *= 0.45
        elif group_count == 2:
            scale *= 0.70
        if populated_stages <= 2:
            scale *= 0.80
        if stage_counts.get("mid", 0) <= 0:
            scale *= 0.90
        return float(min(max(scale, 0.25), 1.0))

    def _blend_concept_geometry_weight(self, weight: float) -> float:
        scale = min(max(float(self._concept_geometry_structure_scale or 1.0), 0.0), 1.0)
        return float(1.0 + (float(weight) - 1.0) * scale)

    def _concept_geometry_stage_sampling_weights(self) -> Dict[str, float]:
        progress = self._concept_geometry_progress()
        return {
            "core": self._blend_concept_geometry_weight(1.75 - 1.00 * progress),
            "mid": 1.00,
            "edge": self._blend_concept_geometry_weight(0.35 + 1.10 * progress),
        }

    def _concept_geometry_stage_loss_weights(self) -> Dict[str, float]:
        progress = self._concept_geometry_progress()
        return {
            "core": self._blend_concept_geometry_weight(1.10 - 0.15 * progress),
            "mid": 1.00,
            "edge": self._blend_concept_geometry_weight(0.90 + 0.20 * progress),
        }

    def _density_factor(self, density: float) -> float:
        stabilized = min(max(float(density), 0.0), 1.0)
        factor = 0.5 + 0.5 * (stabilized ** max(self.concept_geometry_density_power, 1e-6))
        return self._blend_concept_geometry_weight(factor)

    def _compute_concept_geometry_loss_weight(self, geometry: ConceptGeometryEntry) -> float:
        weight = geometry.loss_weight
        if self.concept_geometry_sampler_mode in {"curriculum", "density_curriculum", "concept_batch"}:
            stage_weights = self._concept_geometry_stage_loss_weights()
            weight *= stage_weights.get(geometry.stage, 1.0)
        if self.concept_geometry_sampler_mode in {"density", "density_curriculum", "concept_batch"}:
            weight *= self._density_factor(geometry.density)
        return float(max(weight, 1e-6))

    def get_concept_geometry_sampling_weights(self, global_indices: Optional[List[int]] = None) -> List[float]:
        if not self.is_concept_geometry_enabled():
            return []
        stage_weights = self._concept_geometry_stage_sampling_weights()
        indices = global_indices if global_indices is not None else list(range(len(self.samples)))
        weights: List[float] = []
        for index in indices:
            sample = self.samples[index]
            geometry = self._concept_geometry.get(sample.sample_id) or self._concept_geometry.get(sample.stem)
            if geometry is None:
                weights.append(1.0)
                continue
            weight = 1.0
            if self.concept_geometry_sampler_mode in {"curriculum", "density_curriculum", "concept_batch"}:
                weight *= stage_weights.get(geometry.stage, 1.0)
            if self.concept_geometry_sampler_mode in {"density", "density_curriculum", "concept_batch"}:
                weight *= self._density_factor(geometry.density)
            weights.append(float(max(weight, 1e-6)))
        return weights

    def get_h_lora_sampling_weights(self, global_indices: Optional[List[int]] = None) -> List[float]:
        """Legacy alias for callers that have not migrated yet."""
        return self.get_concept_geometry_sampling_weights(global_indices)

    def get_concept_geometry_concept_batch(self, global_indices: List[int], rng: Any) -> List[int]:
        if self.concept_geometry_sampler_mode != "concept_batch" or len(global_indices) <= 1:
            return []
        index_set = set(int(index) for index in global_indices)
        stem_to_index = {sample.stem: idx for idx, sample in enumerate(self.samples) if idx in index_set}
        stem_to_index.update({sample.sample_id: idx for idx, sample in enumerate(self.samples) if idx in index_set})
        if not stem_to_index:
            return []

        weights = self.get_concept_geometry_sampling_weights(global_indices)
        if len(weights) != len(global_indices):
            weights = [1.0] * len(global_indices)
        total = sum(max(float(weight), 1e-6) for weight in weights)
        pick = rng.random() * total
        cursor = 0.0
        anchor = int(global_indices[0])
        for index, weight in zip(global_indices, weights):
            cursor += max(float(weight), 1e-6)
            if cursor >= pick:
                anchor = int(index)
                break

        batch = [anchor]
        anchor_sample = self.samples[anchor]
        anchor_entry = self._concept_geometry.get(anchor_sample.sample_id) or self._concept_geometry.get(anchor_sample.stem)
        if anchor_entry is None or anchor_entry.geometry_version < 2:
            return []

        def add_from(stems: tuple[str, ...]) -> None:
            candidates = [stem_to_index[stem] for stem in stems if stem in stem_to_index and stem_to_index[stem] not in batch]
            if candidates:
                batch.append(rng.choice(candidates))

        add_from(anchor_entry.neighbor_ids)
        add_from(anchor_entry.sibling_ids)
        conflict_stems = tuple(stem for stem in anchor_entry.neighbor_ids if stem not in anchor_entry.sibling_ids)
        add_from(conflict_stems)

        if len(batch) < min(2, len(global_indices)):
            return []
        remaining = [idx for idx in global_indices if int(idx) not in batch]
        while remaining and len(batch) < len(global_indices):
            pick_idx = rng.randrange(len(remaining))
            batch.append(int(remaining.pop(pick_idx)))
        return batch

    def get_h_lora_concept_batch(self, global_indices: List[int], rng: Any) -> List[int]:
        """Legacy alias for callers that have not migrated yet."""
        return self.get_concept_geometry_concept_batch(global_indices, rng)

    def get_concept_geometry_summary(self) -> Dict[str, Any]:
        if not self.is_concept_geometry_enabled():
            return {
                "enabled": False,
                "geometry_path": str(self.concept_geometry_path),
            }
        stage_counts = {"core": 0, "mid": 0, "edge": 0}
        attached = 0
        for sample in self.samples:
            geometry = self._concept_geometry.get(sample.sample_id) or self._concept_geometry.get(sample.stem)
            if geometry is None:
                continue
            attached += 1
            stage_counts[geometry.stage] = stage_counts.get(geometry.stage, 0) + 1
        return {
            "enabled": True,
            "geometry_path": str(self.concept_geometry_path),
            "sampler_mode": self.concept_geometry_sampler_mode,
            "loss_weighting": self.concept_geometry_loss_weighting,
            "sample_count": len(self.samples),
            "attached_count": attached,
            "stage_counts": stage_counts,
            "concept_group_count": len(self._concept_geometry_concept_groups),
            "structure_scale": self._concept_geometry_structure_scale,
            "counts": stage_counts,
        }

    def get_h_lora_summary(self) -> Dict[str, Any]:
        """Legacy alias for callers that have not migrated yet."""
        return self.get_concept_geometry_summary()

    def _load_loss_mask(self, path: Path, latents: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            from PIL import Image as PILImage
            img = PILImage.open(path).convert("L")
            mask = torch.from_numpy(np.asarray(img)).float() / 255.0
            latent_h, latent_w = int(latents.shape[-2]), int(latents.shape[-1])
            if mask.shape[0] != latent_h or mask.shape[1] != latent_w:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(latent_h, latent_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            return mask
        except Exception:
            return None

    def _build_bucket_indices(self) -> Dict[str, List[int]]:
        """Group sample indices by latent spatial resolution for bucket-based batching."""
        buckets: Dict[str, List[int]] = {}
        for idx, sample in enumerate(self.samples):
            shape = self._latent_shape_for_sample(sample)
            if shape:
                resolution_key = f"{shape[-2]}x{shape[-1]}"
            else:
                resolution_key = "unknown"
            buckets.setdefault(resolution_key, []).append(idx)
        return buckets

    def get_bucket_indices(self) -> Optional[Dict[str, List[int]]]:
        """Return bucket index map, or None if bucketing is disabled."""
        return self._bucket_indices

    def get_token_bucket_summary(self) -> Dict[str, object]:
        """Return native DiT visual-token bucket stats without loading tensors.

        Anima patchifies 2x2 latent cells, so visual tokens are
        ``latent_h // 2 * latent_w // 2``.  This summary is used by the
        trainer manifest and by no-pad compile planning.
        """
        buckets: Dict[str, Dict[str, object]] = {}
        for idx, sample in enumerate(self.samples):
            shape = self._latent_shape_for_sample(sample)
            if not shape:
                key = "unknown"
                latent_h = latent_w = visual_tokens = 0
            else:
                latent_h = int(shape[-2])
                latent_w = int(shape[-1])
                if self.latent_crop_size > 0:
                    latent_h = min(latent_h, self.latent_crop_size)
                    latent_w = min(latent_w, self.latent_crop_size)
                if self.fixed_visual_tokens > 0:
                    visual_tokens = int(self.fixed_visual_tokens)
                    key = f"fixed:{visual_tokens}"
                else:
                    visual_tokens = (latent_h // 2) * (latent_w // 2)
                    key = f"{visual_tokens}:{latent_h}x{latent_w}"
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
            "family": "anima",
            "mode": "fixed_pad" if self.fixed_visual_tokens > 0 else "no_pad",
            "bucket_count": len(buckets),
            "buckets": buckets,
        }

    def get_cache_metadata_summary(self) -> Dict[str, object]:
        """Return startup/cache metadata profile for manifest diagnostics."""
        return {
            "family": "anima",
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
        return build_cache_shape_index((sample.latent_path for sample in self.samples))

    def _load_cache_metadata(self) -> Dict[str, Dict[str, Any]]:
        metadata_path = next((self.data_dir / name for name in ANIMA_CACHE_METADATA_FILENAMES if (self.data_dir / name).is_file()), None)
        if metadata_path is None:
            return {}
        self._cache_metadata_path = metadata_path
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._cache_metadata_error = f"{type(exc).__name__}: {exc}"
            logger.debug("Anima cache metadata ignored: %s", exc)
            return {}
        if not isinstance(payload, dict):
            self._cache_metadata_error = "metadata payload is not an object"
            return {}
        family = str(payload.get("family", "anima") or "anima").strip().lower()
        if family not in {"anima", "qwen_image", "native_anima"}:
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
                str(record.get("sample_id", "") or ""),
                str(record.get("latent_path", "") or ""),
                str(record.get("cache_path", "") or ""),
            ]
            for key in keys:
                normalized = key.replace("\\", "/").strip()
                if normalized:
                    index[normalized] = record
        if index:
            logger.info("Anima cache metadata loaded: %s (%d records)", metadata_path.name, len(index))
        return index

    def _metadata_record_for_sample(self, sample: AnimaCachedSample) -> Optional[Dict[str, Any]]:
        rel_path = self._relative_metadata_path(sample.latent_path)
        for key in (rel_path, sample.latent_path.name, sample.stem, sample.sample_id):
            record = self._cache_metadata.get(key)
            if record is not None:
                return record
        return None

    def _native_shape_record_for_sample(self, sample: AnimaCachedSample) -> Optional[Dict[str, Any]]:
        rel_path = self._relative_metadata_path(sample.latent_path)
        for key in (str(sample.latent_path), sample.latent_path.as_posix(), rel_path, sample.latent_path.name):
            record = self._native_shape_metadata_index.get(key) if self._native_shape_metadata_index else None
            if record is not None:
                return record
        return None

    def _latent_shape_for_sample(self, sample: AnimaCachedSample) -> Optional[tuple]:
        cache_key = self._relative_metadata_path(sample.latent_path)
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
                shape = shape_from_record(native_record)
            except Exception:
                shape = None
            if shape is not None:
                self._shape_profile["metadata_shape_hits"] += 1
                self._shape_cache[cache_key] = shape
                return shape
        self._shape_profile["metadata_shape_misses"] += 1
        try:
            data = self._load_cache_file(sample.latent_path)
            keys = self._cache_keys(data)
            latent_keys = [k for k in keys if k.startswith("latents_")]
            if not latent_keys:
                self._shape_profile["fallback_shape_failures"] += 1
                self._shape_cache[cache_key] = None
                return None
            shape = self._array_shape(data, latent_keys[0])
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

    def _load_cache_file(self, path: Path) -> Dict[str, object]:
        """Load a cache file, dispatching on suffix (.npz / .safetensors / .pt)."""
        suffix = path.suffix.lower()
        sidecar_arrays, sidecar_report = load_lossless_cache_arrays_for_dataset(
            path,
            config=self.lossless_cache_sidecar_config,
        )
        self.lossless_cache_sidecar_last_report = sidecar_report
        if sidecar_arrays is not None:
            return sidecar_arrays

        # Check file handle cache for lazy loading
        if self.cache_lazy and path in self._file_handle_cache:
            self._file_handle_cache.move_to_end(path)
            return self._file_handle_cache[path]

        if suffix == ".npz":
            if self.cache_mmap:
                data = np.load(str(path), mmap_mode="r")  # NpzFile with mmap
            else:
                data = np.load(str(path))  # NpzFile — has .files attribute
        elif suffix == ".safetensors":
            from safetensors.torch import load_file
            # safetensors supports lazy loading by default
            data = load_file(str(path))
        elif suffix in (".pt", ".pth"):
            data = torch.load(str(path), map_location="cpu", weights_only=True)
        else:
            raise ValueError(f"Unsupported cache format: {suffix}")

        # Cache file handle if lazy loading is enabled
        if self.cache_lazy:
            self._update_file_handle_cache(path, data)

        return data

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

    @staticmethod
    def _cache_keys(data: Dict[str, object]) -> List[str]:
        """Return available keys from any cache format (NpzFile or plain dict)."""
        if hasattr(data, "files"):
            return list(data.files)
        return list(data.keys())

    def _load_latents(self, path: Path) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        data = self._load_cache_file(path)
        keys = self._cache_keys(data)
        latent_keys = sorted(
            (key for key in keys if key.startswith("latents_")),
            key=lambda key: int(np.prod(self._array_shape(data, key)[-2:])),
        )
        if not latent_keys:
            raise ValueError(f"No latents_* arrays found in {path}")
        latents = self._tensor_from_cache(data, latent_keys[0]).float()
        loss_mask = self._tensor_from_cache(data, "loss_mask").float() if "loss_mask" in keys else None
        if loss_mask is not None and loss_mask.dim() == 3 and loss_mask.shape[0] == 1:
            loss_mask = loss_mask[0]
        if self.latent_crop_size > 0:
            crop = self.latent_crop_size
            latents = latents[:, :crop, :crop].contiguous()
            if loss_mask is not None:
                loss_mask = loss_mask[:crop, :crop].contiguous()
        if self.fixed_visual_tokens > 0:
            original_h = int(latents.shape[-2])
            original_w = int(latents.shape[-1])
            latents = _pad_latents_to_visual_tokens(latents, self.fixed_visual_tokens)
            if loss_mask is not None and (loss_mask.shape[-2] != latents.shape[-2] or loss_mask.shape[-1] != latents.shape[-1]):
                padded_mask = loss_mask.new_zeros((latents.shape[-2], latents.shape[-1]))
                padded_mask[:original_h, :original_w] = loss_mask[:original_h, :original_w]
                loss_mask = padded_mask
        return latents, loss_mask

    def _load_text(self, path: Path) -> Dict[str, Optional[torch.Tensor]]:
        data = self._load_cache_file(path)
        keys = self._cache_keys(data)
        if "prompt_embeds" not in keys:
            raise ValueError(f"No prompt_embeds array found in {path}")
        prefix = self._select_caption_source_variant_prefix(keys, path)
        prompt_key = f"{prefix}prompt_embeds" if prefix else "prompt_embeds"

        # Detect has_loss_mask from schema_version (v2+ caches record this)
        has_loss_mask = False
        if "schema_version" in keys:
            sv = self._tensor_from_cache(data, "schema_version")
            version = int(sv.flatten()[0]) if sv.numel() > 0 else 0
            if version >= 2 and "has_loss_mask" in keys:
                lm_flag = self._tensor_from_cache(data, "has_loss_mask")
                has_loss_mask = bool(lm_flag.flatten()[0])

        result: Dict[str, Optional[torch.Tensor]] = {
            "encoder_hidden_states": self._tensor_from_cache(data, prompt_key).float(),
            "attention_mask": None,
            "t5_input_ids": None,
            "t5_attention_mask": None,
        }
        limit = self.text_token_limit
        if self.fixed_text_tokens > 0:
            limit = min(limit, self.fixed_text_tokens) if limit > 0 else self.fixed_text_tokens
        if limit > 0:
            result["encoder_hidden_states"] = result["encoder_hidden_states"][:limit]
        attn_key = self._prefixed_cache_key(keys, prefix, "attn_mask")
        if attn_key:
            attention_mask = self._tensor_from_cache(data, attn_key).bool()
            result["attention_mask"] = attention_mask[:limit] if limit > 0 else attention_mask
        t5_ids_key = self._prefixed_cache_key(keys, prefix, "t5_input_ids")
        if t5_ids_key:
            input_ids = self._tensor_from_cache(data, t5_ids_key).long()
            result["t5_input_ids"] = input_ids[:limit] if limit > 0 else input_ids
        t5_mask_key = self._prefixed_cache_key(keys, prefix, "t5_attn_mask")
        if t5_mask_key:
            t5_mask = self._tensor_from_cache(data, t5_mask_key).bool()
            result["t5_attention_mask"] = t5_mask[:limit] if limit > 0 else t5_mask
        # Qwen3 secondary conditioning
        q3_hs_key = self._prefixed_cache_key(keys, prefix, "qwen3_hidden_states")
        if q3_hs_key:
            q3_hs = self._tensor_from_cache(data, q3_hs_key).float()
            result["qwen3_hidden_states"] = q3_hs[:limit] if limit > 0 else q3_hs
        q3_mask_key = self._prefixed_cache_key(keys, prefix, "qwen3_attention_mask")
        if q3_mask_key:
            q3_mask = self._tensor_from_cache(data, q3_mask_key).bool()
            result["qwen3_attention_mask"] = q3_mask[:limit] if limit > 0 else q3_mask

        # Apply static padding if fixed token counts are set
        if self.fixed_text_tokens > 0 or self.fixed_qwen3_tokens > 0 or self.fixed_t5_tokens > 0:
            try:
                from .fixed_token_padding import (
                    AnimaMultiEncoderPaddingConfig,
                    apply_anima_multi_encoder_padding,
                )
            except ImportError:
                from fixed_token_padding import (
                    AnimaMultiEncoderPaddingConfig,
                    apply_anima_multi_encoder_padding,
                )
            padding_config = AnimaMultiEncoderPaddingConfig(
                fixed_text_tokens=self.fixed_text_tokens,
                fixed_qwen3_tokens=self.fixed_qwen3_tokens,
                fixed_t5_tokens=self.fixed_t5_tokens,
                fixed_visual_tokens=self.fixed_visual_tokens,
                warn_on_truncation=True,
            )
            result = apply_anima_multi_encoder_padding(result, padding_config)

        return result

    def _select_caption_source_variant_prefix(self, keys: List[str], path: Path) -> str:
        if not self.caption_source_mix.enabled:
            return ""
        available = {
            name
            for name in ("nl", "tag", "trigger_only", "empty")
            if f"caption_variant_{name}_prompt_embeds" in keys
        }
        if not available:
            if not self._caption_source_mix_missing_variant_warned:
                logger.warning(
                    "Anima caption_source_mix enabled but no caption_variant_* keys found in %s; "
                    "rebuild text caches to enable cache-first mixing.",
                    path.name,
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
            if "empty" in available and source == "empty":
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
    def _array_shape(data: Dict[str, object], key: str) -> tuple:
        """Get shape of an array regardless of cache format."""
        val = data[key]
        if isinstance(val, torch.Tensor):
            return tuple(val.shape)
        if isinstance(val, np.ndarray):
            return val.shape
        return np.array(val).shape

    @staticmethod
    def _tensor_from_cache(data: Dict[str, object], key: str) -> torch.Tensor:
        """Extract a tensor from any cache format."""
        val = data[key]
        if isinstance(val, torch.Tensor):
            return val
        return torch.from_numpy(np.asarray(val))

    def _load_caption(self, sample: AnimaCachedSample) -> tuple[str, float]:
        if sample.caption_path is None:
            return sample.stem, 1.0
        try:
            raw_caption = sample.caption_path.read_text(encoding="utf-8").strip()
        except Exception:
            return sample.stem, 1.0
        structured = json_caption_to_training_parts(raw_caption)
        raw_caption = str(structured.get("text") or "")
        if not raw_caption:
            return sample.stem, 1.0
        structured_tags = None
        structured_nl = None
        if self.shuffle_caption_tags_only and structured.get("structured"):
            structured_tags = list(structured.get("tags") or [])
            structured_nl = list(structured.get("nl") or [])
        return self._process_caption(
            raw_caption,
            structured_tags=structured_tags,
            structured_nl=structured_nl,
        )

    def _process_caption(
        self,
        caption: str,
        *,
        structured_tags: Optional[List[str]] = None,
        structured_nl: Optional[List[str]] = None,
    ) -> tuple[str, float]:
        tags = list(structured_tags) if structured_tags is not None else self._split_caption_tags(caption)
        nl_parts = list(structured_nl) if structured_nl else []
        weight = 1.0
        if self.weighted_captions and tags:
            last_tag = tags[-1]
            if ":" in last_tag:
                parts = last_tag.rsplit(":", 1)
                try:
                    weight = float(parts[1])
                    tags = tags[:-1]
                except (ValueError, IndexError):
                    pass
        if self.keep_tokens > 0:
            kept = tags[:self.keep_tokens]
            rest = tags[self.keep_tokens:]
            if self.shuffle_caption:
                import random

                random.shuffle(rest)
            if kept and self.keep_tokens_separator:
                tags = kept + [self.keep_tokens_separator] + rest
            else:
                tags = kept + rest
        elif self.shuffle_caption:
            import random

            random.shuffle(tags)

        if self.shuffle_caption_tags_only and nl_parts:
            final_parts = list(tags)
            if final_parts and self.keep_tokens_separator:
                final_parts.append(self.keep_tokens_separator)
            final_parts.extend(nl_parts)
            normalized = ", ".join(part for part in final_parts if str(part or "").strip()).strip()
            return normalized, weight
        normalized = ", ".join(tags).strip()
        return normalized, weight

    @staticmethod
    def _split_caption_tags(caption: str) -> List[str]:
        if "," in caption:
            return [tag.strip() for tag in caption.split(",") if tag.strip()]
        return [tag.strip() for tag in caption.split() if tag.strip()]


def _pad_latents_to_visual_tokens(latents: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Pad BCHW Anima latents so 2x2 patchification sees a fixed token budget."""
    if latents.dim() != 3:
        raise ValueError("Cached Anima latents must be shaped [channels, height, width]")
    if latents.shape[-2] % 2 or latents.shape[-1] % 2:
        raise ValueError("Cached Anima latent spatial dimensions must be divisible by 2")
    current_tokens = (int(latents.shape[-2]) // 2) * (int(latents.shape[-1]) // 2)
    if current_tokens > target_tokens:
        raise ValueError(
            f"Cached Anima latent has {current_tokens} visual tokens, exceeding fixed target {target_tokens}"
        )
    if current_tokens == target_tokens:
        return latents

    target_side = int(target_tokens**0.5)
    if target_side * target_side != target_tokens:
        raise ValueError("fixed_visual_tokens must be a square token count for this Warehouse padder")
    target_h = target_side * 2
    target_w = target_side * 2
    if latents.shape[-2] > target_h or latents.shape[-1] > target_w:
        raise ValueError(
            f"Cached Anima latent shape {tuple(latents.shape[-2:])} cannot fit fixed token canvas "
            f"{target_h}x{target_w}"
        )
    out = latents.new_zeros((latents.shape[0], target_h, target_w))
    out[:, : latents.shape[-2], : latents.shape[-1]] = latents
    return out


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


def _collate_optional_1d(items: List[Optional[torch.Tensor]], collate_mode: str = "auto") -> Optional[torch.Tensor]:
    present = [item for item in items if item is not None]
    if not present:
        return None
    max_len = max(int(item.shape[0]) for item in present)
    dtype = present[0].dtype
    if _normalize_cached_collate_mode(collate_mode) in {"auto", "pad_sequence"}:
        padded_items = [
            item if item is not None else torch.zeros((0,), dtype=dtype)
            for item in items
        ]
        return pad_sequence(padded_items, batch_first=True, padding_value=0)
    output = torch.zeros((len(items), max_len), dtype=dtype)
    for index, item in enumerate(items):
        if item is not None:
            output[index, : item.shape[0]] = item
    return output


def _collate_latents(items: List[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    channels = int(items[0].shape[0])
    if any(item.dim() != 3 for item in items):
        raise ValueError("Cached Anima latents must be shaped [channels, height, width]")
    if any(int(item.shape[0]) != channels for item in items):
        raise ValueError("Cached Anima latents in a batch must have the same channel count")

    max_h = max(int(item.shape[-2]) for item in items)
    max_w = max(int(item.shape[-1]) for item in items)
    if all(int(item.shape[-2]) == max_h and int(item.shape[-1]) == max_w for item in items):
        return torch.stack(items), None
    if max_h % 2 or max_w % 2:
        raise ValueError("Cached Anima latent batch padding requires even spatial dimensions")

    latents = items[0].new_zeros((len(items), channels, max_h, max_w))
    padding_mask = torch.ones((len(items), 1, max_h, max_w), dtype=torch.bool)
    for index, item in enumerate(items):
        height = int(item.shape[-2])
        width = int(item.shape[-1])
        if height % 2 or width % 2:
            raise ValueError("Cached Anima latent spatial dimensions must be divisible by 2")
        latents[index, :, :height, :width] = item
        padding_mask[index, :, :height, :width] = False
    return latents, padding_mask


def _pad_hidden_and_mask(
    batch: List[Dict[str, object]],
    *,
    fixed_text_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_items: List[torch.Tensor] = []
    mask_items: List[torch.Tensor] = []
    for item in batch:
        hidden = item["encoder_hidden_states"]  # type: ignore[assignment]
        hidden_tensor = hidden.to(dtype=torch.float32)  # type: ignore[union-attr]
        token_count = int(hidden_tensor.shape[0])
        hidden_items.append(hidden_tensor)
        mask = item.get("attention_mask")
        if isinstance(mask, torch.Tensor):
            mask_items.append(mask.to(dtype=torch.bool))
        else:
            mask_items.append(torch.ones((token_count,), dtype=torch.bool))
    encoder_hidden_states = pad_sequence(hidden_items, batch_first=True, padding_value=0.0)
    attention_mask = pad_sequence(mask_items, batch_first=True, padding_value=False)
    target_len = max(int(fixed_text_tokens or 0), int(encoder_hidden_states.shape[1]))
    if target_len > int(encoder_hidden_states.shape[1]):
        text_dim = int(encoder_hidden_states.shape[-1])
        padded_hidden = encoder_hidden_states.new_zeros((len(batch), target_len, text_dim))
        padded_mask = attention_mask.new_zeros((len(batch), target_len))
        padded_hidden[:, : encoder_hidden_states.shape[1]] = encoder_hidden_states
        padded_mask[:, : attention_mask.shape[1]] = attention_mask
        encoder_hidden_states = padded_hidden
        attention_mask = padded_mask
    return encoder_hidden_states, attention_mask


def anima_cached_collate(
    batch: List[Dict[str, object]],
    fixed_text_tokens: int = 0,
    collate_mode: str = "auto",
) -> Dict[str, object]:
    latents, padding_mask = _collate_latents([item["latents"] for item in batch])  # type: ignore[list-item]
    max_text_len = max(int(item["encoder_hidden_states"].shape[0]) for item in batch)  # type: ignore[index]
    if fixed_text_tokens > 0:
        max_text_len = max(max_text_len, int(fixed_text_tokens))
    text_dim = int(batch[0]["encoder_hidden_states"].shape[-1])  # type: ignore[index]
    resolved_mode = _normalize_cached_collate_mode(collate_mode)
    if resolved_mode in {"auto", "pad_sequence"}:
        encoder_hidden_states, attention_mask = _pad_hidden_and_mask(batch, fixed_text_tokens=fixed_text_tokens)
    else:
        encoder_hidden_states = torch.zeros((len(batch), max_text_len, text_dim), dtype=torch.float32)
        attention_mask = torch.zeros((len(batch), max_text_len), dtype=torch.bool)
        for index, item in enumerate(batch):
            hidden = item["encoder_hidden_states"]  # type: ignore[assignment]
            mask = item.get("attention_mask")  # type: ignore[union-attr]
            token_count = int(hidden.shape[0])
            encoder_hidden_states[index, :token_count] = hidden
            if mask is None:
                attention_mask[index, :token_count] = True
            else:
                attention_mask[index, : mask.shape[0]] = mask

    result: Dict[str, object] = {
        "latents": latents,
        "padding_mask": padding_mask,
        "encoder_hidden_states": encoder_hidden_states,
        "attention_mask": attention_mask,
        "caption_weights": torch.tensor(
            [float(item.get("caption_weight", 1.0)) for item in batch],
            dtype=torch.float32,
        ),
        "t5_input_ids": _collate_optional_1d([item.get("t5_input_ids") for item in batch], collate_mode=resolved_mode),  # type: ignore[arg-type]
        "t5_attention_mask": _collate_optional_1d([item.get("t5_attention_mask") for item in batch], collate_mode=resolved_mode),  # type: ignore[arg-type]
        "captions": [str(item.get("captions", "")) for item in batch],
        "sample_ids": [str(item.get("sample_id", "")) for item in batch],
    }
    if all("geometry_weight" in item for item in batch):
        result["geometry_weights"] = torch.tensor(
            [float(item.get("geometry_weight", 1.0)) for item in batch],
            dtype=torch.float32,
        )
    geometry_stages = [str(item.get("geometry_stage", "")) for item in batch if item.get("geometry_stage")]
    if len(geometry_stages) == len(batch):
        result["geometry_stages"] = geometry_stages
    geometry_groups = [str(item.get("geometry_concept_group", "")) for item in batch]
    if any(geometry_groups):
        result["geometry_concept_groups"] = geometry_groups

    loss_mask_items = [item.get("loss_mask") for item in batch]
    if all(isinstance(m, torch.Tensor) for m in loss_mask_items):
        result_loss_masks: list[torch.Tensor] = loss_mask_items  # type: ignore[assignment]
        max_h_loss = max(int(m.shape[-2]) for m in result_loss_masks)
        max_w_loss = max(int(m.shape[-1]) for m in result_loss_masks)
        loss_mask_batch = result_loss_masks[0].new_zeros(
            (len(result_loss_masks), max_h_loss, max_w_loss)
        )
        for idx, m in enumerate(result_loss_masks):
            loss_mask_batch[idx, : m.shape[-2], : m.shape[-1]] = m
        result["loss_masks"] = loss_mask_batch

    # Qwen3 secondary conditioning — pad to max length
    qwen3_items = [item.get("qwen3_hidden_states") for item in batch]
    qwen3_mask_items = [item.get("qwen3_attention_mask") for item in batch]
    if any(isinstance(t, torch.Tensor) for t in qwen3_items):
        max_q3_len = max(
            int(t.shape[0]) for t in qwen3_items if isinstance(t, torch.Tensor)
        )
        q3_dim = int(next(t for t in qwen3_items if isinstance(t, torch.Tensor)).shape[-1])
        q3_batch = torch.zeros((len(batch), max_q3_len, q3_dim), dtype=torch.float32)
        q3_mask_batch = torch.zeros((len(batch), max_q3_len), dtype=torch.bool)
        for idx, (hs, msk) in enumerate(zip(qwen3_items, qwen3_mask_items)):
            if isinstance(hs, torch.Tensor):
                n = int(hs.shape[0])
                q3_batch[idx, :n] = hs
                if isinstance(msk, torch.Tensor):
                    q3_mask_batch[idx, :msk.shape[0]] = msk
                else:
                    q3_mask_batch[idx, :n] = True
        result["qwen3_hidden_states"] = q3_batch
        result["qwen3_attention_mask"] = q3_mask_batch

    return result


class _AnimaCachedCollator:
    """Pickle-safe collator for DataLoader worker processes."""

    def __init__(self, fixed_text_tokens: int = 0, collate_mode: str = "auto"):
        self.fixed_text_tokens = int(fixed_text_tokens or 0)
        self.collate_mode = _normalize_cached_collate_mode(collate_mode)

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, object]:
        return anima_cached_collate(
            batch,
            fixed_text_tokens=self.fixed_text_tokens,
            collate_mode=self.collate_mode,
        )


def _attach_native_cache_prefetch_shadow_adapter_if_enabled(
    dataloader: DataLoader,
    dataset: "AnimaCachedDataset",
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


def create_anima_cached_dataloader(
    dataset: AnimaCachedDataset,
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
    """Create DataLoader for AnimaCachedDataset with memory optimization support.

    Parameters
    ----------
    dataset : AnimaCachedDataset
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
    rebuild_route = str(getattr(dataset, "dataloader_rebuild_route", "anima_cached") or "anima_cached")
    multiprocessing_safe = bool(getattr(dataset, "dataloader_multiprocessing_safe", True))
    if not multiprocessing_safe:
        num_workers = 0
        persistent_workers = False
        prefetch_factor = None
    dl_kwargs: Dict[str, object] = {
        "num_workers": num_workers,
        "collate_fn": _AnimaCachedCollator(dataset.fixed_text_tokens, collate_mode=collate_mode),
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = prefetch_factor
    mutable_descriptor_fields = (
        ("pin_memory",)
        if not multiprocessing_safe
        else ("num_workers", "prefetch_factor", "pin_memory", "persistent_workers")
    )

    def rebuild_factory(descriptor):
        workers = max(int(descriptor.get("num_workers", num_workers) or 0), 0)
        return create_anima_cached_dataloader(
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
            route=rebuild_route,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            uses_batch_sampler=getattr(attached, "batch_sampler", None) is not None,
            rebuild_factory=rebuild_factory,
            mutable_descriptor_fields=mutable_descriptor_fields,
        )
        attached = attach_dataloader_batching_contract(
            attached,
            requested_physical_batch_size=batch_size,
        )
        return attach_lulynx_dataloader_data_pipeline_report(
            attached,
            requested_physical_batch_size=batch_size,
            route=rebuild_route,
            required_fields=("latents", "encoder_hidden_states", "captions"),
        )

    concept_geometry_active = bool(
        shuffle
        and hasattr(dataset, "is_concept_geometry_enabled")
        and callable(getattr(dataset, "is_concept_geometry_enabled"))
        and dataset.is_concept_geometry_enabled()
    )
    if concept_geometry_active:
        bucket_indices = dataset.get_bucket_indices()
        if bucket_indices and len(bucket_indices) > 1:
            bucket_samplers = [
                _ConceptGeometryCurriculumBatchSampler(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=drop_last,
                    global_indices=indices,
                )
                for indices in bucket_indices.values()
            ]
            combined_sampler = _ConcatBatchSampler(bucket_samplers, shuffle=True)
            return attach(DataLoader(
                dataset,
                batch_sampler=combined_sampler,
                **dl_kwargs,
            ))
        return attach(DataLoader(
            dataset,
            batch_sampler=_ConceptGeometryCurriculumBatchSampler(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last,
            ),
            **dl_kwargs,
        ))

    # When bucketing is enabled, use aBatchSampler that groups samples by resolution
    bucket_indices = dataset.get_bucket_indices()
    if bucket_indices and len(bucket_indices) > 1:
        from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
        bucket_samplers = []
        for indices in bucket_indices.values():
            if shuffle:
                sub_sampler = RandomSampler(indices)
            else:
                sub_sampler = SequentialSampler(indices)
            # Wrap with a custom index-mapping BatchSampler
            bucket_samplers.append(_IndexMappingBatchSampler(indices, sub_sampler, batch_size, drop_last=drop_last))
        combined_sampler = _ConcatBatchSampler(bucket_samplers, shuffle=shuffle)
        return attach(DataLoader(
            dataset,
            batch_sampler=combined_sampler,
            **dl_kwargs,
        ))

    return attach(DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **dl_kwargs,
    ))


class _IndexMappingBatchSampler:
    """BatchSampler that maps local indices to global dataset indices."""

    def __init__(self, global_indices: List[int], sampler, batch_size: int, drop_last: bool = False):
        self.global_indices = global_indices
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for local_idx in self.sampler:
            batch.append(self.global_indices[local_idx])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.global_indices) // self.batch_size
        return (len(self.global_indices) + self.batch_size - 1) // self.batch_size


class _ConceptGeometryCurriculumBatchSampler:
    """Weighted sampler with optional v2 concept-aware batch construction."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
        global_indices: Optional[List[int]] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.global_indices = list(global_indices) if global_indices is not None else list(range(len(dataset)))

    def __iter__(self):
        if not self.global_indices:
            return
        concept_mode = (
            self.shuffle
            and self.batch_size > 1
            and str(getattr(self.dataset, "concept_geometry_sampler_mode", "") or "") == "concept_batch"
            and hasattr(self.dataset, "get_concept_geometry_concept_batch")
        )
        if concept_mode:
            import random as _rng
            epoch = int(getattr(self.dataset, "_current_epoch", 0) or 0)
            seed = int(getattr(self.dataset, "concept_geometry_seed", 42) or 42)
            rng = _rng.Random(seed + epoch)
            remaining = list(self.global_indices)
            emitted = 0
            while remaining:
                ordered = list(remaining)
                concept_batch = self.dataset.get_concept_geometry_concept_batch(ordered, rng)[: self.batch_size]
                if len(concept_batch) < self.batch_size and self.drop_last:
                    break
                if len(concept_batch) < 2:
                    break
                yield concept_batch
                emitted += len(concept_batch)
                used = set(concept_batch)
                remaining = [idx for idx in remaining if idx not in used]
            if emitted > 0:
                return

        if not self.shuffle:
            ordered_indices = list(self.global_indices)
        else:
            weights = []
            if hasattr(self.dataset, "get_concept_geometry_sampling_weights"):
                weights = list(self.dataset.get_concept_geometry_sampling_weights(self.global_indices))
            if len(weights) != len(self.global_indices):
                weights = [1.0] * len(self.global_indices)
            probabilities = torch.tensor(weights, dtype=torch.float64)
            probabilities = probabilities.clamp_min(1e-6)
            generator = torch.Generator()
            epoch = int(getattr(self.dataset, "_current_epoch", 0) or 0)
            seed = int(getattr(self.dataset, "concept_geometry_seed", 42) or 42)
            generator.manual_seed(seed + epoch)
            order = torch.multinomial(
                probabilities,
                num_samples=len(self.global_indices),
                replacement=False,
                generator=generator,
            ).tolist()
            ordered_indices = [self.global_indices[index] for index in order]
        batch: List[int] = []
        for index in ordered_indices:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.global_indices) // self.batch_size
        return (len(self.global_indices) + self.batch_size - 1) // self.batch_size


class _ConcatBatchSampler:
    """Concatenate multiple BatchSamplers, optionally shuffling across buckets."""

    def __init__(self, samplers: List[object], shuffle: bool = True):
        self.samplers = samplers
        self.shuffle = shuffle
        self.batch_size = getattr(self.samplers[0], "batch_size", 1) if self.samplers else 1

    def __iter__(self):
        import random as _rng
        all_batches = []
        for s in self.samplers:
            all_batches.extend(list(s))
        if self.shuffle:
            _rng.shuffle(all_batches)
        for batch in all_batches:
            yield batch

    def __len__(self):
        return sum(len(s) for s in self.samplers)


HLoraGeometryEntry = ConceptGeometryEntry
_HLoraCurriculumBatchSampler = _ConceptGeometryCurriculumBatchSampler



