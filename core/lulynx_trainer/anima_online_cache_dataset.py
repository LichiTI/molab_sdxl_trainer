# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Online cache dataset wrapper for Anima training.

Wraps AnimaCachedDataset to support on-demand cache generation:
- Detects missing cache files per sample
- Calls encode functions on cache miss
- Atomically saves generated cache
- Returns tensors after cache hit or generation

This enables ``online_cache`` mode where missing cache is generated lazily
during training rather than upfront.
"""

from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

try:
    from .anima_cache_builder import (
        AnimaCacheBuilderConfig,
        build_anima_cache_sample,
    )
    from .anima_cached_dataset import AnimaCachedDataset, ConceptGeometryEntry
except ImportError:
    # Standalone mode
    from anima_cache_builder import (
        AnimaCacheBuilderConfig,
        build_anima_cache_sample,
    )
    from anima_cached_dataset import AnimaCachedDataset, ConceptGeometryEntry


logger = logging.getLogger(__name__)


class AnimaOnlineCacheDataset(Dataset):
    """Dataset that generates Anima cache on-demand for missing samples.

    Wraps a raw image directory and generates cache files lazily when
    accessed. Once generated, cache files are reused on subsequent access.
    """

    def __init__(
        self,
        data_dir: str | Path,
        vae_encode_fn: Callable[[torch.Tensor], torch.Tensor],
        text_encode_fn: Callable[[str], Dict[str, torch.Tensor]],
        cache_config: AnimaCacheBuilderConfig,
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
        enable_bucket: bool = False,
        concept_geometry_enabled: bool = False,
        concept_geometry_path: str = "",
        concept_geometry_sampler_mode: str = "density_curriculum",
        concept_geometry_loss_weighting: bool = False,
        concept_geometry_density_power: float = 1.0,
        concept_geometry_seed: int = 42,
        concept_geometry_total_epochs: int = 1,
        concept_geometry_total_steps: int = 0,
        **legacy_geometry_kwargs: Any,
    ):
        """Initialize online cache dataset.

        Parameters
        ----------
        data_dir:
            Directory containing raw images and captions.
        vae_encode_fn:
            Callable accepting image tensor [1, 3, H, W] and returning
            latents [1, 16, h, w].
        text_encode_fn:
            Callable accepting caption string and returning dict with
            prompt_embeds, attn_mask, and optional qwen3/t5 fields.
        cache_config:
            Cache builder configuration.
        latent_crop_size:
            Latent spatial crop size (0 = no crop).
        text_token_limit:
            Text token limit for consumption (0 = no limit).
        fixed_text_tokens:
            Fixed text token count for static shape.
        fixed_visual_tokens:
            Fixed visual token count for static shape.
        caption_extension:
            Caption file extension.
        weighted_captions:
            Enable weighted caption parsing.
        enable_bucket:
            Enable resolution-based bucketing.
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
                raise TypeError(f"Unexpected AnimaOnlineCacheDataset keyword argument(s): {unknown}")

        self.data_dir = Path(data_dir)
        self.vae_encode_fn = vae_encode_fn
        self.text_encode_fn = text_encode_fn
        self.cache_config = cache_config
        self.caption_extension = caption_extension
        self.fixed_text_tokens = max(int(fixed_text_tokens or 0), 0)
        self.fixed_visual_tokens = max(int(fixed_visual_tokens or 0), 0)
        self.fixed_qwen3_tokens = max(int(fixed_qwen3_tokens or 0), 0)
        self.fixed_t5_tokens = max(int(fixed_t5_tokens or 0), 0)
        self._current_epoch = 0
        self._current_step = 0
        self.dataloader_rebuild_route = "anima_online_cache"
        self.dataloader_multiprocessing_safe = False

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Anima online cache data_dir does not exist: {self.data_dir}")

        # Discover raw images
        self.image_paths = self._discover_images()
        if not self.image_paths:
            raise ValueError(f"No images found in {self.data_dir}")

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
        self._concept_geometry = self._load_concept_geometry()

        # Create cached dataset wrapper for consuming generated cache
        self._cached_dataset: Optional[AnimaCachedDataset] = None
        self._cached_dataset_kwargs = {
            "data_dir": data_dir,
            "latent_crop_size": latent_crop_size,
            "text_token_limit": text_token_limit,
            "fixed_text_tokens": self.fixed_text_tokens,
            "fixed_visual_tokens": self.fixed_visual_tokens,
            "fixed_qwen3_tokens": self.fixed_qwen3_tokens,
            "fixed_t5_tokens": self.fixed_t5_tokens,
            "caption_extension": caption_extension,
            "shuffle_caption": shuffle_caption,
            "shuffle_caption_tags_only": shuffle_caption_tags_only,
            "keep_tokens": keep_tokens,
            "keep_tokens_separator": keep_tokens_separator,
            "weighted_captions": weighted_captions,
            "enable_bucket": enable_bucket,
            "concept_geometry_enabled": concept_geometry_enabled,
            "concept_geometry_path": concept_geometry_path,
            "concept_geometry_sampler_mode": concept_geometry_sampler_mode,
            "concept_geometry_loss_weighting": concept_geometry_loss_weighting,
            "concept_geometry_density_power": concept_geometry_density_power,
            "concept_geometry_seed": concept_geometry_seed,
            "concept_geometry_total_epochs": concept_geometry_total_epochs,
            "concept_geometry_total_steps": concept_geometry_total_steps,
        }

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, object]:
        """Get sample, generating cache if missing."""
        image_path = self.image_paths[index]

        # Check if cache exists
        if not self._cache_exists(image_path):
            # Generate cache atomically
            self._generate_cache(image_path)
            self._cached_dataset = None

        # Load from cache using AnimaCachedDataset
        if self._cached_dataset is None:
            self._cached_dataset = AnimaCachedDataset(**self._cached_dataset_kwargs)
            self._cached_dataset.set_current_epoch(self._current_epoch)
            self._cached_dataset.set_global_step(self._current_step)
            self._cached_dataset.set_concept_geometry_total_steps(self.concept_geometry_total_steps)

        # Find sample index in cached dataset by stem
        stem = image_path.stem
        for idx, sample in enumerate(self._cached_dataset.samples):
            if sample.stem == stem:
                return self._cached_dataset[idx]

        raise RuntimeError(f"Cache generation succeeded but sample not found: {stem}")

    def _discover_images(self) -> list[Path]:
        """Discover all image files in data_dir."""
        image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = []
        for path in sorted(self.data_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in image_suffixes:
                # Skip mask/alpha sidecars
                if any(path.stem.endswith(s) for s in ("_mask", "_alpha")):
                    continue
                images.append(path)
        return images

    def _cache_exists(self, image_path: Path) -> bool:
        """Check if cache files exist for this image."""
        stem = image_path.stem
        out_root = Path(self.cache_config.output_dir) if self.cache_config.output_dir else image_path.parent

        # Check for text cache
        text_cache_exists = False
        for ext in (".npz", ".safetensors", ".pt"):
            text_path = out_root / f"{stem}_anima_te{ext}"
            if text_path.exists():
                text_cache_exists = True
                break

        if not text_cache_exists:
            return False

        # Check for latent cache (any resolution)
        latent_cache_exists = False
        for ext in (".npz", ".safetensors", ".pt"):
            latent_candidates = list(out_root.glob(f"{stem}_*_anima{ext}"))
            if latent_candidates:
                latent_cache_exists = True
                break

        return latent_cache_exists

    def _generate_cache(self, image_path: Path) -> None:
        """Generate cache for a single image atomically."""
        try:
            build_anima_cache_sample(
                image_path=image_path,
                vae_encode_fn=self.vae_encode_fn,
                text_encode_fn=self.text_encode_fn,
                config=self.cache_config,
                caption_extension=self.caption_extension,
                force=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to generate Anima cache for {image_path.name}: {type(exc).__name__}: {exc}"
            ) from exc

    def get_bucket_indices(self) -> Optional[Dict[str, list[int]]]:
        """Return bucket index map if bucketing is enabled."""
        if self._cached_dataset is None:
            # Force initialization by accessing first sample
            if len(self) > 0:
                _ = self[0]
        if self._cached_dataset is not None:
            return self._cached_dataset.get_bucket_indices()
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
        return Path(raw).expanduser()

    def _load_concept_geometry(self) -> Dict[str, ConceptGeometryEntry]:
        if not self.concept_geometry_enabled:
            return {}
        if not self.concept_geometry_path.is_file():
            logger.warning(
                "[concept-geometry] metadata not found at %s; online-cache sampler will be disabled.",
                self.concept_geometry_path,
            )
            self.concept_geometry_enabled = False
            return {}
        try:
            payload = json.loads(self.concept_geometry_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "[concept-geometry] failed to load geometry metadata from %s: %s; online-cache sampler will be disabled.",
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
                "[concept-geometry] metadata at %s had no usable samples; online-cache sampler will be disabled.",
                self.concept_geometry_path,
            )
            self.concept_geometry_enabled = False
            return {}

        stage_counts = {"core": 0, "mid": 0, "edge": 0}
        attached = 0
        concept_groups: set[str] = set()
        for image_path in self.image_paths:
            entry = geometry.get(image_path.stem)
            if entry is None:
                continue
            attached += 1
            stage_counts[entry.stage] = stage_counts.get(entry.stage, 0) + 1
            if entry.concept_group:
                concept_groups.add(entry.concept_group)
        self._concept_geometry_attached_count = attached
        self._concept_geometry_stage_counts = dict(stage_counts)
        self._concept_geometry_concept_groups = set(concept_groups)
        self._concept_geometry_structure_scale = self._compute_concept_geometry_structure_scale(
            attached=attached,
            stage_counts=stage_counts,
            concept_groups=concept_groups,
        )
        logger.info(
            "[concept-geometry] online-cache geometry loaded from %s: attached=%s/%s core=%s mid=%s edge=%s "
            "groups=%s structure_scale=%.2f mode=%s",
            self.concept_geometry_path,
            attached,
            len(self.image_paths),
            stage_counts.get("core", 0),
            stage_counts.get("mid", 0),
            stage_counts.get("edge", 0),
            len(concept_groups),
            self._concept_geometry_structure_scale,
            self.concept_geometry_sampler_mode,
        )
        return geometry

    def is_concept_geometry_enabled(self) -> bool:
        return self.concept_geometry_enabled and bool(self._concept_geometry)

    def is_h_lora_enabled(self) -> bool:
        """Legacy alias for callers that have not migrated yet."""
        return self.is_concept_geometry_enabled()

    def set_current_epoch(self, epoch: int) -> None:
        self._current_epoch = max(int(epoch or 0), 0)
        if self._cached_dataset is not None:
            self._cached_dataset.set_current_epoch(self._current_epoch)

    def set_global_step(self, step: int) -> None:
        self._current_step = max(int(step or 0), 0)
        if self._cached_dataset is not None:
            self._cached_dataset.set_global_step(self._current_step)

    def set_concept_geometry_total_steps(self, total_steps: int) -> None:
        self.concept_geometry_total_steps = max(int(total_steps or 0), 0)
        if self._cached_dataset is not None:
            self._cached_dataset.set_concept_geometry_total_steps(self.concept_geometry_total_steps)

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
            "core": self._blend_concept_geometry_weight(1.35 - 0.55 * progress),
            "mid": 1.00,
            "edge": self._blend_concept_geometry_weight(0.65 + 0.85 * progress),
        }

    def _concept_geometry_stage_loss_weights(self) -> Dict[str, float]:
        progress = self._concept_geometry_progress()
        return {
            "core": self._blend_concept_geometry_weight(1.05),
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
        indices = list(global_indices) if global_indices is not None else list(range(len(self.image_paths)))
        weights: List[float] = []
        for index in indices:
            image_path = self.image_paths[int(index)]
            geometry = self._concept_geometry.get(image_path.stem)
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

    def get_concept_geometry_summary(self) -> Dict[str, Any]:
        if not self.is_concept_geometry_enabled():
            return {
                "enabled": False,
                "geometry_path": str(self.concept_geometry_path),
            }
        stage_counts = {"core": 0, "mid": 0, "edge": 0}
        attached = 0
        for image_path in self.image_paths:
            geometry = self._concept_geometry.get(image_path.stem)
            if geometry is None:
                continue
            attached += 1
            stage_counts[geometry.stage] = stage_counts.get(geometry.stage, 0) + 1
        return {
            "enabled": True,
            "geometry_path": str(self.concept_geometry_path),
            "sampler_mode": self.concept_geometry_sampler_mode,
            "loss_weighting": self.concept_geometry_loss_weighting,
            "sample_count": len(self.image_paths),
            "attached_count": attached,
            "stage_counts": stage_counts,
            "concept_group_count": len(self._concept_geometry_concept_groups),
            "structure_scale": self._concept_geometry_structure_scale,
        }

    def get_h_lora_summary(self) -> Dict[str, Any]:
        """Legacy alias for callers that have not migrated yet."""
        return self.get_concept_geometry_summary()


HLoraGeometryEntry = ConceptGeometryEntry


