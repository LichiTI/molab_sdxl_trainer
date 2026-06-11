"""Cache helpers consumed by the Flux LoRA preview trainer."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import torch

from .dataset_loader import LatentDiskCache, TextEncoderDiskCache
from .flux_lora_utils import pack_flux_latents


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _cache_disk_dtype(value: Any, default: str = "float16") -> str:
    raw = str(value or "").strip().lower()
    aliases = {"": default, "auto": default, "default": default, "fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
    raw = aliases.get(raw, raw)
    return raw if raw in {"float16", "bfloat16", "float32"} else default


def _cache_disk_format(value: Any, default: str = "npz") -> str:
    raw = str(value or "").strip().lower()
    aliases = {"": default, "auto": default, "default": default, "safe_tensors": "safetensors"}
    raw = aliases.get(raw, raw)
    return raw if raw in {"npz", "safetensors", "pt"} else default


def _stable_flux_text_conditioning(config: Any) -> bool:
    if _truthy(getattr(config, "shuffle_caption", True)) and not _truthy(getattr(config, "shuffle_caption_tags_only", False)):
        return False
    if float(getattr(config, "caption_dropout_rate", 0.0) or 0.0) > 0.0:
        return False
    if float(getattr(config, "tag_dropout_rate", 0.0) or 0.0) > 0.0:
        return False
    return not str(getattr(config, "caption_tag_dropout_targets", "") or "").strip()


class FluxTrainingCache:
    def __init__(self, config: Any, *, model_id: str, log: Callable[[str], None] | None = None) -> None:
        self.config = config
        self.model_id = model_id
        self.log = log
        self.cache_latents = _truthy(getattr(config, "cache_latents_to_disk", False))
        self.cache_text = (
            _truthy(getattr(config, "cache_text_encoder_outputs_to_disk", False))
            and _stable_flux_text_conditioning(config)
        )
        root = Path(str(getattr(config, "train_data_dir", "") or ".")) / ".lulynx_cache" / "flux" / model_id
        self.latent_cache = LatentDiskCache(
            str(root / "latents"),
            enabled=self.cache_latents,
            disk_format=_cache_disk_format(getattr(config, "latent_cache_disk_format", "npz")),
            disk_dtype=_cache_disk_dtype(getattr(config, "latent_cache_disk_dtype", "float16")),
        )
        self.text_cache = TextEncoderDiskCache(
            str(root / "text_encoder"),
            enabled=self.cache_text,
            disk_format=_cache_disk_format(getattr(config, "text_encoder_outputs_cache_disk_format", "npz")),
            disk_dtype=_cache_disk_dtype(getattr(config, "text_encoder_outputs_cache_disk_dtype", "float16")),
        )
        self.profile: dict[str, Any] = {
            "enabled": self.cache_latents or self.cache_text,
            "cache_dir": str(root),
            "latents_enabled": self.cache_latents,
            "text_enabled": self.cache_text,
            "latent_hits": 0,
            "latent_misses": 0,
            "text_hits": 0,
            "text_misses": 0,
        }
        if not self.cache_text and _truthy(getattr(config, "cache_text_encoder_outputs_to_disk", False)):
            self.profile["text_disabled_reason"] = "dynamic caption conditioning"

    @classmethod
    def from_config(cls, config: Any, *, log: Callable[[str], None] | None = None) -> "FluxTrainingCache":
        model_id_source = "|".join(
            [
                str(getattr(config, "pretrained_model_name_or_path", "") or ""),
                str(getattr(config, "flux_transformer_path", "") or ""),
                str(getattr(config, "ae_path", "") or ""),
                str(getattr(config, "t5xxl_path", "") or ""),
                str(getattr(config, "clip_l_path", "") or ""),
                str(getattr(config, "mixed_precision", "") or ""),
                str(getattr(config, "t5_max_token_length", "") or ""),
            ]
        )
        model_id = hashlib.sha256(model_id_source.encode("utf-8")).hexdigest()[:16]
        return cls(config, model_id=model_id, log=log)

    @staticmethod
    def _sample_key(batch: dict[str, Any], index: int) -> str:
        filenames = batch.get("filenames") or []
        if isinstance(filenames, (list, tuple)) and index < len(filenames):
            return str(filenames[index] or index)
        return str(index)

    @staticmethod
    def _target_size(batch: dict[str, Any], index: int, images: torch.Tensor) -> tuple[int, int]:
        sizes = batch.get("target_sizes") or []
        if isinstance(sizes, (list, tuple)) and index < len(sizes):
            try:
                return tuple(int(v) for v in sizes[index])[:2]  # type: ignore[return-value]
            except Exception:
                pass
        return int(images.shape[-1]), int(images.shape[-2])

    @staticmethod
    def _to_device(value: torch.Tensor, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
        if dtype is not None and value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)

    def resolve_latents(
        self,
        batch: dict[str, Any],
        images: torch.Tensor,
        *,
        vae: Any,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
        ensure_vae: Callable[[], None] | None = None,
        release_vae: Callable[[], None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cache_latents:
            if ensure_vae is not None:
                ensure_vae()
            try:
                encoded = vae.encode(images.to(device=device, dtype=dtype)).latent_dist.sample(generator=generator)
                shift = float(getattr(getattr(vae, "config", None), "shift_factor", 0.0) or 0.0)
                scale = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0) or 1.0)
                latents = ((encoded - shift) * scale).to(device=device, dtype=dtype)
            finally:
                if release_vae is not None:
                    release_vae()
            return latents, pack_flux_latents(latents).to(dtype=dtype)

        latents: list[torch.Tensor] = []
        missing: list[int] = []
        for index in range(int(images.shape[0])):
            cached = self.latent_cache.get(self._sample_key(batch, index), self._target_size(batch, index, images)) if self.cache_latents else None
            if cached and "latents" in cached:
                self.profile["latent_hits"] += 1
                latents.append(cached["latents"].float())
            else:
                self.profile["latent_misses"] += 1
                missing.append(index)

        if missing:
            if ensure_vae is not None:
                ensure_vae()
            try:
                encoded = vae.encode(images[missing].to(device=device, dtype=dtype)).latent_dist.sample(generator=generator)
                shift = float(getattr(getattr(vae, "config", None), "shift_factor", 0.0) or 0.0)
                scale = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0) or 1.0)
                encoded = ((encoded - shift) * scale).detach().cpu().float()
            finally:
                if release_vae is not None:
                    release_vae()
            for offset, index in enumerate(missing):
                sample_latent = encoded[offset]
                latents.insert(index, sample_latent)
                self.latent_cache.put(self._sample_key(batch, index), self._target_size(batch, index, images), {"latents": sample_latent})

        stacked = torch.stack(latents).to(device=device, dtype=dtype)
        return stacked, pack_flux_latents(stacked).to(dtype=dtype)

    def resolve_text(
        self,
        captions: list[str],
        *,
        encode: Callable[[list[str]], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.cache_text:
            return encode(captions)

        cached_items: list[dict[str, torch.Tensor] | None] = []
        missing_indices: list[int] = []
        missing_captions: list[str] = []
        for index, caption in enumerate(captions):
            cached = self.text_cache.get(caption, self.model_id)
            if cached and {"prompt_embeds", "pooled_embeds", "text_ids"}.issubset(cached):
                self.profile["text_hits"] += 1
                cached_items.append(cached)
            else:
                self.profile["text_misses"] += 1
                cached_items.append(None)
                missing_indices.append(index)
                missing_captions.append(caption)

        if missing_indices:
            prompt, pooled, ids = encode(missing_captions)
            for offset, index in enumerate(missing_indices):
                item = {
                    "prompt_embeds": prompt[offset].detach().cpu(),
                    "pooled_embeds": pooled[offset].detach().cpu(),
                    "text_ids": ids.detach().cpu(),
                }
                self.text_cache.put(captions[index], item, self.model_id)
                cached_items[index] = item

        prompt_embeds = torch.stack([item["prompt_embeds"] for item in cached_items if item is not None])
        pooled_embeds = torch.stack([item["pooled_embeds"] for item in cached_items if item is not None])
        text_ids = next(item["text_ids"] for item in cached_items if item is not None)
        return (
            self._to_device(prompt_embeds, device=device, dtype=dtype),
            self._to_device(pooled_embeds, device=device, dtype=dtype),
            self._to_device(text_ids, device=device),
        )


__all__ = ["FluxTrainingCache"]
