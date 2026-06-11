"""Shared contracts for converting model inputs into training conditions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

import torch


@dataclass(frozen=True)
class ConditionCacheKey:
    """Stable identity for a generated training condition."""

    family: str
    sample_id: str
    model_id: str = ""
    resolution: tuple[int, int] = (0, 0)
    dtype: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ConditionBatch:
    """Model-family neutral condition payload consumed by the training loop."""

    latents: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None
    pooled_prompt_embeds: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    original_size: Any = None
    target_size: Any = None
    crop_coords: Any = None
    caption: str = ""
    caption_weight: float = 1.0
    loss_mask: Optional[torch.Tensor] = None
    filename: str = ""
    cache_key: Optional[ConditionCacheKey] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_training_item(self) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "caption": self.caption,
            "original_size": self.original_size,
            "target_size": self.target_size,
            "crop_coords": self.crop_coords,
            "caption_weight": self.caption_weight,
            "loss_mask": self.loss_mask,
            "filenames": self.filename,
        }
        if self.latents is not None:
            item["latents"] = self.latents
        if self.encoder_hidden_states is not None:
            item["encoder_hidden_states"] = self.encoder_hidden_states
        if self.pooled_prompt_embeds is not None:
            item["pooled_prompt_embeds"] = self.pooled_prompt_embeds
        if self.attention_mask is not None:
            item["attention_mask"] = self.attention_mask
        if self.cache_key is not None:
            item["condition_cache_key"] = self.cache_key
        if self.metadata:
            item["condition_metadata"] = dict(self.metadata)
        return item


class ModelToCondition:
    """Base class for family-specific condition builders."""

    family: str = "generic"

    def build(self, sample: Mapping[str, Any]) -> ConditionBatch:
        raise NotImplementedError


class SDXLModelToCondition(ModelToCondition):
    """SDXL adapter for the cache-first compatibility runtime."""

    family = "sdxl"

    def __init__(
        self,
        *,
        model_id: str,
        dtype: str,
        family: str = "sdxl",
        encode_latents: Callable[[Mapping[str, Any]], torch.Tensor],
        encode_text: Callable[[str], Mapping[str, torch.Tensor]],
    ) -> None:
        self.model_id = str(model_id or "")
        self.dtype = str(dtype or "")
        self.family = str(family or "sdxl").strip().lower()
        self._encode_latents = encode_latents
        self._encode_text = encode_text

    def build(self, sample: Mapping[str, Any]) -> ConditionBatch:
        caption = str(sample.get("caption") or "")
        target_size_raw = sample.get("target_size", (0, 0))
        try:
            target_size = tuple(int(v) for v in target_size_raw)
        except Exception:
            target_size = (0, 0)
        filename = str(sample.get("filenames") or "")
        text = self._encode_text(caption)
        return ConditionBatch(
            latents=self._encode_latents(sample),
            encoder_hidden_states=text["encoder_hidden_states"],
            pooled_prompt_embeds=text.get("pooled_prompt_embeds"),
            original_size=sample.get("original_size"),
            target_size=sample.get("target_size"),
            crop_coords=sample.get("crop_coords"),
            caption=caption,
            caption_weight=float(sample.get("caption_weight", 1.0) or 1.0),
            loss_mask=sample.get("loss_mask"),
            filename=filename,
            cache_key=ConditionCacheKey(
                family=self.family,
                sample_id=filename or caption,
                model_id=self.model_id,
                resolution=target_size,
                dtype=self.dtype,
            ),
        )
