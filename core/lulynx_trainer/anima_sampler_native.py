"""Native-Anima sampler adaptations (clean-room Lulynx).

Additive detours that let :func:`anima_sampler.sample_anima` /
:func:`anima_sampler.sample_anima_ersde` drive the *native* Anima stack
(Qwen3 text-encode + qwen-image VAE) without disturbing the existing
CLIP / standard-VAE path. Each helper is inert unless native conditions hold.

* :func:`resolve_text_embeds` — prefer caller-injected ``prompt_embeds`` (Qwen3
  hidden states from :func:`anima_native_inference.encode_qwen3_prompt`); when
  absent, signal the sampler to run its own CLIP encode (``handled=False``).
* :func:`decode_anima_image` — qwen-image VAE needs the inverse latent
  normalisation (``latents*std+mean``) and a 5D ``[B,C,1,H,W]`` decode in fp32
  (the VAE is precision-sensitive — the encode side uses fp32 too); standard
  VAEs keep the original ``latents/vae_scaling_factor`` 4D decode.

PolyForm Noncommercial. Shares no source with any reference repository.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _qwen_image_helpers():
    """Lazily import the qwen-image VAE probe + denormaliser (with smoke fallback)."""
    try:
        from .anima_cache_runtime import (
            _denormalize_qwen_image_latents,
            _is_qwen_image_vae,
        )
    except ImportError:  # pragma: no cover - direct-file smoke fallback.
        from core.lulynx_trainer.anima_cache_runtime import (
            _denormalize_qwen_image_latents,
            _is_qwen_image_vae,
        )
    return _is_qwen_image_vae, _denormalize_qwen_image_latents


def resolve_text_embeds(
    prompt_embeds: Optional[torch.Tensor],
    negative_prompt_embeds: Optional[torch.Tensor],
    *,
    device: str,
    dtype: torch.dtype,
    guidance_scale: float,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
    """Return ``(text_emb, neg_emb, handled)`` for the injected-embeds path.

    ``handled=False`` means no ``prompt_embeds`` were injected and the caller
    should run its own (CLIP) text-encode. When ``handled=True`` the returned
    embeds are device/dtype-aligned; ``neg_emb`` is only kept when CFG is on.
    """
    if prompt_embeds is None:
        return None, None, False
    text_emb = prompt_embeds.to(device=device, dtype=dtype)
    neg_emb = None
    if guidance_scale > 1.0 and negative_prompt_embeds is not None:
        neg_emb = negative_prompt_embeds.to(device=device, dtype=dtype)
    return text_emb, neg_emb, True


@torch.no_grad()
def decode_anima_image(
    vae,
    latents: torch.Tensor,
    vae_scaling_factor: float,
) -> torch.Tensor:
    """Decode latents → pixel tensor ``[B, 3, H, W]`` in ``[-1, 1]``.

    qwen-image VAE: inverse-normalise (``latents*std+mean``), add the singleton
    temporal axis (``[B,C,H,W]`` → ``[B,C,1,H,W]``), decode in fp32, then drop
    the temporal frame. Standard VAE: original ``latents/vae_scaling_factor``
    4D decode in the VAE's own dtype.
    """
    is_qwen_image, denormalize = _qwen_image_helpers()
    if is_qwen_image(vae):
        latents = denormalize(vae, latents)
        if latents.dim() == 4:
            latents = latents.unsqueeze(2)  # [B, C, 1, H, W] for the video VAE
        vae.to(dtype=torch.float32)
        image = vae.decode(latents.to(torch.float32)).sample
        if image.dim() == 5:
            image = image[:, :, 0]  # drop the singleton temporal frame
        return image
    latents = latents / vae_scaling_factor
    return vae.decode(latents.to(vae.dtype)).sample
