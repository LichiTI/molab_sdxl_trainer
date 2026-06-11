# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""
micro_vae.py — Tiny VAE decoder for fast latent→image preview during training.

Warehouse implementation. Architecture is a simple convolutional decoder
compatible with pretrained weights from HuggingFace TAESD models.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MicroDecoder(nn.Module):
    """Tiny convolutional decoder: latent → RGB image.

    Applies 3 rounds of Conv2d + ReLU + nearest-neighbour 2x upsample,
    giving 8x total spatial upscale (e.g. 64x64 latent → 512x512 image
    for SDXL, or 32x32 → 256x256 for SD1.5 low-res previews).
    """

    def __init__(self, latent_channels: int = 4, out_channels: int = 3) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def load_micro_decoder(
    model_type: str = "sd15",
    cache_dir: Optional[str] = None,
) -> MicroDecoder:
    """Load a MicroDecoder, optionally with pretrained weights.

    Attempts to download pretrained weights from HuggingFace:
      - SD1.5: madebyollin/taesd
      - SDXL:  madebyollin/taesdxl

    Falls back to random initialisation if the download or weight loading
    fails for any reason. The returned decoder is in eval mode with all
    parameters frozen.

    Args:
        model_type: ``"sd15"`` or ``"sdxl"``.
        cache_dir: Optional directory for HuggingFace Hub cache.

    Returns:
        Frozen :class:`MicroDecoder` in eval mode.
    """
    # Both SD1.5 and SDXL use 4 latent channels.
    latent_channels = 4
    decoder = MicroDecoder(latent_channels=latent_channels)

    try:
        from huggingface_hub import hf_hub_download  # type: ignore

        repo_id = (
            "madebyollin/taesdxl" if model_type == "sdxl" else "madebyollin/taesd"
        )
        logger.debug("Downloading MicroDecoder weights from %s", repo_id)
        path = hf_hub_download(
            repo_id=repo_id,
            filename="diffusion_pytorch_model.safetensors",
            cache_dir=cache_dir,
        )

        from safetensors.torch import load_file  # type: ignore

        state = load_file(path)

        # Retain only decoder sub-keys and strip the "decoder." prefix.
        decoder_state = {
            k.replace("decoder.", ""): v
            for k, v in state.items()
            if "decoder" in k
        }

        if decoder_state:
            missing, unexpected = decoder.load_state_dict(
                decoder_state, strict=False
            )
            logger.debug(
                "MicroDecoder weights loaded — missing: %d, unexpected: %d",
                len(missing),
                len(unexpected),
            )
        else:
            logger.warning(
                "No decoder keys found in %s; using random init.", repo_id
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "MicroDecoder pretrained load failed (%s); using random init.", exc
        )

    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    return decoder


@torch.no_grad()
def micro_decode(
    latents: torch.Tensor,
    decoder: MicroDecoder,
) -> "Image.Image":  # type: ignore[name-defined]  # PIL imported lazily
    """Decode a batch of latents to a PIL image.

    Runs the decoder on *latents*, clamps the output to ``[-1, 1]``,
    rescales to ``[0, 255]``, and returns the first image in the batch as
    a PIL ``Image``.

    Args:
        latents: Float tensor of shape ``(B, C, H, W)``.
        decoder: A :class:`MicroDecoder` instance (eval, frozen).

    Returns:
        PIL ``Image`` corresponding to ``latents[0]``.
    """
    device = next(decoder.parameters()).device
    latents = latents.to(device=device, dtype=next(decoder.parameters()).dtype)

    output = decoder(latents)
    # output: (B, 3, H, W) — values roughly in [-1, 1] (pretrained) or unbounded (random init).
    output = output.clamp(-1.0, 1.0)
    output = (output + 1.0) / 2.0  # [-1, 1] → [0, 1]
    output = (output * 255.0).clamp(0, 255).byte()

    # Return the first image in the batch.
    img_array = output[0].permute(1, 2, 0).cpu().numpy()

    from PIL import Image  # type: ignore

    return Image.fromarray(img_array)

