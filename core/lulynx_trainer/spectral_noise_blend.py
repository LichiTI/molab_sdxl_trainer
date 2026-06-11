# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Spectral Noise Blending — low-frequency noise mixing for diffusion training.

Blends standard Gaussian noise with a Gaussian-blurred copy to boost
low-frequency energy.  This helps the model learn dark tones and saturated
colours that pure white-noise sampling under-represents.

    noise_out = (1 - alpha) * noise + alpha * gaussian_blur(noise, sigma)

Integration: called in training_loop.py after `noise = torch.randn_like(latents)`
and before noise_offset application.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def blend_spectral_noise(
    noise: torch.Tensor,
    alpha: float,
    sigma: float = 4.0,
) -> torch.Tensor:
    """Blend *noise* with its Gaussian-blurred version.

    Parameters
    ----------
    noise : (B, C, H, W) tensor
    alpha : blend ratio in [0, 1]. 0 returns noise unchanged.
    sigma : standard deviation of the Gaussian blur kernel.

    Returns
    -------
    Blended noise tensor, same shape and dtype as input.
    """
    if alpha <= 0.0:
        return noise
    alpha = min(alpha, 1.0)
    blurred = _gaussian_blur_4d(noise, sigma)
    return (1.0 - alpha) * noise + alpha * blurred


def _gaussian_blur_4d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 2D Gaussian blur on a (B, C, H, W) tensor."""
    kernel_size = int(math.ceil(sigma * 6)) | 1  # ensure odd
    half = kernel_size // 2

    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - half
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    B, C, H, W = x.shape
    flat = x.reshape(B * C, 1, H, W)

    # Blur along H
    k_h = kernel_1d.view(1, 1, -1, 1)
    flat = F.pad(flat, (0, 0, half, half), mode="reflect")
    flat = F.conv2d(flat, k_h)

    # Blur along W
    k_w = kernel_1d.view(1, 1, 1, -1)
    flat = F.pad(flat, (half, half, 0, 0), mode="reflect")
    flat = F.conv2d(flat, k_w)

    return flat.reshape(B, C, H, W)
