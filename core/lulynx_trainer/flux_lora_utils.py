"""Small FLUX LoRA training helpers.

These functions keep tensor contracts testable without importing Diffusers or
loading a model. The trainer calls the same helpers for latent packing and the
rectified-flow objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


FLUX_LATENT_PACK_FACTOR = 2
FLUX_NUM_TRAIN_TIMESTEPS = 1000


@dataclass(frozen=True)
class FluxFlowConfig:
    timestep_sampling: str = "shift"
    sigmoid_scale: float = 1.0
    discrete_flow_shift: float = 1.0
    weighting_scheme: str = "none"
    mode_scale: float = 1.0
    logit_mean: float = 0.0
    logit_std: float = 1.0


def pack_flux_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack VAE latents from BCHW to Flux token layout.

    Shape: ``[B, C, H, W] -> [B, H/2 * W/2, C * 4]``.
    """

    if latents.ndim != 4:
        raise ValueError("Flux latents must be shaped [batch, channels, height, width]")
    batch, channels, height, width = latents.shape
    if height % FLUX_LATENT_PACK_FACTOR or width % FLUX_LATENT_PACK_FACTOR:
        raise ValueError("Flux latent height and width must be divisible by 2")
    latents = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch, (height // 2) * (width // 2), channels * 4)


def prepare_flux_image_ids(
    packed_height: int,
    packed_width: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return Flux image-position ids for packed latent tokens."""

    if packed_height <= 0 or packed_width <= 0:
        raise ValueError("packed_height and packed_width must be positive")
    ids = torch.zeros((packed_height, packed_width, 3), device=device, dtype=dtype)
    ids[..., 1] = torch.arange(packed_height, device=device, dtype=dtype)[:, None]
    ids[..., 2] = torch.arange(packed_width, device=device, dtype=dtype)[None, :]
    return ids.reshape(packed_height * packed_width, 3)


def apply_flux_flow_shift(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply Flux-style monotonic shift to sigma values."""

    shift = float(shift or 1.0)
    if shift <= 0:
        return sigmas
    return (sigmas * shift) / (1.0 + (shift - 1.0) * sigmas)


def sample_flux_sigmas(
    batch_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    config: FluxFlowConfig | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample per-item Flux sigmas in ``[0, 1]``."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    cfg = config or FluxFlowConfig()
    mode = str(cfg.timestep_sampling or "shift").strip().lower()
    if mode in {"", "sigma", "uniform"}:
        sigmas = torch.rand((batch_size,), device=device, dtype=dtype, generator=generator)
    elif mode == "sigmoid":
        noise = torch.randn((batch_size,), device=device, dtype=dtype, generator=generator)
        sigmas = torch.sigmoid(noise * float(cfg.sigmoid_scale or 1.0))
    elif mode in {"shift", "flux_shift"}:
        noise = torch.randn((batch_size,), device=device, dtype=dtype, generator=generator)
        sigmas = torch.sigmoid(noise * float(cfg.sigmoid_scale or 1.0))
        sigmas = apply_flux_flow_shift(sigmas, cfg.discrete_flow_shift)
    elif mode == "logit_normal":
        noise = torch.randn((batch_size,), device=device, dtype=dtype, generator=generator)
        sigmas = torch.sigmoid(float(cfg.logit_mean) + float(cfg.logit_std or 1.0) * noise)
    else:
        raise ValueError(
            f"Unsupported Flux timestep_sampling={cfg.timestep_sampling!r}. "
            "Expected one of: sigma, uniform, sigmoid, shift, flux_shift, logit_normal."
        )
    return sigmas.clamp(0.0, 1.0)


def build_flux_flow_inputs(
    packed_latents: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(noisy_latents, target, timesteps)`` for Flux training.

    FluxTransformer2DModel expects timesteps in ``[0, 1]`` and internally
    scales them by 1000. The velocity target follows the native DiT convention
    used elsewhere in the trainer: ``noise - clean_latents``.
    """

    if packed_latents.shape != noise.shape:
        raise ValueError(f"packed_latents/noise shape mismatch: {packed_latents.shape} vs {noise.shape}")
    if packed_latents.shape[0] != sigmas.shape[0]:
        raise ValueError("sigmas batch dimension must match packed_latents")
    view_shape = (packed_latents.shape[0],) + (1,) * (packed_latents.ndim - 1)
    sigma_view = sigmas.to(device=packed_latents.device, dtype=packed_latents.dtype).view(view_shape)
    noisy_latents = (1.0 - sigma_view) * packed_latents + sigma_view * noise
    target = noise - packed_latents
    timesteps = sigmas.to(device=packed_latents.device, dtype=packed_latents.dtype)
    return noisy_latents, target, timesteps


def compute_flux_loss_weights(
    sigmas: torch.Tensor,
    scheme: str = "none",
    mode_scale: float = 1.0,
) -> torch.Tensor:
    """Compute optional per-sample loss weights."""

    scheme = str(scheme or "none").strip().lower()
    sigmas_f = sigmas.float().clamp(1e-6, 1.0 - 1e-6)
    if scheme in {"", "none", "uniform"}:
        return torch.ones_like(sigmas_f)
    if scheme == "logit_normal":
        logits = torch.log(sigmas_f) - torch.log1p(-sigmas_f)
        return torch.exp(-0.5 * logits * logits)
    if scheme == "mode":
        return (1.0 - sigmas_f * float(mode_scale or 1.0)).clamp_min(0.0)
    if scheme in {"cosine", "cosmap"}:
        return 2.0 / (math.pi * (1.0 - 2.0 * sigmas_f + 2.0 * sigmas_f * sigmas_f))
    raise ValueError(
        f"Unsupported Flux weighting_scheme={scheme!r}. "
        "Expected one of: none, uniform, logit_normal, mode, cosine."
    )


__all__ = [
    "FLUX_LATENT_PACK_FACTOR",
    "FLUX_NUM_TRAIN_TIMESTEPS",
    "FluxFlowConfig",
    "apply_flux_flow_shift",
    "build_flux_flow_inputs",
    "compute_flux_loss_weights",
    "pack_flux_latents",
    "prepare_flux_image_ids",
    "sample_flux_sigmas",
]
