"""Warehouse Anima rectified-flow training contracts.

This module contains small, testable pieces of the Anima training path:
sigma sampling, noisy-latent construction, flow targets, optional loss
weighting, and fixed-length text conditioning.  It deliberately avoids
model-specific forward code; the native DiT loader must still prove real
forward/conditioning/save support before the trainer guard can be lifted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


ANIMA_LATENT_CHANNELS = 16
ANIMA_DEFAULT_PATCH_SIZE = 2
ANIMA_DEFAULT_TEXT_TOKENS = 512
ANIMA_DEFAULT_STATIC_VISUAL_TOKENS = 4096


@dataclass(frozen=True)
class AnimaFlowConfig:
    """User-facing knobs for Anima's flow-matching objective."""

    timestep_sampling: str = "sigma"
    sigmoid_scale: float = 1.0
    discrete_flow_shift: float = 1.0
    weighting_scheme: str = "none"
    guidance_scale: float = 1.0
    model_prediction_type: str = "velocity"  # "velocity", "noise", "epsilon", "sample"
    mode_scale: float = 1.0
    logit_mean: float = 0.0
    logit_std: float = 1.0


def sample_anima_sigmas(
    batch_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    config: Optional[AnimaFlowConfig] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample per-item flow sigmas in ``[0, 1]``.

    The returned value is the interpolation coefficient used by
    ``x_t = (1 - sigma) * latent + sigma * noise``.  A shift can be applied
    to bias sampling toward high-noise or low-noise regions without changing
    the downstream training contract.
    """

    cfg = config or AnimaFlowConfig()
    mode = (cfg.timestep_sampling or "sigma").lower()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if mode in {"sigma", "uniform"}:
        sigmas = torch.rand((batch_size,), device=device, dtype=dtype, generator=generator)
    elif mode == "sigmoid":
        scale = float(cfg.sigmoid_scale or 1.0)
        sigmas = torch.randn((batch_size,), device=device, dtype=dtype, generator=generator)
        sigmas = torch.sigmoid(sigmas * scale)
    elif mode in {"shift", "flux_shift"}:
        sigmas = torch.randn((batch_size,), device=device, dtype=dtype, generator=generator)
        sigmas = torch.sigmoid(sigmas * float(cfg.sigmoid_scale or 1.0))
        sigmas = apply_anima_flow_shift(sigmas, cfg.discrete_flow_shift)
    elif mode == "logit_normal":
        mean = float(cfg.logit_mean or 0.0)
        std = float(cfg.logit_std or 1.0)
        normals = torch.randn((batch_size,), device=device, dtype=dtype, generator=generator)
        sigmas = torch.sigmoid(mean + std * normals)
    else:
        raise ValueError(
            f"Unsupported Anima timestep_sampling={cfg.timestep_sampling!r}. "
            "Expected one of: sigma, uniform, sigmoid, shift, flux_shift, logit_normal."
        )

    return sigmas.clamp(0.0, 1.0)


def apply_anima_flow_shift(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply the monotonic flow-shift transform to sigma values."""

    shift = float(shift or 1.0)
    if shift <= 0:
        return sigmas
    return (sigmas * shift) / (1.0 + (shift - 1.0) * sigmas)


def build_anima_flow_inputs(
    latents: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    num_train_timesteps: int = 1000,
    model_prediction_type: str = "velocity",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(noisy_latents, target, timesteps)`` for Anima.

    The *target* depends on ``model_prediction_type``:
      - ``"velocity"``: ``noise - latent``  (default, standard v-prediction)
      - ``"noise"`` / ``"epsilon"``: ``noise`` (predict the added noise)
      - ``"sample"``: ``latent`` (predict the clean sample)
    """

    if latents.shape != noise.shape:
        raise ValueError(f"latents/noise shape mismatch: {latents.shape} vs {noise.shape}")
    if latents.shape[0] != sigmas.shape[0]:
        raise ValueError("sigmas batch dimension must match latents")

    view_shape = (latents.shape[0],) + (1,) * (latents.dim() - 1)
    sigma_view = sigmas.to(device=latents.device, dtype=latents.dtype).view(view_shape)
    noisy_latents = (1.0 - sigma_view) * latents + sigma_view * noise

    pred_type = (model_prediction_type or "velocity").lower()
    if pred_type == "velocity":
        target = noise - latents
    elif pred_type in {"noise", "epsilon"}:
        target = noise
    elif pred_type == "sample":
        target = latents
    else:
        raise ValueError(
            f"Unsupported model_prediction_type={pred_type!r}. "
            "Expected one of: velocity, noise, epsilon, sample."
        )

    timesteps = (sigmas.to(device=latents.device, dtype=latents.dtype) * float(num_train_timesteps))
    return noisy_latents, target, timesteps


def compute_anima_loss_weighting(
    sigmas: torch.Tensor,
    scheme: str = "none",
    mode_scale: float = 1.0,
) -> torch.Tensor:
    """Compute optional per-sample loss weights for Anima flow training."""

    scheme = (scheme or "none").lower()
    sigmas_f = sigmas.float().clamp(1e-6, 1.0 - 1e-6)
    if scheme in {"", "none"}:
        return torch.ones_like(sigmas_f)
    if scheme == "sigma_sqrt":
        return sigmas_f.rsqrt()
    if scheme == "cosmap":
        return 2.0 / (math.pi * (1.0 - 2.0 * sigmas_f + 2.0 * sigmas_f * sigmas_f))
    if scheme == "mode":
        return 1.0 - sigmas_f * mode_scale
    if scheme == "logit_normal":
        logits = torch.log(sigmas_f) - torch.log1p(-sigmas_f)
        return torch.exp(-0.5 * logits * logits)
    raise ValueError(
        f"Unsupported Anima weighting_scheme={scheme!r}. "
        "Expected one of: none, sigma_sqrt, cosmap, mode, logit_normal."
    )


def pad_anima_text_condition(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    target_tokens: int = ANIMA_DEFAULT_TEXT_TOKENS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad/truncate text conditioning to a fixed token count.

    Anima's native path expects stable text-token shapes for compiled
    training.  Padding uses zeros and the returned boolean mask marks valid
    tokens so attention code can distinguish content from padding.
    """

    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must be shaped [batch, tokens, channels]")
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    batch, tokens, channels = hidden_states.shape
    valid_tokens = min(tokens, target_tokens)
    output = hidden_states.new_zeros((batch, target_tokens, channels))
    output[:, :valid_tokens] = hidden_states[:, :valid_tokens]

    if attention_mask is None:
        mask = torch.zeros((batch, target_tokens), device=hidden_states.device, dtype=torch.bool)
        mask[:, :valid_tokens] = True
    else:
        mask = torch.zeros((batch, target_tokens), device=hidden_states.device, dtype=torch.bool)
        mask[:, :valid_tokens] = attention_mask[:, :valid_tokens].to(
            device=hidden_states.device,
            dtype=torch.bool,
        )
    return output, mask


