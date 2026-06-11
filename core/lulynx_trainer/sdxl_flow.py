"""SDXL-specific flow-matching training contracts.

Provides the flow-matching training path for SDXL models, analogous to the
Anima flow module but using the SDXL UNet's epsilon/velocity/sample convention.

Key concepts:
- Interpolation: x_t = (1 - sigma) * x_0 + sigma * noise
- Target depends on model_prediction_type:
    - "epsilon": target = noise (predict added noise)
    - "velocity": target = noise - x_0 (v-prediction)
    - "sample": target = x_0 (predict clean sample)
- Timestep sampling strategies: uniform, sigmoid, logit_normal, shift
- Loss weighting schemes: none, sigma_sqrt, cosmap, logit_normal
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class SDXLFlowConfig:
    """User-facing knobs for SDXL flow-matching training."""

    timestep_sampling: str = "uniform"  # "uniform" | "sigma" | "sigmoid" | "shift" | "logit_normal"
    sigmoid_scale: float = 1.0
    discrete_flow_shift: float = 1.0  # Alias for "shift"; same formula as Anima
    weighting_scheme: str = "none"  # "none" | "sigma_sqrt" | "cosmap" | "logit_normal"
    model_prediction_type: str = "epsilon"  # "epsilon" | "velocity" | "sample"
    logit_mean: float = 0.0
    logit_std: float = 1.0


def apply_flow_shift(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply the monotonic flow-shift transform.

    Formula: sigma' = (sigma * shift) / (1 + (shift - 1) * sigma)
    This is the same transform used by Anima's ``apply_anima_flow_shift``.
    """
    shift = float(shift or 1.0)
    if shift <= 0:
        return sigmas
    return (sigmas * shift) / (1.0 + (shift - 1.0) * sigmas)


def sample_sdxl_flow_sigmas(
    batch_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    config: Optional[SDXLFlowConfig] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample per-item flow sigmas in ``[0, 1]`` for SDXL.

    The returned value is the interpolation coefficient used by
    ``x_t = (1 - sigma) * latent + sigma * noise``.

    Supported sampling modes:
      - "uniform" / "sigma": uniform random in [0, 1]
      - "sigmoid": sample from Normal, apply sigmoid with configurable scale
      - "shift": sigmoid sampling then apply flow-shift transform
      - "logit_normal": logit-normal sampling with configurable mean/std
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    cfg = config or SDXLFlowConfig()
    mode = (cfg.timestep_sampling or "uniform").lower()

    if mode in {"uniform", "sigma"}:
        sigmas = torch.rand((batch_size,), device=device, dtype=dtype, generator=generator)
    elif mode == "sigmoid":
        scale = float(cfg.sigmoid_scale or 1.0)
        normals = torch.randn((batch_size,), device=device, dtype=dtype, generator=generator)
        sigmas = torch.sigmoid(scale * normals)
    elif mode == "shift":
        scale = float(cfg.sigmoid_scale or 1.0)
        normals = torch.randn((batch_size,), device=device, dtype=dtype, generator=generator)
        sigmas = torch.sigmoid(scale * normals)
        sigmas = apply_flow_shift(sigmas, cfg.discrete_flow_shift)
    elif mode == "logit_normal":
        mean = float(cfg.logit_mean or 0.0)
        std = float(cfg.logit_std or 1.0)
        normals = torch.randn((batch_size,), device=device, dtype=dtype, generator=generator)
        logit_samples = mean + std * normals
        sigmas = torch.sigmoid(logit_samples)
    else:
        raise ValueError(
            f"Unsupported SDXL timestep_sampling={cfg.timestep_sampling!r}. "
            "Expected one of: uniform, sigma, sigmoid, shift, logit_normal."
        )

    return sigmas.clamp(1e-5, 1.0 - 1e-5)


def build_sdxl_flow_inputs(
    latents: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    num_train_timesteps: int = 1000,
    model_prediction_type: str = "epsilon",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``(noisy_latents, target, timesteps)`` for SDXL flow matching.

    Interpolation: x_t = (1 - sigma) * x_0 + sigma * noise

    The *target* depends on ``model_prediction_type``:
      - ``"epsilon"``: target = noise (predict the added noise)
      - ``"velocity"``: target = noise - latents (v-prediction)
      - ``"sample"``: target = latents (predict the clean sample)
    """
    if latents.shape != noise.shape:
        raise ValueError(f"latents/noise shape mismatch: {latents.shape} vs {noise.shape}")
    if latents.shape[0] != sigmas.shape[0]:
        raise ValueError("sigmas batch dimension must match latents")

    view_shape = (latents.shape[0],) + (1,) * (latents.dim() - 1)
    sigma_view = sigmas.to(device=latents.device, dtype=latents.dtype).view(view_shape)

    noisy_latents = (1.0 - sigma_view) * latents + sigma_view * noise

    pred_type = (model_prediction_type or "epsilon").lower()
    if pred_type == "epsilon":
        target = noise
    elif pred_type == "velocity":
        target = noise - latents
    elif pred_type == "sample":
        target = latents
    else:
        raise ValueError(
            f"Unsupported model_prediction_type={pred_type!r}. "
            "Expected one of: epsilon, velocity, sample."
        )

    timesteps = (sigmas.to(device=latents.device, dtype=latents.dtype) * float(num_train_timesteps))
    return noisy_latents, target, timesteps


def compute_sdxl_loss_weighting(
    sigmas: torch.Tensor,
    scheme: str = "none",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
) -> torch.Tensor:
    """Compute optional per-sample loss weights for SDXL flow training.

    Supported weighting schemes:
      - "none": uniform weights (all 1.0)
      - "sigma_sqrt": weight by (1 + sigma^2)^0.5
      - "cosmap": cosine-map weighting from EDM2
      - "logit_normal": logit-normal density weighting with configurable mean/std
    """
    scheme = (scheme or "none").lower()
    sigmas_f = sigmas.float().clamp(1e-6, 1.0 - 1e-6)

    if scheme in {"", "none"}:
        return torch.ones_like(sigmas_f)

    if scheme == "sigma_sqrt":
        return (1.0 + sigmas_f ** 2).sqrt()

    if scheme == "cosmap":
        pi = torch.tensor(math.pi, device=sigmas_f.device, dtype=sigmas_f.dtype)
        return 2.0 / (pi * (1.0 - 2.0 * sigmas_f + 2.0 * sigmas_f * sigmas_f))

    if scheme == "logit_normal":
        mean = float(logit_mean or 0.0)
        std = float(logit_std or 1.0)
        logit = torch.log(sigmas_f) - torch.log1p(-sigmas_f)
        density = torch.exp(-0.5 * ((logit - mean) / max(std, 1e-8)) ** 2)
        return density / (sigmas_f * (1.0 - sigmas_f) + 1e-8).sqrt()

    raise ValueError(
        f"Unsupported SDXL weighting_scheme={scheme!r}. "
        "Expected one of: none, sigma_sqrt, cosmap, logit_normal."
    )
