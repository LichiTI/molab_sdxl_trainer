"""FasterDiT-style SNR optimization for accelerated DiT training.

This module implements enhanced SNR-based training optimizations inspired by
FasterDiT research, including:
1. SNR-aware timestep importance sampling
2. Enhanced loss weighting strategies
3. Adaptive timestep distribution

References:
- FasterDiT: Fast Training of Diffusion Transformers
- Min-SNR weighting strategy (Hang et al., 2023)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FasterDiTSNRConfig:
    """Configuration for FasterDiT SNR optimization.

    Attributes
    ----------
    mode : str
        Weighting mode: 'standard', 'sqrt', 'log', 'adaptive'
    gamma : float
        SNR clamp value (max SNR for weighting)
    timestep_sampling : str
        Timestep sampling strategy: 'uniform', 'snr_weighted', 'low_snr_bias'
    low_snr_weight : float
        Extra weight for low-SNR (high noise) timesteps
    """
    mode: Literal['standard', 'sqrt', 'log', 'adaptive'] = 'standard'
    gamma: float = 5.0
    timestep_sampling: Literal['uniform', 'snr_weighted', 'low_snr_bias'] = 'uniform'
    low_snr_weight: float = 1.5

    def validate(self) -> None:
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.low_snr_weight < 1.0:
            raise ValueError("low_snr_weight must be >= 1.0")


class FasterDiTSNRWeighter(nn.Module):
    """FasterDiT-style SNR loss weighting.

    Implements enhanced SNR weighting strategies for faster convergence.

    Example
    -------
    >>> config = FasterDiTSNRConfig(mode='sqrt', gamma=5.0)
    >>> weighter = FasterDiTSNRWeighter(config)
    >>> snr = compute_snr(timesteps)
    >>> weights = weighter(snr)
    >>> loss = (pred - target).pow(2) * weights
    """

    def __init__(self, config: FasterDiTSNRConfig):
        super().__init__()
        self.config = config
        config.validate()

    def forward(self, snr: torch.Tensor, v_parameterization: bool = False) -> torch.Tensor:
        """Compute loss weights from SNR.

        Parameters
        ----------
        snr : torch.Tensor
            Signal-to-noise ratio, shape (B,)
        v_parameterization : bool
            Whether using v-parameterization

        Returns
        -------
        torch.Tensor
            Loss weights, shape (B,)
        """
        clamped_snr = torch.clamp(snr, max=self.config.gamma)
        divisor = (snr + 1.0) if v_parameterization else snr

        if self.config.mode == 'standard':
            # Standard min-SNR weighting
            weights = clamped_snr / (divisor + 1e-8)
        elif self.config.mode == 'sqrt':
            # Square root for smoother weighting
            weights = torch.sqrt(clamped_snr / (divisor + 1e-8))
        elif self.config.mode == 'log':
            # Log scale for more aggressive low-SNR emphasis
            weights = torch.log1p(clamped_snr) / (torch.log1p(divisor) + 1e-8)
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

        # Optional: boost low-SNR timesteps
        if self.config.low_snr_weight > 1.0:
            # SNR < 1.0 means more noise than signal
            low_snr_mask = (snr < 1.0).float()
            boost = 1.0 + (self.config.low_snr_weight - 1.0) * low_snr_mask
            weights = weights * boost

        return weights.clamp(min=0.01, max=10.0)


def sample_timesteps_with_snr_bias(
    batch_size: int,
    num_train_timesteps: int,
    alphas_cumprod: torch.Tensor,
    device: torch.device,
    strategy: str = 'low_snr_bias',
    bias_strength: float = 1.5,
) -> torch.Tensor:
    """Sample timesteps with SNR-aware importance sampling.

    FasterDiT research shows that focusing more on challenging (high-noise)
    timesteps can accelerate convergence.

    Parameters
    ----------
    batch_size : int
        Number of timesteps to sample
    num_train_timesteps : int
        Total timesteps in schedule
    alphas_cumprod : torch.Tensor
        Cumulative product of alphas from scheduler
    device : torch.device
        Device for tensors
    strategy : str
        Sampling strategy: 'uniform', 'low_snr_bias', 'snr_weighted'
    bias_strength : float
        Strength of bias towards low-SNR timesteps (> 1.0)

    Returns
    -------
    torch.Tensor
        Sampled timesteps, shape (B,)
    """
    if strategy == 'uniform':
        return torch.randint(0, num_train_timesteps, (batch_size,), device=device)

    # Compute SNR for all timesteps
    alphas = alphas_cumprod.to(device)
    snr = alphas / (1.0 - alphas + 1e-8)

    if strategy == 'low_snr_bias':
        # Bias towards low SNR (high noise, early diffusion steps)
        # Inverse SNR as sampling weight
        inv_snr = 1.0 / (snr + 1e-8)
        weights = torch.pow(inv_snr, bias_strength)
        weights = weights / weights.sum()

    elif strategy == 'snr_weighted':
        # Sample proportional to SNR difficulty
        # Use variance of SNR gradient as proxy
        snr_normalized = (snr - snr.min()) / (snr.max() - snr.min() + 1e-8)
        # Peak weight at mid-SNR where learning is most effective
        weights = torch.exp(-((snr_normalized - 0.5) ** 2) / 0.2)
        weights = weights / weights.sum()

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Multinomial sampling with replacement
    indices = torch.multinomial(weights, batch_size, replacement=True)
    return indices.long()


def build_faster_dit_snr_scorecard(
    *,
    config: FasterDiTSNRConfig | None = None,
    weighting_tested: bool = False,
    sampling_tested: bool = False,
    convergence_improved: bool = False,
) -> dict[str, Any]:
    """Build scorecard for FasterDiT SNR optimization readiness.

    Parameters
    ----------
    config : FasterDiTSNRConfig, optional
        Configuration to validate
    weighting_tested : bool
        Whether loss weighting has been tested
    sampling_tested : bool
        Whether timestep sampling has been tested
    convergence_improved : bool
        Whether convergence improvement has been verified

    Returns
    -------
    dict
        Scorecard with readiness status
    """
    cfg = config or FasterDiTSNRConfig()
    blockers: list[str] = []

    try:
        cfg.validate()
    except ValueError as exc:
        blockers.append(f"invalid_config:{exc}")

    if not weighting_tested:
        blockers.append("loss_weighting_not_tested")
    if not sampling_tested:
        blockers.append("timestep_sampling_not_tested")
    if not convergence_improved:
        blockers.append("convergence_improvement_not_verified")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "faster_dit_snr_optimization_v0",
        "ok": ready,
        "optimization_ready": ready,
        "mode": cfg.mode,
        "gamma": cfg.gamma,
        "timestep_sampling": cfg.timestep_sampling,
        "low_snr_weight": cfg.low_snr_weight,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "enable behind config flag after A/B testing"
            if ready
            else "complete testing and convergence verification"
        ),
    }


__all__ = [
    "FasterDiTSNRConfig",
    "FasterDiTSNRWeighter",
    "sample_timesteps_with_snr_bias",
    "build_faster_dit_snr_scorecard",
]
