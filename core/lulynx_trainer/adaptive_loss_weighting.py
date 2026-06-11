# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Adaptive Loss Weighting — learnable SNR-based loss reweighting.

Replaces the fixed min-SNR gamma formula with a small set of trainable
parameters that learn the optimal loss-to-timestep weighting during
training.  Acts as an optional *alternative* to ``min_snr_gamma`` — when
enabled, the fixed clamp path is skipped.

The module has three learnable scalars:

    log_gamma  — controls the SNR clamp point (exp → gamma)
    offset     — additive bias on the final weight
    log_scale  — multiplicative scaling (exp → scale)

Weight formula:
    w(t) = clamp(snr, max=gamma) / (snr + 1e-8) * scale + offset
    w(t) = clamp(w(t), min=0.01, max=10.0)

Integration: created in trainer.py _create_optimizer(), passed into
TrainingLoop, called at the SNR weighting site in training_loop.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class AdaptiveLossWeighter(nn.Module):
    """Learnable per-timestep loss weighting based on SNR."""

    def __init__(self, init_gamma: float = 5.0) -> None:
        super().__init__()
        self.log_gamma = nn.Parameter(torch.tensor(math.log(max(init_gamma, 1e-3))))
        self.offset = nn.Parameter(torch.zeros(1))
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, snr: torch.Tensor, v_parameterization: bool = False) -> torch.Tensor:
        """Compute per-sample loss weights from signal-to-noise ratio.

        Parameters
        ----------
        snr : (B,) tensor of per-sample SNR values.
        v_parameterization : if True, use (snr + 1) as divisor instead of snr.

        Returns
        -------
        (B,) tensor of loss weights.
        """
        gamma = self.log_gamma.exp()
        scale = self.log_scale.exp()
        divisor = (snr + 1.0) if v_parameterization else snr
        weights = torch.clamp(snr, max=gamma) / (divisor + 1e-8) * scale + self.offset
        return weights.clamp(min=0.01, max=10.0)
