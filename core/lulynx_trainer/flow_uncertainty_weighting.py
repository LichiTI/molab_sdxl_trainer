"""EDM2-style learned per-sigma loss balancing for flow-matching training.

Karras et al., "Analyzing and Improving the Training Dynamics of Diffusion
Models" (EDM2, arXiv:2312.02696, eq. 17) replace hand-tuned timestep loss
weights with a tiny learned log-uncertainty ``u(sigma)``::

    L' = L / exp(u(sigma)) + u(sigma)

High-loss sigma regions are down-weighted automatically while the ``+ u``
term keeps the objective honest (``u -> inf`` is penalized), so the net
effect is an adaptive per-timestep weighting with no scheme to tune.

This is a default-off reserve for the Anima rectified-flow route: the
weighter is created only when ``flow_uncertainty_weighting_enabled`` is set,
its parameters join the optimizer as a separate no-decay param group, and at
init the head is zero so the very first steps are bitwise-identical to the
unweighted loss.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# Fixed seed for the Fourier feature bank so checkpoints/resume and parity
# smokes see the same feature basis regardless of global RNG state.
_FOURIER_SEED = 0x1E3D2


class FlowUncertaintyWeighter(nn.Module):
    """Learned ``u(sigma)`` on Fourier features of the flow sigma in [0, 1].

    ``forward`` consumes a per-sample loss vector and the matching sigmas and
    returns the rebalanced per-sample loss ``loss / exp(u) + u``.
    """

    def __init__(self, num_channels: int = 128) -> None:
        super().__init__()
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        generator = torch.Generator().manual_seed(_FOURIER_SEED)
        self.register_buffer("freqs", 2.0 * math.pi * torch.randn((num_channels,), generator=generator))
        self.register_buffer("phases", 2.0 * math.pi * torch.rand((num_channels,), generator=generator))
        self.head = nn.Linear(num_channels, 1, bias=False)
        # Zero head -> u(sigma) == 0 -> exp(u) == 1 -> identity at init.
        nn.init.zeros_(self.head.weight)

    def log_uncertainty(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Return ``u(sigma)`` as a float32 vector shaped like ``sigmas``."""

        x = sigmas.detach().float().view(-1, 1)
        features = torch.cos(x * self.freqs.view(1, -1) + self.phases.view(1, -1))
        return self.head(features).squeeze(-1)

    def forward(self, per_sample_loss: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        if per_sample_loss.dim() != 1:
            raise ValueError("per_sample_loss must be a per-sample vector")
        if per_sample_loss.shape[0] != sigmas.view(-1).shape[0]:
            raise ValueError("per_sample_loss/sigmas batch mismatch")
        u = self.log_uncertainty(sigmas).to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)
        return per_sample_loss / torch.exp(u) + u


__all__ = ["FlowUncertaintyWeighter"]
