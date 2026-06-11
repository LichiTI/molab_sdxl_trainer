# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""EasyControl for DiT (Phase 8.9 / #117).

Lightweight spatial conditioning for DiT-style models, similar in spirit
to ControlNet but with a much smaller parameter cost.  The control image
(canny edge map, depth, pose skeleton, etc) is:

  1. Resized + downsampled to the latent spatial grid.
  2. Encoded by a tiny CNN into the same channel count as the noisy latent.
  3. Added as a residual to the noisy latent before it enters the DiT.

Only the small CNN encoder is trainable; the DiT and base VAE remain
frozen.  This gives ~50–200K extra parameters vs. ControlNet's ~360M.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class EasyControlConfig:
    """Configuration for the EasyControl encoder."""

    in_channels: int = 3            # control image channels (3 for RGB, 1 for depth)
    latent_channels: int = 16       # match the DiT latent channels (Anima=16, SDXL=4)
    hidden_dim: int = 32            # CNN hidden width
    num_blocks: int = 3             # depth of the CNN
    downsample_factor: int = 8      # how much to downsample the control image
    scale: float = 1.0              # multiplier on residual contribution
    init_zero_out: bool = True      # zero-init the final conv (no-op at start)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class EasyControlEncoder(nn.Module):
    """Tiny CNN that maps a control image to a latent residual.

    Forward signature: ``encoder(control) -> [B, latent_channels, H_lat, W_lat]``.
    """

    def __init__(self, config: EasyControlConfig) -> None:
        super().__init__()
        self.config = config

        # Initial projection
        layers: List[nn.Module] = [
            nn.Conv2d(config.in_channels, config.hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        ]

        # Downsampling stack
        ch = config.hidden_dim
        cur_factor = 1
        while cur_factor < config.downsample_factor:
            new_ch = min(ch * 2, 256)
            layers.append(nn.Conv2d(ch, new_ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.SiLU())
            ch = new_ch
            cur_factor *= 2

        # Body convolutions at the bottleneck
        for _ in range(max(1, config.num_blocks)):
            layers.append(nn.Conv2d(ch, ch, kernel_size=3, padding=1))
            layers.append(nn.SiLU())

        # Output projection to latent_channels
        out_conv = nn.Conv2d(ch, config.latent_channels, kernel_size=1)
        if config.init_zero_out:
            nn.init.zeros_(out_conv.weight)
            nn.init.zeros_(out_conv.bias)
        layers.append(out_conv)

        self.net = nn.Sequential(*layers)

    def forward(self, control: torch.Tensor) -> torch.Tensor:
        return self.net(control) * self.config.scale


# ---------------------------------------------------------------------------
# High-level helper
# ---------------------------------------------------------------------------

class EasyControl(nn.Module):
    """Apply EasyControl residual to noisy latents in a DiT training step.

    Use::

        ctrl = EasyControl(config)
        residual = ctrl(control_image, target_size=latent.shape[-2:])
        latent_with_control = latent + residual
    """

    def __init__(self, config: EasyControlConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = EasyControlEncoder(config)

    def forward(
        self,
        control: torch.Tensor,
        *,
        target_size: Optional[tuple] = None,
    ) -> torch.Tensor:
        """Encode the control image and (optionally) resize to ``target_size``."""
        if control.dim() == 3:
            control = control.unsqueeze(0)

        residual = self.encoder(control)
        if target_size and residual.shape[-2:] != target_size:
            residual = F.interpolate(
                residual, size=target_size, mode="bilinear", align_corners=False,
            )
        return residual

    def get_trainable_params(self) -> List[nn.Parameter]:
        return list(self.encoder.parameters())
