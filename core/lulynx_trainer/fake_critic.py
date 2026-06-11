"""Fake critic network for adversarial distillation.

Used in Mode A (Online Adversarial) distillation.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LightweightCritic(nn.Module):
    """Lightweight discriminator for distillation.

    A simple convolutional network that scores image quality.
    Used to provide adversarial signal during distillation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
    ):
        """
        Parameters
        ----------
        in_channels : int, default 3
            Number of input channels (3 for RGB).
        base_channels : int, default 64
            Base number of feature channels.
        """
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Global average pooling + final score
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : torch.Tensor
            Input images [batch, channels, height, width].

        Returns
        -------
        torch.Tensor
            Quality scores [batch, 1].
        """
        features = self.conv_blocks(images)
        scores = self.head(features)
        return scores


class StandardCritic(nn.Module):
    """Standard-size critic with more capacity.

    Used when lightweight critic is not sufficient.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 2x2
            nn.Conv2d(base_channels * 8, base_channels * 8, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.conv_blocks(images)
        scores = self.head(features)
        return scores


def create_critic(
    arch: str = "lightweight",
    in_channels: int = 3,
    base_channels: int = 64,
) -> nn.Module:
    """Create a critic network.

    Parameters
    ----------
    arch : str, default "lightweight"
        Architecture: "lightweight" or "standard".
    in_channels : int, default 3
        Number of input channels.
    base_channels : int, default 64
        Base channel count.

    Returns
    -------
    nn.Module
        Critic network.
    """
    if arch == "lightweight":
        return LightweightCritic(in_channels, base_channels)
    elif arch == "standard":
        return StandardCritic(in_channels, base_channels)
    else:
        logger.warning(f"Unknown critic arch '{arch}', using lightweight")
        return LightweightCritic(in_channels, base_channels)


def compute_critic_loss(
    critic: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """Compute critic loss (discriminator loss).

    Parameters
    ----------
    critic : nn.Module
        Critic network.
    real_images : torch.Tensor
        Real images (teacher outputs).
    fake_images : torch.Tensor
        Fake images (student outputs).

    Returns
    -------
    tuple of (torch.Tensor, dict)
        Loss and metrics dict.
    """
    # Score real and fake
    real_scores = critic(real_images)
    fake_scores = critic(fake_images.detach())

    # Hinge loss
    real_loss = torch.relu(1.0 - real_scores).mean()
    fake_loss = torch.relu(1.0 + fake_scores).mean()

    critic_loss = real_loss + fake_loss

    metrics = {
        "critic_loss": critic_loss.item(),
        "real_score_mean": real_scores.mean().item(),
        "fake_score_mean": fake_scores.mean().item(),
    }

    return critic_loss, metrics


def compute_generator_loss(
    critic: nn.Module,
    fake_images: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """Compute generator loss (student adversarial loss).

    Parameters
    ----------
    critic : nn.Module
        Critic network.
    fake_images : torch.Tensor
        Fake images (student outputs).

    Returns
    -------
    tuple of (torch.Tensor, dict)
        Loss and metrics dict.
    """
    fake_scores = critic(fake_images)

    # Generator tries to maximize critic score (fool discriminator)
    gen_loss = -fake_scores.mean()

    metrics = {
        "gen_adv_loss": gen_loss.item(),
        "student_score_mean": fake_scores.mean().item(),
    }

    return gen_loss, metrics
