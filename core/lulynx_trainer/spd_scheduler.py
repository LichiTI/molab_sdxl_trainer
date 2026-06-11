"""SPD multi-resolution inference integration for samplers.

This module integrates SPD (Scale-Preserving Diffusion) multi-resolution
inference into the sampling pipeline.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

from .spd_inference import SpdInferenceConfig, build_spd_inference_plan, resize_latents_for_spd

logger = logging.getLogger(__name__)


class SPDMultiResolutionScheduler:
    """Scheduler for SPD multi-resolution inference.

    SPD progressively increases resolution during sampling, starting from
    lower resolution for initial steps and gradually moving to full resolution.

    Example
    -------
    >>> config = SpdInferenceConfig(scale_factors=(0.5, 1.0), steps_per_level=(10, 20))
    >>> scheduler = SPDMultiResolutionScheduler(config, latent_shape=(1, 4, 64, 64))
    >>> for step in range(30):
    ...     level_info = scheduler.get_level_for_step(step)
    ...     scaled_latents = scheduler.scale_latents(latents, step)
    """

    def __init__(
        self,
        config: SpdInferenceConfig,
        latent_shape: Tuple[int, int, int, int],
    ):
        """
        Parameters
        ----------
        config : SpdInferenceConfig
            SPD configuration.
        latent_shape : tuple of int
            Base latent shape (batch, channels, height, width).
        """
        self.config = config
        self.config.validate()
        self.latent_shape = latent_shape

        # Build inference plan
        self.plan = build_spd_inference_plan(latent_shape, config)
        self.levels = self.plan["levels"]
        self.total_steps = self.plan["total_steps"]

        logger.info(
            f"SPD scheduler initialized: {len(self.levels)} levels, "
            f"{self.total_steps} total steps"
        )

    def get_level_for_step(self, step: int) -> Dict:
        """Get the resolution level for a given step.

        Parameters
        ----------
        step : int
            Current sampling step.

        Returns
        -------
        dict
            Level information including scale, latent_shape, etc.
        """
        for level in self.levels:
            if level["step_start"] <= step < level["step_end_exclusive"]:
                return level

        # If step is beyond total steps, return last level
        return self.levels[-1]

    def scale_latents(
        self,
        latents: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Scale latents to the appropriate resolution for the current step.

        Parameters
        ----------
        latents : torch.Tensor
            Input latents at base resolution.
        step : int
            Current sampling step.

        Returns
        -------
        torch.Tensor
            Scaled latents.
        """
        level = self.get_level_for_step(step)
        target_shape = level["latent_shape"]
        target_h, target_w = target_shape[2], target_shape[3]

        # If already at target size, return as-is
        if latents.shape[2] == target_h and latents.shape[3] == target_w:
            return latents

        # Resize
        scaled = resize_latents_for_spd(
            latents,
            (target_h, target_w),
            mode=self.config.resize_mode,
        )

        return scaled

    def upscale_to_next_level(
        self,
        latents: torch.Tensor,
        current_step: int,
    ) -> Optional[torch.Tensor]:
        """Upscale latents when transitioning to next resolution level.

        Parameters
        ----------
        latents : torch.Tensor
            Current latents.
        current_step : int
            Current step.

        Returns
        -------
        torch.Tensor or None
            Upscaled latents, or None if no transition needed.
        """
        current_level = self.get_level_for_step(current_step)
        next_step = current_step + 1

        if next_step >= self.total_steps:
            return None

        next_level = self.get_level_for_step(next_step)

        # Check if transitioning to next level
        if next_level["level"] != current_level["level"]:
            target_shape = next_level["latent_shape"]
            target_h, target_w = target_shape[2], target_shape[3]

            upscaled = resize_latents_for_spd(
                latents,
                (target_h, target_w),
                mode=self.config.resize_mode,
            )

            logger.debug(
                f"SPD level transition: {current_level['level']} -> {next_level['level']}, "
                f"scale {current_level['scale']:.2f} -> {next_level['scale']:.2f}"
            )

            return upscaled

        return None

    def get_schedule_info(self) -> Dict:
        """Get full schedule information.

        Returns
        -------
        dict
            Complete SPD schedule plan.
        """
        return self.plan.copy()


class SPDSamplerWrapper:
    """Wraps a standard sampler to add SPD multi-resolution support.

    This wrapper intercepts latent processing to scale them according
    to the SPD schedule.
    """

    def __init__(
        self,
        scheduler: SPDMultiResolutionScheduler,
        enabled: bool = True,
    ):
        """
        Parameters
        ----------
        scheduler : SPDMultiResolutionScheduler
            SPD scheduler.
        enabled : bool, default True
            Whether SPD is enabled.
        """
        self.scheduler = scheduler
        self.enabled = enabled
        self._current_step = 0

    def prepare_latents(
        self,
        latents: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Prepare latents for current step (scale if needed).

        Parameters
        ----------
        latents : torch.Tensor
            Input latents.
        step : int
            Current step.

        Returns
        -------
        torch.Tensor
            Scaled latents.
        """
        if not self.enabled:
            return latents

        return self.scheduler.scale_latents(latents, step)

    def post_step_processing(
        self,
        latents: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Post-step processing (handle level transitions).

        Parameters
        ----------
        latents : torch.Tensor
            Latents after current step.
        step : int
            Current step.

        Returns
        -------
        torch.Tensor
            Processed latents (upscaled if transitioning).
        """
        if not self.enabled:
            return latents

        upscaled = self.scheduler.upscale_to_next_level(latents, step)
        return upscaled if upscaled is not None else latents


def create_spd_scheduler_from_config(
    config,
    latent_shape: Tuple[int, int, int, int],
) -> Optional[SPDMultiResolutionScheduler]:
    """Create SPD scheduler from training/sampling config.

    Parameters
    ----------
    config : object
        Configuration object.
    latent_shape : tuple of int
        Base latent shape (batch, channels, height, width).

    Returns
    -------
    SPDMultiResolutionScheduler or None
        Scheduler if enabled, None otherwise.
    """
    if not getattr(config, "spd_enabled", False):
        return None

    scale_factors = getattr(config, "spd_scale_factors", (0.5, 1.0))
    steps_per_level = getattr(config, "spd_steps_per_level", (10, 20))
    resize_mode = getattr(config, "spd_resize_mode", "bilinear")
    model_family = getattr(config, "model_arch", "anima")

    spd_config = SpdInferenceConfig(
        model_family=model_family,
        scale_factors=tuple(scale_factors),
        steps_per_level=tuple(steps_per_level),
        resize_mode=resize_mode,
    )

    scheduler = SPDMultiResolutionScheduler(spd_config, latent_shape)

    logger.info(
        f"SPD scheduler created: scales={scale_factors}, "
        f"steps={steps_per_level}, total={scheduler.total_steps}"
    )

    return scheduler
