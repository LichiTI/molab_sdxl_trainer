"""Consistency distillation (Mode D) training integration.

LICENSE: PolyForm Noncommercial 1.0.0
Cleanroom implementation - no AGPL code referenced.

Mode D implements LCM-style consistency distillation:
- Teacher runs N steps (e.g., 50)
- Student runs few steps (e.g., 4)
- Loss = MSE(student_output, teacher_output) + optional perceptual loss
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn

from .distillation_integration import DistillationIntegration

logger = logging.getLogger(__name__)


class ConsistencyDistillationTrainer:
    """Handles consistency distillation (Mode D) training steps.

    Integrates into the main training loop to add distillation loss.
    """

    def __init__(
        self,
        integration: DistillationIntegration,
        enabled: bool = True,
    ):
        """
        Parameters
        ----------
        integration : DistillationIntegration
            Distillation integration coordinator.
        enabled : bool, default True
            Whether distillation is enabled.
        """
        self.integration = integration
        self.enabled = enabled

        logger.info(
            f"Consistency distillation trainer initialized: "
            f"enabled={enabled}, "
            f"teacher_steps={integration.config.teacher_steps}, "
            f"student_steps={integration.config.student_steps}"
        )

    def compute_consistency_distillation_loss(
        self,
        model: nn.Module,
        sampler_fn: Callable,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        **sampler_kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute consistency distillation loss for one training step.

        Parameters
        ----------
        model : nn.Module
            Student model (trainable).
        sampler_fn : callable
            Sampling function.
        latents : torch.Tensor
            Initial noise latents [batch, channels, height, width].
        prompt_embeds : torch.Tensor
            Text prompt embeddings [batch, seq_len, hidden_dim].
        **sampler_kwargs
            Additional sampler parameters.

        Returns
        -------
        tuple of (torch.Tensor, dict)
            (distillation_loss, metrics)
        """
        if not self.enabled:
            return torch.tensor(0.0, device=latents.device), {}

        # Compute distillation step via integration
        loss, metrics = self.integration.compute_distillation_step(
            model=model,
            sampler_fn=sampler_fn,
            latents=latents,
            prompt_embeds=prompt_embeds,
            teacher_output=None,  # No pre-cached output in online mode
            vae_decoder=None,  # No VAE needed for consistency mode
            **sampler_kwargs,
        )

        return loss, metrics

    def should_apply_distillation(
        self,
        global_step: int,
        total_steps: int,
        warmup_steps: int = 0,
    ) -> bool:
        """Check if distillation should be applied at current step.

        Parameters
        ----------
        global_step : int
            Current global training step.
        total_steps : int
            Total training steps.
        warmup_steps : int, default 0
            Number of warmup steps before enabling distillation.

        Returns
        -------
        bool
            Whether to apply distillation.
        """
        if not self.enabled:
            return False

        # Skip during warmup
        if global_step < warmup_steps:
            return False

        return True

    def get_status_summary(self) -> dict:
        """Get trainer status summary.

        Returns
        -------
        dict
            Status summary.
        """
        return {
            "enabled": self.enabled,
            "mode": "consistency",
            "teacher_steps": self.integration.config.teacher_steps,
            "student_steps": self.integration.config.student_steps,
            "use_perceptual": self.integration.config.consistency_use_perceptual,
            "perceptual_weight": self.integration.config.consistency_perceptual_weight,
        }


def create_consistency_distillation_trainer_from_config(
    config,
    device: torch.device,
) -> Optional[ConsistencyDistillationTrainer]:
    """Create consistency distillation trainer from training config.

    Parameters
    ----------
    config : object
        Training configuration object.
    device : torch.device
        Device for computation.

    Returns
    -------
    ConsistencyDistillationTrainer or None
        Trainer instance if enabled and mode is consistency, None otherwise.
    """
    from .distillation_integration import create_distillation_integration_from_config

    integration = create_distillation_integration_from_config(config, device)

    if integration is None:
        return None

    # Only create trainer if mode is consistency
    if integration.config.mode != "consistency":
        return None

    trainer = ConsistencyDistillationTrainer(
        integration=integration,
        enabled=True,
    )

    return trainer


def add_consistency_distillation_to_training_step(
    trainer_instance,
    model: nn.Module,
    sampler_fn: Callable,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    base_loss: torch.Tensor,
    global_step: int,
    **sampler_kwargs,
) -> Tuple[torch.Tensor, dict]:
    """Add consistency distillation loss to training step.

    This is a convenience function to be called from the main training loop.

    Parameters
    ----------
    trainer_instance : ConsistencyDistillationTrainer
        Distillation trainer instance.
    model : nn.Module
        Student model.
    sampler_fn : callable
        Sampling function.
    latents : torch.Tensor
        Initial latents.
    prompt_embeds : torch.Tensor
        Prompt embeddings.
    base_loss : torch.Tensor
        Base training loss (e.g., diffusion loss).
    global_step : int
        Current global step.
    **sampler_kwargs
        Additional sampler parameters.

    Returns
    -------
    tuple of (torch.Tensor, dict)
        (combined_loss, metrics)
    """
    if trainer_instance is None:
        return base_loss, {}

    # Check if should apply distillation
    if not trainer_instance.should_apply_distillation(
        global_step=global_step,
        total_steps=100000,  # Placeholder, should come from config
        warmup_steps=0,
    ):
        return base_loss, {}

    # Compute distillation loss
    distill_loss, distill_metrics = trainer_instance.compute_consistency_distillation_loss(
        model=model,
        sampler_fn=sampler_fn,
        latents=latents,
        prompt_embeds=prompt_embeds,
        **sampler_kwargs,
    )

    # Combine with base loss
    combined_loss = base_loss + distill_loss

    # Add combined loss to metrics
    distill_metrics["combined_loss"] = combined_loss.item()
    distill_metrics["base_loss"] = base_loss.item()

    return combined_loss, distill_metrics
