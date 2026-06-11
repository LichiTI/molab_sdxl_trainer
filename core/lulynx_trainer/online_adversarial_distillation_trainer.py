"""Online adversarial distillation (Mode A) training integration.

LICENSE: PolyForm Noncommercial 1.0.0
Cleanroom implementation - no AGPL code referenced.

Mode A implements full GAN-style distillation:
- Teacher generates real outputs (N steps)
- Student generates fake outputs (few steps)
- Critic discriminates between real/fake
- Student trained with consistency loss + adversarial loss
- Critic trained to distinguish teacher/student outputs
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn

from .distillation_integration import DistillationIntegration
from .fake_critic import compute_critic_loss

logger = logging.getLogger(__name__)


class OnlineAdversarialDistillationTrainer:
    """Handles online adversarial distillation (Mode A) training steps.

    Coordinates teacher, student, and critic training.
    """

    def __init__(
        self,
        integration: DistillationIntegration,
        vae_decoder: nn.Module,
        critic_update_ratio: int = 1,
        enabled: bool = True,
    ):
        """
        Parameters
        ----------
        integration : DistillationIntegration
            Distillation integration coordinator.
        vae_decoder : nn.Module
            VAE decoder to convert latents to images for critic.
        critic_update_ratio : int, default 1
            How many critic updates per generator update.
        enabled : bool, default True
            Whether distillation is enabled.
        """
        self.integration = integration
        self.vae_decoder = vae_decoder
        self.critic_update_ratio = critic_update_ratio
        self.enabled = enabled

        # Validate critic exists
        if self.enabled and integration.critic is None:
            logger.warning(
                "Online adversarial mode enabled but no critic found. "
                "Distillation will be disabled."
            )
            self.enabled = False

        self._critic_update_counter = 0

        logger.info(
            f"Online adversarial distillation trainer initialized: "
            f"enabled={self.enabled}, "
            f"has_critic={integration.critic is not None}, "
            f"critic_update_ratio={critic_update_ratio}"
        )

    def compute_adversarial_distillation_loss(
        self,
        model: nn.Module,
        sampler_fn: Callable,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        **sampler_kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute adversarial distillation loss (generator step).

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

        # Compute distillation step via integration (includes adversarial loss)
        loss, metrics = self.integration.compute_distillation_step(
            model=model,
            sampler_fn=sampler_fn,
            latents=latents,
            prompt_embeds=prompt_embeds,
            teacher_output=None,
            vae_decoder=self.vae_decoder,
            **sampler_kwargs,
        )

        return loss, metrics

    def update_critic_step(
        self,
        model: nn.Module,
        sampler_fn: Callable,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        **sampler_kwargs,
    ) -> dict:
        """Update critic network (discriminator step).

        Parameters
        ----------
        model : nn.Module
            Student model.
        sampler_fn : callable
            Sampling function.
        latents : torch.Tensor
            Initial latents.
        prompt_embeds : torch.Tensor
            Prompt embeddings.
        **sampler_kwargs
            Additional sampler parameters.

        Returns
        -------
        dict
            Critic metrics.
        """
        if not self.enabled or self.integration.critic is None:
            return {}

        # Generate teacher and student outputs
        with torch.no_grad():
            # Teacher (real)
            teacher_latents = self.integration.sampler.sample_for_distillation(
                model=model,
                sampler_fn=sampler_fn,
                latents=latents,
                prompt_embeds=prompt_embeds,
                **sampler_kwargs,
            )[0]  # Get teacher output only

            teacher_images = self.vae_decoder.decode(teacher_latents)

        # Student (fake)
        student_latents = self.integration.sampler.sample_for_distillation(
            model=model,
            sampler_fn=sampler_fn,
            latents=latents,
            prompt_embeds=prompt_embeds,
            **sampler_kwargs,
        )[1]  # Get student output only

        with torch.no_grad():
            student_images = self.vae_decoder.decode(student_latents)

        # Update critic
        metrics = self.integration.update_critic(
            teacher_images=teacher_images,
            student_images=student_images,
        )

        self._critic_update_counter += 1

        return metrics

    def should_update_critic(self, global_step: int) -> bool:
        """Check if critic should be updated at current step.

        Parameters
        ----------
        global_step : int
            Current global training step.

        Returns
        -------
        bool
            Whether to update critic.
        """
        if not self.enabled:
            return False

        # Update critic every N generator updates
        return (self._critic_update_counter % self.critic_update_ratio) == 0

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
            "mode": "online_adversarial",
            "teacher_steps": self.integration.config.teacher_steps,
            "student_steps": self.integration.config.student_steps,
            "has_critic": self.integration.critic is not None,
            "critic_arch": self.integration.config.online_adv_critic_arch,
            "critic_weight": self.integration.config.online_adv_critic_weight,
            "critic_update_ratio": self.critic_update_ratio,
        }


def create_online_adversarial_distillation_trainer_from_config(
    config,
    device: torch.device,
    vae_decoder: nn.Module,
    critic_update_ratio: int = 1,
) -> Optional[OnlineAdversarialDistillationTrainer]:
    """Create online adversarial distillation trainer from training config.

    Parameters
    ----------
    config : object
        Training configuration object.
    device : torch.device
        Device for computation.
    vae_decoder : nn.Module
        VAE decoder for latent-to-image conversion.
    critic_update_ratio : int, default 1
        Critic updates per generator update.

    Returns
    -------
    OnlineAdversarialDistillationTrainer or None
        Trainer instance if enabled and mode is online_adv, None otherwise.
    """
    from .distillation_integration import create_distillation_integration_from_config

    integration = create_distillation_integration_from_config(config, device)

    if integration is None:
        return None

    # Only create trainer if mode is online_adv
    if integration.config.mode != "online_adv":
        return None

    trainer = OnlineAdversarialDistillationTrainer(
        integration=integration,
        vae_decoder=vae_decoder,
        critic_update_ratio=critic_update_ratio,
        enabled=True,
    )

    return trainer
