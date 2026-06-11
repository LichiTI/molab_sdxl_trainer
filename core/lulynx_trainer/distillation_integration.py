"""Distillation integration coordinator for training loop.

LICENSE: PolyForm Noncommercial 1.0.0
Cleanroom implementation - no AGPL code referenced.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn

from .distillation_config import DistillationConfig, DistillationMode
from .distillation_loss import create_distillation_loss, compute_distillation_loss
from .distillation_sampler import create_distillation_sampler
from .fake_critic import create_critic, compute_critic_loss, compute_generator_loss

logger = logging.getLogger(__name__)


class DistillationIntegration:
    """Coordinates distillation components in training loop.

    Manages:
    - Configuration
    - Loss computation
    - Sampler coordination
    - Critic network (for adversarial mode)
    - Metrics tracking
    """

    def __init__(
        self,
        config: DistillationConfig,
        device: torch.device,
    ):
        """
        Parameters
        ----------
        config : DistillationConfig
            Distillation configuration.
        device : torch.device
            Device for computation.
        """
        self.config = config.normalized()
        self.device = device
        self.mode = self.config.get_mode_enum()

        # Create loss function
        self.loss_fn = create_distillation_loss(
            mode=self.config.mode,
            use_perceptual=self.config.consistency_use_perceptual,
            perceptual_weight=self.config.consistency_perceptual_weight,
        )

        # Create sampler
        self.sampler = create_distillation_sampler(
            mode=self.config.mode,
            teacher_steps=self.config.teacher_steps,
            student_steps=self.config.student_steps,
            teacher_guidance=self.config.consistency_teacher_guidance,
            student_guidance=self.config.consistency_student_guidance,
        )

        # Create critic for adversarial mode
        self.critic = None
        self.critic_optimizer = None
        if self.mode == DistillationMode.ONLINE_ADVERSARIAL:
            if self.config.online_adv_critic_enabled:
                self.critic = create_critic(
                    arch=self.config.online_adv_critic_arch,
                    in_channels=3,
                    base_channels=64,
                ).to(device)

                logger.info(
                    f"Created critic for adversarial distillation: "
                    f"arch={self.config.online_adv_critic_arch}"
                )

        logger.info(
            f"Distillation integration initialized: mode={self.config.mode}, "
            f"teacher_steps={self.config.teacher_steps}, "
            f"student_steps={self.config.student_steps}"
        )

    def set_critic_optimizer(self, optimizer):
        """Set optimizer for critic network.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer for critic.
        """
        self.critic_optimizer = optimizer

    def compute_distillation_step(
        self,
        model: nn.Module,
        sampler_fn: Callable,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        teacher_output: Optional[torch.Tensor] = None,
        vae_decoder: Optional[nn.Module] = None,
        **sampler_kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute distillation loss for one training step.

        Parameters
        ----------
        model : nn.Module
            Student model (trainable).
        sampler_fn : callable
            Sampling function.
        latents : torch.Tensor
            Initial latents.
        prompt_embeds : torch.Tensor
            Prompt embeddings.
        teacher_output : torch.Tensor, optional
            Pre-cached teacher output (for guided dataset mode).
        vae_decoder : nn.Module, optional
            VAE decoder (for adversarial mode, decode latents to images).
        **sampler_kwargs
            Additional sampler kwargs.

        Returns
        -------
        tuple of (torch.Tensor, dict)
            (total_loss, metrics_dict)
        """
        metrics = {}

        # Sample teacher and student outputs
        if self.mode == DistillationMode.GUIDED_DATASET:
            if teacher_output is None:
                raise ValueError("Guided dataset mode requires teacher_output")

            teacher_out, student_out, sample_metrics = self.sampler.sample_with_cached_teacher(
                model=model,
                sampler_fn=sampler_fn,
                latents=latents,
                prompt_embeds=prompt_embeds,
                teacher_output=teacher_output,
                **sampler_kwargs,
            )
        else:
            # Online modes
            teacher_out, student_out, sample_metrics = self.sampler.sample_for_distillation(
                model=model,
                sampler_fn=sampler_fn,
                latents=latents,
                prompt_embeds=prompt_embeds,
                **sampler_kwargs,
            )

        metrics.update(sample_metrics)

        # Compute distillation loss
        distill_loss, distill_metrics = compute_distillation_loss(
            loss_fn=self.loss_fn,
            student_output=student_out,
            teacher_output=teacher_out,
            weight=self.config.weight,
        )

        metrics.update(distill_metrics)

        total_loss = distill_loss

        # Add adversarial loss if enabled
        if self.mode == DistillationMode.ONLINE_ADVERSARIAL and self.critic is not None:
            if vae_decoder is None:
                logger.warning(
                    "Adversarial mode requires vae_decoder to decode latents, "
                    "skipping critic loss"
                )
            else:
                # Decode latents to images
                with torch.no_grad():
                    teacher_images = vae_decoder.decode(teacher_out)
                student_images = vae_decoder.decode(student_out)

                # Generator loss (fool discriminator)
                gen_loss, gen_metrics = compute_generator_loss(
                    critic=self.critic,
                    fake_images=student_images,
                )

                adv_loss = self.config.online_adv_critic_weight * gen_loss
                total_loss = total_loss + adv_loss

                metrics.update(gen_metrics)
                metrics["distill_adv_loss"] = adv_loss.item()

        metrics["distill_total_loss"] = total_loss.item()

        return total_loss, metrics

    def update_critic(
        self,
        teacher_images: torch.Tensor,
        student_images: torch.Tensor,
    ) -> dict:
        """Update critic network (for adversarial mode).

        Parameters
        ----------
        teacher_images : torch.Tensor
            Real images from teacher.
        student_images : torch.Tensor
            Fake images from student.

        Returns
        -------
        dict
            Critic metrics.
        """
        if self.critic is None or self.critic_optimizer is None:
            return {}

        # Compute critic loss
        critic_loss, critic_metrics = compute_critic_loss(
            critic=self.critic,
            real_images=teacher_images,
            fake_images=student_images,
        )

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_metrics

    def get_config_summary(self) -> dict:
        """Get configuration summary.

        Returns
        -------
        dict
            Configuration summary.
        """
        return {
            "enabled": self.config.enabled,
            "mode": self.config.mode,
            "teacher_steps": self.config.teacher_steps,
            "student_steps": self.config.student_steps,
            "weight": self.config.weight,
            "has_critic": self.critic is not None,
        }


def create_distillation_integration_from_config(
    config,
    device: torch.device,
) -> Optional[DistillationIntegration]:
    """Create distillation integration from training config.

    Parameters
    ----------
    config : object
        Training configuration object.
    device : torch.device
        Device for computation.

    Returns
    -------
    DistillationIntegration or None
        Integration instance if enabled, None otherwise.
    """
    from .distillation_config import create_distillation_config_from_training_config

    distill_config = create_distillation_config_from_training_config(config)

    if distill_config is None or not distill_config.enabled:
        return None

    integration = DistillationIntegration(
        config=distill_config,
        device=device,
    )

    return integration
