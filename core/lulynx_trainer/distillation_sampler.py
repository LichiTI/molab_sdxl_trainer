"""Distillation sampler for teacher-student inference.

LICENSE: PolyForm Noncommercial 1.0.0
Cleanroom implementation - no AGPL code referenced.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DistillationSamplerWrapper:
    """Wraps sampling logic for distillation training.

    Performs teacher inference (many steps) and student inference (few steps)
    on the same input, producing outputs for distillation loss computation.
    """

    def __init__(
        self,
        teacher_steps: int = 50,
        student_steps: int = 4,
        teacher_guidance: float = 7.5,
        student_guidance: float = 7.5,
    ):
        """
        Parameters
        ----------
        teacher_steps : int, default 50
            Number of sampling steps for teacher.
        student_steps : int, default 4
            Number of sampling steps for student.
        teacher_guidance : float, default 7.5
            Guidance scale for teacher.
        student_guidance : float, default 7.5
            Guidance scale for student.
        """
        self.teacher_steps = teacher_steps
        self.student_steps = student_steps
        self.teacher_guidance = teacher_guidance
        self.student_guidance = student_guidance

    def run_teacher_student_pair(
        self,
        model: nn.Module,
        sampler_fn: Callable,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        **sampler_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Run teacher and student sampling on the same input.

        Parameters
        ----------
        model : nn.Module
            Diffusion model (UNet or DiT).
        sampler_fn : callable
            Sampling function (e.g., sample_anima, sample_euler).
        latents : torch.Tensor
            Initial noise latents.
        prompt_embeds : torch.Tensor
            Text prompt embeddings.
        timesteps : torch.Tensor
            Timestep schedule (optional, can be None).
        **sampler_kwargs
            Additional kwargs passed to sampler_fn.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, dict)
            (teacher_output, student_output, metrics)
        """
        metrics = {}

        # Teacher inference (frozen, no gradients)
        with torch.no_grad():
            teacher_output = sampler_fn(
                model=model,
                latents=latents.clone(),
                prompt_embeds=prompt_embeds,
                num_steps=self.teacher_steps,
                guidance_scale=self.teacher_guidance,
                **sampler_kwargs,
            )

        # Student inference (trainable)
        student_output = sampler_fn(
            model=model,
            latents=latents.clone(),
            prompt_embeds=prompt_embeds,
            num_steps=self.student_steps,
            guidance_scale=self.student_guidance,
            **sampler_kwargs,
        )

        metrics["teacher_steps"] = self.teacher_steps
        metrics["student_steps"] = self.student_steps

        return teacher_output, student_output, metrics


class OnlineDistillationSampler:
    """Sampler for online distillation (Mode D and Mode A).

    Runs teacher and student inference in the same forward pass.
    """

    def __init__(
        self,
        wrapper: DistillationSamplerWrapper,
    ):
        """
        Parameters
        ----------
        wrapper : DistillationSamplerWrapper
            Sampler wrapper instance.
        """
        self.wrapper = wrapper

    def sample_for_distillation(
        self,
        model: nn.Module,
        sampler_fn: Callable,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Sample teacher and student outputs for distillation.

        Parameters
        ----------
        model : nn.Module
            Diffusion model.
        sampler_fn : callable
            Sampling function.
        latents : torch.Tensor
            Initial latents.
        prompt_embeds : torch.Tensor
            Prompt embeddings.
        **kwargs
            Additional sampler kwargs.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, dict)
            (teacher_output, student_output, metrics)
        """
        return self.wrapper.run_teacher_student_pair(
            model=model,
            sampler_fn=sampler_fn,
            latents=latents,
            prompt_embeds=prompt_embeds,
            timesteps=None,
            **kwargs,
        )


class CachedDistillationSampler:
    """Sampler for cached distillation (Mode C and Hybrid Mode D+C).

    Loads pre-generated teacher outputs from dataset instead of
    running teacher inference.
    """

    def __init__(
        self,
        student_steps: int = 4,
        student_guidance: float = 7.5,
    ):
        """
        Parameters
        ----------
        student_steps : int, default 4
            Number of sampling steps for student.
        student_guidance : float, default 7.5
            Guidance scale for student.
        """
        self.student_steps = student_steps
        self.student_guidance = student_guidance

    def sample_with_cached_teacher(
        self,
        model: nn.Module,
        sampler_fn: Callable,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        teacher_output: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Sample student output with pre-cached teacher output.

        Parameters
        ----------
        model : nn.Module
            Diffusion model.
        sampler_fn : callable
            Sampling function.
        latents : torch.Tensor
            Initial latents.
        prompt_embeds : torch.Tensor
            Prompt embeddings.
        teacher_output : torch.Tensor
            Pre-generated teacher output from dataset.
        **kwargs
            Additional sampler kwargs.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, dict)
            (teacher_output, student_output, metrics)
        """
        # Student inference
        student_output = sampler_fn(
            model=model,
            latents=latents,
            prompt_embeds=prompt_embeds,
            num_steps=self.student_steps,
            guidance_scale=self.student_guidance,
            **kwargs,
        )

        metrics = {
            "student_steps": self.student_steps,
            "teacher_cached": True,
        }

        return teacher_output, student_output, metrics


def create_distillation_sampler(
    mode: str,
    teacher_steps: int = 50,
    student_steps: int = 4,
    teacher_guidance: float = 7.5,
    student_guidance: float = 7.5,
):
    """Create distillation sampler for specified mode.

    Parameters
    ----------
    mode : str
        Distillation mode: "consistency", "guided_dataset", "online_adv", "hybrid".
    teacher_steps : int, default 50
        Teacher sampling steps.
    student_steps : int, default 4
        Student sampling steps.
    teacher_guidance : float, default 7.5
        Teacher guidance scale.
    student_guidance : float, default 7.5
        Student guidance scale.

    Returns
    -------
    OnlineDistillationSampler or CachedDistillationSampler
        Sampler instance for the specified mode.
    """
    mode = str(mode).strip().lower()

    if mode in ("consistency", "online_adv", "hybrid"):
        # Online modes use wrapper
        wrapper = DistillationSamplerWrapper(
            teacher_steps=teacher_steps,
            student_steps=student_steps,
            teacher_guidance=teacher_guidance,
            student_guidance=student_guidance,
        )
        return OnlineDistillationSampler(wrapper)

    elif mode == "guided_dataset":
        # Cached mode
        return CachedDistillationSampler(
            student_steps=student_steps,
            student_guidance=student_guidance,
        )

    else:
        logger.warning(f"Unknown distillation mode '{mode}', using online sampler")
        wrapper = DistillationSamplerWrapper(
            teacher_steps=teacher_steps,
            student_steps=student_steps,
            teacher_guidance=teacher_guidance,
            student_guidance=student_guidance,
        )
        return OnlineDistillationSampler(wrapper)
