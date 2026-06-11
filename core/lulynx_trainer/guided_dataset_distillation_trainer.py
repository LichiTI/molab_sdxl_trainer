"""Guided dataset distillation (Mode C) training integration.

LICENSE: PolyForm Noncommercial 1.0.0
Cleanroom implementation - no AGPL code referenced.

Mode C implements guided dataset distillation:
- Pre-generate teacher outputs offline
- Load teacher outputs from dataset during training
- Student trains on cached teacher outputs
- No online teacher inference during training
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn

from .distillation_integration import DistillationIntegration

logger = logging.getLogger(__name__)


class GuidedDatasetDistillationTrainer:
    """Handles guided dataset distillation (Mode C) training steps.

    Uses pre-generated teacher outputs instead of online inference.
    """

    def __init__(
        self,
        integration: DistillationIntegration,
        dataset_path: str,
        enabled: bool = True,
    ):
        """
        Parameters
        ----------
        integration : DistillationIntegration
            Distillation integration coordinator.
        dataset_path : str
            Path to pre-generated dataset.
        enabled : bool, default True
            Whether distillation is enabled.
        """
        self.integration = integration
        self.dataset_path = dataset_path
        self.enabled = enabled

        # Validate dataset path
        if self.enabled and dataset_path:
            if not os.path.exists(dataset_path):
                logger.warning(
                    f"Guided dataset path does not exist: {dataset_path}. "
                    f"Distillation will be disabled."
                )
                self.enabled = False

        logger.info(
            f"Guided dataset distillation trainer initialized: "
            f"enabled={self.enabled}, "
            f"dataset_path={dataset_path}, "
            f"student_steps={integration.config.student_steps}"
        )

    def compute_guided_distillation_loss(
        self,
        model: nn.Module,
        sampler_fn: Callable,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        teacher_output: torch.Tensor,
        **sampler_kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute guided distillation loss using cached teacher output.

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
        teacher_output : torch.Tensor
            Pre-generated teacher output from dataset.
        **sampler_kwargs
            Additional sampler parameters.

        Returns
        -------
        tuple of (torch.Tensor, dict)
            (distillation_loss, metrics)
        """
        if not self.enabled:
            return torch.tensor(0.0, device=latents.device), {}

        # Compute distillation step via integration (with cached teacher)
        loss, metrics = self.integration.compute_distillation_step(
            model=model,
            sampler_fn=sampler_fn,
            latents=latents,
            prompt_embeds=prompt_embeds,
            teacher_output=teacher_output,
            vae_decoder=None,
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
            "mode": "guided_dataset",
            "dataset_path": self.dataset_path,
            "student_steps": self.integration.config.student_steps,
        }


class GuidedDatasetPreGenerator:
    """Pre-generates teacher outputs for guided dataset distillation.

    This is run offline before training to cache teacher outputs.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        teacher_sampler_fn: Callable,
        teacher_steps: int = 50,
        teacher_guidance: float = 7.5,
        output_dir: str = "./distilled_data",
    ):
        """
        Parameters
        ----------
        teacher_model : nn.Module
            Teacher model for inference.
        teacher_sampler_fn : callable
            Teacher sampling function.
        teacher_steps : int, default 50
            Number of steps for teacher inference.
        teacher_guidance : float, default 7.5
            Teacher guidance scale.
        output_dir : str, default "./distilled_data"
            Output directory for cached data.
        """
        self.teacher_model = teacher_model
        self.teacher_sampler_fn = teacher_sampler_fn
        self.teacher_steps = teacher_steps
        self.teacher_guidance = teacher_guidance
        self.output_dir = output_dir

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Guided dataset pre-generator initialized: "
            f"teacher_steps={teacher_steps}, "
            f"output_dir={output_dir}"
        )

    @torch.no_grad()
    def generate_teacher_outputs(
        self,
        latents_batch: torch.Tensor,
        prompt_embeds_batch: torch.Tensor,
        batch_idx: int,
        **sampler_kwargs,
    ) -> torch.Tensor:
        """Generate teacher outputs for a batch.

        Parameters
        ----------
        latents_batch : torch.Tensor
            Batch of initial latents.
        prompt_embeds_batch : torch.Tensor
            Batch of prompt embeddings.
        batch_idx : int
            Batch index for saving.
        **sampler_kwargs
            Additional sampler parameters.

        Returns
        -------
        torch.Tensor
            Teacher outputs.
        """
        # Run teacher inference
        teacher_outputs = self.teacher_sampler_fn(
            model=self.teacher_model,
            latents=latents_batch,
            prompt_embeds=prompt_embeds_batch,
            num_steps=self.teacher_steps,
            guidance_scale=self.teacher_guidance,
            **sampler_kwargs,
        )

        # Save to disk
        save_path = Path(self.output_dir) / f"teacher_output_batch_{batch_idx}.pt"
        torch.save(
            {
                "teacher_output": teacher_outputs.cpu(),
                "latents": latents_batch.cpu(),
                "prompt_embeds": prompt_embeds_batch.cpu(),
            },
            save_path,
        )

        logger.info(f"Saved teacher outputs for batch {batch_idx} to {save_path}")

        return teacher_outputs

    def get_generation_summary(self) -> dict:
        """Get generation summary.

        Returns
        -------
        dict
            Summary of generated data.
        """
        # Count generated files
        num_files = len(list(Path(self.output_dir).glob("teacher_output_batch_*.pt")))

        return {
            "output_dir": self.output_dir,
            "num_batches_generated": num_files,
            "teacher_steps": self.teacher_steps,
        }


def create_guided_dataset_distillation_trainer_from_config(
    config,
    device: torch.device,
) -> Optional[GuidedDatasetDistillationTrainer]:
    """Create guided dataset distillation trainer from training config.

    Parameters
    ----------
    config : object
        Training configuration object.
    device : torch.device
        Device for computation.

    Returns
    -------
    GuidedDatasetDistillationTrainer or None
        Trainer instance if enabled and mode is guided_dataset, None otherwise.
    """
    from .distillation_integration import create_distillation_integration_from_config

    integration = create_distillation_integration_from_config(config, device)

    if integration is None:
        return None

    # Only create trainer if mode is guided_dataset
    if integration.config.mode != "guided_dataset":
        return None

    dataset_path = integration.config.guided_dataset_path

    trainer = GuidedDatasetDistillationTrainer(
        integration=integration,
        dataset_path=dataset_path,
        enabled=True,
    )

    return trainer


def load_teacher_output_from_dataset(
    dataset_path: str,
    batch_idx: int,
    device: torch.device,
) -> Optional[dict]:
    """Load pre-generated teacher output from dataset.

    Parameters
    ----------
    dataset_path : str
        Path to dataset directory.
    batch_idx : int
        Batch index to load.
    device : torch.device
        Device to load data to.

    Returns
    -------
    dict or None
        Dictionary with teacher_output, latents, prompt_embeds, or None if not found.
    """
    file_path = Path(dataset_path) / f"teacher_output_batch_{batch_idx}.pt"

    if not file_path.exists():
        logger.warning(f"Teacher output file not found: {file_path}")
        return None

    data = torch.load(file_path, map_location=device)

    return data
