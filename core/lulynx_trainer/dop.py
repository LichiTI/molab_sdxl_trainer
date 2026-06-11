# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""DOP — Differential Output Preservation.

Prevents concept forgetting by regularizing the training model's noise
predictions to stay close to a frozen reference copy. The reference is
a deepcopy of the model taken before LoRA injection, ensuring it
represents the original pretrained behavior.

Integration: created in trainer.py before LoRA injection, loss computed
in training_loop.py after the main diffusion loss.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentialOutputPreservation:
    """Regularize model output against a frozen reference."""

    def __init__(
        self,
        reference_model: nn.Module,
        weight: float = 0.1,
        target: str = "output",
        start_step: int = 0,
        interval: int = 1,
    ) -> None:
        self._reference = reference_model
        self._reference.eval()
        for p in self._reference.parameters():
            p.requires_grad_(False)
        self._weight = weight
        self._target = target
        self._start_step = start_step
        self._interval = max(1, interval)

    def should_compute(self, global_step: int) -> bool:
        """Return True if DOP loss should be computed at this step."""
        if global_step < self._start_step:
            return False
        return (global_step - self._start_step) % self._interval == 0

    def compute_loss(
        self,
        current_output: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """Compute DOP regularization loss.

        Args:
            current_output: noise prediction from the training model
            noisy_latents: input to the UNet
            timesteps: diffusion timesteps
            encoder_hidden_states: text encoder hidden states
            added_cond_kwargs: optional additional conditioning (SDXL)

        Returns:
            Scalar DOP loss (weight already applied).
        """
        fwd_kwargs = dict(
            sample=noisy_latents.detach(),
            timestep=timesteps.detach(),
            encoder_hidden_states=encoder_hidden_states.detach(),
        )
        if added_cond_kwargs is not None:
            fwd_kwargs["added_cond_kwargs"] = {
                k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in added_cond_kwargs.items()
            }

        with torch.no_grad():
            ref_output = self._reference(**fwd_kwargs)
            if hasattr(ref_output, "sample"):
                ref_output = ref_output.sample

        return self._weight * F.mse_loss(
            current_output.float(), ref_output.float().detach()
        )

    def to_device(self, device: torch.device | str) -> "DifferentialOutputPreservation":
        """Move reference model to device."""
        self._reference = self._reference.to(device)
        return self

    def offload_to_cpu(self) -> None:
        """Move reference to CPU to save VRAM between steps."""
        self._reference = self._reference.cpu()

    @property
    def reference_model(self) -> nn.Module:
        return self._reference

    @property
    def weight(self) -> float:
        return self._weight
