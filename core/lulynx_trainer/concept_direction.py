# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Concept Direction Training — learn LoRA weights encoding concept directions.

Trains a LoRA to capture the difference between paired prompts (e.g.
"a smiling person" vs "a person"). At inference, scaling the LoRA
controls concept intensity without retraining.

The loss encourages the LoRA-augmented model to amplify the base model's
concept direction: MSE(lora_direction, guidance_scale * base_direction).
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConceptDirectionPair:
    """A (positive, negative) prompt pair defining a concept direction."""
    positive: str
    negative: str


class ConceptDirectionTrainer:
    """Concept Direction Training: learn LoRA from paired prompts."""

    def __init__(
        self,
        pairs: List[ConceptDirectionPair],
        weight: float = 1.0,
        guidance_scale: float = 1.0,
        timestep_range: Optional[Tuple[int, int]] = None,
        neutral_reg: float = 0.0,
    ) -> None:
        if not pairs:
            raise ValueError("At least one concept direction pair is required")
        self._pairs = pairs
        self._weight = weight
        self._guidance_scale = guidance_scale
        self._timestep_range = timestep_range
        self._neutral_reg = neutral_reg

    @staticmethod
    def parse_pairs(json_str: str) -> List[ConceptDirectionPair]:
        """Parse JSON config into concept direction pairs."""
        if not json_str or not json_str.strip():
            return []
        try:
            raw = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Invalid concept_direction_pairs JSON")
            return []
        if not isinstance(raw, list):
            return []
        pairs = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            pos = str(item.get("positive", "")).strip()
            neg = str(item.get("negative", "")).strip()
            if pos and neg:
                pairs.append(ConceptDirectionPair(positive=pos, negative=neg))
        return pairs

    @staticmethod
    def parse_timestep_range(range_str: str) -> Optional[Tuple[int, int]]:
        """Parse 'lo,hi' string into (lo, hi) tuple."""
        if not range_str or not range_str.strip():
            return None
        parts = range_str.strip().split(",")
        if len(parts) != 2:
            return None
        try:
            lo, hi = int(parts[0].strip()), int(parts[1].strip())
            if lo < hi:
                return (lo, hi)
        except ValueError:
            pass
        return None

    def sample_pair(self, rng: Optional[random.Random] = None) -> ConceptDirectionPair:
        """Sample a random pair for this training step."""
        if rng is not None:
            return rng.choice(self._pairs)
        return random.choice(self._pairs)

    def sample_timesteps(
        self,
        batch_size: int,
        num_train_timesteps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample timesteps, optionally restricted to configured range."""
        if self._timestep_range is not None:
            lo, hi = self._timestep_range
            return torch.randint(lo, min(hi, num_train_timesteps), (batch_size,), device=device)
        return torch.randint(0, num_train_timesteps, (batch_size,), device=device)

    def compute_direction_loss(
        self,
        pred_positive: torch.Tensor,
        pred_negative: torch.Tensor,
        base_positive: torch.Tensor,
        base_negative: torch.Tensor,
    ) -> torch.Tensor:
        """Compute concept direction loss from model outputs.

        Args:
            pred_positive: UNet output with positive prompt (LoRA active)
            pred_negative: UNet output with negative prompt (LoRA active)
            base_positive: UNet output with positive prompt (LoRA disabled)
            base_negative: UNet output with negative prompt (LoRA disabled)

        Returns:
            Scalar direction loss (weight already applied).
        """
        base_direction = (base_positive - base_negative).detach().float()
        lora_direction = (pred_positive - pred_negative).float()

        loss = self._weight * torch.nn.functional.mse_loss(
            lora_direction, self._guidance_scale * base_direction
        )

        if self._neutral_reg > 0:
            loss = loss + self._neutral_reg * torch.nn.functional.mse_loss(
                pred_negative.float(), base_negative.detach().float()
            )

        return loss

    @property
    def pairs(self) -> List[ConceptDirectionPair]:
        return list(self._pairs)

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale
