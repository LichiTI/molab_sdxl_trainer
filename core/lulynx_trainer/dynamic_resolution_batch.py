# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Resolution-Aware Dynamic Micro-Batch — adjust effective batch size based on
input resolution to maintain consistent VRAM usage.

## Concept

Higher resolution images consume quadratically more VRAM (latent size scales
with resolution²).  A batch of 4 images at 512×512 uses roughly the same
VRAM as a batch of 1 image at 1024×1024.  This module dynamically adjusts
the gradient accumulation steps per resolution bucket so that:
- Small images (< base_resolution): more micro-batches per step → better gradient estimates
- Large images (> base_resolution): fewer micro-batches → fits in VRAM

The effective batch size is: ``train_batch_size * adjusted_accumulation_steps``

This is different from ``auto_controller.py``'s GSNR-based dynamic batch,
which adjusts batch size based on gradient signal-to-noise ratio.  This
module adjusts based on resolution alone (simpler, predictable).

Warehouse implementation using only public PyTorch APIs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "DynamicMicroBatchScheduler",
    "ResolutionBatchConfig",
]


@dataclass
class ResolutionBatchConfig:
    """Configuration for resolution-aware batch sizing."""

    base_resolution: int = 1024
    base_accumulation_steps: int = 1
    max_factor: float = 4.0
    min_factor: float = 0.25

    def validate(self) -> None:
        if self.base_resolution <= 0:
            raise ValueError(f"base_resolution must be positive, got {self.base_resolution}")
        if self.base_accumulation_steps <= 0:
            raise ValueError(f"base_accumulation_steps must be positive")
        if self.max_factor < 1.0:
            raise ValueError(f"max_factor must be >= 1.0, got {self.max_factor}")
        if self.min_factor <= 0.0 or self.min_factor > 1.0:
            raise ValueError(f"min_factor must be in (0, 1], got {self.min_factor}")


class DynamicMicroBatchScheduler:
    """Compute effective gradient accumulation steps based on image resolution.

    Usage::

        scheduler = DynamicMicroBatchScheduler(
            config=ResolutionBatchConfig(base_resolution=1024, base_accumulation_steps=4),
        )

        # Before each micro-batch group:
        batch_resolution = images.shape[-1]  # or max(H, W)
        effective_steps = scheduler.compute_accumulation_steps(batch_resolution)

        # Use effective_steps instead of the fixed gradient_accumulation_steps

    """

    def __init__(self, config: ResolutionBatchConfig) -> None:
        config.validate()
        self.config = config
        self._resolution_history: List[Tuple[int, int]] = []
        self._adjustment_count = 0

    def compute_accumulation_steps(
        self,
        resolution: int,
        total_microbatches_remaining: Optional[int] = None,
    ) -> int:
        """Compute the effective gradient accumulation steps for a given resolution.

        The relationship is: VRAM ∝ resolution².  So if the base resolution is
        1024 with accumulation_steps=4, then at 512 (1/4 the VRAM per sample),
        we can do 4× more accumulation steps.

        Args:
            resolution: The current batch resolution (max of H, W).
            total_microbatches_remaining: If provided, clamps the result to not
                exceed remaining micro-batches in the epoch.

        Returns:
            Adjusted gradient accumulation steps (always >= 1).
        """
        base_res = self.config.base_resolution
        base_steps = self.config.base_accumulation_steps

        if resolution <= 0:
            return base_steps

        vram_ratio = (base_res / resolution) ** 2

        factor = max(
            self.config.min_factor,
            min(self.config.max_factor, vram_ratio),
        )

        adjusted = max(1, round(base_steps * factor))

        if total_microbatches_remaining is not None:
            adjusted = min(adjusted, max(1, total_microbatches_remaining))

        if adjusted != base_steps:
            self._adjustment_count += 1

        self._resolution_history.append((resolution, adjusted))
        if len(self._resolution_history) > 1000:
            self._resolution_history = self._resolution_history[-500:]

        return adjusted

    def get_batch_resolution(self, batch: Dict[str, Any]) -> int:
        """Extract resolution from a training batch.

        Supports multiple batch formats:
        - ``batch["images"]``: tensor with shape [B, C, H, W]
        - ``batch["latents"]``: tensor with shape [B, C, H//8, W//8] (×8 for pixel)
        - ``batch["original_size"]``: explicit (H, W) tuple/list
        - ``batch["resolution"]``: explicit int
        """
        if "resolution" in batch:
            r = batch["resolution"]
            return int(r) if not isinstance(r, (list, tuple)) else max(int(x) for x in r)

        if "original_size" in batch:
            size = batch["original_size"]
            if isinstance(size, (list, tuple)):
                return max(int(x) for x in size[:2])

        if "images" in batch:
            t = batch["images"]
            if isinstance(t, torch.Tensor) and t.ndim >= 3:
                return max(t.shape[-2], t.shape[-1])

        if "latents" in batch:
            t = batch["latents"]
            if isinstance(t, torch.Tensor) and t.ndim >= 3:
                return max(t.shape[-2], t.shape[-1]) * 8

        return self.config.base_resolution

    @property
    def stats(self) -> Dict[str, Any]:
        if not self._resolution_history:
            return {
                "adjustment_count": 0,
                "avg_resolution": self.config.base_resolution,
                "avg_accumulation_steps": self.config.base_accumulation_steps,
            }

        resolutions = [r for r, _ in self._resolution_history]
        steps = [s for _, s in self._resolution_history]
        return {
            "adjustment_count": self._adjustment_count,
            "total_batches": len(self._resolution_history),
            "avg_resolution": sum(resolutions) / len(resolutions),
            "min_resolution": min(resolutions),
            "max_resolution": max(resolutions),
            "avg_accumulation_steps": sum(steps) / len(steps),
            "min_accumulation_steps": min(steps),
            "max_accumulation_steps": max(steps),
        }

