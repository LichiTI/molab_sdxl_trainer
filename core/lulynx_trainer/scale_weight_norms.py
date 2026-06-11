# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""LoRA weight-norm clipping (#67).

After each optimizer step, clip the L2 norm of each LoRA pair so the
combined ``up @ down`` delta never exceeds a configured threshold.
This is a common stabilisation trick for high learning rates and
prevents adapter weights from drifting into degenerate magnitudes.

The clipping is applied **in-place** on the adapter weight tensors.

Usage::

    from .scale_weight_norms import LoRAWeightNormClipper

    clipper = LoRAWeightNormClipper(max_norm=1.0)
    clipper.register_from_injector(injector)

    # In the training loop, after optimizer.step():
    clipper.step()
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRAWeightNormClipper:
    """Per-step L2 clipping of LoRA up/down weights.

    Parameters
    ----------
    max_norm : float
        L2 ceiling for the combined ``up @ down`` matrix.  ``0.0`` disables.
    interval : int
        Apply clipping every N calls.  Default 1 = every step.
    eps : float
        Numerical safety constant for the norm denominator.
    """

    def __init__(
        self,
        *,
        max_norm: float = 1.0,
        interval: int = 1,
        eps: float = 1e-6,
    ) -> None:
        self.max_norm = float(max_norm)
        self.interval = max(1, int(interval))
        self.eps = float(eps)
        self._step_counter = 0
        self._layers: List[Tuple[str, nn.Module]] = []

    def register_layer(self, name: str, layer: nn.Module) -> None:
        self._layers.append((name, layer))

    def register_from_injector(self, injector: object) -> int:
        layers = getattr(injector, "injected_layers", None)
        if not isinstance(layers, dict):
            return 0
        added = 0
        for name, layer in layers.items():
            self.register_layer(name, layer)
            added += 1
        return added

    @torch.no_grad()
    def step(self) -> int:
        """Apply one round of weight-norm clipping.  Returns the number of layers clipped."""
        if self.max_norm <= 0.0:
            return 0

        self._step_counter += 1
        if self._step_counter % self.interval != 0:
            return 0

        clipped = 0
        for _, layer in self._layers:
            if self._clip_layer_inplace(layer):
                clipped += 1
        return clipped

    def reset(self) -> None:
        self._step_counter = 0
        self._layers.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_pair(self, layer: nn.Module) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return (up_weight, down_weight) for known LoRA wrapper layouts."""
        adapter = getattr(layer, "lora", layer)

        for down_attr, up_attr in (
            ("lora_down", "lora_up"),
            ("lora_A", "lora_B"),
        ):
            down = getattr(adapter, down_attr, None)
            up = getattr(adapter, up_attr, None)
            down_w = getattr(down, "weight", down) if down is not None else None
            up_w = getattr(up, "weight", up) if up is not None else None
            if isinstance(down_w, torch.Tensor) and isinstance(up_w, torch.Tensor):
                return (up_w, down_w)
        return None

    def _clip_layer_inplace(self, layer: nn.Module) -> bool:
        pair = self._find_pair(layer)
        if pair is None:
            return False
        up, down = pair

        # Compute the L2 norm of the combined delta matrix
        delta = (up.float() @ down.float())
        current_norm = float(torch.linalg.norm(delta).item())
        if current_norm <= self.max_norm + self.eps:
            return False

        # Distribute the clipping factor evenly between up and down by sqrt
        scale = self.max_norm / (current_norm + self.eps)
        sqrt_scale = scale ** 0.5
        up.data.mul_(sqrt_scale)
        down.data.mul_(sqrt_scale)
        return True


def apply_weight_norm_clipping(
    injector: object,
    *,
    max_norm: float = 1.0,
) -> int:
    """One-shot helper: clip every LoRA layer in the injector once.

    Returns the number of layers that needed clipping.
    """
    clipper = LoRAWeightNormClipper(max_norm=max_norm)
    clipper.register_from_injector(injector)
    return clipper.step()
