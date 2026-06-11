"""
LoRA-FA (Frozen-A LoRA)

Standard LoRA structure where the down-projection (A) is frozen after
initialization and only the up-projection (B) is trained.  This halves
the trainable parameter count while preserving expressivity through the
random projection in A.

Compatible with standard LoRA checkpoint format — saved weights can be
loaded by any LoRA loader without special handling.
"""

from __future__ import annotations

import logging
import math
import torch
import torch.nn as nn
from typing import Dict

logger = logging.getLogger(__name__)


class LoRAFALinear(nn.Module):
    """LoRA-FA adapter: frozen A, trainable B."""

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # A (down) — frozen after init
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        std = math.sqrt(2.0 / (in_features + rank))
        nn.init.normal_(self.lora_down.weight, std=std)
        self.lora_down.weight.requires_grad = False

        # B (up) — trainable, zero-init
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        nn.init.zeros_(self.lora_up.weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original
        for param in self.original.parameters():
            param.requires_grad = False

        # Mark adapter leaves for BlockSwap
        self.lora_down._lora_leaf = True
        self.lora_up._lora_leaf = True
        self._block_weight_lr_scale = 1.0
        self._block_weight_frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.lora_up(self.dropout(self.lora_down(x)))
        return self.original(x) + delta * self.scaling

    def get_weight_matrix(self) -> torch.Tensor:
        """Effective delta weight for analysis."""
        return (self.lora_up.weight @ self.lora_down.weight) * self.scaling

    def get_trainable_params(self):
        """Only lora_up is trainable."""
        return list(self.lora_up.parameters())
