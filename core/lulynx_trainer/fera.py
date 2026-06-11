# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""FeRA adapter layer for Linear modules."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class FeRALinear(nn.Module):
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.original = original_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_down = nn.Linear(original_layer.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, original_layer.out_features, bias=False)
        self.residual_gate = nn.Parameter(torch.full((original_layer.out_features,), float(gate_init)))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        delta = self.lora_up(self.dropout(self.lora_down(x))) * self.scaling
        return base + delta * self.residual_gate.view(*([1] * (delta.dim() - 1)), -1)

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [self.lora_down.weight, self.lora_up.weight, self.residual_gate]
