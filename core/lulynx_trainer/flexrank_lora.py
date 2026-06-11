# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""FlexRank LoRA — dynamic rank low-rank adaptation.

Each forward pass randomly samples an active rank from [min_rank, max_rank]
during training.  At inference time, the full max_rank is used.  Because
training touches every rank slice, a single checkpoint works at *any* rank
<= max_rank after training — no separate export needed.

The implementation uses rank-slicing on shared lora_down / lora_up weight
matrices, so memory cost is identical to standard LoRA at max_rank.

Integration: added as a new branch in lora_injector.py factory chain.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlexRankLoRALinear(nn.Module):
    """Drop-in LoRA linear with dynamic rank sampling during training."""

    def __init__(
        self,
        original_layer: nn.Linear,
        max_rank: int = 4,
        min_rank: int = 1,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.original = original_layer
        self.original.requires_grad_(False)

        self.max_rank = max_rank
        self.min_rank = max(1, min(min_rank, max_rank))
        self.alpha = alpha

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_down = nn.Linear(in_features, max_rank, bias=False)
        self.lora_up = nn.Linear(max_rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_down._lora_leaf = True
        self.lora_up._lora_leaf = True

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original(x)

        if self.training:
            r = torch.randint(self.min_rank, self.max_rank + 1, (1,)).item()
        else:
            r = self.max_rank

        scaling = self.alpha / r
        hidden = F.linear(x, self.lora_down.weight[:r, :])
        hidden = self.dropout(hidden)
        delta = F.linear(hidden, self.lora_up.weight[:, :r])
        return original_out + delta * scaling

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias
