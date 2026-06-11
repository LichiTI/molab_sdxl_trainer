"""Blockwise fused optimizer: groups parameters by model block for fused updates with per-group LR scheduling."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn


class BlockwiseFusedOptimizer:
    """Groups parameters by block for fused updates with per-group LR scheduling."""

    def __init__(self, model: nn.Module, base_lr: float = 1e-3, lr_decay: float = 0.9):
        self.param_groups = []
        self._schedulers = []
        self.lr_decay = lr_decay

        # Group parameters by block name
        block_params = defaultdict(list)
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            parts = name.split(".")
            # Find "blocks.N" pattern
            block_key = None
            for i, p in enumerate(parts):
                if p == "blocks" and i + 1 < len(parts):
                    block_key = f"block_{parts[i + 1]}"
                    break
            if block_key:
                block_params[block_key].append(param)
            else:
                other_params.append(param)

        # Create per-block optimizer param groups
        for block_name, params in sorted(block_params.items()):
            group = {"params": params, "lr": base_lr, "name": block_name}
            self.param_groups.append(group)

        if other_params:
            self.param_groups.append({"params": other_params, "lr": base_lr, "name": "other"})

        # Create the underlying optimizer
        self.optimizer = torch.optim.AdamW(self.param_groups)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def scheduler_step(self, group_name: str):
        """Apply a single LR scheduler step to a specific parameter group."""
        for group in self.optimizer.param_groups:
            if group["name"] == group_name:
                group["lr"] *= self.lr_decay
                return True
        return False


def maybe_replace_with_blockwise_fused(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    config: Any,
) -> torch.optim.Optimizer:
    """Create a BlockwiseFusedOptimizer when ``config.blockwise_fused_optimizers`` is True.

    Returns the original *optimizer* unchanged when the flag is False.
    """
    if not getattr(config, "blockwise_fused_optimizers", False):
        return optimizer

    base_lr = optimizer.defaults.get("lr", 1e-3)
    return BlockwiseFusedOptimizer(model, base_lr=base_lr)
