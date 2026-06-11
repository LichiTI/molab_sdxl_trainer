# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""HydraLoRA / MoE-LoRA — mixture-of-experts LoRA layer (Phase 8.3 / #111).

A standard LoRA layer applies a single rank-r update.  HydraLoRA
maintains *N* parallel LoRA experts and a small gating MLP that
produces a soft-mix of expert outputs::

    y = base(x) + sum_e gate_e(x) * (lora_up_e @ lora_down_e @ x) * scale

Two routing modes are supported:

  * ``"dense"`` — soft-mix all experts (smooth, fully differentiable).
  * ``"top_k"`` — pick the top-k experts per sample, normalize, mix.

The module is a drop-in replacement for ``LoRALinear``: it wraps an
existing ``nn.Linear`` and exposes the same interface (``forward``,
``get_trainable_params``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class HydraLoRAConfig:
    """Configuration for a HydraLoRA layer."""

    num_experts: int = 4
    rank: int = 8
    alpha: float = 8.0
    routing: str = "top_k"     # "dense" or "top_k"
    top_k: int = 2
    sparse_top_k: bool = False
    gate_init_std: float = 0.01
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# Core layer
# ---------------------------------------------------------------------------

class HydraLoRALinear(nn.Module):
    """LoRA wrapper with ``num_experts`` parallel rank-r adapters and a gate."""

    def __init__(self, original: nn.Linear, config: HydraLoRAConfig) -> None:
        super().__init__()
        if config.num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if config.routing not in {"dense", "top_k"}:
            raise ValueError(f"unknown routing: {config.routing}")
        if config.routing == "top_k" and config.top_k > config.num_experts:
            raise ValueError("top_k cannot exceed num_experts")

        self.original = original
        for p in self.original.parameters():
            p.requires_grad = False

        in_features = original.in_features
        out_features = original.out_features

        self.config = config
        self.scaling = config.alpha / max(config.rank, 1)

        # Per-expert down/up projections, all stored in single tensors for speed
        self.lora_down = nn.Parameter(
            torch.empty(config.num_experts, config.rank, in_features)
        )
        self.lora_up = nn.Parameter(
            torch.empty(config.num_experts, out_features, config.rank)
        )
        nn.init.kaiming_uniform_(self.lora_down, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up)

        # Gate: maps input features to expert logits
        self.gate = nn.Linear(in_features, config.num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=config.gate_init_std)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        x_d = self.dropout(x)
        logits = self.gate(x)  # [..., E]

        if self.config.routing == "dense":
            weights = F.softmax(logits, dim=-1)
            mixed = self._dense_mixed_delta(x_d, weights)
        else:
            k = self._top_k_size()
            if self.config.sparse_top_k and k < self.config.num_experts:
                mixed = self._top_k_sparse_mixed_delta(x_d, logits)
            else:
                weights = self._top_k_weights(logits)
                mixed = self._dense_mixed_delta(x_d, weights)

        return base_out + mixed

    def _dense_mixed_delta(self, x_d: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # Compute per-expert deltas: x -> down -> up.
        proj = torch.einsum("...i,eri->...er", x_d, self.lora_down)
        deltas = torch.einsum("...er,eor->...eo", proj, self.lora_up)
        deltas = deltas * self.scaling
        return (weights.unsqueeze(-1) * deltas).sum(dim=-2)

    def _top_k_size(self) -> int:
        return max(1, min(int(self.config.top_k), int(self.config.num_experts)))

    def _top_k_values_and_indices(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = self._top_k_size()
        topk_vals, topk_idx = logits.topk(k, dim=-1)
        normalized = F.softmax(topk_vals, dim=-1)
        return normalized.to(dtype=logits.dtype), topk_idx

    def _top_k_sparse_mixed_delta(self, x_d: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        weights, topk_idx = self._top_k_values_and_indices(logits)
        flat_x = x_d.reshape(-1, x_d.shape[-1])
        flat_weights = weights.reshape(-1, weights.shape[-1])
        flat_indices = topk_idx.reshape(-1, topk_idx.shape[-1])
        mixed_flat = flat_x.new_zeros(flat_x.shape[0], self.original.out_features)

        for slot_idx in range(flat_indices.shape[-1]):
            selected = flat_indices[:, slot_idx]
            selected_down = self.lora_down.index_select(0, selected)
            selected_up = self.lora_up.index_select(0, selected)
            projected = torch.bmm(selected_down, flat_x.unsqueeze(-1)).squeeze(-1)
            delta = torch.bmm(selected_up, projected.unsqueeze(-1)).squeeze(-1)
            mixed_flat = mixed_flat + delta * flat_weights[:, slot_idx].unsqueeze(-1)

        mixed_flat = mixed_flat * self.scaling
        return mixed_flat.reshape(*x_d.shape[:-1], self.original.out_features)

    def _top_k_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute sparse normalised weights selecting top-k experts."""
        normalized, topk_idx = self._top_k_values_and_indices(logits)
        weights = torch.zeros_like(logits, dtype=logits.dtype)
        weights.scatter_(-1, topk_idx, normalized)
        return weights

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [self.lora_down, self.lora_up, self.gate.weight]

    def expert_balance_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Importance-balance auxiliary loss (variance across experts).

        Encourages roughly uniform expert utilisation.  Caller should add
        this to the main loss with a small weight (e.g. 0.01).
        """
        weights = F.softmax(logits, dim=-1)
        mean_per_expert = weights.flatten(end_dim=-2).mean(dim=0)
        return ((mean_per_expert - mean_per_expert.mean()) ** 2).sum()
