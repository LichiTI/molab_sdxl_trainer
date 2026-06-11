"""T-LoRA (Temporal LoRA) implementation.

T-LoRA adapts the effective rank during training using a schedule:
- constant: rank stays at min_rank
- linear: rank increases linearly from min_rank to max_rank over total_steps
- geometric: rank increases geometrically (exponentially) from min_rank to max_rank

The rank schedule is driven by the global training step, which the
training loop must push via ``set_global_step()`` on each step.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TLoRALinear(nn.Module):
    """LoRA layer with temporal rank scheduling.

    Internally allocates ``max_rank`` capacity but only uses the first
    ``current_rank`` columns/rows during forward.  Unused columns are
    zeroed so they contribute nothing to the output.

    Parameters
    ----------
    original_layer : nn.Linear
        The base linear layer being adapted.
    max_rank : int
        Maximum LoRA rank (allocated capacity).
    min_rank : int
        Starting / minimum rank.
    alpha : float
        LoRA alpha scaling factor.
    dropout : float
        Dropout probability (0 = no dropout).
    schedule : str
        ``"constant"``, ``"linear"``, or ``"geometric"``.
    total_steps : int
        Total training steps over which rank ramps up (used by
        ``"linear"`` and ``"geometric"`` schedules).
    orthogonal_init : bool
        If True, initialise lora_down with a random orthogonal matrix
        instead of Kaiming uniform.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        max_rank: int = 32,
        min_rank: int = 1,
        alpha: float = 16.0,
        dropout: float = 0.0,
        schedule: str = "constant",
        total_steps: int = 1000,
        orthogonal_init: bool = False,
    ):
        super().__init__()
        self.original = original_layer
        self.max_rank = max_rank
        self.min_rank = min(min_rank, max_rank)
        self.alpha = alpha
        self.schedule = schedule
        self.total_steps = max(total_steps, 1)
        self.orthogonal_init = orthogonal_init
        self._current_rank = self.min_rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Allocate full capacity
        self.lora_down = nn.Linear(in_features, max_rank, bias=False)
        self.lora_up = nn.Linear(max_rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        if orthogonal_init:
            self._orthogonal_init(max_rank, in_features)
        else:
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))

        # lora_up starts at zero so T-LoRA is a no-op at init
        nn.init.zeros_(self.lora_up.weight)

        # Mark LoRA leaves for BlockSwap compatibility
        self.lora_down._lora_leaf = True
        self.lora_up._lora_leaf = True

        # Mask buffer — ones for active columns, zeros for inactive
        self.register_buffer(
            "_rank_mask",
            torch.zeros(max_rank, dtype=torch.float32),
        )
        self._update_rank_mask()

    # -- initialisation helpers -------------------------------------------

    def _orthogonal_init(self, rows: int, cols: int) -> None:
        """Random orthogonal initialisation for lora_down."""
        with torch.no_grad():
            q = min(rows, cols)
            mat = torch.randn(q, cols, device=self.lora_down.weight.device)
            q_mat, _ = torch.linalg.qr(mat.T)  # (cols, q)
            self.lora_down.weight[:q].copy_(q_mat.T)
            if rows > q:
                self.lora_down.weight[q:].zero_()

    # -- rank scheduling --------------------------------------------------

    @property
    def current_rank(self) -> int:
        return self._current_rank

    def set_global_step(self, step: int) -> None:
        """Update the effective rank based on the current training step."""
        new_rank = self._compute_rank(step)
        if new_rank != self._current_rank:
            self._current_rank = new_rank
            self._update_rank_mask()

    def _compute_rank(self, step: int) -> int:
        if self.schedule == "constant":
            return self.min_rank

        progress = min(max(step / self.total_steps, 0.0), 1.0)

        if self.schedule == "linear":
            rank = self.min_rank + (self.max_rank - self.min_rank) * progress
        elif self.schedule == "geometric":
            if self.min_rank <= 0:
                return self.max_rank
            ratio = self.max_rank / self.min_rank
            rank = self.min_rank * (ratio ** progress)
        else:
            rank = self.min_rank

        return int(max(self.min_rank, min(round(rank), self.max_rank)))

    def _update_rank_mask(self) -> None:
        with torch.no_grad():
            self._rank_mask.zero_()
            self._rank_mask[: self._current_rank] = 1.0

    # -- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Base output + masked LoRA contribution."""
        base_out = self.original(x)

        # Masked LoRA: zero out inactive columns
        mask = self._rank_mask.to(x.device)
        down_out = self.lora_down(self.dropout(x))
        down_out = down_out * mask  # zero inactive dimensions
        lora_out = self.lora_up(down_out)

        scaling = self.alpha / self._current_rank
        return base_out + lora_out * scaling

    def get_weight_matrix(self) -> torch.Tensor:
        """Get merged LoRA weight (for analysis)."""
        mask = self._rank_mask.to(self.lora_down.weight.device)
        down = self.lora_down.weight * mask.unsqueeze(1)
        return (self.lora_up.weight @ down) * (self.alpha / self._current_rank)

    def merge_weights(self) -> None:
        """Merge LoRA weights into the base layer permanently."""
        with torch.no_grad():
            merged_w = self.original.weight.data + self.get_weight_matrix().to(self.original.weight.dtype)
            self.original.weight.data.copy_(merged_w)
            nn.init.zeros_(self.lora_up.weight)

    def state_dict_for_save(self) -> dict:
        """Return a serialisable state dict with only active rank weights."""
        return {
            "lora_down": self.lora_down.weight.data[: self._current_rank].clone(),
            "lora_up": self.lora_up.weight.data[:, : self._current_rank].clone(),
            "alpha": self.alpha,
            "current_rank": self._current_rank,
            "max_rank": self.max_rank,
            "min_rank": self.min_rank,
            "schedule": self.schedule,
        }
