"""
VeRA (Vector-based Random Matrix Adaptation)

Shares frozen random projection matrices across all layers and learns
only per-layer scaling vectors (lambda_d, lambda_b), drastically reducing
trainable parameter count compared to standard LoRA.

Reference insight: delta = (lambda_b * B_shared[:out, :]) @ diag(lambda_d) @ A_shared[:, :in] @ x
"""

from __future__ import annotations

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class VeRASharedBuffers:
    """Global shared random projection buffers for all VeRA modules.

    Buffers are lazily grown to accommodate the largest layer dimensions
    encountered across all injected modules.  They are deterministic
    given the same PRNG seed.
    """

    def __init__(self, rank: int, prng_key: int = 0, device: torch.device = None):
        self.rank = rank
        self.prng_key = prng_key
        self.device = device or torch.device("cpu")
        self._max_in: int = 0
        self._max_out: int = 0
        # Registered as non-trainable buffers on a dummy module so they
        # follow .to() / .cuda() calls made by the host model.
        self._container = nn.Module()
        self._A: Optional[nn.Parameter] = None  # (rank, max_in)
        self._B: Optional[nn.Parameter] = None  # (max_out, rank)

    def _kaiming_init(self, shape: Tuple[int, ...], seed: int) -> torch.Tensor:
        """Deterministic Kaiming uniform init with given seed."""
        # Use manual_seed for determinism, then create tensor
        state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        tensor = torch.empty(shape, device=self.device)
        nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
        torch.random.set_rng_state(state)
        return tensor

    def ensure(self, in_features: int, out_features: int) -> None:
        """Grow shared buffers if necessary to cover *in_features* / *out_features*."""
        if in_features <= self._max_in and out_features <= self._max_out:
            return

        new_max_in = max(self._max_in, in_features)
        new_max_out = max(self._max_out, out_features)

        new_A = self._kaiming_init((self.rank, new_max_in), seed=self.prng_key)
        new_B = self._kaiming_init((new_max_out, self.rank), seed=self.prng_key + 1)

        # Preserve existing data in the top-left corner
        if self._A is not None:
            new_A[:, : self._max_in] = self._A.data[:, : self._max_in]
        if self._B is not None:
            new_B[: self._max_out, :] = self._B.data[: self._max_out, :]

        self._A = nn.Parameter(new_A, requires_grad=False)
        self._B = nn.Parameter(new_B, requires_grad=False)

        # Re-register on container so .to() propagates
        self._container._parameters.pop("vera_A", None)
        self._container._parameters.pop("vera_B", None)
        self._container.register_parameter("vera_A", self._A)
        self._container.register_parameter("vera_B", self._B)

        self._max_in = new_max_in
        self._max_out = new_max_out

    @property
    def shared_A(self) -> nn.Parameter:
        assert self._A is not None, "Call ensure() before accessing shared_A"
        return self._A

    @property
    def shared_B(self) -> nn.Parameter:
        assert self._B is not None, "Call ensure() before accessing shared_B"
        return self._B

    def to(self, *args, **kwargs):
        self._container.to(*args, **kwargs)
        self.device = next(self._container.parameters()).device
        return self


class VeRALinear(nn.Module):
    """VeRA adapter wrapping an nn.Linear layer.

    Forward:
        h = x @ (lambda_d * A[:, :in]).T      -- shared down with per-rank scale
        delta = h @ (lambda_b * B[:out, :]).T  -- shared up with per-output scale
        out = original(x) + delta
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        shared_buffers: VeRASharedBuffers,
        d_initial: float = 0.1,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.original = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = shared_buffers.rank
        self.alpha = alpha
        self.scaling = alpha / self.rank
        self.d_initial = d_initial

        # Ensure shared buffers cover this layer's dimensions
        shared_buffers.ensure(self.in_features, self.out_features)
        self._buffers_ref = shared_buffers

        # Per-layer trainable scaling vectors
        self.vera_lambda_d = nn.Parameter(
            torch.full((self.rank,), d_initial)
        )
        self.vera_lambda_b = nn.Parameter(
            torch.zeros(self.out_features)
        )

        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

        # Mark adapter leaves for BlockSwap
        self.vera_lambda_d._lora_leaf = True
        self.vera_lambda_b._lora_leaf = True
        self._block_weight_lr_scale = 1.0
        self._block_weight_frozen = False

    def _aligned_shared_slices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return shared VeRA buffers aligned to this layer's runtime device/dtype."""
        target_device = self.vera_lambda_d.device
        target_dtype = self.vera_lambda_d.dtype
        shared_a = self._buffers_ref.shared_A[:, : self.in_features].to(device=target_device, dtype=target_dtype)
        shared_b = self._buffers_ref.shared_B[: self.out_features, :].to(device=target_device, dtype=target_dtype)
        return shared_a, shared_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A_slice, B_slice = self._aligned_shared_slices()

        # Scale shared A by per-rank lambda_d
        scaled_A = self.vera_lambda_d.unsqueeze(1) * A_slice  # (rank, in)
        # Scale shared B by per-output lambda_b
        scaled_B = self.vera_lambda_b.unsqueeze(1) * B_slice  # (out, rank)

        h = F.linear(x, scaled_A)  # (..., rank)
        delta = F.linear(h, scaled_B)  # (..., out)

        original_out = self.original(x)
        return original_out + delta * self.scaling

    def get_weight_matrix(self) -> torch.Tensor:
        """Effective delta weight for analysis / export."""
        A, B = self._aligned_shared_slices()
        scaled_A = self.vera_lambda_d.unsqueeze(1) * A
        scaled_B = self.vera_lambda_b.unsqueeze(1) * B
        return (scaled_B @ scaled_A) * self.scaling

    def export_standard_lora_weights(self) -> Dict[str, torch.Tensor]:
        """Materialize into standard LoRA down/up weights for export."""
        A, B = self._aligned_shared_slices()
        down_weight = self.vera_lambda_d.unsqueeze(1) * A  # (rank, in)
        up_weight = self.vera_lambda_b.unsqueeze(1) * B  # (out, rank)
        return {
            "lora_down.weight": down_weight,
            "lora_up.weight": up_weight,
        }
