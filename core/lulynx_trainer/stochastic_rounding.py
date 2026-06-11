# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Stochastic Rounding — unbiased rounding for low-precision parameter updates.

## Problem

When training in bf16/fp16, the standard round-to-nearest mode introduces
systematic downward bias: if a parameter update is smaller than half the
precision gap at the parameter's magnitude, it always rounds to zero.
Over many steps, this means tiny-but-consistent gradient signals are lost.

## Solution

Stochastic rounding rounds up with probability proportional to the fractional
part, and down otherwise.  In expectation, the rounded value equals the true
value — no systematic bias.

    Given true value x and representable neighbors x_lo, x_hi:
        P(round to x_hi) = (x - x_lo) / (x_hi - x_lo)
        P(round to x_lo) = 1 - P(round to x_hi)

## Implementation

Two integration modes:
1. **Wrapper** (`stochastic_round_`): call after any optimizer step to
   re-round the param data in-place.
2. **Optimizer mixin** (`StochasticRoundingMixin`): wraps .step() to
   automatically apply stochastic rounding after every update.

Warehouse implementation using only public PyTorch APIs.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

import torch
from torch import Tensor

__all__ = [
    "stochastic_round_",
    "apply_stochastic_rounding_to_optimizer",
    "StochasticRoundingOptimizerWrapper",
]

logger = logging.getLogger(__name__)


def stochastic_round_(tensor: Tensor) -> Tensor:
    """In-place stochastic rounding of a low-precision (bf16/fp16) tensor.

    For each element, computes the full-precision value, finds the two nearest
    representable values in the target dtype, and probabilistically rounds to
    one of them.

    Only acts on bf16/fp16 tensors.  fp32 tensors are returned unchanged
    (their precision gap is too small for this to matter in practice).

    Returns the tensor for chaining.
    """
    if tensor.dtype not in (torch.bfloat16, torch.float16):
        return tensor

    with torch.no_grad():
        fp32 = tensor.float()

        floored = fp32.to(tensor.dtype).float()

        diff = fp32 - floored

        ulp = _compute_ulp(floored, tensor.dtype)

        frac = diff / ulp.clamp(min=1e-45)
        frac = frac.clamp(0.0, 1.0)

        noise = torch.rand_like(frac)
        rounded = torch.where(noise < frac, floored + ulp, floored)

        tensor.copy_(rounded.to(tensor.dtype))

    return tensor


def _compute_ulp(floored: Tensor, dtype: torch.dtype) -> Tensor:
    """Compute the unit-in-the-last-place (ULP) for each element.

    ULP is the gap between the floored value and the next representable value
    in the target dtype.
    """
    next_up = (floored.to(dtype) + torch.finfo(dtype).tiny).float()
    ulp = (next_up - floored).abs()
    ulp = ulp.clamp(min=torch.finfo(dtype).tiny)
    return ulp


def stochastic_round_params_(params: Iterable[torch.nn.Parameter]) -> int:
    """Apply stochastic rounding to all parameters in the iterable.

    Returns the number of parameters rounded.
    """
    count = 0
    for p in params:
        if p.dtype in (torch.bfloat16, torch.float16):
            stochastic_round_(p.data)
            count += 1
    return count


class StochasticRoundingOptimizerWrapper:
    """Wraps an optimizer to apply stochastic rounding after each step.

    Usage::

        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer = StochasticRoundingOptimizerWrapper(base_optimizer)

        # Use optimizer normally:
        optimizer.step()      # calls base.step() + stochastic rounding
        optimizer.zero_grad() # delegates to base

    The wrapper exposes the same interface as the base optimizer for
    compatibility with lr schedulers and training loops.
    """

    def __init__(self, base_optimizer: torch.optim.Optimizer) -> None:
        self._base = base_optimizer
        self._round_count = 0
        self._step_count = 0

    def step(self, closure=None):
        result = self._base.step(closure)
        self._step_count += 1

        for group in self._base.param_groups:
            for p in group["params"]:
                if p.dtype in (torch.bfloat16, torch.float16) and p.grad is not None or p.requires_grad:
                    stochastic_round_(p.data)
                    self._round_count += 1

        return result

    def zero_grad(self, set_to_none: bool = True):
        return self._base.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self._base.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self._base.param_groups = value

    @property
    def state(self):
        return self._base.state

    def state_dict(self):
        return self._base.state_dict()

    def load_state_dict(self, state_dict):
        return self._base.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        return self._base.add_param_group(param_group)

    @property
    def defaults(self):
        return self._base.defaults

    @property
    def stats(self):
        return {
            "step_count": self._step_count,
            "round_count": self._round_count,
        }

    def __repr__(self):
        return f"StochasticRoundingOptimizerWrapper({self._base!r})"


def apply_stochastic_rounding_to_optimizer(
    optimizer: torch.optim.Optimizer,
    enabled: bool = True,
) -> torch.optim.Optimizer:
    """Conditionally wrap an optimizer with stochastic rounding.

    If *enabled* is False, returns the optimizer unchanged.
    """
    if not enabled:
        return optimizer
    if isinstance(optimizer, StochasticRoundingOptimizerWrapper):
        return optimizer
    logger.info("Stochastic rounding enabled for %s", type(optimizer).__name__)
    return StochasticRoundingOptimizerWrapper(optimizer)

