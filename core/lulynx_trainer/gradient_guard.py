# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Anima Gradient Guard — Adaptive Gradient Clipping (AGC) and Gradient Centralization.

Two orthogonal gradient pre-processing techniques wrapped as an optimizer
decorator (same pattern as StochasticRoundingOptimizerWrapper).

AGC (Adaptive Gradient Clipping)
    Clips each parameter's gradient so that its norm does not exceed
    ``clip_factor * ||param||``.  Unlike fixed max_grad_norm, the threshold
    adapts to each layer's weight magnitude — large weights tolerate larger
    gradients, small weights are protected.

Gradient Centralization
    Subtracts the mean from each gradient tensor along all dimensions except
    the first (output) dimension, for tensors with 2+ dimensions.  This acts
    as a free regulariser and can accelerate convergence.

Both techniques are applied *before* the base optimizer's step().

Integration: inserted into the optimizer wrapper chain in trainer.py after
stochastic rounding.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

__all__ = ["GradientGuardOptimizerWrapper", "apply_gradient_guard"]

logger = logging.getLogger(__name__)


class GradientGuardOptimizerWrapper:
    """Wraps an optimizer to apply AGC and/or gradient centralization before each step."""

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        strategy: str = "none",
        agc_clip_factor: float = 0.01,
        agc_eps: float = 1e-3,
    ) -> None:
        self._base = base_optimizer
        self._strategy = strategy
        self._agc_clip_factor = agc_clip_factor
        self._agc_eps = agc_eps
        self._step_count = 0

    def step(self, closure=None):
        if "agc" in self._strategy:
            self._apply_agc()
        if "centralized" in self._strategy:
            self._apply_centralization()
        self._step_count += 1
        return self._base.step(closure)

    @torch.no_grad()
    def _apply_agc(self):
        for group in self._base.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p_norm = p.data.norm(2).clamp(min=self._agc_eps)
                g_norm = p.grad.data.norm(2)
                if g_norm < 1e-12:
                    continue
                max_norm = p_norm * self._agc_clip_factor
                if g_norm > max_norm:
                    p.grad.data.mul_(max_norm / g_norm)

    @torch.no_grad()
    def _apply_centralization(self):
        for group in self._base.param_groups:
            for p in group["params"]:
                if p.grad is None or p.grad.dim() < 2:
                    continue
                p.grad.data.sub_(
                    p.grad.data.mean(dim=tuple(range(1, p.grad.dim())), keepdim=True)
                )

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

    def __repr__(self):
        return (
            f"GradientGuardOptimizerWrapper(strategy={self._strategy!r}, "
            f"base={self._base!r})"
        )


def apply_gradient_guard(
    optimizer: torch.optim.Optimizer,
    strategy: str = "none",
    agc_clip_factor: float = 0.01,
    agc_eps: float = 1e-3,
) -> torch.optim.Optimizer:
    """Conditionally wrap an optimizer with gradient guard."""
    if strategy == "none" or not strategy:
        return optimizer
    if isinstance(optimizer, GradientGuardOptimizerWrapper):
        return optimizer
    logger.info("Gradient guard enabled: strategy=%s for %s", strategy, type(optimizer).__name__)
    return GradientGuardOptimizerWrapper(
        optimizer,
        strategy=strategy,
        agc_clip_factor=agc_clip_factor,
        agc_eps=agc_eps,
    )
