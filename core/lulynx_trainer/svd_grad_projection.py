# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""SVD Gradient Projection — low-rank gradient projection for memory-efficient training.

Projects gradients into a low-rank SVD subspace, reducing optimizer state
memory from O(m*n) to O(r*(m+n)) per 2D parameter where r << min(m, n).

The projection basis is periodically recomputed from the current gradient
via truncated SVD. Between updates, gradients are projected into and
reconstructed from the stored basis.

Integration: optimizer wrapper in the chain, same pattern as
StochasticRoundingOptimizerWrapper and GradientGuardOptimizerWrapper.
"""

from __future__ import annotations

from math import prod
from typing import Any, Dict, Optional

import torch


class _GradProjector:
    """Per-parameter SVD projection basis."""

    def __init__(
        self,
        shape: torch.Size,
        rank: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        m = shape[0]
        n = prod(shape[1:]) if len(shape) > 1 else 1
        self._rank = min(rank, min(m, n))
        self._shape = shape
        self._m = m
        self._n = n
        self._P: Optional[torch.Tensor] = None  # (m, r)
        self._Q: Optional[torch.Tensor] = None  # (r, n)

    def update_basis(self, grad: torch.Tensor) -> None:
        """Recompute projection basis from current gradient via truncated SVD."""
        G = grad.reshape(self._m, self._n).float()
        try:
            U, S, Vt = torch.linalg.svd(G, full_matrices=False)
        except torch.linalg.LinAlgError:
            return
        self._P = U[:, : self._rank].to(grad.dtype)
        self._Q = Vt[: self._rank, :].to(grad.dtype)

    def project(self, grad: torch.Tensor) -> torch.Tensor:
        """Project gradient into low-rank subspace and reconstruct."""
        if self._P is None or self._Q is None:
            return grad
        G = grad.reshape(self._m, self._n)
        projected = self._P.T @ G @ self._Q.T  # (r, r)
        reconstructed = self._P @ projected @ self._Q  # (m, n)
        return reconstructed.reshape(self._shape)


class SVDGradientProjectionWrapper(torch.optim.Optimizer):
    """Optimizer wrapper that projects gradients into a low-rank SVD subspace."""

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        rank: int = 128,
        update_interval: int = 200,
        scale: float = 1.0,
        warmup_steps: int = 0,
    ) -> None:
        self._base = base_optimizer
        self._initializing_optimizer = True
        torch.optim.Optimizer.__init__(self, base_optimizer.param_groups, base_optimizer.defaults)
        self._initializing_optimizer = False
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state
        self.defaults = base_optimizer.defaults
        self._rank = rank
        self._update_interval = max(1, update_interval)
        self._scale = scale
        self._warmup_steps = warmup_steps
        self._step_count = 0
        self._projectors: Dict[int, _GradProjector] = {}

    def step(self, closure=None):
        self._step_count += 1

        if self._step_count <= self._warmup_steps:
            return self._base.step(closure)

        effective_step = self._step_count - self._warmup_steps
        should_update = (effective_step - 1) % self._update_interval == 0

        for group in self._base.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.dim() < 2:
                    continue
                proj = self._get_or_create_projector(p)
                if should_update:
                    proj.update_basis(p.grad.data)
                p.grad.data = proj.project(p.grad.data) * self._scale

        return self._base.step(closure)

    def _get_or_create_projector(self, param: torch.Tensor) -> _GradProjector:
        pid = id(param)
        if pid not in self._projectors:
            self._projectors[pid] = _GradProjector(
                param.shape, self._rank, param.device, param.dtype
            )
        return self._projectors[pid]

    def zero_grad(self, set_to_none: bool = True):
        return self._base.zero_grad(set_to_none)

    def state_dict(self):
        return self._base.state_dict()

    def load_state_dict(self, state_dict):
        return self._base.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        if getattr(self, "_initializing_optimizer", False):
            return torch.optim.Optimizer.add_param_group(self, param_group)
        if hasattr(self, "_base"):
            return self._base.add_param_group(param_group)
        return torch.optim.Optimizer.add_param_group(self, param_group)

    def __repr__(self):
        return (
            f"SVDGradientProjectionWrapper(rank={self._rank}, "
            f"update_interval={self._update_interval}, "
            f"scale={self._scale}, base={self._base!r})"
        )


def apply_svd_gradient_projection(
    optimizer: torch.optim.Optimizer,
    enabled: bool = False,
    rank: int = 128,
    update_interval: int = 200,
    scale: float = 1.0,
    warmup_steps: int = 0,
) -> torch.optim.Optimizer:
    """Conditionally wrap an optimizer with SVD gradient projection."""
    if not enabled:
        return optimizer
    return SVDGradientProjectionWrapper(
        optimizer, rank=rank, update_interval=update_interval,
        scale=scale, warmup_steps=warmup_steps,
    )

