# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Zero-overhead gradient covariance tracker.

Captures the return value of ``clip_grad_norm_`` (already computed, currently
discarded) and derives lightweight statistics every step:

* ``grad_norm``       — raw total norm (free)
* ``grad_norm_ema``   — exponential moving average
* ``grad_norm_var``   — Welford online variance
* ``grad_cosine``     — cosine similarity with previous step (optional)
* ``fisher_diag``     — mean(grad²) across trainable parameters
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


@dataclass
class GradSnapshot:
    norm: float
    norm_ema: float
    norm_var: float
    cosine_sim: Optional[float]
    fisher_diag: float

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class GradientCovarianceTracker:
    """Accumulates gradient statistics with near-zero compute overhead."""

    def __init__(self, smooth_factor: float = 0.1) -> None:
        self._alpha = smooth_factor
        self._ema = 0.0
        self._welford_count = 0
        self._welford_mean = 0.0
        self._welford_m2 = 0.0
        self._prev_flat_grad: Optional[torch.Tensor] = None
        self._track_cosine = False
        self._last_snapshot: Optional[GradSnapshot] = None

    def enable_cosine(self) -> None:
        self._track_cosine = True

    def disable_cosine(self) -> None:
        self._track_cosine = False
        self._prev_flat_grad = None

    def update(
        self,
        grad_norm: float,
        trainable_params: Iterable[nn.Parameter],
    ) -> GradSnapshot:
        self._ema = self._alpha * grad_norm + (1.0 - self._alpha) * self._ema
        self._welford_count += 1
        delta = grad_norm - self._welford_mean
        self._welford_mean += delta / self._welford_count
        delta2 = grad_norm - self._welford_mean
        self._welford_m2 += delta * delta2

        variance = (
            self._welford_m2 / self._welford_count
            if self._welford_count > 1
            else 0.0
        )

        cosine_sim: Optional[float] = None
        fisher_diag = 0.0
        total_numel = 0

        params_list = [
            p for p in trainable_params if p.grad is not None
        ]

        if params_list:
            sq_sum = 0.0
            for p in params_list:
                g = p.grad
                sq_sum += float(g.pow(2).sum())
                total_numel += g.numel()
            fisher_diag = sq_sum / max(total_numel, 1)

            if self._track_cosine:
                flat_grad = torch.cat(
                    [p.grad.detach().flatten() for p in params_list]
                )
                if self._prev_flat_grad is not None:
                    dot = torch.dot(flat_grad, self._prev_flat_grad)
                    n1 = flat_grad.norm()
                    n2 = self._prev_flat_grad.norm()
                    denom = n1 * n2
                    cosine_sim = float(dot / denom) if denom > 0 else 0.0
                self._prev_flat_grad = flat_grad

        snap = GradSnapshot(
            norm=grad_norm,
            norm_ema=self._ema,
            norm_var=variance,
            cosine_sim=cosine_sim,
            fisher_diag=fisher_diag,
        )
        self._last_snapshot = snap
        return snap

    @property
    def last_snapshot(self) -> Optional[GradSnapshot]:
        return self._last_snapshot

    def reset(self) -> None:
        self._ema = 0.0
        self._welford_count = 0
        self._welford_mean = 0.0
        self._welford_m2 = 0.0
        self._prev_flat_grad = None
        self._last_snapshot = None
