# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Manifold tracker — PCA-based weight trajectory visualization.

Periodically snapshots the flattened LoRA weight vector to CPU and, on
request, runs PCA (via ``torch.pca_lowrank``) to project the trajectory
into 3D for interactive visualization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ManifoldPoint:
    step: int
    x: float
    y: float
    z: float
    loss: float

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ManifoldResult:
    points: List[ManifoldPoint]
    variance_explained: List[float]
    num_params: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "points": [p.as_dict() for p in self.points],
            "variance_explained": self.variance_explained,
            "num_params": self.num_params,
        }


class ManifoldTracker:
    """Collects weight snapshots and projects them to 3D via PCA."""

    def __init__(self, max_snapshots: int = 500) -> None:
        self._snapshots: List[Tuple[int, torch.Tensor, float]] = []
        self._max_snapshots = max_snapshots
        self._last_result: Optional[ManifoldResult] = None

    @property
    def num_snapshots(self) -> int:
        return len(self._snapshots)

    def snapshot(
        self,
        step: int,
        trainable_params: Iterable[nn.Parameter],
        loss: float,
    ) -> None:
        """Capture current LoRA weights to CPU."""
        parts = [p.detach().cpu().flatten() for p in trainable_params if p.requires_grad]
        if not parts:
            return
        flat = torch.cat(parts)
        self._snapshots.append((step, flat, loss))
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots :]

    def compute_pca(self) -> Optional[ManifoldResult]:
        """Run PCA on all snapshots, returning 3D coordinates.

        Needs at least 4 snapshots to produce a meaningful projection.
        """
        if len(self._snapshots) < 4:
            return None

        steps = [s[0] for s in self._snapshots]
        losses = [s[2] for s in self._snapshots]
        matrix = torch.stack([s[1] for s in self._snapshots]).float()
        num_params = matrix.shape[1]

        mean = matrix.mean(dim=0, keepdim=True)
        centered = matrix - mean

        q = min(3, centered.shape[0] - 1, centered.shape[1])
        if q < 3:
            return None

        U, S, V = torch.pca_lowrank(centered, q=q)
        coords = U[:, :3] * S[:3].unsqueeze(0)

        total_var = (centered ** 2).sum()
        explained = [(float(S[i] ** 2) / max(float(total_var), 1e-12)) for i in range(min(3, len(S)))]

        points = [
            ManifoldPoint(
                step=steps[i],
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                z=float(coords[i, 2]),
                loss=losses[i],
            )
            for i in range(len(steps))
        ]

        result = ManifoldResult(
            points=points,
            variance_explained=explained,
            num_params=num_params,
        )
        self._last_result = result
        return result

    @property
    def last_result(self) -> Optional[ManifoldResult]:
        return self._last_result

    def reset(self) -> None:
        self._snapshots = []
        self._last_result = None
