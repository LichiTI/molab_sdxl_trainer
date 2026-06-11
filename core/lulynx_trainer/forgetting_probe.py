# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Forgetting probe — fixed-anchor loss tracker.

Captures a small fixed set of batches from the validation/eval dataloader at
the start of training.  Periodically re-evaluates loss on these *same*
batches and compares against the baseline (step-0) loss.  A rising ratio
indicates the model is forgetting prior knowledge.

Requires a validation or eval dataloader to be available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ForgettingSnapshot:
    step: int
    anchor_loss: float
    baseline_loss: float
    ratio: float
    trend: str
    score: float

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class ForgettingProbe:
    """Track forgetting by periodically evaluating loss on fixed anchor batches."""

    def __init__(
        self,
        num_anchors: int = 4,
        warning_ratio: float = 1.3,
        critical_ratio: float = 1.8,
    ) -> None:
        self._num_anchors = max(num_anchors, 1)
        self._warning_ratio = warning_ratio
        self._critical_ratio = critical_ratio
        self._anchor_batches: List[Any] = []
        self._baseline_loss: Optional[float] = None
        self._history: List[ForgettingSnapshot] = []

    @property
    def has_anchors(self) -> bool:
        return len(self._anchor_batches) > 0

    @property
    def history(self) -> List[ForgettingSnapshot]:
        return list(self._history)

    def capture_anchors(self, dataloader: Any) -> int:
        """Grab fixed batches from the dataloader at training start.

        Returns the number of batches captured.
        """
        self._anchor_batches = []
        it = iter(dataloader)
        for _ in range(self._num_anchors):
            try:
                batch = next(it)
                self._anchor_batches.append(batch)
            except StopIteration:
                break
        return len(self._anchor_batches)

    def probe(
        self,
        validation_step_fn: Callable[[Any], float],
        step: int,
    ) -> Optional[ForgettingSnapshot]:
        """Run forward-only pass on anchor batches, compare against baseline.

        ``validation_step_fn`` should accept a single batch and return a scalar
        loss value.  It is expected to run under ``torch.no_grad()`` and
        ``model.eval()`` internally.
        """
        if not self._anchor_batches:
            return None

        losses: List[float] = []
        for batch in self._anchor_batches:
            try:
                loss_val = float(validation_step_fn(batch))
                losses.append(loss_val)
            except Exception:
                continue

        if not losses:
            return None

        avg_loss = sum(losses) / len(losses)

        if self._baseline_loss is None:
            self._baseline_loss = avg_loss

        ratio = avg_loss / self._baseline_loss if self._baseline_loss > 0 else 1.0

        if ratio >= self._critical_ratio:
            trend = "critical"
        elif ratio >= self._warning_ratio:
            trend = "warning"
        else:
            trend = "stable"

        score = max(0.0, min(100.0, 100.0 * (1.0 - max(0.0, ratio - 1.0))))

        snap = ForgettingSnapshot(
            step=step,
            anchor_loss=avg_loss,
            baseline_loss=self._baseline_loss,
            ratio=round(ratio, 4),
            trend=trend,
            score=round(score, 1),
        )
        self._history.append(snap)
        return snap

    def reset(self) -> None:
        self._anchor_batches = []
        self._baseline_loss = None
        self._history = []
