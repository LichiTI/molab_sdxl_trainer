# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Lightweight loss modification tracker for advanced monitoring.

Records every loss transformation in the training step (SNR weighting,
debiased estimation, prior loss, REPA, plugin mutation, etc.) and exposes
the modification chain as a snapshot for the info dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LossModification:
    stage: str
    before: float
    after: float
    scale: float = 1.0
    bias: float = 0.0


class LossTracker:
    """Track loss modifications within a single training step."""

    __slots__ = ("_modifications", "_enabled")

    def __init__(self) -> None:
        self._modifications: List[LossModification] = []
        self._enabled: bool = False

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(
        self,
        stage: str,
        before: float,
        after: float,
        scale: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        if not self._enabled:
            return
        self._modifications.append(
            LossModification(
                stage=stage, before=before, after=after,
                scale=scale, bias=bias,
            )
        )

    def snapshot(self) -> Optional[List[Dict[str, Any]]]:
        """Return current step's modifications and clear for next step."""
        if not self._modifications:
            return None
        result = [
            {
                "stage": m.stage,
                "before": round(m.before, 6),
                "after": round(m.after, 6),
                "scale": round(m.scale, 6),
                "bias": round(m.bias, 6),
            }
            for m in self._modifications
        ]
        self._modifications.clear()
        return result

    def summary(self) -> Dict[str, Any]:
        """Aggregate view of current step's modifications."""
        if not self._modifications:
            return {"active_modifiers": 0, "total_scale": 1.0, "total_bias": 0.0}
        total_scale = 1.0
        total_bias = 0.0
        for m in self._modifications:
            total_scale *= m.scale
            total_bias += m.bias
        return {
            "active_modifiers": len(self._modifications),
            "total_scale": round(total_scale, 6),
            "total_bias": round(total_bias, 6),
            "stages": [m.stage for m in self._modifications],
        }
