# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Stepped Loss Schedule — change loss type and weight at configurable step thresholds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class _SteppedLossEntry:
    step: int
    loss_type: str
    weight: float


_VALID_LOSS_TYPES = {"l2", "l1", "huber", "smooth_l1"}
_DEFAULT = ("l2", 1.0)


class SteppedLossSchedule:
    """Step-based loss type/weight schedule.

    Given a JSON-encoded schedule string, determines the active loss type
    and weight multiplier for any given training step.
    """

    def __init__(self, schedule_json: str) -> None:
        self._entries: List[_SteppedLossEntry] = []
        if not schedule_json or not schedule_json.strip():
            return
        try:
            raw = json.loads(schedule_json)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(raw, list):
            return
        for item in raw:
            if not isinstance(item, dict):
                continue
            step = int(item.get("step", 0))
            lt = str(item.get("loss_type", "l2")).lower()
            w = float(item.get("weight", 1.0))
            if lt not in _VALID_LOSS_TYPES:
                lt = "l2"
            self._entries.append(_SteppedLossEntry(step=step, loss_type=lt, weight=w))
        self._entries.sort(key=lambda e: e.step)

    def resolve(self, global_step: int) -> Tuple[str, float]:
        """Return (loss_type, weight) active at *global_step*."""
        if not self._entries:
            return _DEFAULT
        active = self._entries[0]
        for entry in self._entries:
            if entry.step <= global_step:
                active = entry
            else:
                break
        return (active.loss_type, active.weight)

    @property
    def enabled(self) -> bool:
        return len(self._entries) > 0
