"""HASTE-style schedule guard for representation-alignment losses.

The guard is deliberately loss-agnostic. It decides whether an auxiliary
alignment loss should be active at a given step and can stop it by fixed step,
linear decay, or plateau evidence from recent loss history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class HasteAlignmentSchedulePolicy:
    enabled: bool = False
    base_weight: float = 1.0
    start_step: int = 0
    stop_step: int = -1
    decay_start_step: int = -1
    decay_end_step: int = -1
    min_weight: float = 0.0
    plateau_patience: int = 0
    min_relative_improvement: float = 0.0

    def normalized(self) -> "HasteAlignmentSchedulePolicy":
        base_weight = 1.0 if self.base_weight is None else float(self.base_weight)
        min_weight = 0.0 if self.min_weight is None else float(self.min_weight)
        patience = 0 if self.plateau_patience is None else int(self.plateau_patience)
        min_improvement = 0.0 if self.min_relative_improvement is None else float(self.min_relative_improvement)
        return HasteAlignmentSchedulePolicy(
            enabled=bool(self.enabled),
            base_weight=max(base_weight, 0.0),
            start_step=max(int(self.start_step or 0), 0),
            stop_step=int(self.stop_step if self.stop_step is not None else -1),
            decay_start_step=int(self.decay_start_step if self.decay_start_step is not None else -1),
            decay_end_step=int(self.decay_end_step if self.decay_end_step is not None else -1),
            min_weight=max(min_weight, 0.0),
            plateau_patience=max(patience, 0),
            min_relative_improvement=max(min_improvement, 0.0),
        )


@dataclass(frozen=True)
class HasteAlignmentWeightDecision:
    active: bool
    weight: float
    reason: str
    step: int
    total_steps: int
    base_weight: float
    min_weight: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "active": bool(self.active),
            "weight": float(self.weight),
            "reason": self.reason,
            "step": int(self.step),
            "total_steps": int(self.total_steps),
            "base_weight": float(self.base_weight),
            "min_weight": float(self.min_weight),
        }


def resolve_haste_alignment_weight(
    *,
    step: int,
    total_steps: int,
    policy: HasteAlignmentSchedulePolicy | Mapping[str, Any] | None = None,
    loss_history: Sequence[float] | None = None,
) -> HasteAlignmentWeightDecision:
    cfg = _policy(policy)
    current_step = max(int(step or 0), 0)
    total = max(int(total_steps or 0), 0)

    if not cfg.enabled or cfg.base_weight <= 0.0:
        return _decision(False, 0.0, "disabled", current_step, total, cfg)
    if current_step < cfg.start_step:
        return _decision(False, 0.0, "before_start", current_step, total, cfg)
    if cfg.stop_step >= 0 and current_step >= cfg.stop_step:
        return _decision(False, 0.0, "stop_step", current_step, total, cfg)
    if _plateau_reached(loss_history, cfg):
        return _decision(False, 0.0, "plateau_stop", current_step, total, cfg)

    weight = cfg.base_weight
    if cfg.decay_start_step >= 0 and current_step >= cfg.decay_start_step:
        end_step = cfg.decay_end_step if cfg.decay_end_step > cfg.decay_start_step else max(total - 1, cfg.decay_start_step)
        if current_step >= end_step:
            weight = cfg.min_weight
        else:
            progress = (current_step - cfg.decay_start_step) / max(float(end_step - cfg.decay_start_step), 1.0)
            weight = cfg.base_weight + (cfg.min_weight - cfg.base_weight) * progress
        reason = "decay"
    else:
        reason = "active"

    active = weight > 0.0
    return _decision(active, weight if active else 0.0, reason if active else "zero_weight", current_step, total, cfg)


def _policy(policy: HasteAlignmentSchedulePolicy | Mapping[str, Any] | None) -> HasteAlignmentSchedulePolicy:
    if isinstance(policy, Mapping):
        return HasteAlignmentSchedulePolicy(**policy).normalized()
    return (policy or HasteAlignmentSchedulePolicy()).normalized()


def _plateau_reached(loss_history: Sequence[float] | None, cfg: HasteAlignmentSchedulePolicy) -> bool:
    if cfg.plateau_patience <= 0 or cfg.min_relative_improvement <= 0.0 or not loss_history:
        return False
    needed = cfg.plateau_patience + 1
    if len(loss_history) < needed:
        return False
    window = [float(v) for v in loss_history[-needed:]]
    improvements: list[float] = []
    for previous, current in zip(window, window[1:]):
        denom = max(abs(previous), 1e-12)
        improvements.append((previous - current) / denom)
    return max(improvements, default=0.0) < cfg.min_relative_improvement


def _decision(
    active: bool,
    weight: float,
    reason: str,
    step: int,
    total_steps: int,
    cfg: HasteAlignmentSchedulePolicy,
) -> HasteAlignmentWeightDecision:
    return HasteAlignmentWeightDecision(
        active=bool(active),
        weight=max(float(weight), 0.0),
        reason=reason,
        step=int(step),
        total_steps=int(total_steps),
        base_weight=float(cfg.base_weight),
        min_weight=float(cfg.min_weight),
    )


__all__ = [
    "HasteAlignmentSchedulePolicy",
    "HasteAlignmentWeightDecision",
    "resolve_haste_alignment_weight",
]
