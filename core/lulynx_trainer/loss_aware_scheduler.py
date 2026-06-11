"""Loss-aware cosine learning-rate schedulers.

These schedulers keep the familiar cosine shape, but make the cosine phase
advance depend on recent loss behavior.  The training loop passes the current
optimizer-step loss into ``step(loss)``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _auto_max_hold_steps(active_steps: int) -> int:
    """Cap loss-triggered holds so training loss cannot stall the cosine forever."""

    active = max(int(active_steps), 1)
    return min(active, max(4, min(200, int(math.ceil(active * 0.05)))))


class _LossTrend:
    def __init__(
        self,
        ema_alpha: float = 0.1,
        min_delta: float = 5e-4,
        relative_delta: float = 1e-3,
        patience: int = 8,
        cooldown: int = 0,
    ) -> None:
        self.ema_alpha = _clamp(float(ema_alpha), 0.0, 1.0)
        self.min_delta = max(float(min_delta), 0.0)
        self.relative_delta = max(float(relative_delta), 0.0)
        self.patience = max(int(patience), 1)
        self.cooldown = max(int(cooldown), 0)

        self.ema_loss: Optional[float] = None
        self.best_loss: Optional[float] = None
        self.bad_steps = 0
        self.cooldown_remaining = 0
        self.last_improved = False

    def update(self, loss: Optional[float]) -> Dict[str, Any]:
        if loss is None or not math.isfinite(float(loss)):
            self.last_improved = False
            return {
                "ema_loss": self.ema_loss,
                "best_loss": self.best_loss,
                "improved": False,
                "bad_steps": self.bad_steps,
                "plateau": self.bad_steps >= self.patience,
                "cooldown_remaining": self.cooldown_remaining,
            }

        loss_value = float(loss)
        if self.ema_loss is None:
            self.ema_loss = loss_value
        else:
            self.ema_loss = self.ema_alpha * loss_value + (1.0 - self.ema_alpha) * self.ema_loss

        threshold = self.min_delta
        if self.best_loss is not None and self.relative_delta > 0.0:
            threshold = max(threshold, abs(self.best_loss) * self.relative_delta)

        if self.best_loss is None or self.ema_loss < (self.best_loss - threshold):
            self.best_loss = self.ema_loss
            self.bad_steps = 0
            self.cooldown_remaining = self.cooldown
            self.last_improved = True
        else:
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
            else:
                self.bad_steps += 1
            self.last_improved = False

        return {
            "ema_loss": self.ema_loss,
            "best_loss": self.best_loss,
            "improved": self.last_improved,
            "bad_steps": self.bad_steps,
            "plateau": self.bad_steps >= self.patience,
            "cooldown_remaining": self.cooldown_remaining,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ema_loss": self.ema_loss,
            "best_loss": self.best_loss,
            "bad_steps": self.bad_steps,
            "cooldown_remaining": self.cooldown_remaining,
            "last_improved": self.last_improved,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.ema_loss = state.get("ema_loss")
        self.best_loss = state.get("best_loss")
        self.bad_steps = int(state.get("bad_steps", 0) or 0)
        self.cooldown_remaining = int(state.get("cooldown_remaining", 0) or 0)
        self.last_improved = bool(state.get("last_improved", False))


class LossAwareCosineScheduler:
    """Cosine scheduler with loss-gated phase advancement."""

    def __init__(
        self,
        optimizer: Any,
        total_steps: int,
        warmup_steps: int = 0,
        *,
        eta_min: float = 0.0,
        num_cycles: float = 0.5,
        mode: str = "gated",
        ema_alpha: float = 0.1,
        min_delta: float = 5e-4,
        relative_delta: float = 1e-3,
        patience: int = 8,
        cooldown: int = 0,
        max_hold_steps: int = 0,
        hold_on_improvement: bool = True,
        late_loss_gamma: float = 2.0,
        lock_weight_threshold: float = 0.7,
        min_advance_ratio: float = 0.25,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = max(min(int(warmup_steps), self.total_steps - 1), 0)
        self.active_steps = max(self.total_steps - self.warmup_steps, 1)
        self.eta_min = max(float(eta_min), 0.0)
        self.num_cycles = max(float(num_cycles), 0.0)
        self.mode = str(mode or "gated").strip().lower()
        raw_max_hold_steps = int(max_hold_steps)
        self.auto_max_hold_steps = raw_max_hold_steps <= 0
        self.max_hold_steps = (
            _auto_max_hold_steps(self.active_steps)
            if self.auto_max_hold_steps
            else max(raw_max_hold_steps, 1)
        )
        self.hold_on_improvement = bool(hold_on_improvement)
        self.late_loss_gamma = max(float(late_loss_gamma), 0.01)
        self.lock_weight_threshold = _clamp(float(lock_weight_threshold), 0.0, 1.0)
        self.min_advance_ratio = _clamp(float(min_advance_ratio), 0.0, 1.0)

        self.base_lrs = [float(group.get("lr", 0.0) or 0.0) for group in optimizer.param_groups]
        self.last_lrs = list(self.base_lrs)
        self.last_epoch = -1
        self.phase_step = 0.0
        self.hold_steps = 0
        self.last_action = "init"
        self._last_loss_weight = 0.0
        self._trend = _LossTrend(
            ema_alpha=ema_alpha,
            min_delta=min_delta,
            relative_delta=relative_delta,
            patience=patience,
            cooldown=cooldown,
        )

        self._apply_lrs(self._compute_lrs())

    @classmethod
    def gated(cls, optimizer: Any, total_steps: int, warmup_steps: int = 0, **kwargs: Any) -> "LossAwareCosineScheduler":
        return cls(optimizer, total_steps, warmup_steps, mode="gated", **kwargs)

    @classmethod
    def weighted(cls, optimizer: Any, total_steps: int, warmup_steps: int = 0, **kwargs: Any) -> "LossAwareCosineScheduler":
        return cls(optimizer, total_steps, warmup_steps, mode="weighted", **kwargs)

    def _compute_lrs(self) -> List[float]:
        if self.last_epoch < self.warmup_steps and self.warmup_steps > 0:
            factor = float(self.last_epoch + 1) / float(self.warmup_steps)
            factor = _clamp(factor, 0.0, 1.0)
            return [base_lr * factor for base_lr in self.base_lrs]

        progress = _clamp(self.phase_step / float(self.active_steps), 0.0, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * 2.0 * self.num_cycles * progress))
        return [self.eta_min + (base_lr - self.eta_min) * cosine for base_lr in self.base_lrs]

    def _apply_lrs(self, lrs: List[float]) -> None:
        self.last_lrs = list(lrs)
        for group, lr in zip(self.optimizer.param_groups, self.last_lrs):
            group["lr"] = lr

    def _hold_or_cap(self, hold_action: str, cap_action: str, cap_advance: float) -> float:
        self.hold_steps += 1
        if self.hold_steps < self.max_hold_steps:
            self.last_action = hold_action
            return 0.0
        self.hold_steps = 0
        self.last_action = cap_action
        return cap_advance

    def _weighted_advance(self, trend: Dict[str, Any]) -> float:
        progress = _clamp(self.last_epoch / float(max(self.total_steps - 1, 1)), 0.0, 1.0)
        loss_weight = progress ** self.late_loss_gamma
        self._last_loss_weight = loss_weight

        if trend["improved"] and loss_weight >= self.lock_weight_threshold:
            advance = max(self.min_advance_ratio, 1.0 - loss_weight)
            return self._hold_or_cap("weighted_lock", "weighted_max_hold_advance", advance)

        if trend["plateau"]:
            self.hold_steps = 0
            self._trend.bad_steps = 0
            self.last_action = "weighted_plateau_advance"
            return 1.0

        self.hold_steps = 0
        advance = 1.0 - loss_weight
        self.last_action = "weighted_slow" if advance < 0.999 else "weighted_normal"
        return max(self.min_advance_ratio, advance)

    def _gated_advance(self, trend: Dict[str, Any]) -> float:
        if self.hold_on_improvement and trend["improved"]:
            return self._hold_or_cap("hold_improving", "max_hold_advance", 1.0)

        if trend["plateau"]:
            self.hold_steps = 0
            self._trend.bad_steps = 0
            self.last_action = "plateau_advance"
            return 1.0

        return self._hold_or_cap("plateau_wait", "max_hold_advance", 1.0)

    def step(self, loss: Optional[float] = None, epoch: Optional[int] = None) -> None:
        if epoch is not None:
            self.last_epoch = int(epoch)
        else:
            self.last_epoch += 1

        trend = self._trend.update(loss)
        if self.last_epoch >= self.warmup_steps:
            if self.mode == "weighted":
                advance = self._weighted_advance(trend)
            else:
                self._last_loss_weight = 1.0
                advance = self._gated_advance(trend)
            self.phase_step = _clamp(self.phase_step + advance, 0.0, float(self.active_steps))
        else:
            self.last_action = "warmup"

        self._apply_lrs(self._compute_lrs())

    def get_last_lr(self) -> List[float]:
        return list(self.last_lrs)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "base_lrs": list(self.base_lrs),
            "last_lrs": list(self.last_lrs),
            "last_epoch": self.last_epoch,
            "phase_step": self.phase_step,
            "hold_steps": self.hold_steps,
            "max_hold_steps": self.max_hold_steps,
            "auto_max_hold_steps": self.auto_max_hold_steps,
            "last_action": self.last_action,
            "loss_weight": self._last_loss_weight,
            "trend": self._trend.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.base_lrs = [float(v) for v in state.get("base_lrs", self.base_lrs)]
        self.last_lrs = [float(v) for v in state.get("last_lrs", self.last_lrs)]
        self.last_epoch = int(state.get("last_epoch", self.last_epoch) or 0)
        self.phase_step = float(state.get("phase_step", self.phase_step) or 0.0)
        self.hold_steps = int(state.get("hold_steps", self.hold_steps) or 0)
        self.auto_max_hold_steps = bool(state.get("auto_max_hold_steps", self.auto_max_hold_steps))
        loaded_max_hold_steps = state.get("max_hold_steps")
        if loaded_max_hold_steps is not None:
            self.max_hold_steps = max(int(loaded_max_hold_steps), 1)
        elif self.auto_max_hold_steps:
            self.max_hold_steps = _auto_max_hold_steps(self.active_steps)
        self.last_action = str(state.get("last_action", self.last_action) or self.last_action)
        self._last_loss_weight = float(state.get("loss_weight", self._last_loss_weight) or 0.0)
        trend_state = state.get("trend")
        if isinstance(trend_state, dict):
            self._trend.load_state_dict(trend_state)
        self._apply_lrs(self.last_lrs)

    def get_loss_aware_state(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "action": self.last_action,
            "phase_step": round(float(self.phase_step), 6),
            "phase_progress": round(_clamp(self.phase_step / float(self.active_steps), 0.0, 1.0), 6),
            "hold_steps": int(self.hold_steps),
            "max_hold_steps": int(self.max_hold_steps),
            "auto_max_hold_steps": bool(self.auto_max_hold_steps),
            "loss_weight": round(float(self._last_loss_weight), 6),
            "ema_loss": self._trend.ema_loss,
            "best_loss": self._trend.best_loss,
            "bad_steps": int(self._trend.bad_steps),
        }
