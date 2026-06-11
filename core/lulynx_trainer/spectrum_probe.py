"""Observe-only Spectrum scheduling probe and execution layer.

This module mirrors Spectrum's actual-vs-cached step scheduler and records
which DiT block calls would have been skipped.

The execution layer implements activation caching and linear extrapolation
for block skip (first version; Chebyshev forecast to be added later).
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterator, Optional

import torch


@dataclass(frozen=True)
class SpectrumProbePolicy:
    enabled: bool = False
    window_size: float = 2.0
    flex_window: float = 0.25
    warmup_steps: int = 6
    stop_caching_step: int = -1

    def normalized(self) -> "SpectrumProbePolicy":
        window_size = 2.0 if self.window_size is None else float(self.window_size)
        flex_window = 0.0 if self.flex_window is None else float(self.flex_window)
        warmup_steps = 0 if self.warmup_steps is None else int(self.warmup_steps)
        return SpectrumProbePolicy(
            enabled=bool(self.enabled),
            window_size=max(window_size, 1.0),
            flex_window=max(flex_window, 0.0),
            warmup_steps=max(warmup_steps, 0),
            stop_caching_step=int(self.stop_caching_step if self.stop_caching_step is not None else -1),
        )


@dataclass(frozen=True)
class SpectrumStepDecision:
    step_index: int
    total_steps: int
    actual_forward: bool
    would_cache: bool
    window_size: float
    consecutive_cached: int
    reason: str


@dataclass(frozen=True)
class SpectrumStepContext:
    decision: SpectrumStepDecision


class SpectrumProbeState:
    def __init__(self, policy: SpectrumProbePolicy, *, total_steps: int) -> None:
        self.policy = policy.normalized()
        self.total_steps = max(int(total_steps or 0), 0)
        self.current_window = self.policy.window_size
        self.consecutive_cached = 0
        self.actual_forwards = 0
        self.cached_steps = 0
        self.decisions: list[SpectrumStepDecision] = []

    def decide(self, step_index: int) -> SpectrumStepDecision:
        step = int(step_index)
        stop_at = self.total_steps - 3 if self.policy.stop_caching_step < 0 else self.policy.stop_caching_step
        if not self.policy.enabled:
            actual = True
            reason = "disabled"
        elif step < self.policy.warmup_steps:
            actual = True
            reason = "warmup"
        elif step >= stop_at:
            actual = True
            reason = "tail_stop"
        else:
            interval = max(1, math.floor(self.current_window))
            actual = (self.consecutive_cached + 1) % interval == 0
            reason = "scheduled_actual" if actual else "would_cache"

        decision = SpectrumStepDecision(
            step_index=step,
            total_steps=self.total_steps,
            actual_forward=actual,
            would_cache=not actual,
            window_size=float(self.current_window),
            consecutive_cached=int(self.consecutive_cached),
            reason=reason,
        )
        self.decisions.append(decision)
        _STATS["steps_observed"] += 1
        if actual:
            self.actual_forwards += 1
            _STATS["actual_forward_steps"] += 1
            if step >= self.policy.warmup_steps and self.policy.enabled:
                self.current_window = round(self.current_window + self.policy.flex_window, 3)
            self.consecutive_cached = 0
        else:
            self.cached_steps += 1
            self.consecutive_cached += 1
            _STATS["would_cache_steps"] += 1
        return decision

    def summary(self) -> Dict[str, float | int | bool]:
        actual = max(int(self.actual_forwards), 1)
        return {
            "enabled": bool(self.policy.enabled),
            "total_steps": int(self.total_steps),
            "actual_forward_steps": int(self.actual_forwards),
            "would_cache_steps": int(self.cached_steps),
            "theoretical_step_speedup": float(self.total_steps / actual) if self.total_steps else 1.0,
        }


_CURRENT_CONTEXT: ContextVar[Optional[SpectrumStepContext]] = ContextVar("lulynx_spectrum_context", default=None)
_STATS: Dict[str, int] = {
    "steps_observed": 0,
    "actual_forward_steps": 0,
    "would_cache_steps": 0,
    "block_calls_observed": 0,
    "would_skip_block_calls": 0,
    "missing_step_context_block_calls": 0,
}


@contextmanager
def spectrum_step_context(decision: SpectrumStepDecision) -> Iterator[None]:
    token = _CURRENT_CONTEXT.set(SpectrumStepContext(decision=decision))
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)


def observe_block_call(*, block_index: int) -> bool:
    """Record a block call and return whether Spectrum would have skipped it."""
    _STATS["block_calls_observed"] += 1
    context = _CURRENT_CONTEXT.get()
    if context is None:
        _STATS["missing_step_context_block_calls"] += 1
        return False
    if context.decision.would_cache:
        _STATS["would_skip_block_calls"] += 1
        return True
    return False


def has_spectrum_step_context() -> bool:
    return _CURRENT_CONTEXT.get() is not None


def snapshot_spectrum_probe_stats() -> Dict[str, int]:
    return dict(_STATS)


def reset_spectrum_probe_stats() -> None:
    for key in _STATS:
        _STATS[key] = 0


# ============================================================================
# Spectrum Execution Layer: Cache and Linear Extrapolation
# ============================================================================


class SpectrumCache:
    """Cache DiT block activations for Spectrum skip/forecast.

    Stores a sliding window of recent activations per block, enabling
    linear extrapolation (or future Chebyshev forecast) during cached steps.
    """

    def __init__(self, window_size: int = 3):
        """
        Parameters
        ----------
        window_size : int
            Number of recent activations to store per block for forecasting.
        """
        self._history: Dict[int, deque] = {}
        self._window_size = max(int(window_size), 2)
        self._forecasts = 0
        self._actual_computes = 0

    def push(self, block_index: int, activation: torch.Tensor):
        """Store a block activation (detached)."""
        if block_index not in self._history:
            self._history[block_index] = deque(maxlen=self._window_size)
        self._history[block_index].append(activation.detach())

    def forecast_linear(self, block_index: int) -> Optional[torch.Tensor]:
        """Forecast next activation via linear extrapolation.

        Given activations at steps t-1 and t-2, predict step t via:
            y_t = 2 * y_{t-1} - y_{t-2}

        Returns None if insufficient history.
        """
        history = self._history.get(block_index)
        if history is None or len(history) < 2:
            return None

        recent = list(history)
        # Linear extrapolation: forecast = 2*recent - prev
        forecast = 2.0 * recent[-1] - recent[-2]
        self._forecasts += 1
        return forecast

    def clear(self):
        """Clear all cached activations."""
        self._history.clear()
        self._forecasts = 0
        self._actual_computes = 0

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            "cached_blocks": len(self._history),
            "forecasts_generated": self._forecasts,
            "actual_computes": self._actual_computes,
        }


def run_with_spectrum_skip(
    block_fn: Callable[..., torch.Tensor],
    block_index: int,
    cache: SpectrumCache,
    *args,
    **kwargs
) -> torch.Tensor:
    """Wrap DiT block call with Spectrum skip logic.

    If the current step is a cached step according to Spectrum schedule,
    attempt to forecast the block output. Otherwise, compute normally and cache.

    Parameters
    ----------
    block_fn : callable
        The actual block forward function.
    block_index : int
        Block identifier (e.g., 0, 1, 2, ...).
    cache : SpectrumCache
        Shared cache for this generation.
    *args, **kwargs
        Arguments to pass to block_fn.

    Returns
    -------
    torch.Tensor
        Block output (forecasted or computed).
    """
    # Check if this step should skip via existing probe logic
    should_skip = observe_block_call(block_index=block_index)

    if not should_skip:
        # Actual forward step: compute and cache
        output = block_fn(*args, **kwargs)
        cache.push(block_index, output)
        cache._actual_computes += 1
        return output

    # Cached step: try to forecast
    forecast = cache.forecast_linear(block_index)
    if forecast is not None:
        # Successful forecast: skip actual computation
        return forecast

    # Forecast failed (insufficient history): fall back to actual computation
    output = block_fn(*args, **kwargs)
    cache.push(block_index, output)
    cache._actual_computes += 1
    return output


@dataclass(frozen=True)
class SpectrumExecutionStats:
    """Statistics for Spectrum execution."""
    total_steps: int
    actual_forward_steps: int
    cached_steps: int
    block_calls_observed: int
    block_calls_skipped: int
    forecasts_generated: int
    actual_computes: int
    step_speedup: float  # theoretical
    block_skip_rate: float  # forecasts / (forecasts + actual_computes)

    @classmethod
    def from_stats(cls, probe_stats: Dict[str, int], cache_stats: Dict[str, int], state_summary: Dict) -> "SpectrumExecutionStats":
        total = state_summary.get("total_steps", 0)
        actual = state_summary.get("actual_forward_steps", 1)
        cached = state_summary.get("would_cache_steps", 0)
        block_obs = probe_stats.get("block_calls_observed", 0)
        block_skip = probe_stats.get("would_skip_block_calls", 0)
        forecasts = cache_stats.get("forecasts_generated", 0)
        computes = cache_stats.get("actual_computes", 0)

        step_speedup = total / max(actual, 1)
        block_skip_rate = forecasts / max(forecasts + computes, 1)

        return cls(
            total_steps=total,
            actual_forward_steps=actual,
            cached_steps=cached,
            block_calls_observed=block_obs,
            block_calls_skipped=block_skip,
            forecasts_generated=forecasts,
            actual_computes=computes,
            step_speedup=step_speedup,
            block_skip_rate=block_skip_rate,
        )

