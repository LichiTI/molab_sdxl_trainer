# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Error-guided SmoothCache probe and execution layer (cleanroom Lulynx).

SmoothCache accelerates the multi-step DiT denoising loop by reusing a block's
previously computed output on timesteps where that block changes little.  Unlike
Spectrum's window heuristic (``spectrum_probe.py``) or T-GATE's late-step
cross-attention reuse (``tgate.py``), SmoothCache derives a **per-block, error-
bounded** caching schedule from a calibration pass:

    L_b(t) = ||O_b(t) - O_b(t-1)||_1 / ||O_b(t-1)||_1

A step ``t`` is cacheable for block ``b`` while the accumulated relative change
since the last real compute stays under an error threshold ``alpha``; once it
would exceed ``alpha`` the block is recomputed and the accumulator resets.

Two segments, mirroring ``spectrum_probe.py``:

* **Probe (observe-only, default off):** records which block calls SmoothCache
  *would* have reused.  This is the part wired into the live DiT / sampler.
* **Execution layer (library primitive):** ``SmoothCacheStore`` +
  ``run_with_smoothcache`` actually reuse cached outputs.  Not auto-wired into
  the live block loop; at ``alpha == 0`` / no schedule it is bit-identical to a
  plain block call.

Clean-room Lulynx module; references no external caching source.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Iterator, List, Optional

import torch

Schedule = Dict[int, FrozenSet[int]]


@dataclass(frozen=True)
class SmoothCachePolicy:
    """Configuration for the SmoothCache schedule (default off)."""

    enabled: bool = False
    error_threshold: float = 0.08
    warmup_steps: int = 2
    schedule: Optional[Schedule] = None

    def normalized(self) -> "SmoothCachePolicy":
        threshold = 0.0 if self.error_threshold is None else float(self.error_threshold)
        warmup = 0 if self.warmup_steps is None else int(self.warmup_steps)
        return SmoothCachePolicy(
            enabled=bool(self.enabled),
            error_threshold=max(threshold, 0.0),
            warmup_steps=max(warmup, 0),
            schedule=self.schedule,
        )


@dataclass(frozen=True)
class SmoothCacheStepDecision:
    step_index: int
    total_steps: int
    cacheable_blocks: FrozenSet[int]
    reason: str

    @property
    def any_cacheable(self) -> bool:
        return len(self.cacheable_blocks) > 0


# ---------------------------------------------------------------------------
# Calibration: error-guided per-block schedule (the SmoothCache contribution).
# ---------------------------------------------------------------------------


class SmoothCacheCalibrator:
    """Record per-block relative-L1 inter-step change and build a schedule.

    Feed each block's output once per denoising step via :meth:`record`, in
    block order, repeating across steps.  :meth:`build_schedule` then turns the
    accumulated change series into a per-block set of cacheable step indices for
    a given error threshold.
    """

    def __init__(self) -> None:
        self._prev: Dict[int, torch.Tensor] = {}
        self._rel_l1: Dict[int, List[float]] = {}
        self._steps_seen: Dict[int, int] = {}

    def record(self, block_index: int, output: torch.Tensor) -> None:
        out = output.detach()
        self._steps_seen[block_index] = self._steps_seen.get(block_index, 0) + 1
        prev = self._prev.get(block_index)
        if prev is not None and prev.shape == out.shape:
            num = (out - prev).abs().sum()
            den = prev.abs().sum().clamp_min(1e-8)
            self._rel_l1.setdefault(block_index, []).append(float((num / den).item()))
        self._prev[block_index] = out

    def relative_changes(self) -> Dict[int, List[float]]:
        return {block: list(series) for block, series in self._rel_l1.items()}

    def build_schedule(self, error_threshold: float, *, warmup_steps: int = 0) -> Schedule:
        """Cacheable step set per block under ``error_threshold``.

        ``series[i]`` is the relative change from step ``i`` to step ``i + 1``,
        so it keys step ``i + 1``.  Steps below ``warmup_steps`` are always
        recomputed.  Larger ``error_threshold`` yields a (weakly) larger set,
        which the smoke test asserts.
        """
        threshold = max(float(error_threshold), 0.0)
        warmup = max(int(warmup_steps), 0)
        schedule: Schedule = {}
        for block, series in self._rel_l1.items():
            cacheable: set[int] = set()
            acc = 0.0
            for i, change in enumerate(series):
                step = i + 1
                if step < warmup:
                    acc = 0.0
                    continue
                acc += change
                if acc < threshold:
                    cacheable.add(step)
                else:
                    acc = 0.0
            schedule[block] = frozenset(cacheable)
        return schedule

    def summary(self) -> Dict[str, object]:
        return {
            "blocks_calibrated": len(self._rel_l1),
            "steps_per_block": {b: n for b, n in self._steps_seen.items()},
        }


# ---------------------------------------------------------------------------
# Per-step decision state.
# ---------------------------------------------------------------------------


class SmoothCacheState:
    def __init__(self, policy: SmoothCachePolicy, *, total_steps: int) -> None:
        self.policy = policy.normalized()
        self.total_steps = max(int(total_steps or 0), 0)
        self.decisions: List[SmoothCacheStepDecision] = []

    def decide(self, step_index: int) -> SmoothCacheStepDecision:
        step = int(step_index)
        schedule = self.policy.schedule
        if not self.policy.enabled:
            decision = SmoothCacheStepDecision(step, self.total_steps, frozenset(), "disabled")
        elif schedule is None:
            decision = SmoothCacheStepDecision(step, self.total_steps, frozenset(), "no_schedule")
        elif step < self.policy.warmup_steps:
            decision = SmoothCacheStepDecision(step, self.total_steps, frozenset(), "warmup")
        else:
            cacheable = frozenset(b for b, steps in schedule.items() if step in steps)
            reason = "scheduled_cache" if cacheable else "scheduled_actual"
            decision = SmoothCacheStepDecision(step, self.total_steps, cacheable, reason)
        self.decisions.append(decision)
        return decision

    def summary(self) -> Dict[str, float | int | bool]:
        cached_block_calls = sum(len(d.cacheable_blocks) for d in self.decisions)
        return {
            "enabled": bool(self.policy.enabled),
            "total_steps": int(self.total_steps),
            "error_threshold": float(self.policy.error_threshold),
            "cacheable_block_calls": int(cached_block_calls),
        }


# ---------------------------------------------------------------------------
# Observe-only probe (wired into the live DiT / sampler).
# ---------------------------------------------------------------------------


_CURRENT_CONTEXT: ContextVar[Optional[SmoothCacheStepDecision]] = ContextVar(
    "lulynx_smoothcache_context", default=None
)
_STATS: Dict[str, int] = {
    "steps_observed": 0,
    "block_calls_observed": 0,
    "would_reuse_block_calls": 0,
    "missing_step_context_block_calls": 0,
}


@contextmanager
def smoothcache_step_context(decision: SmoothCacheStepDecision) -> Iterator[None]:
    token = _CURRENT_CONTEXT.set(decision)
    _STATS["steps_observed"] += 1
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)


def observe_block_call(*, block_index: int) -> bool:
    """Record a block call and return whether SmoothCache would reuse it."""
    _STATS["block_calls_observed"] += 1
    decision = _CURRENT_CONTEXT.get()
    if decision is None:
        _STATS["missing_step_context_block_calls"] += 1
        return False
    if int(block_index) in decision.cacheable_blocks:
        _STATS["would_reuse_block_calls"] += 1
        return True
    return False


def has_smoothcache_step_context() -> bool:
    return _CURRENT_CONTEXT.get() is not None


def snapshot_smoothcache_probe_stats() -> Dict[str, int]:
    return dict(_STATS)


def reset_smoothcache_probe_stats() -> None:
    for key in _STATS:
        _STATS[key] = 0


# ---------------------------------------------------------------------------
# Execution layer: reuse cached block outputs (library primitive).
# ---------------------------------------------------------------------------


class SmoothCacheStore:
    """Per-block cache of the most recent computed output."""

    def __init__(self) -> None:
        self._cache: Dict[int, torch.Tensor] = {}
        self._hits = 0
        self._misses = 0

    def get(self, block_index: int) -> Optional[torch.Tensor]:
        cached = self._cache.get(block_index)
        if cached is not None:
            self._hits += 1
        else:
            self._misses += 1
        return cached

    def push(self, block_index: int, output: torch.Tensor) -> None:
        self._cache[block_index] = output.detach()

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, int]:
        return {
            "cached_blocks": len(self._cache),
            "cache_hits": self._hits,
            "cache_misses": self._misses,
        }


def run_with_smoothcache(
    block_fn: Callable[..., torch.Tensor],
    block_index: int,
    store: SmoothCacheStore,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Wrap a DiT block call with SmoothCache reuse.

    On a cacheable step (per the active probe decision) with a stored output,
    return the cached tensor; otherwise compute and store.  With ``alpha == 0``
    or no schedule the probe never marks a block cacheable, so this returns the
    freshly computed output unchanged.
    """
    should_reuse = observe_block_call(block_index=block_index)
    if should_reuse:
        cached = store.get(block_index)
        if cached is not None:
            return cached
    output = block_fn(*args, **kwargs)
    store.push(block_index, output)
    return output


@dataclass(frozen=True)
class SmoothCacheExecutionStats:
    total_steps: int
    block_calls_observed: int
    block_calls_reused: int
    cache_hits: int
    cache_misses: int
    block_reuse_rate: float
    theoretical_step_speedup: float

    @classmethod
    def from_stats(
        cls,
        probe_stats: Dict[str, int],
        store_stats: Dict[str, int],
        state_summary: Dict,
    ) -> "SmoothCacheExecutionStats":
        observed = probe_stats.get("block_calls_observed", 0)
        reused = probe_stats.get("would_reuse_block_calls", 0)
        hits = store_stats.get("cache_hits", 0)
        actual = max(observed - hits, 1)
        return cls(
            total_steps=int(state_summary.get("total_steps", 0)),
            block_calls_observed=observed,
            block_calls_reused=reused,
            cache_hits=hits,
            cache_misses=store_stats.get("cache_misses", 0),
            block_reuse_rate=reused / max(observed, 1),
            theoretical_step_speedup=observed / actual,
        )


__all__ = [
    "Schedule",
    "SmoothCachePolicy",
    "SmoothCacheStepDecision",
    "SmoothCacheCalibrator",
    "SmoothCacheState",
    "smoothcache_step_context",
    "observe_block_call",
    "has_smoothcache_step_context",
    "snapshot_smoothcache_probe_stats",
    "reset_smoothcache_probe_stats",
    "SmoothCacheStore",
    "run_with_smoothcache",
    "SmoothCacheExecutionStats",
]
