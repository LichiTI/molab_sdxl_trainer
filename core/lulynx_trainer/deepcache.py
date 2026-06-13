# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Interval deep-block reuse cache (cleanroom Lulynx "DeepCache"-family).

DeepCache accelerates multi-step DiT denoising by recomputing every block only on
periodic *key steps* and, on the *non-key steps* in between, reusing the cached
outputs of the deeper blocks (whose contribution changes slowly), while still
recomputing the shallow blocks. Unlike SmoothCache (error-guided, per-block) or
Spectrum (window heuristic), the schedule here is a simple, deterministic
interval + depth split.

This store is **self-contained**: it is driven purely by the unified cache seam's
``run_block(block_fn, block_index, *args)`` calls inside the live ``_run_blocks``
loop. It detects step boundaries from the ``block_index`` wrap-around (the loop
always visits blocks ``0..N-1`` in order each step), so it needs no per-step
probe context. Default-off parity is owned by the seam: it only builds this store
when ``backend="deepcache"`` is explicitly opted in.

Clean-room Lulynx module; references no external caching source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch


@dataclass(frozen=True)
class DeepCachePolicy:
    """Configuration for interval deep-block reuse (default off)."""

    enabled: bool = False
    interval: int = 3          # recompute all blocks every ``interval`` steps
    deep_fraction: float = 0.4  # blocks with index >= floor(N*frac) are reusable
    warmup_steps: int = 1       # always fully compute the first ``warmup`` steps

    def normalized(self) -> "DeepCachePolicy":
        return DeepCachePolicy(
            enabled=bool(self.enabled),
            interval=max(int(self.interval or 1), 1),
            deep_fraction=min(max(float(self.deep_fraction or 0.0), 0.0), 1.0),
            warmup_steps=max(int(self.warmup_steps or 0), 0),
        )


class DeepCacheStore:
    """Per-block output cache with interval + depth reuse scheduling.

    Mirrors the ``get/push/clear/stats`` surface the unified cache seam expects;
    the reuse decision lives in :meth:`decide` (called by ``run_with_deepcache``).
    """

    def __init__(
        self,
        *,
        interval: int = 3,
        deep_fraction: float = 0.4,
        warmup_steps: int = 1,
    ) -> None:
        self.interval = max(int(interval), 1)
        self.deep_fraction = min(max(float(deep_fraction), 0.0), 1.0)
        self.warmup_steps = max(int(warmup_steps), 0)
        self._cache: Dict[int, torch.Tensor] = {}
        self._step = -1
        self._last_block_index: Optional[int] = None
        self._total_blocks = 0
        self._cache_block_start: Optional[int] = None
        self._hits = 0
        self._misses = 0
        self._key_steps = 0
        self._reuse_block_calls = 0

    # -- step tracking ---------------------------------------------------- #
    def _advance_step(self, block_index: int) -> None:
        """Increment the step counter when the block loop wraps to a new step."""
        if self._last_block_index is None or block_index <= self._last_block_index:
            self._step += 1
            if self._is_key_step():
                self._key_steps += 1
        # Track the widest block index seen and (re)derive the deep-block split.
        # ``_total_blocks`` only grows during step 0 (then stabilises at N); step 0
        # is always a key step, so the split is never *used* until it is final.
        self._total_blocks = max(self._total_blocks, int(block_index) + 1)
        if self._total_blocks > 0:
            self._cache_block_start = int(self._total_blocks * self.deep_fraction)
        self._last_block_index = int(block_index)

    def _is_key_step(self) -> bool:
        if self._step < self.warmup_steps:
            return True
        return (self._step % self.interval) == 0

    def decide(self, block_index: int) -> bool:
        """Return True if this block call should reuse its cached output."""
        self._advance_step(block_index)
        if self._is_key_step():
            return False
        start = self._cache_block_start if self._cache_block_start is not None else 0
        reuse = int(block_index) >= start
        if reuse:
            self._reuse_block_calls += 1
        return reuse

    # -- cache surface ---------------------------------------------------- #
    def get(self, block_index: int) -> Optional[torch.Tensor]:
        cached = self._cache.get(int(block_index))
        if cached is not None:
            self._hits += 1
        else:
            self._misses += 1
        return cached

    def push(self, block_index: int, output: torch.Tensor) -> None:
        self._cache[int(block_index)] = output.detach()

    def clear(self) -> None:
        self._cache.clear()
        self._step = -1
        self._last_block_index = None
        self._total_blocks = 0
        self._cache_block_start = None
        self._hits = 0
        self._misses = 0
        self._key_steps = 0
        self._reuse_block_calls = 0

    def stats(self) -> Dict[str, int]:
        return {
            "cached_blocks": len(self._cache),
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "key_steps": self._key_steps,
            "reuse_block_calls": self._reuse_block_calls,
            "total_blocks": self._total_blocks,
        }


def run_with_deepcache(
    block_fn: Callable[..., torch.Tensor],
    block_index: int,
    store: DeepCacheStore,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Wrap a DiT block call with interval deep-block reuse.

    On a non-key step a deep block returns its cached output; otherwise the block
    is computed and its output stored. On key steps and during warmup every block
    is computed, so the only divergence from a plain forward is the intentional
    reuse on non-key deep blocks.
    """
    should_reuse = store.decide(int(block_index))
    if should_reuse:
        cached = store.get(int(block_index))
        if cached is not None:
            return cached
    output = block_fn(*args, **kwargs)
    store.push(int(block_index), output)
    return output


__all__ = [
    "DeepCachePolicy",
    "DeepCacheStore",
    "run_with_deepcache",
]
