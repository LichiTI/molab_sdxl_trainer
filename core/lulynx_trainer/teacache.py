# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Timestep-aware per-block reuse cache (cleanroom Lulynx "TeaCache"-family).

TeaCache accelerates multi-step DiT denoising by skipping a block's recompute
while its *input* is changing slowly. It accumulates the rescaled relative-L1
distance of a block's input between consecutive steps; while the accumulated
distance since the last real compute stays under a threshold the cached output is
reused, and once it would exceed the threshold the block is recomputed and the
accumulator resets. This is the timestep-embedding-aware reuse criterion adapted
to the block granularity of the live ``_run_blocks`` loop.

Difference from siblings: SmoothCache derives its schedule from an offline
calibration pass; DeepCache uses a fixed interval+depth split; TeaCache decides
*online* from the running input-distance signal, needing no calibration.

This store is **self-contained**: the unified cache seam drives it via
``run_block(block_fn, block_index, x, ...)``, so it reads the live block input
(``x``) directly and tracks step boundaries from the ``block_index`` wrap-around.
Default-off parity is owned by the seam (built only when ``backend="teacache"``).

Clean-room Lulynx module; references no external caching source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch


@dataclass(frozen=True)
class TeaCachePolicy:
    """Configuration for timestep-aware per-block reuse (default off)."""

    enabled: bool = False
    rel_l1_threshold: float = 0.05  # accumulated input rel-L1 budget before recompute
    warmup_steps: int = 2           # always compute the first ``warmup`` steps
    max_consecutive_skips: int = 4  # bound drift: force a recompute after N skips

    def normalized(self) -> "TeaCachePolicy":
        return TeaCachePolicy(
            enabled=bool(self.enabled),
            rel_l1_threshold=max(float(self.rel_l1_threshold or 0.0), 0.0),
            warmup_steps=max(int(self.warmup_steps or 0), 0),
            max_consecutive_skips=max(int(self.max_consecutive_skips or 0), 0),
        )


class TeaCacheStore:
    """Per-block output cache with an online input-distance reuse criterion."""

    def __init__(
        self,
        *,
        rel_l1_threshold: float = 0.05,
        warmup_steps: int = 2,
        max_consecutive_skips: int = 4,
    ) -> None:
        self.threshold = max(float(rel_l1_threshold), 0.0)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.max_consecutive_skips = max(int(max_consecutive_skips), 0)
        self._cache: Dict[int, torch.Tensor] = {}
        self._prev_input: Dict[int, torch.Tensor] = {}
        self._acc: Dict[int, float] = {}
        self._skips: Dict[int, int] = {}
        self._step = -1
        self._last_block_index: Optional[int] = None
        self._hits = 0
        self._misses = 0
        self._reuse_block_calls = 0

    def _advance_step(self, block_index: int) -> None:
        if self._last_block_index is None or block_index <= self._last_block_index:
            self._step += 1
        self._last_block_index = int(block_index)

    @staticmethod
    def _rel_l1(current: torch.Tensor, prev: torch.Tensor) -> float:
        num = (current - prev).abs().sum()
        den = prev.abs().sum().clamp_min(1e-8)
        return float((num / den).item())

    def decide(self, block_index: int, x: Optional[torch.Tensor]) -> bool:
        """Update distance state and return True to reuse the cached output."""
        self._advance_step(block_index)
        bi = int(block_index)
        if not isinstance(x, torch.Tensor):
            return False
        prev = self._prev_input.get(bi)
        # Always refresh the per-block previous input to the current step's input.
        self._prev_input[bi] = x.detach()
        if self._step < self.warmup_steps or prev is None or prev.shape != x.shape:
            self._acc[bi] = 0.0
            self._skips[bi] = 0
            return False
        acc = self._acc.get(bi, 0.0) + self._rel_l1(x, prev)
        skips = self._skips.get(bi, 0)
        if acc < self.threshold and skips < self.max_consecutive_skips:
            self._acc[bi] = acc
            self._skips[bi] = skips + 1
            self._reuse_block_calls += 1
            return True
        # Recompute: reset the accumulator and the consecutive-skip counter.
        self._acc[bi] = 0.0
        self._skips[bi] = 0
        return False

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
        self._prev_input.clear()
        self._acc.clear()
        self._skips.clear()
        self._step = -1
        self._last_block_index = None
        self._hits = 0
        self._misses = 0
        self._reuse_block_calls = 0

    def stats(self) -> Dict[str, int]:
        return {
            "cached_blocks": len(self._cache),
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "reuse_block_calls": self._reuse_block_calls,
        }


def run_with_teacache(
    block_fn: Callable[..., torch.Tensor],
    block_index: int,
    store: TeaCacheStore,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Wrap a DiT block call with timestep-aware reuse.

    Reuses the cached output while the block's input is changing slowly (small
    accumulated relative-L1); otherwise computes and caches. During warmup, on a
    cold block, or after ``max_consecutive_skips`` reuses, the block is computed,
    so the only divergence from a plain forward is the intentional reuse steps.
    """
    x = args[0] if args else None
    if store.decide(int(block_index), x):
        cached = store.get(int(block_index))
        if cached is not None:
            return cached
    output = block_fn(*args, **kwargs)
    store.push(int(block_index), output)
    return output


__all__ = [
    "TeaCachePolicy",
    "TeaCacheStore",
    "run_with_teacache",
]
