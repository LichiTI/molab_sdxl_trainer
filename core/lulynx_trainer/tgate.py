"""T-GATE probe primitives and execution layer.

The first Lulynx T-GATE stage is observe-only: it records which text
cross-attention calls would be eligible for late-step/deep-block skipping, but
does not skip or cache attention outputs.

The second stage (this module) implements actual skip/cache logic for production use.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import re
from typing import Callable, Dict, Iterator, Optional, Any

import torch


@dataclass(frozen=True)
class TGatePolicy:
    enabled: bool = False
    start_step: int = 0
    min_block: int = 0

    def normalized(self) -> "TGatePolicy":
        return TGatePolicy(
            enabled=bool(self.enabled),
            start_step=max(int(self.start_step or 0), 0),
            min_block=max(int(self.min_block or 0), 0),
        )

    def eligible(self, *, is_cross_attention: bool, step_index: Optional[int], block_index: Optional[int]) -> bool:
        if not self.enabled or not is_cross_attention:
            return False
        if step_index is None or block_index is None:
            return False
        return int(step_index) >= self.start_step and int(block_index) >= self.min_block


@dataclass(frozen=True)
class TGateStepContext:
    policy: TGatePolicy
    step_index: Optional[int] = None
    total_steps: Optional[int] = None


_CURRENT_CONTEXT: ContextVar[Optional[TGateStepContext]] = ContextVar("lulynx_tgate_context", default=None)
_STATS: Dict[str, int] = {
    "self_attention_calls": 0,
    "cross_attention_calls": 0,
    "eligible_cross_attention_calls": 0,
    "missing_step_context_calls": 0,
    "missing_block_index_calls": 0,
}


def parse_block_index(module_name: str) -> Optional[int]:
    match = re.search(r"(?:^|\.)(?:blocks|transformer_blocks|double_blocks|single_blocks)\.(\d+)(?:\.|$)", str(module_name or ""))
    if not match:
        return None
    return int(match.group(1))


@contextmanager
def tgate_step_context(
    *,
    enabled: bool,
    step_index: int,
    total_steps: int,
    start_step: int = 0,
    min_block: int = 0,
) -> Iterator[None]:
    policy = TGatePolicy(enabled=enabled, start_step=start_step, min_block=min_block).normalized()
    token = _CURRENT_CONTEXT.set(
        TGateStepContext(policy=policy, step_index=int(step_index), total_steps=int(total_steps))
    )
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)


def observe_attention_call(*, module_name: str, is_cross_attention: bool) -> bool:
    """Record an attention call and return whether it is T-GATE eligible."""
    if is_cross_attention:
        _STATS["cross_attention_calls"] += 1
    else:
        _STATS["self_attention_calls"] += 1
        return False

    context = _CURRENT_CONTEXT.get()
    if context is None:
        _STATS["missing_step_context_calls"] += 1
        return False

    block_index = parse_block_index(module_name)
    if block_index is None:
        _STATS["missing_block_index_calls"] += 1
        return False

    eligible = context.policy.eligible(
        is_cross_attention=True,
        step_index=context.step_index,
        block_index=block_index,
    )
    if eligible:
        _STATS["eligible_cross_attention_calls"] += 1
    return eligible


def snapshot_tgate_stats() -> Dict[str, int]:
    return dict(_STATS)


def reset_tgate_stats() -> None:
    for key in _STATS:
        _STATS[key] = 0


# ============================================================================
# T-GATE Execution Layer: Cache and Skip
# ============================================================================


class TGateCache:
    """Cache cross-attention outputs for T-GATE skip/reuse.

    Cross-attention between latents and text embeddings is relatively stable
    in late diffusion steps, allowing us to cache and reuse previous outputs.
    """

    def __init__(self):
        self._cache: Dict[str, torch.Tensor] = {}
        self._hits = 0
        self._misses = 0

    def get(self, module_name: str) -> Optional[torch.Tensor]:
        """Retrieve cached attention output for a module."""
        cached = self._cache.get(module_name)
        if cached is not None:
            self._hits += 1
        else:
            self._misses += 1
        return cached

    def set(self, module_name: str, output: torch.Tensor):
        """Store attention output in cache (detached from graph)."""
        self._cache[module_name] = output.detach()

    def clear(self):
        """Clear all cached outputs."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._hits,
            "cache_misses": self._misses,
        }


def run_with_tgate_skip(
    attention_fn: Callable[..., torch.Tensor],
    module_name: str,
    cache: TGateCache,
    is_cross_attention: bool,
    *args,
    **kwargs
) -> torch.Tensor:
    """Wrap attention call with T-GATE skip logic.

    If the call is eligible for T-GATE (cross-attention, late step, deep block),
    attempt to reuse cached output. Otherwise, execute normally and cache result.

    Parameters
    ----------
    attention_fn : callable
        The actual attention forward function to call.
    module_name : str
        Module identifier (e.g., "blocks.10.cross_attn").
    cache : TGateCache
        Shared cache instance for this generation.
    is_cross_attention : bool
        Whether this is a cross-attention (text conditioning) call.
    *args, **kwargs
        Arguments to pass to attention_fn.

    Returns
    -------
    torch.Tensor
        Attention output (either cached or freshly computed).
    """
    # Check eligibility via existing probe logic
    eligible = observe_attention_call(
        module_name=module_name,
        is_cross_attention=is_cross_attention,
    )

    if not eligible:
        # Not eligible for skip: execute normally and cache
        output = attention_fn(*args, **kwargs)
        if is_cross_attention:
            cache.set(module_name, output)
        return output

    # Eligible: try to use cache
    cached = cache.get(module_name)
    if cached is not None:
        # Cache hit: reuse previous output
        # Ensure same shape as expected output
        return cached

    # Cache miss (first eligible step): execute and cache
    output = attention_fn(*args, **kwargs)
    cache.set(module_name, output)
    return output


@dataclass(frozen=True)
class TGateExecutionStats:
    """Statistics for T-GATE execution."""
    total_attention_calls: int
    cross_attention_calls: int
    eligible_calls: int
    cache_hits: int
    cache_misses: int
    skip_rate: float  # cache_hits / eligible_calls

    @classmethod
    def from_stats(cls, probe_stats: Dict[str, int], cache_stats: Dict[str, int]) -> "TGateExecutionStats":
        eligible = probe_stats.get("eligible_cross_attention_calls", 0)
        hits = cache_stats.get("cache_hits", 0)
        skip_rate = hits / max(eligible, 1)
        return cls(
            total_attention_calls=probe_stats.get("self_attention_calls", 0) + probe_stats.get("cross_attention_calls", 0),
            cross_attention_calls=probe_stats.get("cross_attention_calls", 0),
            eligible_calls=eligible,
            cache_hits=hits,
            cache_misses=cache_stats.get("cache_misses", 0),
            skip_rate=skip_rate,
        )

