# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Unified DiT block cache seam (roadmap 624 ``deepcache_spectrum_cache_ab``).

The Spectrum and SmoothCache probes are already wired observe-only into the
live Anima/Newbie DiT ``_run_blocks`` loop, and each ships a block-level
execution primitive (``run_with_spectrum_skip`` / ``run_with_smoothcache``).
This seam is the single opt-in switch that actually *drives* one of those
primitives during a generation, so the cache execution layer stops being a
library-only primitive.

Design invariants
-----------------
* **Default off, parity red-line.** ``backend="none"`` (or a disabled policy)
  makes ``run_block`` call the block verbatim, so the live forward is bitwise
  identical to today's behavior.  Even when enabled, a step that the active
  Spectrum/SmoothCache decision does not mark cacheable computes normally, so
  the only divergence is on intentionally cached steps.
* **Block granularity.** Only Spectrum and SmoothCache are block-level and
  share the ``run_*(block_fn, block_index, cache, *args)`` contract.  T-GATE is
  cross-attention granularity (``run_with_tgate_skip`` operates inside a block)
  and is *not* driven here; it stays the already-wired observe probe + library
  primitive.  ``backend="tgate"`` is accepted but leaves the block seam in the
  parity passthrough so callers are never silently given a wrong granularity.
* **ContextVar scoped.** The seam holds a cache that must persist across the
  whole denoise loop, so it is published via a ContextVar for the generation,
  mirroring the per-step ``*_step_context`` pattern the probes already use.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional

try:  # package import
    from .spectrum_probe import SpectrumCache, run_with_spectrum_skip
    from .smoothcache import SmoothCacheStore, run_with_smoothcache
except ImportError:  # pragma: no cover - direct-file smoke fallback
    from core.lulynx_trainer.spectrum_probe import SpectrumCache, run_with_spectrum_skip
    from core.lulynx_trainer.smoothcache import SmoothCacheStore, run_with_smoothcache


BLOCK_LEVEL_BACKENDS = ("spectrum", "smoothcache")
KNOWN_BACKENDS = ("none", "spectrum", "smoothcache", "tgate")


@dataclass(frozen=True)
class UnifiedCacheSeamPolicy:
    enabled: bool = False
    backend: str = "none"
    spectrum_window_size: int = 3

    def normalized(self) -> "UnifiedCacheSeamPolicy":
        backend = str(self.backend or "none").strip().lower().replace("-", "").replace("_", "")
        if backend not in KNOWN_BACKENDS:
            backend = "none"
        return UnifiedCacheSeamPolicy(
            enabled=bool(self.enabled) and backend in BLOCK_LEVEL_BACKENDS,
            backend=backend,
            spectrum_window_size=max(int(self.spectrum_window_size), 2),
        )

    def to_dict(self) -> Dict[str, Any]:
        n = self.normalized()
        return {"enabled": n.enabled, "backend": n.backend, "spectrum_window_size": n.spectrum_window_size}


class UnifiedCacheSeam:
    """Opt-in block-cache dispatcher for one block-level backend."""

    def __init__(self, policy: UnifiedCacheSeamPolicy | None = None) -> None:
        self.policy = (policy or UnifiedCacheSeamPolicy()).normalized()
        self.backend = self.policy.backend
        self.enabled = self.policy.enabled
        self._cache: Any = None
        if self.enabled:
            if self.backend == "spectrum":
                self._cache = SpectrumCache(window_size=self.policy.spectrum_window_size)
            elif self.backend == "smoothcache":
                self._cache = SmoothCacheStore()

    def run_block(self, block_fn: Callable[..., Any], block_index: int, *args, **kwargs) -> Any:
        """Drive the active backend for one DiT block, or pass through verbatim."""
        if not self.enabled or self._cache is None:
            return block_fn(*args, **kwargs)
        if self.backend == "spectrum":
            return run_with_spectrum_skip(block_fn, block_index, self._cache, *args, **kwargs)
        if self.backend == "smoothcache":
            return run_with_smoothcache(block_fn, block_index, self._cache, *args, **kwargs)
        return block_fn(*args, **kwargs)

    def clear(self) -> None:
        if self._cache is not None and hasattr(self._cache, "clear"):
            self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        cache_stats = self._cache.stats() if self._cache is not None else {}
        return {"backend": self.backend, "enabled": self.enabled, **cache_stats}


_CURRENT_SEAM: ContextVar[Optional[UnifiedCacheSeam]] = ContextVar("lulynx_cache_seam", default=None)


@contextmanager
def cache_seam_context(seam: Optional[UnifiedCacheSeam]) -> Iterator[Optional[UnifiedCacheSeam]]:
    """Publish ``seam`` for the duration of a generation (denoise loop)."""
    token = _CURRENT_SEAM.set(seam)
    try:
        yield seam
    finally:
        _CURRENT_SEAM.reset(token)


def get_active_cache_seam() -> Optional[UnifiedCacheSeam]:
    seam = _CURRENT_SEAM.get()
    if seam is not None and seam.enabled:
        return seam
    return None


def build_cache_seam(
    *,
    enabled: bool = False,
    backend: str = "none",
    spectrum_window_size: int = 3,
) -> UnifiedCacheSeam:
    return UnifiedCacheSeam(
        UnifiedCacheSeamPolicy(enabled=enabled, backend=backend, spectrum_window_size=spectrum_window_size)
    )


@dataclass(frozen=True)
class InferenceAccelResolution:
    """Low-level switches a high-level inference-accel scheme resolves to.

    A real block-cache skip needs *two* switches flipped together: the probe
    (per-step decision source published via ``*_step_context``) and the seam
    backend (the execution driver that actually calls ``run_with_*``).  A
    single user-facing scheme maps to that pair, so callers (generation
    request, training-preview config, UI) pass one value and never wire the
    two switches by hand.  ``none`` / ``tgate`` (observe-only, cross-attention
    granularity) / unknown all resolve to a full-off parity passthrough.
    """

    scheme: str
    spectrum_probe: bool
    smoothcache_probe: bool
    cache_seam_backend: str

    @property
    def enabled(self) -> bool:
        return self.cache_seam_backend in BLOCK_LEVEL_BACKENDS


def resolve_inference_accel_scheme(scheme: str | None) -> InferenceAccelResolution:
    """Map a high-level accel scheme to its (probe, seam) switch pair.

    Only ``spectrum`` / ``smoothcache`` are block-level real-skip schemes;
    anything else resolves to off (bitwise parity).  Accepts hyphen/underscore
    spellings (``smooth-cache`` / ``smooth_cache``).
    """
    token = str(scheme or "none").strip().lower().replace("-", "").replace("_", "")
    if token == "spectrum":
        return InferenceAccelResolution("spectrum", True, False, "spectrum")
    if token == "smoothcache":
        return InferenceAccelResolution("smoothcache", False, True, "smoothcache")
    return InferenceAccelResolution("none", False, False, "none")


__all__ = [
    "UnifiedCacheSeamPolicy",
    "UnifiedCacheSeam",
    "cache_seam_context",
    "get_active_cache_seam",
    "build_cache_seam",
    "InferenceAccelResolution",
    "resolve_inference_accel_scheme",
    "BLOCK_LEVEL_BACKENDS",
    "KNOWN_BACKENDS",
]
