# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Unified DiT block compute-reducer seam (roadmap TREAD / DiffCR / BlockSkip).

The TREAD token-routing, DiffCR token-compression, and BlockSkip block-skipping
primitives each already ship a ``run_*`` wrapper whose *disabled* path is an
exact ``block(tokens)`` call.  This seam is the single opt-in switch that
actually *drives* one of those primitives inside the live Anima/Newbie DiT
``_run_blocks`` loop, so the compute-reduction layer stops being a library-only
primitive -- mirroring how ``unified_cache_seam`` drives Spectrum/SmoothCache.

Design invariants (identical posture to ``unified_cache_seam``)
---------------------------------------------------------------
* **Default off, parity red-line.** ``strategy="none"`` (or a disabled policy)
  makes ``run_block`` call the block verbatim, so the live forward is bitwise
  identical to today's behaviour.  ``get_active_compute_reducer_seam`` returns
  ``None`` unless a seam is *explicitly* published, so an un-wired run never
  pays more than one ``ContextVar`` read.
* **One reducer at a time.** Only one block-level strategy drives a forward;
  the seam never composes two reducers.  An unknown strategy normalises to
  ``"none"`` (parity passthrough) so a caller is never silently handed a wrong
  reduction.
* **ContextVar scoped.** Published for the duration of one model forward,
  mirroring the cache seam, so it auto-resets and never leaks across runs.

Loss-parity and quality-drift A/B remain the operator's real-model job; this
seam plus its smoke only establish the *local* shape-stability and
disabled-path bitwise parity evidence the trainer gate asks for before wiring.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional

try:  # package import
    from .tread_token_routing import TreadTokenRoutePolicy, run_tread_routed_block
    from .diffcr_token_compression import (
        DiffCRTokenCompressionPolicy,
        run_diffcr_compressed_block,
    )
    from .dit_blockskip_training_spike import (
        DiTBlockSkipPolicy,
        apply_dit_blockskip_decision,
        build_dit_blockskip_plan,
    )
except ImportError:  # pragma: no cover - direct-file smoke fallback
    from core.lulynx_trainer.tread_token_routing import TreadTokenRoutePolicy, run_tread_routed_block
    from core.lulynx_trainer.diffcr_token_compression import (
        DiffCRTokenCompressionPolicy,
        run_diffcr_compressed_block,
    )
    from core.lulynx_trainer.dit_blockskip_training_spike import (
        DiTBlockSkipPolicy,
        apply_dit_blockskip_decision,
        build_dit_blockskip_plan,
    )


BLOCK_LEVEL_STRATEGIES = ("tread", "diffcr", "blockskip")
KNOWN_STRATEGIES = ("none", "tread", "diffcr", "blockskip")


@dataclass(frozen=True)
class DiTComputeReducerSeamPolicy:
    """Normalised, default-off selection of one block-level compute reducer."""

    enabled: bool = False
    strategy: str = "none"
    # TREAD token routing
    keep_ratio: float = 1.0
    min_keep_tokens: int = 1
    # DiffCR token compression
    compression_ratio: float = 1.0
    min_tokens: int = 1
    # BlockSkip block skipping
    skip_ratio: float = 0.0
    skip_every: int = 0
    warmup_steps: int = 0
    min_block: int = 0
    # shared scoring mode for token reducers
    score_mode: str = "l2"

    def normalized(self) -> "DiTComputeReducerSeamPolicy":
        strategy = str(self.strategy or "none").strip().lower().replace("-", "").replace("_", "")
        if strategy not in KNOWN_STRATEGIES:
            strategy = "none"
        return DiTComputeReducerSeamPolicy(
            enabled=bool(self.enabled) and strategy in BLOCK_LEVEL_STRATEGIES,
            strategy=strategy,
            keep_ratio=min(max(float(self.keep_ratio), 0.0), 1.0),
            min_keep_tokens=max(int(self.min_keep_tokens), 1),
            compression_ratio=min(max(float(self.compression_ratio), 0.0), 1.0),
            min_tokens=max(int(self.min_tokens), 1),
            skip_ratio=min(max(float(self.skip_ratio), 0.0), 0.95),
            skip_every=max(int(self.skip_every), 0),
            warmup_steps=max(int(self.warmup_steps), 0),
            min_block=max(int(self.min_block), 0),
            score_mode=str(self.score_mode or "l2").strip().lower(),
        )

    def to_dict(self) -> Dict[str, Any]:
        n = self.normalized()
        return {
            "enabled": n.enabled,
            "strategy": n.strategy,
            "keep_ratio": n.keep_ratio,
            "min_keep_tokens": n.min_keep_tokens,
            "compression_ratio": n.compression_ratio,
            "min_tokens": n.min_tokens,
            "skip_ratio": n.skip_ratio,
            "skip_every": n.skip_every,
            "warmup_steps": n.warmup_steps,
            "min_block": n.min_block,
            "score_mode": n.score_mode,
        }


class DiTComputeReducerSeam:
    """Opt-in block-level compute-reducer dispatcher for one strategy."""

    def __init__(
        self,
        policy: DiTComputeReducerSeamPolicy | None = None,
        *,
        total_blocks: int = 0,
        step_index: int = 0,
        total_steps: int = 0,
    ) -> None:
        self.policy = (policy or DiTComputeReducerSeamPolicy()).normalized()
        self.strategy = self.policy.strategy
        self.enabled = self.policy.enabled
        self._total_blocks = max(int(total_blocks), 0)
        self._step_index = max(int(step_index), 0)
        self._total_steps = max(int(total_steps), 0)
        self._blockskip_plan: Any = None
        self._blockskip_dirty = True
        self._last_output: Any = None

        self._tread_policy = (
            TreadTokenRoutePolicy(
                enabled=True,
                keep_ratio=self.policy.keep_ratio,
                min_keep_tokens=self.policy.min_keep_tokens,
                score_mode=self.policy.score_mode,
            )
            if self.strategy == "tread"
            else None
        )
        self._diffcr_policy = (
            DiffCRTokenCompressionPolicy(
                enabled=True,
                compression_ratio=self.policy.compression_ratio,
                min_tokens=self.policy.min_tokens,
                score_mode=self.policy.score_mode,
            )
            if self.strategy == "diffcr"
            else None
        )
        self._blockskip_policy = (
            DiTBlockSkipPolicy(
                enabled=True,
                skip_ratio=self.policy.skip_ratio,
                skip_every=self.policy.skip_every,
                warmup_steps=self.policy.warmup_steps,
                min_block=self.policy.min_block,
            )
            if self.strategy == "blockskip"
            else None
        )

    # -- live step / topology updates (cheap; only blockskip consumes them) ---
    def set_total_blocks(self, total_blocks: int) -> None:
        value = max(int(total_blocks), 0)
        if value != self._total_blocks:
            self._total_blocks = value
            self._blockskip_dirty = True

    def set_step(self, step_index: int, total_steps: int = 0) -> None:
        step = max(int(step_index), 0)
        steps = max(int(total_steps), 0)
        if step != self._step_index or steps != self._total_steps:
            self._step_index = step
            self._total_steps = steps
            self._blockskip_dirty = True

    def _ensure_blockskip_plan(self) -> Any:
        if self.strategy != "blockskip" or self._total_blocks <= 0:
            return None
        if self._blockskip_plan is None or self._blockskip_dirty:
            self._blockskip_plan = build_dit_blockskip_plan(
                total_blocks=self._total_blocks,
                step_index=self._step_index,
                total_steps=self._total_steps,
                policy=self._blockskip_policy,
            )
            self._blockskip_dirty = False
        return self._blockskip_plan

    def run_block(self, block_fn: Callable[..., Any], block_index: int, *args: Any, **kwargs: Any) -> Any:
        """Drive the active reducer for one DiT block, or pass through verbatim."""
        if not self.enabled or not args:
            return block_fn(*args, **kwargs)
        tokens = args[0]
        rest = args[1:]

        def inner(toks: Any) -> Any:
            return block_fn(toks, *rest, **kwargs)

        strategy = self.strategy
        if strategy == "tread":
            output, _ = run_tread_routed_block(tokens, inner, self._tread_policy)
            return output
        if strategy == "diffcr":
            output, _ = run_diffcr_compressed_block(
                tokens, inner, self._diffcr_policy, layer_index=int(block_index)
            )
            return output
        if strategy == "blockskip":
            plan = self._ensure_blockskip_plan()
            if plan is None or int(block_index) >= len(plan.decisions):
                return inner(tokens)
            if int(block_index) == 0:
                self._last_output = None
            output = apply_dit_blockskip_decision(
                tokens, inner, plan.decisions[int(block_index)], cached_residual=self._last_output
            )
            self._last_output = output
            return output
        return inner(tokens)

    def should_skip_block(self, block_index: int) -> bool:
        """Whether blockskip's deterministic plan elects to skip this block.

        Token-preserving identity skip: when ``True`` the caller passes ``x``
        through unchanged. This lets the faithful native forward drive blockskip
        from its own block loop (bypassing :meth:`run_block`) so ``rope_emb`` and
        native block-checkpointing both stay intact — TREAD/DiffCR cannot do this
        because they change the token count. Always ``False`` for non-blockskip
        strategies (and when disabled), so the seam is inert under faithful unless
        blockskip is the active strategy.
        """
        if not self.enabled or self.strategy != "blockskip":
            return False
        plan = self._ensure_blockskip_plan()
        if plan is None or int(block_index) >= len(plan.decisions):
            return False
        return bool(plan.decisions[int(block_index)].skip)

    def clear(self) -> None:
        self._blockskip_plan = None
        self._blockskip_dirty = True
        self._last_output = None

    def stats(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "enabled": self.enabled,
            "total_blocks": self._total_blocks,
            "step_index": self._step_index,
        }


_CURRENT_SEAM: ContextVar[Optional[DiTComputeReducerSeam]] = ContextVar(
    "lulynx_dit_compute_reducer_seam", default=None
)


@contextmanager
def compute_reducer_seam_context(
    seam: Optional[DiTComputeReducerSeam],
) -> Iterator[Optional[DiTComputeReducerSeam]]:
    """Publish ``seam`` for the duration of one model forward."""
    token = _CURRENT_SEAM.set(seam)
    try:
        yield seam
    finally:
        _CURRENT_SEAM.reset(token)


def get_active_compute_reducer_seam() -> Optional[DiTComputeReducerSeam]:
    seam = _CURRENT_SEAM.get()
    if seam is not None and seam.enabled:
        return seam
    return None


def build_compute_reducer_seam(
    *,
    enabled: bool = False,
    strategy: str = "none",
    total_blocks: int = 0,
    step_index: int = 0,
    total_steps: int = 0,
    keep_ratio: float = 1.0,
    min_keep_tokens: int = 1,
    compression_ratio: float = 1.0,
    min_tokens: int = 1,
    skip_ratio: float = 0.0,
    skip_every: int = 0,
    warmup_steps: int = 0,
    min_block: int = 0,
    score_mode: str = "l2",
) -> DiTComputeReducerSeam:
    policy = DiTComputeReducerSeamPolicy(
        enabled=enabled,
        strategy=strategy,
        keep_ratio=keep_ratio,
        min_keep_tokens=min_keep_tokens,
        compression_ratio=compression_ratio,
        min_tokens=min_tokens,
        skip_ratio=skip_ratio,
        skip_every=skip_every,
        warmup_steps=warmup_steps,
        min_block=min_block,
        score_mode=score_mode,
    )
    return DiTComputeReducerSeam(
        policy, total_blocks=total_blocks, step_index=step_index, total_steps=total_steps
    )


__all__ = [
    "DiTComputeReducerSeamPolicy",
    "DiTComputeReducerSeam",
    "compute_reducer_seam_context",
    "get_active_compute_reducer_seam",
    "build_compute_reducer_seam",
    "BLOCK_LEVEL_STRATEGIES",
    "KNOWN_STRATEGIES",
]
