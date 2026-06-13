# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Dynamo/AOT budget knobs for the compile path.

Two traps these helpers exist to avoid (both absorbed from real-world
debugging records of compiled DiT training; see
plans/absorbed-pitfalls-anima-lora-190.md):

1. ``torch._dynamo.config.recompile_limit`` is ContextVar-backed: a plain
   ``config.recompile_limit = N`` assignment only takes effect in the thread/
   context that ran it. The backward pass is compiled in a *different*
   context (AOTAutograd), where the override is absent and the read falls
   back to the entry's ``default`` (8) — the budget silently reverts and the
   first grad-bearing forward spills to eager. :func:`pin_recompile_limit`
   raises both the override (same-context reads + log visibility) and the
   canonical entry's ``.default`` (cross-context fallback every thread reads).

2. ``torch._functorch.config.activation_memory_budget`` < 1.0 makes the AOT
   min-cut partitioner recompute the cheapest intermediates in backward
   instead of saving them — the sanctioned lever when the partitioner keeps
   too many activations (try it before gradient checkpointing on OOM). It is
   incompatible with ``torch.utils.checkpoint``: repartitioning can make the
   checkpoint recompute pass select a different graph than forward
   (CheckpointError, pytorch #166926) — and grad-ckpt already minimizes saved
   activations, so the budget is also redundant there.
   :func:`apply_activation_memory_budget` guards the combination.

Keep this module dependency-free (torch + logging only) so any layer can
import it without a cycle.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def pin_recompile_limit(limit: int, *, log: Optional[Callable[[str], Any]] = None) -> int:
    """Raise the dynamo recompile budget so it holds in EVERY compile context.

    Sets both the context-local override and the canonical config entry's
    ``.default`` (following aliases such as ``cache_size_limit`` →
    ``recompile_limit``), so the backward-compile thread reads the same
    budget. Returns the effective value: ``max(current, limit)`` — never
    lowers an already-raised budget.
    """
    limit = int(limit)
    if limit <= 0:
        return 0

    import torch._dynamo as _dynamo

    cfg = _dynamo.config
    name = "recompile_limit" if hasattr(cfg, "recompile_limit") else "cache_size_limit"
    effective = max(int(getattr(cfg, name)), limit)
    setattr(cfg, name, effective)  # context-local override
    try:
        entry = cfg._config[name]
        canonical = (getattr(entry, "alias", None) or name).rsplit(".", 1)[-1]
        cfg._config[canonical].default = effective  # cross-context fallback
        message = f"dynamo {name} pinned to {effective} (override + cross-context default)"
    except Exception as exc:  # noqa: BLE001 - defensive against torch internals drift
        message = (
            f"dynamo {name} raised to {effective}, but the cross-context default could not "
            f"be pinned ({type(exc).__name__}: {exc}); the budget may revert in the "
            "backward-compile context and spill to eager"
        )
        logger.warning(message)
    if log is not None:
        log(message)
    return effective


def apply_activation_memory_budget(
    budget: float,
    *,
    gradient_checkpointing: bool,
    log: Optional[Callable[[str], Any]] = None,
) -> bool:
    """Cap the AOT partitioner's saved-for-backward set. Returns True when applied.

    ``budget`` in (0, 1): below 1.0 the min-cut partitioner solves a knapsack
    to recompute the cheapest intermediates in backward. 0 (or unset) means
    leave torch's default untouched. Skipped (with a log line) under
    gradient checkpointing — incompatible (pytorch #166926) and redundant.
    """
    budget = float(budget)
    if budget <= 0.0:
        return False

    def _emit(message: str) -> None:
        if log is not None:
            log(message)

    if budget > 1.0:
        _emit(f"activation_memory_budget={budget} is out of range (0, 1]; ignoring")
        return False
    if gradient_checkpointing:
        _emit(
            f"activation_memory_budget={budget} skipped: incompatible with "
            "gradient_checkpointing (checkpoint recompute can diverge from the "
            "repartitioned forward graph) and redundant there"
        )
        return False

    import torch._functorch.config as functorch_config

    functorch_config.activation_memory_budget = budget
    _emit(f"AOT partitioner activation_memory_budget set to {budget}")
    return True


__all__ = ["pin_recompile_limit", "apply_activation_memory_budget"]
