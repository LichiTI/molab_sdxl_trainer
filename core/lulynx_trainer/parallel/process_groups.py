# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Process-group setup for Lulynx tensor/sequence parallelism (cleanroom).

Builds a Megatron-style 2D mesh: the global world is split into ``tp_degree``-
wide tensor-parallel groups, and the matching ranks across those groups form the
data-parallel groups.  When distributed is unavailable / uninitialized, or
``tp_degree <= 1``, this returns a **degenerate** :class:`ProcessGroups` with
``tp_size == 1`` and no real groups, so every collective downstream becomes an
identity passthrough — the single-GPU path costs nothing and is bit-identical to
the unsharded model.

``backend="cuda_direct"`` flips on NCCL P2P/IPC transport env hints before any
group is created (falling back to plain NCCL if unsupported) — the v1 hook for
CUDA-direct collectives.

Clean-room Lulynx module; references no external parallelism source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


def maybe_activate_backend(backend: str) -> str:
    """Apply transport hints for the requested backend; return the effective one.

    ``cuda_direct`` enables NCCL peer-to-peer / IPC shared-memory transports via
    env hints (a thin, dependency-free stand-in for a custom IPC collective).
    Anything unsupported degrades to ``nccl``.
    """
    name = str(backend or "nccl").strip().lower()
    if name == "cuda_direct":
        if torch.cuda.is_available():
            os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
            os.environ.setdefault("NCCL_SHM_DISABLE", "0")
            os.environ.setdefault("NCCL_P2P_DISABLE", "0")
            return "cuda_direct"
        return "nccl"
    return "nccl" if name not in {"nccl", "gloo"} else name


@dataclass
class ProcessGroups:
    """Resolved tensor-parallel / data-parallel groups and this rank's place."""

    tp_group: Optional[dist.ProcessGroup]
    dp_group: Optional[dist.ProcessGroup]
    _tp_world: int
    _tp_rank: int
    _dp_world: int
    _dp_rank: int
    backend: str = "nccl"

    @property
    def tp_size(self) -> int:
        return self._tp_world

    @property
    def tp_rank(self) -> int:
        return self._tp_rank

    @property
    def dp_size(self) -> int:
        return self._dp_world

    @property
    def dp_rank(self) -> int:
        return self._dp_rank

    @property
    def is_degenerate(self) -> bool:
        """True when there is no real tensor-parallel sharding (single GPU)."""
        return self._tp_world <= 1

    @property
    def is_dp_main(self) -> bool:
        return self._dp_rank == 0


def _degenerate(backend: str = "nccl") -> ProcessGroups:
    return ProcessGroups(
        tp_group=None, dp_group=None,
        _tp_world=1, _tp_rank=0, _dp_world=1, _dp_rank=0, backend=backend,
    )


def init_parallel(tp_degree: int = 1, sequence_parallel: bool = False, backend: str = "nccl") -> ProcessGroups:
    """Resolve process groups for the requested tensor-parallel degree.

    Returns a degenerate (single-process) group unless distributed is
    initialized *and* ``tp_degree > 1``.  Mesh layout: rank ``r`` belongs to
    tp-group ``r // tp_degree`` at tp-rank ``r % tp_degree``.
    """
    effective_backend = maybe_activate_backend(backend)
    if int(tp_degree) <= 1 or not dist.is_available() or not dist.is_initialized():
        return _degenerate(effective_backend)

    world = dist.get_world_size()
    rank = dist.get_rank()
    if world % int(tp_degree) != 0:
        raise ValueError(f"world_size {world} not divisible by tp_degree {tp_degree}")
    tp_degree = int(tp_degree)
    dp_degree = world // tp_degree

    tp_group = None
    dp_group = None
    # Tensor-parallel groups: contiguous blocks of tp_degree ranks.
    for g in range(dp_degree):
        ranks = list(range(g * tp_degree, (g + 1) * tp_degree))
        grp = dist.new_group(ranks)
        if rank in ranks:
            tp_group = grp
    # Data-parallel groups: same tp-rank across the dp blocks.
    for t in range(tp_degree):
        ranks = list(range(t, world, tp_degree))
        grp = dist.new_group(ranks)
        if rank in ranks:
            dp_group = grp

    return ProcessGroups(
        tp_group=tp_group,
        dp_group=dp_group,
        _tp_world=tp_degree,
        _tp_rank=rank % tp_degree,
        _dp_world=dp_degree,
        _dp_rank=rank // tp_degree,
        backend=effective_backend,
    )


__all__ = ["ProcessGroups", "init_parallel", "maybe_activate_backend"]
