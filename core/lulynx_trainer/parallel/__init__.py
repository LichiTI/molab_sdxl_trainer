# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Lulynx tensor / sequence parallelism (v1, subsystem-level, cleanroom).

A Megatron-style tensor-parallel toolkit: process-group setup, autograd-aware
collectives, column/row parallel linears, and a declarative apply pass.  Default
off and **not** wired into the live trainer model build yet — it is a tested,
scorecard-gated subsystem that is bit-identical to the unsharded model at
``tp_size == 1`` (verified single-GPU) and ready to shard real models under
``torchrun`` when multi-GPU work resumes.

Clean-room Lulynx implementation; references no external parallelism source.
"""

from __future__ import annotations

from .apply import ParallelSpec, apply_tensor_parallel
from .collectives import (
    copy_to_tp_region,
    gather_to_sp,
    reduce_from_tp_region,
    scatter_to_sp,
    world_size,
)
from .layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    merge_column_shards,
    merge_row_shards,
)
from .process_groups import ProcessGroups, init_parallel, maybe_activate_backend

__version__ = "0.1.0"

__all__ = [
    "ParallelSpec",
    "apply_tensor_parallel",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "merge_column_shards",
    "merge_row_shards",
    "ProcessGroups",
    "init_parallel",
    "maybe_activate_backend",
    "copy_to_tp_region",
    "reduce_from_tp_region",
    "gather_to_sp",
    "scatter_to_sp",
    "world_size",
]
