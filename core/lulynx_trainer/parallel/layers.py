# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Column / row tensor-parallel linear layers (cleanroom Lulynx).

The canonical Megatron pattern for an attention/MLP block:

* **ColumnParallelLinear** shards the weight along ``out_features`` (dim 0).
  Each rank holds ``out/tp`` output columns; with ``gather_output=False`` the
  output stays sharded so the next layer can consume it directly.
* **RowParallelLinear** shards the weight along ``in_features`` (dim 1) and
  expects a column-sharded input; it all-reduces the partial outputs to a full
  result.

Chaining ``ColumnParallelLinear(gather_output=False) → act → RowParallelLinear
(input_is_parallel=True)`` reproduces a dense MLP with all communication folded
into the two collectives.  With ``tp_size == 1`` both layers hold the full
weight and the collectives are identities, so they are bit-identical to
``nn.Linear``.

Clean-room Lulynx module; references no external parallelism source.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .collectives import copy_to_tp_region, gather_last_dim, reduce_from_tp_region
from .process_groups import ProcessGroups


def _tp(groups: Optional[ProcessGroups]):
    if groups is None:
        return None, 1, 0
    return groups.tp_group, groups.tp_size, groups.tp_rank


def _shard(tensor: torch.Tensor, dim: int, world: int, rank: int) -> torch.Tensor:
    if world <= 1:
        return tensor.clone()
    return tensor.chunk(world, dim=dim)[rank].clone()


class ColumnParallelLinear(nn.Module):
    """Linear sharded along ``out_features``; optionally gathers the output."""

    def __init__(self, in_features, out_features, bias=True, gather_output=False,
                 groups: Optional[ProcessGroups] = None):
        super().__init__()
        group, world, rank = _tp(groups)
        if out_features % world != 0:
            raise ValueError(f"out_features {out_features} not divisible by tp_size {world}")
        self.in_features = in_features
        self.out_features = out_features
        self.local_out = out_features // world
        self.gather_output = gather_output
        self._group = group
        self.weight = nn.Parameter(torch.empty(self.local_out, in_features))
        self.bias = nn.Parameter(torch.empty(self.local_out)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, groups: Optional[ProcessGroups] = None,
                    gather_output=False) -> "ColumnParallelLinear":
        group, world, rank = _tp(groups)
        out_f, in_f = linear.weight.shape
        mod = cls(in_f, out_f, bias=linear.bias is not None, gather_output=gather_output, groups=groups)
        with torch.no_grad():
            mod.weight.copy_(_shard(linear.weight.data, 0, world, rank))
            if linear.bias is not None:
                mod.bias.copy_(_shard(linear.bias.data, 0, world, rank))
        mod._group = group
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = copy_to_tp_region(x, self._group)
        out = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            out = gather_last_dim(out, self._group)
        return out


class RowParallelLinear(nn.Module):
    """Linear sharded along ``in_features``; all-reduces partial outputs."""

    def __init__(self, in_features, out_features, bias=True, input_is_parallel=True,
                 groups: Optional[ProcessGroups] = None):
        super().__init__()
        group, world, rank = _tp(groups)
        if in_features % world != 0:
            raise ValueError(f"in_features {in_features} not divisible by tp_size {world}")
        self.in_features = in_features
        self.out_features = out_features
        self.local_in = in_features // world
        self.input_is_parallel = input_is_parallel
        self._group = group
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in))
        # Bias is full and added once, after the reduction.
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, groups: Optional[ProcessGroups] = None,
                    input_is_parallel=True) -> "RowParallelLinear":
        group, world, rank = _tp(groups)
        out_f, in_f = linear.weight.shape
        mod = cls(in_f, out_f, bias=linear.bias is not None, input_is_parallel=input_is_parallel, groups=groups)
        with torch.no_grad():
            mod.weight.copy_(_shard(linear.weight.data, 1, world, rank))
            if linear.bias is not None:
                mod.bias.copy_(linear.bias.data.clone())  # full bias, added post-reduce
        mod._group = group
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, None)
        out = reduce_from_tp_region(out, self._group)
        if self.bias is not None:
            out = out + self.bias
        return out


def merge_column_shards(shards: list[torch.Tensor]) -> torch.Tensor:
    """Reassemble a full column-parallel weight/bias from per-rank shards."""
    return torch.cat(shards, dim=0)


def merge_row_shards(shards: list[torch.Tensor]) -> torch.Tensor:
    """Reassemble a full row-parallel weight from per-rank shards (dim 1)."""
    return torch.cat(shards, dim=1)


__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "merge_column_shards",
    "merge_row_shards",
]
