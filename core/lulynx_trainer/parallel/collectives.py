# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Autograd-aware collectives for tensor / sequence parallelism (cleanroom).

These are the Megatron ``f`` / ``g`` conjugate operators:

* :func:`copy_to_tp_region` (``f``) — identity forward, all-reduce backward.
  Used at the *entry* of a column-parallel region so each rank sees the full
  input but gradients sum across the group.
* :func:`reduce_from_tp_region` (``g``) — all-reduce forward, identity backward.
  Used at the *exit* of a row-parallel region to sum the partial outputs.
* :func:`gather_to_sp` / :func:`scatter_to_sp` — sequence-parallel gather /
  reduce-scatter along the token dimension.

Every op short-circuits to a plain passthrough when the group's world size is 1,
so the single-GPU path makes **no** distributed calls and is bit-identical to an
unsharded module.

Clean-room Lulynx module; references no external parallelism source.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist


def world_size(group: Optional[dist.ProcessGroup]) -> int:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


def _all_reduce(t: torch.Tensor, group) -> torch.Tensor:
    t = t.contiguous()
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    return t


def _all_gather(t: torch.Tensor, group, dim: int) -> torch.Tensor:
    ws = world_size(group)
    parts = [torch.empty_like(t) for _ in range(ws)]
    dist.all_gather(parts, t.contiguous(), group=group)
    return torch.cat(parts, dim=dim)


def _reduce_scatter(t: torch.Tensor, group, dim: int) -> torch.Tensor:
    ws = world_size(group)
    chunks = list(t.contiguous().chunk(ws, dim=dim))
    out = torch.empty_like(chunks[0])
    dist.reduce_scatter(out, chunks, op=dist.ReduceOp.SUM, group=group)
    return out


class _CopyToTPRegion(torch.autograd.Function):
    """f: forward identity, backward all-reduce (sum gradients across tp)."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        if world_size(ctx.group) == 1:
            return grad, None
        return _all_reduce(grad, ctx.group), None


class _ReduceFromTPRegion(torch.autograd.Function):
    """g: forward all-reduce (sum partials across tp), backward identity."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        if world_size(group) == 1:
            return x
        return _all_reduce(x.clone(), group)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _GatherToSP(torch.autograd.Function):
    """Sequence-parallel gather along ``dim``; backward reduce-scatters."""

    @staticmethod
    def forward(ctx, x, group, dim):
        ctx.group = group
        ctx.dim = dim
        if world_size(group) == 1:
            return x
        return _all_gather(x, group, dim)

    @staticmethod
    def backward(ctx, grad):
        if world_size(ctx.group) == 1:
            return grad, None, None
        return _reduce_scatter(grad, ctx.group, ctx.dim), None, None


class _ScatterToSP(torch.autograd.Function):
    """Sequence-parallel reduce-scatter along ``dim``; backward all-gathers."""

    @staticmethod
    def forward(ctx, x, group, dim):
        ctx.group = group
        ctx.dim = dim
        if world_size(group) == 1:
            return x
        return _reduce_scatter(x, group, dim)

    @staticmethod
    def backward(ctx, grad):
        if world_size(ctx.group) == 1:
            return grad, None, None
        return _all_gather(grad, ctx.group, ctx.dim), None, None


def copy_to_tp_region(x: torch.Tensor, group) -> torch.Tensor:
    return _CopyToTPRegion.apply(x, group)


def reduce_from_tp_region(x: torch.Tensor, group) -> torch.Tensor:
    return _ReduceFromTPRegion.apply(x, group)


def gather_to_sp(x: torch.Tensor, group, dim: int = 1) -> torch.Tensor:
    return _GatherToSP.apply(x, group, dim)


def scatter_to_sp(x: torch.Tensor, group, dim: int = 1) -> torch.Tensor:
    return _ScatterToSP.apply(x, group, dim)


def gather_last_dim(x: torch.Tensor, group) -> torch.Tensor:
    """Plain (non-autograd-special) all-gather of a column-sharded output."""
    if world_size(group) == 1:
        return x
    return _all_gather(x, group, dim=-1)


__all__ = [
    "world_size",
    "copy_to_tp_region",
    "reduce_from_tp_region",
    "gather_to_sp",
    "scatter_to_sp",
    "gather_last_dim",
]
