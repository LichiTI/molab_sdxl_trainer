# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test for the Lulynx tensor-parallel subsystem (v1). CPU is fine.

Run with the flashattention env:
    backend/env/python-flashattention/python.exe \
        backend/core/lulynx_trainer/parallel/parallel_tp_smoke.py

At ``tp_size == 1`` (no distributed init) every collective is an identity, so the
parallel layers must be bit-identical to ``nn.Linear`` in both forward and
backward.  Checks: (1) ColumnParallelLinear parity; (2) RowParallelLinear parity;
(3) Column→GELU→Row composition == dense MLP; (4) apply_tensor_parallel swaps the
matched Linears and preserves output; (5) shard merge is lossless.  Emits the
subsystem scorecard.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(__file__)
_BACKEND = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.lulynx_trainer.parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelSpec,
    apply_tensor_parallel,
    merge_column_shards,
    merge_row_shards,
    init_parallel,
)
from core.lulynx_trainer.multi_gpu_parallel_scorecard import build_multi_gpu_parallel_scorecard

torch.manual_seed(0)


def _grads_equal(a: nn.Module, b: nn.Module) -> bool:
    return all(
        torch.allclose(pa.grad, pb.grad, atol=1e-6)
        for pa, pb in zip(a.parameters(), b.parameters())
        if pa.grad is not None and pb.grad is not None
    )


def check_column() -> bool:
    print("== ColumnParallelLinear parity (tp=1) ==")
    lin = nn.Linear(64, 128)
    col = ColumnParallelLinear.from_linear(lin, groups=None, gather_output=True)
    x1 = torch.randn(8, 64, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1, y2 = lin(x1), col(x2)
    fwd = torch.equal(y1, y2)
    y1.sum().backward(); y2.sum().backward()
    bwd = torch.allclose(x1.grad, x2.grad, atol=1e-6) and torch.allclose(lin.weight.grad, col.weight.grad, atol=1e-6)
    ok = fwd and bwd
    print(f"  forward_equal={fwd}  backward_equal={bwd}  {'OK' if ok else 'FAIL'}")
    return ok


def check_row() -> bool:
    print("== RowParallelLinear parity (tp=1) ==")
    lin = nn.Linear(128, 64)
    row = RowParallelLinear.from_linear(lin, groups=None, input_is_parallel=True)
    x1 = torch.randn(8, 128, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1, y2 = lin(x1), row(x2)
    fwd = torch.allclose(y1, y2, atol=1e-6)
    y1.sum().backward(); y2.sum().backward()
    bwd = torch.allclose(x1.grad, x2.grad, atol=1e-6) and torch.allclose(lin.weight.grad, row.weight.grad, atol=1e-6)
    ok = fwd and bwd
    print(f"  forward_equal={fwd}  backward_equal={bwd}  {'OK' if ok else 'FAIL'}")
    return ok


def check_mlp_composition() -> bool:
    print("== Column→GELU→Row == dense MLP (tp=1) ==")
    lin1 = nn.Linear(64, 256)
    lin2 = nn.Linear(256, 64)
    col = ColumnParallelLinear.from_linear(lin1, groups=None, gather_output=False)
    row = RowParallelLinear.from_linear(lin2, groups=None, input_is_parallel=True)
    x = torch.randn(8, 64)
    dense = lin2(F.gelu(lin1(x)))
    parallel = row(F.gelu(col(x)))
    ok = torch.allclose(dense, parallel, atol=1e-6)
    print(f"  composition_equal={ok}  {'OK' if ok else 'FAIL'}")
    return ok


def check_apply_pass() -> bool:
    print("== apply_tensor_parallel swaps matched Linears ==")

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(32, 128)
            self.layer2 = nn.Linear(128, 32)
            self.norm = nn.LayerNorm(32)

        def forward(self, x):
            return self.norm(self.layer2(F.gelu(self.layer1(x))))

    model = Block()
    x = torch.randn(4, 32)
    before = model(x)
    spec = ParallelSpec({"layer1": "column", "layer2": "row"})
    groups = init_parallel(tp_degree=1, backend="cuda_direct")  # degenerate, also exercises backend hook
    n = apply_tensor_parallel(model, spec, groups=groups, gather_column_output=False)
    after = model(x)
    ok = (n == 2) and isinstance(model.layer1, ColumnParallelLinear) and isinstance(model.layer2, RowParallelLinear) and torch.allclose(before, after, atol=1e-6)
    print(f"  swapped={n} types_ok={isinstance(model.layer1, ColumnParallelLinear)}/{isinstance(model.layer2, RowParallelLinear)} output_preserved={torch.allclose(before, after, atol=1e-6)}  {'OK' if ok else 'FAIL'}")
    return ok


def check_merge() -> bool:
    print("== shard merge is lossless ==")
    w = torch.randn(128, 64)
    col_ok = torch.equal(merge_column_shards(list(w.chunk(4, dim=0))), w)
    row_ok = torch.equal(merge_row_shards(list(w.chunk(4, dim=1))), w)
    ok = col_ok and row_ok
    print(f"  column_merge={col_ok}  row_merge={row_ok}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    groups = init_parallel(tp_degree=1, backend="cuda_direct")
    print(f"tp_size={groups.tp_size} degenerate={groups.is_degenerate} backend={groups.backend}")
    col_ok = check_column()
    row_ok = check_row()
    mlp_ok = check_mlp_composition()
    apply_ok = check_apply_pass()
    merge_ok = check_merge()

    scorecard = build_multi_gpu_parallel_scorecard(
        column_parity=col_ok,
        row_parity=row_ok,
        mlp_composition_parity=mlp_ok,
        apply_pass_verified=apply_ok,
        merge_lossless=merge_ok,
        world_size_tested=groups.tp_size,
        backend=groups.backend,
    )
    print("\n== scorecard ==")
    for k, v in scorecard.items():
        print(f"  {k}: {v}")

    all_ok = col_ok and row_ok and mlp_ok and apply_ok and merge_ok
    print("\nRESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT", f"| scorecard.ok={scorecard['ok']}")
    sys.exit(0 if all_ok else 1)
