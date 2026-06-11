"""Smoke test for blockwise fused optimizer: parameter grouping by block with separate lr scheduler steps."""
from __future__ import annotations

import os
import sys
import importlib.util

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.blockwise_fused",
    os.path.join(_HERE, "blockwise_fused.py"),
)
_bf = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.blockwise_fused"] = _bf
_spec.loader.exec_module(_bf)
BlockwiseFusedOptimizer = _bf.BlockwiseFusedOptimizer


class _BlockModel(nn.Module):
    """Simple model with 3 named blocks as nn.ModuleList."""
    def __init__(self, dim: int = 16, num_blocks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x) + x  # residual
        return x


def test_param_groups_split_by_block():
    """Parameters are correctly grouped by block, with one group per block."""
    model = _BlockModel(dim=16, num_blocks=3)
    fused_opt = BlockwiseFusedOptimizer(model, base_lr=1e-3)

    # Should have 3 groups (one per block)
    block_groups = [g for g in fused_opt.param_groups if g["name"].startswith("block_")]
    assert len(block_groups) == 3, f"Expected 3 block groups, got {len(block_groups)}"

    # Group names should be block_0, block_1, block_2
    names = sorted(g["name"] for g in block_groups)
    assert names == ["block_0", "block_1", "block_2"], f"Expected block_0/1/2, got {names}"


def test_each_group_has_correct_params():
    """Each block group contains only its block's parameters."""
    model = _BlockModel(dim=16, num_blocks=3)
    fused_opt = BlockwiseFusedOptimizer(model, base_lr=1e-3)

    for group in fused_opt.param_groups:
        if not group["name"].startswith("block_"):
            continue
        block_idx = int(group["name"].split("_")[1])
        block = model.blocks[block_idx]
        block_param_ids = {id(p) for p in block.parameters()}
        group_param_ids = {id(p) for p in group["params"]}
        assert group_param_ids == block_param_ids, (
            f"Group {group['name']} params don't match block {block_idx}"
        )


def test_per_group_scheduler_step():
    """Each group gets a separate LR scheduler step that only affects that group."""
    model = _BlockModel(dim=16, num_blocks=3)
    fused_opt = BlockwiseFusedOptimizer(model, base_lr=1e-3, lr_decay=0.5)

    initial_lrs = {g["name"]: g["lr"] for g in fused_opt.optimizer.param_groups}

    # Step only block_1's LR
    ok = fused_opt.scheduler_step("block_1")
    assert ok, "scheduler_step should return True for existing group"

    lrs_after = {g["name"]: g["lr"] for g in fused_opt.optimizer.param_groups}

    assert lrs_after["block_1"] == initial_lrs["block_1"] * 0.5, (
        f"block_1 LR should have decayed: {initial_lrs['block_1']} -> {lrs_after['block_1']}"
    )
    assert lrs_after["block_0"] == initial_lrs["block_0"], (
        "block_0 LR should be unchanged"
    )
    assert lrs_after["block_2"] == initial_lrs["block_2"], (
        "block_2 LR should be unchanged"
    )


def test_all_groups_same_initial_lr():
    """All parameter groups start with the same base learning rate."""
    model = _BlockModel(dim=16, num_blocks=3)
    fused_opt = BlockwiseFusedOptimizer(model, base_lr=5e-4)

    for group in fused_opt.param_groups:
        assert group["lr"] == 5e-4, f"Group {group['name']} has lr={group['lr']}, expected 5e-4"


def test_fused_optimizer_step_runs():
    """A full optimizer step (forward + backward + optimizer step) runs without error."""
    model = _BlockModel(dim=16, num_blocks=3)
    fused_opt = BlockwiseFusedOptimizer(model, base_lr=1e-3)

    x = torch.randn(2, 16)
    out = model(x)
    loss = out.sum()
    loss.backward()
    fused_opt.step()
    fused_opt.zero_grad()

    # Should not raise
    assert True


if __name__ == "__main__":
    print("Blockwise Fused Optimizer Smoke Tests")
    print("=" * 40)
    test_param_groups_split_by_block()
    print("PASS: param_groups_split_by_block")
    test_each_group_has_correct_params()
    print("PASS: each_group_has_correct_params")
    test_per_group_scheduler_step()
    print("PASS: per_group_scheduler_step")
    test_all_groups_same_initial_lr()
    print("PASS: all_groups_same_initial_lr")
    test_fused_optimizer_step_runs()
    print("PASS: fused_optimizer_step_runs")
    print("=" * 40)
    print("All blockwise fused optimizer smoke tests passed!")
