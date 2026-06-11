"""Smoke tests for per-block torch.compile.

Tests:
  1. RuntimeOptimizationPlan carries torch_compile_scope
  2. apply_per_block_compile compiles each block individually
  3. Scope="" or "full" does NOT trigger per-block compile
  4. Per-block compile with no blocks produces a warning
"""

from __future__ import annotations

import sys
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))

_rt = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.runtime_optimizations",
    os.path.join(_HERE, "runtime_optimizations.py"),
)
_rt_mod = importlib.util.module_from_spec(_rt)
sys.modules["core.lulynx_trainer.runtime_optimizations"] = _rt_mod
_rt.loader.exec_module(_rt_mod)

import torch
import torch.nn as nn


class _FakeBlock(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x) + x


class _FakeUNet(nn.Module):
    def __init__(self, n_down=2, n_up=2):
        super().__init__()
        self.down_blocks = nn.ModuleList([_FakeBlock() for _ in range(n_down)])
        self.mid_block = _FakeBlock()
        self.up_blocks = nn.ModuleList([_FakeBlock() for _ in range(n_up)])


class _FakeConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_plan_carries_scope():
    plan = _rt_mod.RuntimeOptimizationPlan(
        attention_backend="sdpa",
        requested_attention_backend="sdpa",
        torch_compile=True,
        torch_compile_scope="per_block",
    )
    assert plan.torch_compile_scope == "per_block"
    plan2 = _rt_mod.RuntimeOptimizationPlan(
        attention_backend="sdpa",
        requested_attention_backend="sdpa",
    )
    assert plan2.torch_compile_scope == ""
    print("PASS: RuntimeOptimizationPlan carries torch_compile_scope")


def test_per_block_compile():
    """Per-block compile should compile each block."""
    model = _FakeUNet()
    plan = _rt_mod.RuntimeOptimizationPlan(
        attention_backend="sdpa",
        requested_attention_backend="sdpa",
        torch_compile=True,
        torch_compile_scope="per_block",
        torch_compile_backend="inductor",
        torch_compile_mode="default",
    )

    _rt_mod.apply_per_block_compile(model, plan)

    # Check that compile was attempted for each block
    # Since torch.compile may not be available, we check the plan's reasons
    assert any("per_block_compile" in r for r in plan.reasons), \
        f"Expected per_block_compile reason, got: {plan.reasons}"
    print("PASS: per_block_compile compiled blocks")


def test_full_scope_does_not_per_block():
    """Scope "full" should not trigger per-block compile."""
    model = _FakeUNet()
    plan = _rt_mod.RuntimeOptimizationPlan(
        attention_backend="sdpa",
        requested_attention_backend="sdpa",
        torch_compile=True,
        torch_compile_scope="full",
    )

    _rt_mod.apply_per_block_compile(model, plan)
    assert not any("per_block_compile" in r for r in plan.reasons), \
        "per_block_compile should not trigger for scope='full'"
    print("PASS: scope='full' does not trigger per_block_compile")


def test_no_compile_flag_skips_per_block():
    """torch_compile=False should skip per-block compile."""
    model = _FakeUNet()
    plan = _rt_mod.RuntimeOptimizationPlan(
        attention_backend="sdpa",
        requested_attention_backend="sdpa",
        torch_compile=False,
        torch_compile_scope="per_block",
    )

    _rt_mod.apply_per_block_compile(model, plan)
    assert not any("per_block_compile" in r for r in plan.reasons)
    print("PASS: torch_compile=False skips per_block_compile")


def test_no_blocks_warning():
    """Model with no block collections should produce a warning."""
    model = nn.Linear(4, 4)  # No blocks
    plan = _rt_mod.RuntimeOptimizationPlan(
        attention_backend="sdpa",
        requested_attention_backend="sdpa",
        torch_compile=True,
        torch_compile_scope="per_block",
    )

    _rt_mod.apply_per_block_compile(model, plan)
    assert any("no block collections" in w for w in plan.warnings), \
        f"Expected no-blocks warning, got: {plan.warnings}"
    print("PASS: no-block model produces warning")


if __name__ == "__main__":
    test_plan_carries_scope()
    test_per_block_compile()
    test_full_scope_does_not_per_block()
    test_no_compile_flag_skips_per_block()
    test_no_blocks_warning()
    print("\nAll per-block compile smoke tests passed!")
