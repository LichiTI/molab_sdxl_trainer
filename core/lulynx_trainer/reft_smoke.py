# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for reft.py (Phase 8.4 / #112)."""

from __future__ import annotations

import os
import sys
import importlib.util

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.reft",
    os.path.join(_HERE, "reft.py"),
)
_reft = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.reft"] = _reft
_spec.loader.exec_module(_reft)


class _ToyBlock(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class _ToyModel(nn.Module):
    def __init__(self, dim: int = 32, num_blocks: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([_ToyBlock(dim) for _ in range(num_blocks)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def test_intervention_zero_init_is_identity():
    layer = _reft.ReFTIntervention(hidden_size=16, rank=4, init_scale=0.0)
    h = torch.randn(2, 4, 16)
    out = layer(h)
    assert torch.allclose(out, h, atol=1e-6)
    print("PASS: zero-init ReFTIntervention is identity")


def test_intervention_nonzero_init_changes_output():
    torch.manual_seed(0)
    layer = _reft.ReFTIntervention(hidden_size=16, rank=4, init_scale=0.5)
    h = torch.randn(2, 4, 16)
    out = layer(h)
    diff = (out - h).abs().max().item()
    assert diff > 0.0
    print(f"PASS: nonzero-init ReFTIntervention shifts output (max diff={diff:.4f})")


def test_install_reft_attaches_hook():
    model = _ToyModel(dim=32, num_blocks=4)
    targets = ["blocks.0", "blocks.2"]
    interventions = _reft.install_reft(model, targets, rank=4, init_scale=0.0)
    assert len(interventions) == 2
    # Confirm hooks attached
    for path in targets:
        mod = _reft._resolve_module(model, path)
        assert hasattr(mod, "_reft_intervention")
        assert hasattr(mod, "_reft_hook_handle")
    print("PASS: install_reft attaches hooks at requested targets")


def test_zero_init_preserves_forward_output():
    torch.manual_seed(0)
    model = _ToyModel(dim=32, num_blocks=3)
    x = torch.randn(2, 32)
    baseline = model(x).clone()

    _reft.install_reft(model, ["blocks.0", "blocks.1", "blocks.2"], rank=4, init_scale=0.0)
    after = model(x)
    assert torch.allclose(baseline, after, atol=1e-6)
    print("PASS: zero-init ReFT preserves forward output")


def test_nonzero_init_changes_forward_output():
    torch.manual_seed(0)
    model = _ToyModel(dim=32, num_blocks=3)
    x = torch.randn(2, 32)
    baseline = model(x).clone()

    _reft.install_reft(model, ["blocks.1"], rank=4, init_scale=0.5)
    after = model(x)
    assert not torch.allclose(baseline, after, atol=1e-6)
    print("PASS: nonzero-init ReFT changes forward output")


def test_remove_reft_restores_original_forward():
    torch.manual_seed(0)
    model = _ToyModel(dim=32, num_blocks=3)
    x = torch.randn(2, 32)
    baseline = model(x).clone()

    _reft.install_reft(model, ["blocks.0", "blocks.1"], rank=4, init_scale=0.5)
    after_install = model(x)
    assert not torch.allclose(baseline, after_install)

    removed = _reft.remove_reft(model)
    assert removed == 2
    after_remove = model(x)
    assert torch.allclose(baseline, after_remove, atol=1e-6)
    print("PASS: remove_reft restores original behaviour")


def test_install_is_idempotent():
    model = _ToyModel(dim=32, num_blocks=2)
    first = _reft.install_reft(model, ["blocks.0"], rank=4)
    second = _reft.install_reft(model, ["blocks.0"], rank=4)
    assert len(first) == len(second) == 1
    print("PASS: install_reft is idempotent on the same target")


def test_get_reft_params_collects_trainables():
    model = _ToyModel(dim=32, num_blocks=2)
    _reft.install_reft(model, ["blocks.0", "blocks.1"], rank=4)
    params = _reft.get_reft_params(model)
    assert len(params) > 0
    assert all(p.requires_grad for p in params)
    # 2 blocks × (W1.weight + W1.bias + W2.weight) = 6 tensors
    assert len(params) == 6
    print("PASS: get_reft_params returns 3 tensors per intervention")


def test_unknown_target_is_skipped():
    model = _ToyModel(dim=32)
    interventions = _reft.install_reft(model, ["does.not.exist"], rank=4)
    assert interventions == []
    print("PASS: unknown target paths are skipped without crash")


def test_intervention_is_differentiable():
    model = _ToyModel(dim=16, num_blocks=2)
    _reft.install_reft(model, ["blocks.0"], rank=4, init_scale=0.1)
    x = torch.randn(2, 16, requires_grad=True)
    loss = model(x).sum()
    loss.backward()
    params = _reft.get_reft_params(model)
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in params)
    print("PASS: ReFT intervention parameters receive gradients")


if __name__ == "__main__":
    test_intervention_zero_init_is_identity()
    test_intervention_nonzero_init_changes_output()
    test_install_reft_attaches_hook()
    test_zero_init_preserves_forward_output()
    test_nonzero_init_changes_forward_output()
    test_remove_reft_restores_original_forward()
    test_install_is_idempotent()
    test_get_reft_params_collects_trainables()
    test_unknown_target_is_skipped()
    test_intervention_is_differentiable()
    print("\nAll ReFT smoke tests passed!")
