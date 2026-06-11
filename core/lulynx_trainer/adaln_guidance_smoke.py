# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for adaln_guidance.py (Phase 8.6 / #114)."""

from __future__ import annotations

import os
import sys
import importlib.util

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.adaln_guidance",
    os.path.join(_HERE, "adaln_guidance.py"),
)
_ag = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.adaln_guidance"] = _ag
_spec.loader.exec_module(_ag)


class _FakeDiTBlock(nn.Module):
    """Minimal block exposing an `adaLN_modulation` linear that produces 6*dim outputs."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.adaLN_modulation = nn.Linear(dim, dim * 6)

    def forward(self, x):
        # Project to modulation space, then sum the 6 chunks back to dim
        # so blocks can chain together in the test
        mod = self.adaLN_modulation(x)
        return sum(mod.chunk(6, dim=-1)) / 6.0


class _FakeDiT(nn.Module):
    def __init__(self, dim: int = 64, num_blocks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeDiTBlock(dim) for _ in range(num_blocks)])

    def forward(self, x):
        out = x
        for blk in self.blocks:
            out = blk(out)
        return out


def test_init_scale_zero_is_no_op():
    bias = _ag.AdaLNGuidanceBias(modulation_dim=64, init_scale=0.0)
    mod = torch.randn(2, 64 * 6)
    out = bias(mod)
    assert torch.allclose(out, mod)
    print("PASS: AdaLNGuidanceBias is a no-op at init_scale=0.0")


def test_bias_shifts_msa_and_mlp_only():
    bias = _ag.AdaLNGuidanceBias(modulation_dim=8, init_scale=1.0)
    mod = torch.zeros(1, 8 * 6)
    out = bias(mod)
    chunks = out.chunk(6, dim=-1)
    # shift_msa, scale_msa, shift_mlp, scale_mlp shifted by 1.0
    assert torch.allclose(chunks[0], torch.ones_like(chunks[0]))
    assert torch.allclose(chunks[1], torch.ones_like(chunks[1]))
    assert torch.allclose(chunks[3], torch.ones_like(chunks[3]))
    assert torch.allclose(chunks[4], torch.ones_like(chunks[4]))
    # gate_msa, gate_mlp untouched
    assert torch.allclose(chunks[2], torch.zeros_like(chunks[2]))
    assert torch.allclose(chunks[5], torch.zeros_like(chunks[5]))
    print("PASS: bias adds to shift_*/scale_* only, gates untouched")


def test_unexpected_shape_passes_through():
    bias = _ag.AdaLNGuidanceBias(modulation_dim=8, init_scale=5.0)
    mod = torch.randn(2, 16)  # not 6*8
    out = bias(mod)
    assert torch.allclose(out, mod)
    print("PASS: unexpected modulation shape passes through unchanged")


def test_install_finds_modulation_modules():
    model = _FakeDiT(dim=32, num_blocks=4)
    installed = _ag.install_adaln_guidance(model, init_scale=0.0)
    assert len(installed) == 4
    print("PASS: install_adaln_guidance attaches to all 4 blocks")


def test_install_is_idempotent():
    model = _FakeDiT(dim=32, num_blocks=2)
    first = _ag.install_adaln_guidance(model)
    second = _ag.install_adaln_guidance(model)
    # Second call should not double-install
    assert len(first) == len(second) == 2
    print("PASS: install_adaln_guidance is idempotent")


def test_install_preserves_forward_at_zero_init():
    torch.manual_seed(0)
    model = _FakeDiT(dim=32, num_blocks=3)
    x = torch.randn(2, 32)
    baseline = model(x).clone()

    _ag.install_adaln_guidance(model, init_scale=0.0)
    after = model(x)
    assert torch.allclose(baseline, after, atol=1e-6)
    print("PASS: zero-init guidance preserves model output")


def test_remove_restores_forward():
    model = _FakeDiT(dim=32, num_blocks=2)
    _ag.install_adaln_guidance(model, init_scale=2.0)
    removed = _ag.remove_adaln_guidance(model)
    assert removed == 2
    # Confirm subsequent forward runs without the guidance hook
    x = torch.randn(1, 32)
    out = model(x)
    assert out.shape == (1, 32)
    print("PASS: remove_adaln_guidance restores original forward")


def test_get_params_collects_biases():
    model = _FakeDiT(dim=32, num_blocks=3)
    _ag.install_adaln_guidance(model)
    params = _ag.get_adaln_guidance_params(model)
    # 4 bias tensors per block × 3 blocks = 12
    assert len(params) == 12
    assert all(p.requires_grad for p in params)
    print("PASS: get_adaln_guidance_params returns 4 × num_blocks tensors")


def test_guidance_is_differentiable():
    model = _FakeDiT(dim=16, num_blocks=2)
    _ag.install_adaln_guidance(model, init_scale=0.1)
    x = torch.randn(2, 16, requires_grad=True)
    out = model(x).sum()
    out.backward()
    params = _ag.get_adaln_guidance_params(model)
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in params)
    print("PASS: guidance bias receives gradients")


if __name__ == "__main__":
    test_init_scale_zero_is_no_op()
    test_bias_shifts_msa_and_mlp_only()
    test_unexpected_shape_passes_through()
    test_install_finds_modulation_modules()
    test_install_is_idempotent()
    test_install_preserves_forward_at_zero_init()
    test_remove_restores_forward()
    test_get_params_collects_biases()
    test_guidance_is_differentiable()
    print("\nAll adaln_guidance smoke tests passed!")
