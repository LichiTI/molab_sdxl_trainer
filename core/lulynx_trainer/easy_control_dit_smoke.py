# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for easy_control_dit.py (Phase 8.9 / #117)."""

from __future__ import annotations

import os
import sys
import importlib.util

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.easy_control_dit",
    os.path.join(_HERE, "easy_control_dit.py"),
)
_ec = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.easy_control_dit"] = _ec
_spec.loader.exec_module(_ec)


def test_zero_init_output_makes_residual_zero():
    cfg = _ec.EasyControlConfig(in_channels=3, latent_channels=16, init_zero_out=True)
    ctrl = _ec.EasyControl(cfg)
    x = torch.randn(2, 3, 64, 64)
    residual = ctrl(x)
    assert torch.allclose(residual, torch.zeros_like(residual))
    print("PASS: zero-init makes residual exactly zero")


def test_nonzero_init_produces_nonzero_residual():
    cfg = _ec.EasyControlConfig(in_channels=3, latent_channels=16, init_zero_out=False)
    ctrl = _ec.EasyControl(cfg)
    x = torch.randn(2, 3, 64, 64)
    residual = ctrl(x)
    assert residual.abs().sum().item() > 0
    print("PASS: non-zero init produces non-zero residual")


def test_downsample_factor_8_maps_64_to_8():
    cfg = _ec.EasyControlConfig(downsample_factor=8, init_zero_out=False)
    ctrl = _ec.EasyControl(cfg)
    x = torch.randn(1, 3, 64, 64)
    residual = ctrl(x)
    assert residual.shape == (1, 16, 8, 8)
    print("PASS: downsample_factor=8 maps 64x64 -> 8x8")


def test_target_size_resizes_residual():
    cfg = _ec.EasyControlConfig(downsample_factor=8, init_zero_out=False)
    ctrl = _ec.EasyControl(cfg)
    x = torch.randn(1, 3, 64, 64)
    residual = ctrl(x, target_size=(16, 16))
    assert residual.shape == (1, 16, 16, 16)
    print("PASS: target_size resizes residual to requested grid")


def test_3d_input_is_unsqueezed():
    cfg = _ec.EasyControlConfig(in_channels=3, latent_channels=16, init_zero_out=False)
    ctrl = _ec.EasyControl(cfg)
    x = torch.randn(3, 32, 32)  # no batch dim
    residual = ctrl(x)
    assert residual.shape[0] == 1
    print("PASS: 3D input is auto-unsqueezed to add batch dim")


def test_scale_zero_yields_zero_residual():
    cfg = _ec.EasyControlConfig(in_channels=1, latent_channels=4, init_zero_out=False, scale=0.0)
    ctrl = _ec.EasyControl(cfg)
    residual = ctrl(torch.randn(1, 1, 16, 16))
    assert torch.allclose(residual, torch.zeros_like(residual))
    print("PASS: scale=0 produces zero residual")


def test_get_trainable_params_returns_encoder_params():
    cfg = _ec.EasyControlConfig()
    ctrl = _ec.EasyControl(cfg)
    params = ctrl.get_trainable_params()
    assert len(params) > 0
    print(f"PASS: get_trainable_params returns {len(params)} encoder tensors")


def test_encoder_is_differentiable():
    cfg = _ec.EasyControlConfig(in_channels=3, latent_channels=8, init_zero_out=False)
    ctrl = _ec.EasyControl(cfg)
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    out = ctrl(x).sum()
    out.backward()
    grads = [p.grad for p in ctrl.get_trainable_params() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in grads)
    print("PASS: encoder receives gradients during backward")


def test_different_in_channels():
    """Depth maps (1ch) and pose skeletons (3ch) should both work."""
    for in_ch in (1, 3, 4):
        cfg = _ec.EasyControlConfig(in_channels=in_ch, latent_channels=16, init_zero_out=False)
        ctrl = _ec.EasyControl(cfg)
        x = torch.randn(1, in_ch, 32, 32)
        residual = ctrl(x)
        assert residual.shape[1] == 16
    print("PASS: encoder accepts varied in_channels (1, 3, 4)")


if __name__ == "__main__":
    test_zero_init_output_makes_residual_zero()
    test_nonzero_init_produces_nonzero_residual()
    test_downsample_factor_8_maps_64_to_8()
    test_target_size_resizes_residual()
    test_3d_input_is_unsqueezed()
    test_scale_zero_yields_zero_residual()
    test_get_trainable_params_returns_encoder_params()
    test_encoder_is_differentiable()
    test_different_in_channels()
    print("\nAll EasyControl-DiT smoke tests passed!")
