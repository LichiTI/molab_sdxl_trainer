# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for adaptive loss weighting (learnable SNR gamma)."""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_alw = _import_module(
    "adaptive_loss_weighting",
    os.path.join(_HERE, "adaptive_loss_weighting.py"),
)
AdaptiveLossWeighter = _alw.AdaptiveLossWeighter

import torch


def test_forward_produces_valid_weights():

    weighter = AdaptiveLossWeighter(init_gamma=5.0)
    snr = torch.tensor([0.1, 1.0, 5.0, 10.0, 100.0])
    weights = weighter(snr)
    assert weights.shape == snr.shape, f"Shape mismatch: {weights.shape}"
    assert (weights > 0).all(), "Weights should be positive"
    assert (weights <= 10.0).all(), "Weights should be clamped to 10.0"
    print("PASS: forward produces valid weights")


def test_gradients_flow():

    weighter = AdaptiveLossWeighter(init_gamma=5.0)
    snr = torch.tensor([1.0, 5.0, 10.0])
    weights = weighter(snr)
    loss = weights.sum()
    loss.backward()
    assert weighter.log_gamma.grad is not None, "log_gamma should have gradient"
    assert weighter.offset.grad is not None, "offset should have gradient"
    assert weighter.log_scale.grad is not None, "log_scale should have gradient"
    print("PASS: gradients flow through all parameters")


def test_v_parameterization():

    weighter = AdaptiveLossWeighter(init_gamma=5.0)
    snr = torch.tensor([1.0, 5.0, 10.0])
    w_eps = weighter(snr, v_parameterization=False)
    w_v = weighter(snr, v_parameterization=True)
    # v_parameterization uses (snr + 1) divisor, so weights differ
    assert not torch.allclose(w_eps, w_v), "v_param weights should differ from epsilon weights"
    print("PASS: v_parameterization produces different weights")


def test_init_gamma():

    import math
    w5 = AdaptiveLossWeighter(init_gamma=5.0)
    w10 = AdaptiveLossWeighter(init_gamma=10.0)
    assert abs(w5.log_gamma.item() - math.log(5.0)) < 1e-5, "init_gamma=5 should set log_gamma=ln(5)"
    assert abs(w10.log_gamma.item() - math.log(10.0)) < 1e-5, "init_gamma=10 should set log_gamma=ln(10)"
    print("PASS: init_gamma sets correct log_gamma")


def test_config_field():
    cfg_path = os.path.join(_HERE, "..", "configs.py")
    with open(cfg_path, encoding="utf-8") as f:
        src = f.read()
    assert "adaptive_loss_weighting_enabled" in src
    assert "adaptive_loss_weighting_lr" in src
    assert "adaptive_loss_weighting_init_gamma" in src
    print("PASS: config fields exist")


if __name__ == "__main__":
    test_forward_produces_valid_weights()
    test_gradients_flow()
    test_v_parameterization()
    test_init_gamma()
    test_config_field()
    print("\nAll adaptive loss weighting smoke tests passed!")
