# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for pattern loss (frequency-band loss)."""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_pl = _import_module(
    "pattern_loss",
    os.path.join(_HERE, "pattern_loss.py"),
)
pattern_loss = _pl.pattern_loss

import torch


def test_identical_zero_loss():
    pred = torch.ones(2, 4, 32, 32)
    target = torch.ones(2, 4, 32, 32)
    loss = pattern_loss(pred, target)
    assert loss.item() < 1e-5, f"Expected near-zero loss for identical tensors, got {loss.item()}"
    print("PASS: identical tensors produce near-zero loss")


def test_weight_scaling():
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 32, 32)
    target = torch.randn(2, 4, 32, 32)
    loss_high = pattern_loss(pred, target, high_weight=2.0)
    loss_low = pattern_loss(pred, target, high_weight=0.5)
    assert loss_high.item() > loss_low.item(), (
        f"Expected high_weight=2.0 loss ({loss_high.item()}) > high_weight=0.5 loss ({loss_low.item()})"
    )
    print("PASS: higher high_weight produces larger loss when pred != target")


def test_odd_spatial_dims():
    torch.manual_seed(1)
    pred = torch.randn(1, 1, 31, 33)
    target = torch.randn(1, 1, 31, 33)
    loss = pattern_loss(pred, target)
    assert torch.isfinite(loss), f"Expected finite loss for odd spatial dims, got {loss.item()}"
    print("PASS: odd spatial dimensions do not crash")


def test_multi_level():
    torch.manual_seed(2)
    pred = torch.randn(1, 4, 64, 64)
    target = torch.randn(1, 4, 64, 64)
    loss = pattern_loss(pred, target, levels=2)
    assert torch.isfinite(loss), f"Expected finite loss for levels=2, got {loss.item()}"
    print("PASS: levels=2 works without error on [1,4,64,64] inputs")


def test_different_band_types():
    torch.manual_seed(3)
    pred = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 3, 32, 32)
    loss = pattern_loss(pred, target, ll_type="l1", high_type="huber")
    assert torch.isfinite(loss), f"Expected finite loss for ll_type='l1', high_type='huber', got {loss.item()}"
    print("PASS: ll_type='l1', high_type='huber' produce finite loss")


if __name__ == "__main__":
    test_identical_zero_loss()
    test_weight_scaling()
    test_odd_spatial_dims()
    test_multi_level()
    test_different_band_types()
    print("\nAll pattern loss smoke tests passed!")
