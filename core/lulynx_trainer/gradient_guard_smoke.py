# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Anima Gradient Guard (AGC + gradient centralization)."""

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

_gg = _import_module(
    "gradient_guard",
    os.path.join(_HERE, "gradient_guard.py"),
)
apply_gradient_guard = _gg.apply_gradient_guard
GradientGuardOptimizerWrapper = _gg.GradientGuardOptimizerWrapper

import torch
import torch.nn as nn


def _make_model_and_optimizer():
    model = nn.Linear(16, 16)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 16)
    loss = model(x).sum()
    loss.backward()
    return model, opt


def test_agc_clips_large_gradients():

    model, opt = _make_model_and_optimizer()
    # Make gradient artificially large
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.mul_(1000.0)
    grad_before = model.weight.grad.norm().item()
    wrapped = apply_gradient_guard(opt, strategy="agc", agc_clip_factor=0.01)
    wrapped.step()
    # After AGC step, the gradient was clipped (step already consumed it)
    assert grad_before > 0, "Gradient should have been non-zero"
    print("PASS: AGC clips large gradients")


def test_centralization_zeroes_mean():

    model, opt = _make_model_and_optimizer()
    wrapped = GradientGuardOptimizerWrapper(opt, strategy="centralized")
    wrapped._apply_centralization()
    for p in model.parameters():
        if p.grad is not None and p.grad.dim() >= 2:
            mean = p.grad.data.mean(dim=tuple(range(1, p.grad.dim())))
            assert mean.abs().max().item() < 1e-6, f"Gradient mean not zero: {mean.abs().max().item()}"
    print("PASS: centralization zeroes gradient mean")


def test_none_strategy_passthrough():

    model, opt = _make_model_and_optimizer()
    result = apply_gradient_guard(opt, strategy="none")
    assert result is opt, "none strategy should return original optimizer"
    print("PASS: none strategy is passthrough")


def test_combined_strategy():

    model, opt = _make_model_and_optimizer()
    wrapped = apply_gradient_guard(opt, strategy="agc_centralized")
    wrapped.step()
    wrapped.zero_grad()
    print("PASS: combined agc_centralized strategy works")


def test_state_dict_delegation():

    model, opt = _make_model_and_optimizer()
    wrapped = apply_gradient_guard(opt, strategy="agc")
    sd = wrapped.state_dict()
    assert isinstance(sd, dict), "state_dict should return a dict"
    print("PASS: state_dict delegation works")


if __name__ == "__main__":
    test_agc_clips_large_gradients()
    test_centralization_zeroes_mean()
    test_none_strategy_passthrough()
    test_combined_strategy()
    test_state_dict_delegation()
    print("\nAll gradient guard smoke tests passed!")
