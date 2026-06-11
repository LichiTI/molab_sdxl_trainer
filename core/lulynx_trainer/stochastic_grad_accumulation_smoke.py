# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for stochastic gradient accumulation."""

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

_stochastic_rounding = _import_module(
    "stochastic_rounding",
    os.path.join(_HERE, "stochastic_rounding.py"),
)
stochastic_round_ = _stochastic_rounding.stochastic_round_

import torch
import torch.nn as nn


def test_stochastic_round_on_bf16_grad():
    # Simulate a bf16 gradient
    grad = torch.randn(16, 16, dtype=torch.bfloat16)
    rounded = stochastic_round_(grad.clone())
    assert rounded.dtype == torch.bfloat16, "Rounded grad should stay bf16"
    assert rounded.shape == grad.shape, "Shape should be preserved"
    assert torch.isfinite(rounded).all(), "Rounded values should be finite"
    print("PASS: stochastic_round_ works on bf16 gradients")


def test_fp32_passthrough():
    grad = torch.randn(8, 8, dtype=torch.float32)
    original = grad.clone()
    stochastic_round_(grad)
    assert torch.equal(grad, original), "fp32 gradients should be unchanged"
    print("PASS: fp32 gradients pass through unchanged")


def test_accumulation_scenario():
    """Simulate gradient accumulation with stochastic rounding."""
    model = nn.Linear(8, 8)
    model = model.to(torch.bfloat16)
    accumulated = torch.zeros_like(model.weight.data)
    # Simulate 4 micro-batches
    for _ in range(4):
        x = torch.randn(2, 8, dtype=torch.bfloat16)
        loss = model(x).sum()
        loss.backward()
        if model.weight.grad is not None:
            stochastic_round_(model.weight.grad)
            accumulated += model.weight.grad
        model.zero_grad()
    assert torch.isfinite(accumulated).all(), "Accumulated gradients should be finite"
    print("PASS: accumulation scenario with stochastic rounding works")


def test_config_field():
    cfg_path = os.path.join(_HERE, "..", "configs.py")
    with open(cfg_path, encoding="utf-8") as f:
        src = f.read()
    assert "stochastic_grad_accumulation" in src, "Missing stochastic_grad_accumulation in configs.py"
    print("PASS: config field exists")


if __name__ == "__main__":
    test_stochastic_round_on_bf16_grad()
    test_fp32_passthrough()
    test_accumulation_scenario()
    test_config_field()
    print("\nAll stochastic gradient accumulation smoke tests passed!")
