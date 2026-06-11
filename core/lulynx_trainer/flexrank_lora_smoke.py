# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for FlexRank LoRA (dynamic rank)."""

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

_fr = _import_module(
    "flexrank_lora",
    os.path.join(_HERE, "flexrank_lora.py"),
)
FlexRankLoRALinear = _fr.FlexRankLoRALinear

import torch
import torch.nn as nn


def test_forward_training_mode():

    original = nn.Linear(64, 64)
    flex = FlexRankLoRALinear(original, max_rank=8, min_rank=2, alpha=1.0)
    flex.train()
    x = torch.randn(2, 64)
    out = flex(x)
    assert out.shape == (2, 64), f"Shape mismatch: {out.shape}"
    print("PASS: forward in training mode")


def test_forward_eval_mode():

    original = nn.Linear(64, 64)
    flex = FlexRankLoRALinear(original, max_rank=8, min_rank=2, alpha=1.0)
    flex.eval()
    x = torch.randn(2, 64)
    out = flex(x)
    assert out.shape == (2, 64), f"Shape mismatch: {out.shape}"
    print("PASS: forward in eval mode")


def test_dynamic_rank_sampling():

    original = nn.Linear(32, 32)
    flex = FlexRankLoRALinear(original, max_rank=8, min_rank=1, alpha=1.0)
    nn.init.normal_(flex.lora_up.weight, std=0.1)
    flex.train()
    x = torch.randn(1, 32)
    outputs = set()
    for _ in range(50):
        out = flex(x)
        outputs.add(round(out.sum().item(), 4))
    assert len(outputs) > 1, "Dynamic rank should produce varying outputs in training"
    print("PASS: dynamic rank sampling produces varying outputs")


def test_eval_deterministic():

    original = nn.Linear(32, 32)
    flex = FlexRankLoRALinear(original, max_rank=8, min_rank=1, alpha=1.0)
    flex.eval()
    x = torch.randn(1, 32)
    out1 = flex(x)
    out2 = flex(x)
    assert torch.allclose(out1, out2), "Eval mode should be deterministic"
    print("PASS: eval mode is deterministic")


def test_lora_leaf_markers():

    original = nn.Linear(16, 16)
    flex = FlexRankLoRALinear(original, max_rank=4)
    assert hasattr(flex.lora_down, "_lora_leaf"), "lora_down should have _lora_leaf"
    assert hasattr(flex.lora_up, "_lora_leaf"), "lora_up should have _lora_leaf"
    print("PASS: lora_leaf markers set")


def test_original_frozen():

    original = nn.Linear(16, 16)
    flex = FlexRankLoRALinear(original, max_rank=4)
    for p in flex.original.parameters():
        assert not p.requires_grad, "Original layer should be frozen"
    print("PASS: original layer frozen")


def test_gradient_flow():

    original = nn.Linear(32, 32)
    flex = FlexRankLoRALinear(original, max_rank=8, min_rank=1, alpha=1.0)
    flex.train()
    x = torch.randn(2, 32, requires_grad=True)
    out = flex(x)
    loss = out.sum()
    loss.backward()
    assert flex.lora_down.weight.grad is not None, "lora_down should have gradient"
    assert flex.lora_up.weight.grad is not None, "lora_up should have gradient"
    print("PASS: gradient flows through lora_down and lora_up")


if __name__ == "__main__":
    test_forward_training_mode()
    test_forward_eval_mode()
    test_dynamic_rank_sampling()
    test_eval_deterministic()
    test_lora_leaf_markers()
    test_original_frozen()
    test_gradient_flow()
    print("\nAll FlexRank LoRA smoke tests passed!")
