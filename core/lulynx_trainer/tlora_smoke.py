"""Smoke tests for T-LoRA (Temporal LoRA) rank scheduling and integration."""

import sys
import os
import math

# Direct import to avoid __init__.py chain pulling in diffusers/xformers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_tlora = _import_module(
    "tlora",
    os.path.join(os.path.dirname(__file__), "tlora.py"),
)
TLoRALinear = _tlora.TLoRALinear


import torch
import torch.nn as nn


def _make_linear(in_f=64, out_f=32):
    return nn.Linear(in_f, out_f)


def test_tlora_constant_schedule():

    base = _make_linear()
    tlora = TLoRALinear(base, max_rank=16, min_rank=4, schedule="constant", total_steps=100)

    assert tlora.current_rank == 4, f"Expected initial rank 4, got {tlora.current_rank}"

    tlora.set_global_step(50)
    assert tlora.current_rank == 4, f"Constant schedule should stay at min_rank, got {tlora.current_rank}"

    tlora.set_global_step(100)
    assert tlora.current_rank == 4, f"Constant schedule should stay at min_rank, got {tlora.current_rank}"
    print("  [PASS] tlora_constant_schedule")


def test_tlora_linear_schedule():
    """T-LoRA with linear schedule should ramp from min_rank to max_rank."""
    base = _make_linear()
    tlora = TLoRALinear(base, max_rank=16, min_rank=4, schedule="linear", total_steps=100)

    assert tlora.current_rank == 4, f"Expected initial rank 4, got {tlora.current_rank}"

    tlora.set_global_step(50)
    rank_mid = tlora.current_rank
    assert 4 <= rank_mid <= 16, f"Mid rank should be between 4 and 16, got {rank_mid}"

    tlora.set_global_step(100)
    assert tlora.current_rank == 16, f"Final rank should be 16, got {tlora.current_rank}"

    # Beyond total_steps should clamp to max
    tlora.set_global_step(200)
    assert tlora.current_rank == 16, f"Rank beyond total_steps should clamp to max_rank, got {tlora.current_rank}"
    print("  [PASS] tlora_linear_schedule")


def test_tlora_geometric_schedule():
    """T-LoRA with geometric schedule should ramp exponentially."""
    base = _make_linear()
    tlora = TLoRALinear(base, max_rank=32, min_rank=2, schedule="geometric", total_steps=100)

    assert tlora.current_rank == 2, f"Expected initial rank 2, got {tlora.current_rank}"

    tlora.set_global_step(50)
    rank_mid = tlora.current_rank
    # Geometric at 50% should be sqrt(32/2)*2 = ~8
    assert 2 < rank_mid < 32, f"Mid geometric rank should be between 2 and 32, got {rank_mid}"

    tlora.set_global_step(100)
    assert tlora.current_rank == 32, f"Final rank should be 32, got {tlora.current_rank}"
    print("  [PASS] tlora_geometric_schedule")


def test_tlora_forward():
    """T-LoRA forward should produce valid output with rank masking."""
    base = _make_linear()
    tlora = TLoRALinear(base, max_rank=16, min_rank=4, schedule="linear", total_steps=100, alpha=8.0)

    x = torch.randn(2, 64)
    with torch.no_grad():
        out = tlora(x)

    assert out.shape == (2, 32), f"Expected output shape (2, 32), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"
    print("  [PASS] tlora_forward")


def test_tlora_orthogonal_init():
    """T-LoRA with orthogonal init should produce orthogonal rows in lora_down."""
    base = _make_linear()
    tlora = TLoRALinear(base, max_rank=8, min_rank=4, schedule="constant", orthogonal_init=True)

    # lora_down shape: (8, 64) — first 8 rows should be orthogonal
    w = tlora.lora_down.weight.data
    gram = w @ w.T
    # Diagonal should be ~1, off-diagonal ~0
    diag = torch.diag(gram)
    off_diag = gram - torch.diag(diag)
    assert (diag > 0.5).all(), f"Orthogonal rows should have norm ~1, got {diag}"
    assert off_diag.abs().max() < 0.3, f"Off-diagonal should be ~0, got max {off_diag.abs().max()}"
    print("  [PASS] tlora_orthogonal_init")


def test_tlora_rank_mask_effect():
    """Rank mask should zero out inactive columns, affecting output."""
    base = _make_linear()
    tlora = TLoRALinear(base, max_rank=16, min_rank=4, schedule="linear", total_steps=100, alpha=16.0)

    # Set lora_up to non-zero so we can observe the rank mask effect
    with torch.no_grad():
        nn.init.kaiming_uniform_(tlora.lora_up.weight)

    x = torch.randn(2, 64)

    tlora.set_global_step(0)
    rank_4 = tlora.current_rank
    with torch.no_grad():
        out_low = tlora(x).clone()

    tlora.set_global_step(100)
    rank_16 = tlora.current_rank
    with torch.no_grad():
        out_high = tlora(x).clone()

    assert rank_4 < rank_16, f"Rank should increase: {rank_4} -> {rank_16}"
    # Outputs should differ because more columns are active
    assert not torch.allclose(out_low, out_high, atol=1e-6), "Outputs should differ at different ranks"
    print("  [PASS] tlora_rank_mask_effect")


def test_tlora_lora_leaf_marking():
    """T-LoRA layers should have _lora_leaf=True on lora_down and lora_up."""
    base = _make_linear()
    tlora = TLoRALinear(base, max_rank=8, min_rank=2, schedule="constant")

    assert getattr(tlora.lora_down, "_lora_leaf", False) is True, "lora_down should be marked as _lora_leaf"
    assert getattr(tlora.lora_up, "_lora_leaf", False) is True, "lora_up should be marked as _lora_leaf"
    print("  [PASS] tlora_lora_leaf_marking")


def test_tlora_merge_weights():
    """Merged weights should equal base + LoRA contribution."""
    base = _make_linear()
    original_weight = base.weight.data.clone()
    tlora = TLoRALinear(base, max_rank=8, min_rank=8, schedule="constant", alpha=8.0)

    with torch.no_grad():
        nn.init.kaiming_uniform_(tlora.lora_up.weight)

    expected_merged = original_weight + tlora.get_weight_matrix().to(original_weight.dtype)

    tlora.merge_weights()

    assert torch.allclose(tlora.original.weight.data, expected_merged, atol=1e-5), \
        "Merged weight should equal base + LoRA"
    print("  [PASS] tlora_merge_weights")


if __name__ == "__main__":
    print("T-LoRA Smoke Tests")
    print("=" * 40)
    test_tlora_constant_schedule()
    test_tlora_linear_schedule()
    test_tlora_geometric_schedule()
    test_tlora_forward()
    test_tlora_orthogonal_init()
    test_tlora_rank_mask_effect()
    test_tlora_lora_leaf_marking()
    test_tlora_merge_weights()
    print("=" * 40)
    print("All T-LoRA smoke tests passed!")
