"""Smoke tests for VeRA (Vector-based Random Matrix Adaptation) and LoRA-FA.

Tests:
  1. VeRASharedBuffers: deterministic init and growth
  2. VeRALinear: forward shape correctness
  3. VeRALinear: zero-init (newly injected adapter starts as identity)
  4. VeRALinear: only lambda_d and lambda_b are trainable
  5. VeRALinear: export_standard_lora_weights shapes
  6. LoRAFALinear: forward shape correctness
  7. LoRAFALinear: zero-init (B is zeros)
  8. LoRAFALinear: only lora_up is trainable
  9. LoRAFALinear: lora_down is frozen
"""

from __future__ import annotations

import sys
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))

# Load vera_layer via importlib
_vl = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.vera_layer",
    os.path.join(_HERE, "vera_layer.py"),
)
_vl_mod = importlib.util.module_from_spec(_vl)
sys.modules["core.lulynx_trainer.vera_layer"] = _vl_mod
_vl.loader.exec_module(_vl_mod)

# Load lora_fa_layer via importlib
_lf = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.lora_fa_layer",
    os.path.join(_HERE, "lora_fa_layer.py"),
)
_lf_mod = importlib.util.module_from_spec(_lf)
sys.modules["core.lulynx_trainer.lora_fa_layer"] = _lf_mod
_lf.loader.exec_module(_lf_mod)

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# VeRA tests
# ---------------------------------------------------------------------------

def test_vera_shared_buffers_deterministic():
    """Same prng_key should produce identical shared buffers."""
    buf1 = _vl_mod.VeRASharedBuffers(rank=4, prng_key=42)
    buf2 = _vl_mod.VeRASharedBuffers(rank=4, prng_key=42)
    buf1.ensure(64, 64)
    buf2.ensure(64, 64)
    assert torch.allclose(buf1.shared_A, buf2.shared_A), "A buffers differ"
    assert torch.allclose(buf1.shared_B, buf2.shared_B), "B buffers differ"
    print("PASS: VeRASharedBuffers deterministic init")


def test_vera_shared_buffers_growth():
    """Buffers should grow to accommodate larger dimensions."""
    buf = _vl_mod.VeRASharedBuffers(rank=4, prng_key=0)
    buf.ensure(32, 64)
    A1 = buf.shared_A.clone()
    B1 = buf.shared_B.clone()
    buf.ensure(64, 128)
    # Original data should be preserved in top-left
    assert torch.allclose(buf.shared_A[:, :32], A1[:, :32])
    assert torch.allclose(buf.shared_B[:64, :], B1[:64, :])
    assert buf.shared_A.shape == (4, 64)
    assert buf.shared_B.shape == (128, 4)
    print("PASS: VeRASharedBuffers growth preserves existing data")


def test_vera_linear_forward_shape():
    """VeRALinear forward should match expected shape."""
    original = nn.Linear(32, 64)
    buf = _vl_mod.VeRASharedBuffers(rank=4, prng_key=0)
    vera = _vl_mod.VeRALinear(original, shared_buffers=buf, d_initial=0.1, alpha=1.0)
    x = torch.randn(2, 8, 32)
    out = vera(x)
    assert out.shape == (2, 8, 64), f"Shape: {out.shape}"
    print("PASS: VeRALinear forward shape correctness")


def test_vera_linear_zero_init():
    """VeRALinear should start as (near-)identity since lambda_b is zeros."""
    original = nn.Linear(32, 32)
    # Make original easy to check
    with torch.no_grad():
        original.weight.fill_(0.5)
        original.bias.fill_(0.0)
    buf = _vl_mod.VeRASharedBuffers(rank=4, prng_key=0)
    vera = _vl_mod.VeRALinear(original, shared_buffers=buf, d_initial=0.1)
    x = torch.randn(2, 8, 32)
    original_out = original(x)
    vera_out = vera(x)
    # lambda_b is zeros → delta should be zero → vera_out ≈ original_out
    diff = (vera_out - original_out).abs().max().item()
    assert diff < 1e-5, f"VeRA delta should be ~0 at init, got diff={diff}"
    print("PASS: VeRALinear zero-init (lambda_b=0)")


def test_vera_trainable_params():
    """Only lambda_d and lambda_b should be trainable."""
    original = nn.Linear(32, 64)
    buf = _vl_mod.VeRASharedBuffers(rank=4, prng_key=0)
    vera = _vl_mod.VeRALinear(original, shared_buffers=buf)
    trainable = [n for n, p in vera.named_parameters() if p.requires_grad]
    assert "vera_lambda_d" in trainable
    assert "vera_lambda_b" in trainable
    assert len(trainable) == 2, f"Expected 2 trainable params, got: {trainable}"
    print("PASS: VeRALinear only lambda_d/lambda_b trainable")


def test_vera_export_lora_weights():
    """export_standard_lora_weights should produce standard LoRA shapes."""
    original = nn.Linear(32, 64)
    buf = _vl_mod.VeRASharedBuffers(rank=4, prng_key=0)
    vera = _vl_mod.VeRALinear(original, shared_buffers=buf)
    weights = vera.export_standard_lora_weights()
    assert "lora_down.weight" in weights
    assert "lora_up.weight" in weights
    assert weights["lora_down.weight"].shape == (4, 32)
    assert weights["lora_up.weight"].shape == (64, 4)
    print("PASS: VeRALinear export_standard_lora_weights shapes")


# ---------------------------------------------------------------------------
# LoRA-FA tests
# ---------------------------------------------------------------------------

def test_lora_fa_forward_shape():
    """LoRAFALinear forward should match expected shape."""
    original = nn.Linear(32, 64)
    lora_fa = _lf_mod.LoRAFALinear(original, rank=4, alpha=1.0)
    x = torch.randn(2, 8, 32)
    out = lora_fa(x)
    assert out.shape == (2, 8, 64), f"Shape: {out.shape}"
    print("PASS: LoRAFALinear forward shape correctness")


def test_lora_fa_zero_init():
    """LoRAFALinear should start as identity since B is zeros."""
    original = nn.Linear(32, 32)
    with torch.no_grad():
        original.weight.fill_(0.5)
        original.bias.fill_(0.0)
    lora_fa = _lf_mod.LoRAFALinear(original, rank=4, alpha=1.0)
    x = torch.randn(2, 8, 32)
    original_out = original(x)
    fa_out = lora_fa(x)
    diff = (fa_out - original_out).abs().max().item()
    assert diff < 1e-5, f"LoRA-FA delta should be ~0 at init, got diff={diff}"
    print("PASS: LoRAFALinear zero-init (B=0)")


def test_lora_fa_trainable_params():
    """Only lora_up should be trainable."""
    original = nn.Linear(32, 64)
    lora_fa = _lf_mod.LoRAFALinear(original, rank=4, alpha=1.0)
    trainable = [n for n, p in lora_fa.named_parameters() if p.requires_grad]
    assert any("lora_up" in n for n in trainable)
    assert not any("lora_down" in n for n in trainable), "lora_down should be frozen"
    print("PASS: LoRAFALinear only lora_up trainable")


def test_lora_fa_down_frozen():
    """lora_down weight should be frozen."""
    original = nn.Linear(32, 64)
    lora_fa = _lf_mod.LoRAFALinear(original, rank=4, alpha=1.0)
    assert not lora_fa.lora_down.weight.requires_grad
    print("PASS: LoRAFALinear lora_down frozen")


def test_lora_fa_gradient_flow():
    """Gradient should flow through lora_up."""
    original = nn.Linear(32, 64)
    lora_fa = _lf_mod.LoRAFALinear(original, rank=4, alpha=1.0)
    x = torch.randn(2, 8, 32)
    out = lora_fa(x)
    loss = out.sum()
    loss.backward()
    assert lora_fa.lora_up.weight.grad is not None
    assert lora_fa.lora_down.weight.grad is None
    print("PASS: LoRAFALinear gradient only through lora_up")


if __name__ == "__main__":
    test_vera_shared_buffers_deterministic()
    test_vera_shared_buffers_growth()
    test_vera_linear_forward_shape()
    test_vera_linear_zero_init()
    test_vera_trainable_params()
    test_vera_export_lora_weights()
    test_lora_fa_forward_shape()
    test_lora_fa_zero_init()
    test_lora_fa_trainable_params()
    test_lora_fa_down_frozen()
    test_lora_fa_gradient_flow()
    print("\nAll VeRA / LoRA-FA smoke tests passed!")
