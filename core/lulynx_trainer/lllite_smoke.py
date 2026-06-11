"""Smoke tests for ControlNet-LLLite module.

Tests the core LLLite components without requiring a real UNet or GPU:
  1. ConditioningEncoder shape correctness
  2. LLLiteLinear forward / zero-init
  3. LLLiteConv2d forward / zero-init
  4. Inject/remove roundtrip on a minimal UNet-like model
  5. State dict save/load roundtrip
  6. Depth assignment correctness
"""

from __future__ import annotations

import sys
import os

# Use importlib.util to bypass __init__.py import chain failures
import importlib.util
import types


def _load_module_from_file(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_HERE = os.path.dirname(os.path.abspath(__file__))
_lllite = _load_module_from_file(
    "core.lulynx_trainer.lllite",
    os.path.join(_HERE, "lllite.py"),
)

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DummyAttn(nn.Module):
    """Mimics a UNet attention block's linear layers."""
    def __init__(self, dim: int = 64):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)


class _DummyTransformerBlock(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.attn1 = _DummyAttn(dim)
        self.attn2 = _DummyAttn(dim)


class _DummyInputBlock(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.transformer = _DummyTransformerBlock(dim)


class _DummyMidBlock(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.transformer = _DummyTransformerBlock(dim)


class _DummyOutputBlock(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.transformer = _DummyTransformerBlock(dim)


class _DummyUNet(nn.Module):
    """Minimal UNet-like model for injection testing."""
    def __init__(self, dim: int = 64):
        super().__init__()
        self.time_embed = nn.Linear(dim, dim)  # should be skipped
        self.input_blocks = nn.ModuleList([_DummyInputBlock(dim) for _ in range(6)])
        self.middle_block = _DummyMidBlock(dim)
        self.output_blocks = nn.ModuleList([_DummyOutputBlock(dim) for _ in range(6)])

    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_conditioning_encoder_shapes():
    """Encoder should produce embeddings at 3 depth levels."""
    enc = _lllite.ConditioningEncoder(cond_emb_dim=32)
    img = torch.randn(2, 3, 128, 128)
    embs = enc(img)
    assert 1 in embs and 2 in embs and 3 in embs
    # depth-1: 128 / 4 = 32 (first conv), then / 2 = 16
    assert embs[1].shape == (2, 32, 16, 16), f"depth-1 shape: {embs[1].shape}"
    # depth-2: 128 / 4 = 32, then / 4 = 8
    assert embs[2].shape == (2, 32, 8, 8), f"depth-2 shape: {embs[2].shape}"
    # depth-3: 128 / 4 = 32 → / 4 = 8 → / 2 = 4
    assert embs[3].shape == (2, 32, 4, 4), f"depth-3 shape: {embs[3].shape}"
    print("PASS: ConditioningEncoder shapes correct")


def test_lllite_linear_zero_init():
    """LLLiteLinear should start as identity (zero output)."""
    adapter = _lllite.LLLiteLinear(in_dim=64, cond_emb_dim=32, mlp_dim=32)
    x = torch.randn(2, 10, 64)
    # Without conditioning → zero
    out = adapter(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)
    # With conditioning → still zero (up layer zero-init)
    cond = torch.randn(2, 32, 4, 4)
    adapter.set_conditioning(cond)
    out = adapter(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)
    print("PASS: LLLiteLinear zero-init correct")


def test_lllite_conv_zero_init():
    """LLLiteConv2d should start as identity (zero output)."""
    adapter = _lllite.LLLiteConv2d(in_channels=64, cond_emb_dim=32, mlp_dim=32)
    x = torch.randn(2, 64, 8, 8)
    out = adapter(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)
    cond = torch.randn(2, 32, 4, 4)
    adapter.set_conditioning(cond)
    out = adapter(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)
    print("PASS: LLLiteConv2d zero-init correct")


def test_lllite_linear_gradient_flow():
    """After setting conditioning, gradients should flow through adapter."""
    adapter = _lllite.LLLiteLinear(in_dim=64, cond_emb_dim=32, mlp_dim=32)
    # Give non-zero weights to up layer so we get a signal
    nn.init.xavier_normal_(adapter.up.weight)
    nn.init.zeros_(adapter.up.bias)
    x = torch.randn(2, 10, 64, requires_grad=True)
    cond = torch.randn(2, 32, 4, 4)
    adapter.set_conditioning(cond)
    out = adapter(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert adapter.down.weight.grad is not None
    assert adapter.mid.weight.grad is not None
    assert adapter.up.weight.grad is not None
    print("PASS: LLLiteLinear gradient flow correct")


def test_inject_remove_roundtrip():
    """Inject then remove should leave UNet unchanged."""
    unet = _DummyUNet(dim=64)
    original_names = set(n for n, _ in unet.named_modules())
    encoder, injected = _lllite.inject_lllite(
        unet, cond_emb_dim=16, mlp_dim=32,
        skip_input_blocks=False, skip_output_blocks=True,
    )
    assert len(injected) > 0, "No adapters were injected"
    # Adapters should exist as new attributes
    assert hasattr(unet, "_lllite_injected")
    assert hasattr(unet, "_lllite_encoder")

    _lllite.remove_lllite(unet)
    # After removal, no _lllite attributes should remain
    assert not hasattr(unet, "_lllite_injected")
    assert not hasattr(unet, "_lllite_hooks")
    # Original modules should still be present
    for name in original_names:
        assert unet.get_submodule(name) is not None, f"Missing {name} after removal"
    print("PASS: Inject/remove roundtrip correct")


def test_state_dict_save_load():
    """State dict roundtrip should preserve adapter weights."""
    unet = _DummyUNet(dim=64)
    encoder, injected = _lllite.inject_lllite(
        unet, cond_emb_dim=16, mlp_dim=32,
        skip_input_blocks=True, skip_output_blocks=True,
    )
    # Modify weights to have non-zero values
    with torch.no_grad():
        for p in encoder.parameters():
            p.add_(0.1)
        for name in injected:
            adapter = _lllite._get_adapter(unet, name)
            for p in adapter.parameters():
                p.add_(0.1)

    state = _lllite.get_lllite_state_dict(unet)
    assert len(state) > 0

    # Save / load roundtrip
    _lllite.load_lllite_state_dict(unet, state)
    state2 = _lllite.get_lllite_state_dict(unet)
    for key in state:
        assert key in state2, f"Missing key after reload: {key}"
        assert torch.allclose(state[key], state2[key], atol=1e-6), f"Mismatch for {key}"

    _lllite.remove_lllite(unet)
    print("PASS: State dict save/load roundtrip correct")


def test_depth_assignment():
    """Adapter depth should be correctly assigned based on block index."""
    unet = _DummyUNet(dim=64)
    _, injected = _lllite.inject_lllite(
        unet, cond_emb_dim=16, mlp_dim=32,
        skip_input_blocks=False, skip_output_blocks=False,
    )
    # Check depth for a few known blocks
    for name in injected:
        adapter = _lllite._get_adapter(unet, name)
        depth = getattr(adapter, "_depth", None)
        assert depth is not None, f"No depth for {name}"
        # input_blocks.0.* → depth 1
        if "input_blocks.0" in name:
            assert depth == 1, f"input_blocks.0 depth should be 1, got {depth}"
        # input_blocks.5.* → depth 2
        elif "input_blocks.5" in name:
            assert depth == 2, f"input_blocks.5 depth should be 2, got {depth}"
        # middle_block.* → depth 3
        elif "middle_block" in name:
            assert depth == 3, f"middle_block depth should be 3, got {depth}"
        # output_blocks.0.* → depth 3
        elif "output_blocks.0" in name:
            assert depth == 3, f"output_blocks.0 depth should be 3, got {depth}"
    _lllite.remove_lllite(unet)
    print("PASS: Depth assignment correct")


def test_set_conditioning():
    """set_lllite_conditioning should push embeddings to all adapters."""
    unet = _DummyUNet(dim=64)
    encoder, injected = _lllite.inject_lllite(
        unet, cond_emb_dim=16, mlp_dim=32,
        skip_input_blocks=False, skip_output_blocks=True,
    )
    cond_embs = encoder(torch.randn(2, 3, 64, 64))
    _lllite.set_lllite_conditioning(unet, cond_embs)

    for name in injected:
        adapter = _lllite._get_adapter(unet, name)
        assert adapter._cond_emb is not None, f"Adapter {name} has no cond embedding"

    _lllite.remove_lllite(unet)
    print("PASS: set_lllite_conditioning works")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_conditioning_encoder_shapes()
    test_lllite_linear_zero_init()
    test_lllite_conv_zero_init()
    test_lllite_linear_gradient_flow()
    test_inject_remove_roundtrip()
    test_state_dict_save_load()
    test_depth_assignment()
    test_set_conditioning()
    print("\nAll LLLite smoke tests passed!")
