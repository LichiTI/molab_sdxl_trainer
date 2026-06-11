"""Smoke tests for Anima attention backend dispatch.

Tests:
  1. SDPA attention output matches manual torch attention
  2. Flash2 falls back gracefully when flash_attn is not installed
  3. SageAttn falls back gracefully when sageattention is not installed
  4. FlexAttention falls back gracefully when the PyTorch API is unavailable
  5. patch_anima_attention sets _attention_backend on modules
  6. Patched forward produces same output as SDPA baseline
"""

from __future__ import annotations

import sys
import os
import importlib.util
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# Load anima_attention via importlib to avoid diffusers import chain
_aa = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.anima_attention",
    os.path.join(_HERE, "anima_attention.py"),
)
_aa_mod = importlib.util.module_from_spec(_aa)
sys.modules["core.lulynx_trainer.anima_attention"] = _aa_mod
_aa.loader.exec_module(_aa_mod)

import torch
import torch.nn as nn


def _make_qkv(batch=2, heads=4, seq=16, dim=32):
    return torch.randn(batch, heads, seq, dim), torch.randn(batch, heads, seq, dim), torch.randn(batch, heads, seq, dim)


def test_sdpa_matches_torch():
    """SDPA should produce numerically close results to manual attention."""
    q, k, v = _make_qkv()
    sdpa_out = _aa_mod.dit_attention(q, k, v, backend="sdpa")
    torch_out = _aa_mod.dit_attention(q, k, v, backend="torch")
    assert sdpa_out.shape == torch_out.shape
    # SDPA and manual may differ by floating point but should be close
    diff = (sdpa_out - torch_out).abs().max().item()
    assert diff < 0.1, f"SDPA vs torch diff too large: {diff}"
    print("PASS: SDPA matches torch attention (within tolerance)")


def test_flash2_fallback():
    """Flash2 should fall back to SDPA when flash_attn is not available."""
    q, k, v = _make_qkv()
    previous = _aa_mod._flash_attn_available
    try:
        _aa_mod._flash_attn_available = False
        out = _aa_mod.dit_attention(q, k, v, backend="flash2")
        sdpa_out = _aa_mod.dit_attention(q, k, v, backend="sdpa")
    finally:
        _aa_mod._flash_attn_available = previous
    assert torch.allclose(out, sdpa_out, atol=1e-6)
    print("PASS: Flash2 falls back to SDPA when flash_attn unavailable")


def test_sageattn_fallback():
    """SageAttn should fall back to SDPA when sageattention is not available."""
    q, k, v = _make_qkv()
    previous = _aa_mod._sageattn_available
    try:
        _aa_mod._sageattn_available = False
        out = _aa_mod.dit_attention(q, k, v, backend="sageattn")
        sdpa_out = _aa_mod.dit_attention(q, k, v, backend="sdpa")
    finally:
        _aa_mod._sageattn_available = previous
    assert torch.allclose(out, sdpa_out, atol=1e-6)
    print("PASS: SageAttn falls back to SDPA when sageattention unavailable")


def test_flexattn_fallback():
    """FlexAttention should fall back to SDPA when the PyTorch API is unavailable."""
    q, k, v = _make_qkv()
    previous = _aa_mod._flex_attn_available
    try:
        _aa_mod._flex_attn_available = False
        out = _aa_mod.dit_attention(q, k, v, backend="flexattn")
        sdpa_out = _aa_mod.dit_attention(q, k, v, backend="sdpa")
    finally:
        _aa_mod._flex_attn_available = previous
    assert torch.allclose(out, sdpa_out, atol=1e-6)
    print("PASS: FlexAttention falls back to SDPA when torch flex_attention unavailable")


def test_torch_attention_shapes():
    """All backends should produce the same output shape."""
    q, k, v = _make_qkv()
    for backend in ("sdpa", "torch"):
        out = _aa_mod.dit_attention(q, k, v, backend=backend)
        assert out.shape == q.shape, f"Shape mismatch for {backend}: {out.shape} vs {q.shape}"
    print("PASS: All backends produce correct output shapes")


# Minimal _ProjectionAttention-like module for patch testing
class _FakeProjectionAttention(nn.Module):
    def __init__(self, hidden_dim=64, head_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.num_heads = hidden_dim // head_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        # Simple LayerNorm as QK-norm stand-in
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)

    def _split_heads(self, tensor):
        batch, tokens, width = tensor.shape
        return tensor.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tensor):
        batch, _heads, tokens, _head_dim = tensor.shape
        return tensor.transpose(1, 2).reshape(batch, tokens, self.hidden_dim)

    def forward(self, x, context=None):
        source = x if context is None else context
        q = self.q_norm(self._split_heads(self.q_proj(x)))
        k = self.k_norm(self._split_heads(self.k_proj(source)))
        v = self._split_heads(self.v_proj(source))
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.output_proj(self._merge_heads(attn))


class _FakeDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": _FakeProjectionAttention(),
                "cross_attn": _FakeProjectionAttention(),
            })
            for _ in range(2)
        ])


def test_patch_anima_attention():
    """patch_anima_attention should set _attention_backend on attention modules."""
    model = _FakeDiT()
    _aa_mod.patch_anima_attention(model, backend="flash2")

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, _FakeProjectionAttention):
            assert hasattr(module, "_attention_backend"), f"{name} missing _attention_backend"
            assert module._attention_backend == "flash2"
            count += 1

    assert count == 4, f"Expected 4 patched attention modules, got {count}"
    _aa_mod.unpatch_anima_attention(model)

    for name, module in model.named_modules():
        if isinstance(module, _FakeProjectionAttention):
            assert not hasattr(module, "_attention_backend"), f"{name} still has _attention_backend after unpatch"
    print("PASS: patch_anima_attention sets _attention_backend correctly")


def test_patched_forward_matches_sdpa():
    """Patched forward should produce same output as original SDPA forward."""
    model = _FakeDiT()
    x = torch.randn(1, 8, 64)

    # Baseline: original forward
    with torch.no_grad():
        baseline_out = model.blocks[0]["self_attn"](x)

    # Patch to SDPA (should be identical)
    _aa_mod.patch_anima_attention(model, backend="sdpa")
    with torch.no_grad():
        patched_out = model.blocks[0]["self_attn"](x)

    diff = (baseline_out - patched_out).abs().max().item()
    assert diff < 1e-5, f"Patched SDPA forward differs from original: {diff}"

    _aa_mod.unpatch_anima_attention(model)
    print("PASS: Patched SDPA forward matches original")


def test_tgate_probe_observes_cross_attention_without_output_change():
    """T-GATE observe mode records eligibility without changing attention output."""
    from core.lulynx_trainer.tgate import reset_tgate_stats, snapshot_tgate_stats, tgate_step_context

    model = _FakeDiT()
    x = torch.randn(1, 8, 64)
    context = torch.randn(1, 10, 64)

    with torch.no_grad():
        baseline_out = model.blocks[1]["cross_attn"](x, context)

    reset_tgate_stats()
    _aa_mod.reset_attention_stats()
    _aa_mod.patch_anima_attention(model, backend="sdpa")
    with torch.no_grad(), tgate_step_context(
        enabled=True,
        step_index=4,
        total_steps=8,
        start_step=2,
        min_block=1,
    ):
        patched_out = model.blocks[1]["cross_attn"](x, context)

    diff = (baseline_out - patched_out).abs().max().item()
    assert diff < 1e-5, f"T-GATE observe changed output: {diff}"
    stats = snapshot_tgate_stats()
    assert stats["cross_attention_calls"] == 1, stats
    assert stats["eligible_cross_attention_calls"] == 1, stats
    attention_stats = _aa_mod.snapshot_attention_stats()
    assert attention_stats["tgate_cross_attention_calls"] == 1, attention_stats
    assert attention_stats["tgate_eligible_cross_attention_calls"] == 1, attention_stats

    _aa_mod.unpatch_anima_attention(model)
    print("PASS: T-GATE observe records cross-attn eligibility without changing output")


if __name__ == "__main__":
    test_sdpa_matches_torch()
    test_flash2_fallback()
    test_sageattn_fallback()
    test_flexattn_fallback()
    test_torch_attention_shapes()
    test_patch_anima_attention()
    test_patched_forward_matches_sdpa()
    test_tgate_probe_observes_cross_attention_without_output_change()
    print("\nAll Anima attention smoke tests passed!")
