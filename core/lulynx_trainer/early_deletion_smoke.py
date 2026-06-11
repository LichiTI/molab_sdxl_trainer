# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for attention early deletion.

Tests:
  1. Early deletion produces identical outputs to non-early-deletion
  2. Patched forward with early_deletion matches baseline
  3. Config field exists and defaults to False
"""

from __future__ import annotations

import sys
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))

# Set up fake package so relative imports work
import types as _types

_pkg_name = "backend.core.lulynx_trainer"
_pkg = _types.ModuleType(_pkg_name)
_pkg.__path__ = [_HERE]
_pkg.__package__ = _pkg_name
sys.modules.setdefault(_pkg_name, _pkg)
for _parent in ("backend", "backend.core"):
    sys.modules.setdefault(_parent, _types.ModuleType(_parent))

# Load amd_runtime first (dependency of anima_attention)
_amd_spec = importlib.util.spec_from_file_location(
    f"{_pkg_name}.amd_runtime",
    os.path.join(_HERE, "amd_runtime.py"),
    submodule_search_locations=[],
)
_amd = importlib.util.module_from_spec(_amd_spec)
_amd.__package__ = _pkg_name
sys.modules[f"{_pkg_name}.amd_runtime"] = _amd
_amd_spec.loader.exec_module(_amd)

# Load anima_attention via importlib to avoid diffusers import chain
_spec = importlib.util.spec_from_file_location(
    f"{_pkg_name}.anima_attention",
    os.path.join(_HERE, "anima_attention.py"),
    submodule_search_locations=[],
)
_aa = importlib.util.module_from_spec(_spec)
_aa.__package__ = _pkg_name
sys.modules[f"{_pkg_name}.anima_attention"] = _aa
_spec.loader.exec_module(_aa)

import torch
import torch.nn as nn


def _make_qkv(batch=2, heads=4, seq=16, dim=32):
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)
    return q, k, v


def test_early_delete_sdpa_identical():
    """SDPA with early_delete should produce identical output (SDPA is fused, early_delete is a no-op)."""
    q, k, v = _make_qkv()
    out_normal = _aa.dit_attention(q, k, v, backend="sdpa", early_delete=False)
    out_early = _aa.dit_attention(q, k, v, backend="sdpa", early_delete=True)
    assert torch.allclose(out_normal, out_early, atol=1e-6), \
        f"SDPA early_delete mismatch: {(out_normal - out_early).abs().max().item()}"
    print("PASS: SDPA early_delete produces identical output")


def test_early_delete_torch_identical():
    """Torch manual backend with early_delete should produce identical output."""
    q, k, v = _make_qkv()
    out_normal = _aa.dit_attention(q.clone(), k.clone(), v.clone(), backend="torch", early_delete=False)
    out_early = _aa.dit_attention(q.clone(), k.clone(), v.clone(), backend="torch", early_delete=True)
    diff = (out_normal - out_early).abs().max().item()
    assert diff < 1e-6, f"Torch early_delete mismatch: {diff}"
    print("PASS: Torch manual early_delete produces identical output")


def test_early_delete_chunked_identical():
    """Chunked attention with early_delete should produce identical output."""
    q, k, v = _make_qkv()
    out_normal = _aa.dit_attention(q, k, v, backend="sdpa", split_chunks=2, early_delete=False)
    out_early = _aa.dit_attention(q, k, v, backend="sdpa", split_chunks=2, early_delete=True)
    diff = (out_normal - out_early).abs().max().item()
    assert diff < 1e-6, f"Chunked early_delete mismatch: {diff}"
    print("PASS: Chunked attention early_delete produces identical output")


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


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeProjectionAttention() for _ in range(2)])


def test_patched_forward_early_deletion():
    """Patched forward with early_deletion should match baseline."""
    model = _FakeModel()
    x = torch.randn(1, 8, 64)

    with torch.no_grad():
        baseline = model.blocks[0](x).clone()

    _aa.patch_anima_attention(model, backend="sdpa", early_deletion=True)

    for _, m in model.named_modules():
        if isinstance(m, _FakeProjectionAttention):
            assert getattr(m, "_attention_early_deletion", False), \
                "early_deletion attr not set on patched module"

    with torch.no_grad():
        patched_out = model.blocks[0](x)

    diff = (baseline - patched_out).abs().max().item()
    assert diff < 1e-5, f"Patched early_deletion forward differs: {diff}"

    _aa.unpatch_anima_attention(model)
    for _, m in model.named_modules():
        if isinstance(m, _FakeProjectionAttention):
            assert not hasattr(m, "_attention_early_deletion"), \
                "early_deletion attr not cleaned up after unpatch"

    print("PASS: Patched forward with early_deletion matches baseline")


def test_patched_forward_torch_early_deletion():
    """Patched forward with torch backend + early_deletion should match SDPA baseline."""
    model = _FakeModel()
    x = torch.randn(1, 8, 64)

    with torch.no_grad():
        baseline = model.blocks[0](x).clone()

    _aa.patch_anima_attention(model, backend="torch", early_deletion=True)
    with torch.no_grad():
        patched_out = model.blocks[0](x)

    diff = (baseline - patched_out).abs().max().item()
    assert diff < 0.1, f"Torch+early_deletion forward differs too much: {diff}"

    _aa.unpatch_anima_attention(model)
    print("PASS: Patched torch+early_deletion forward matches SDPA baseline")


def test_backward_with_early_deletion():
    """Backward pass should work correctly with early deletion enabled."""
    model = _FakeModel()
    _aa.patch_anima_attention(model, backend="sdpa", early_deletion=True)

    x = torch.randn(1, 8, 64, requires_grad=True)
    out = model.blocks[0](x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Gradients not computed with early_deletion"
    assert x.grad.shape == x.shape, "Gradient shape mismatch"
    assert not torch.all(x.grad == 0), "All-zero gradients"

    _aa.unpatch_anima_attention(model)
    print("PASS: Backward pass works with early_deletion")


def test_config_field():
    """attention_early_deletion config field should exist in configs.py source."""
    cfg_path = os.path.join(_HERE, "..", "configs.py")
    with open(cfg_path, encoding="utf-8") as f:
        src = f.read()
    assert "attention_early_deletion" in src, "Missing attention_early_deletion in configs.py"
    assert "attention_early_deletion: bool = False" in src, "Field should default to False"
    print("PASS: attention_early_deletion config field exists in source")


if __name__ == "__main__":
    test_early_delete_sdpa_identical()
    test_early_delete_torch_identical()
    test_early_delete_chunked_identical()
    test_patched_forward_early_deletion()
    test_patched_forward_torch_early_deletion()
    test_backward_with_early_deletion()
    test_config_field()
    print("\nAll early deletion smoke tests passed!")
