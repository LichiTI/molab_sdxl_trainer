# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Anima split_attn (#54).

Verifies:
  1. dit_attention(split_chunks=N) produces the same result as a single-pass
     attention call (numerically, within bf16 tolerance).
  2. RuntimeOptimizationPlan picks up anima_split_attn from config.
  3. patch_anima_attention propagates split_chunks to the patched module.
"""

from __future__ import annotations

import os
import sys
import importlib.util
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"core.lulynx_trainer.{name}",
        os.path.join(_HERE, f"{name}.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"core.lulynx_trainer.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


_aa = _load("anima_attention")
_ro = _load("runtime_optimizations")


def test_split_attn_equivalent_to_single_pass():
    torch.manual_seed(42)
    B, H, T, D = 2, 8, 16, 16
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    # Reference: single-pass SDPA
    ref = _aa.dit_attention(q, k, v, backend="sdpa", split_chunks=0)
    # Split into 2 head groups
    out2 = _aa.dit_attention(q, k, v, backend="sdpa", split_chunks=2)
    # Split into 4 head groups
    out4 = _aa.dit_attention(q, k, v, backend="sdpa", split_chunks=4)

    assert ref.shape == out2.shape == out4.shape
    assert torch.allclose(ref, out2, atol=1e-5, rtol=1e-5)
    assert torch.allclose(ref, out4, atol=1e-5, rtol=1e-5)
    print(f"PASS: split_attn(2) and split_attn(4) match single-pass SDPA "
          f"(max diff vs ref: {(ref - out2).abs().max().item():.2e}, "
          f"{(ref - out4).abs().max().item():.2e})")


def test_split_attn_handles_non_divisible_heads():
    """When head count doesn't divide evenly, last chunk gets remainder."""
    torch.manual_seed(0)
    B, H, T, D = 1, 7, 8, 8  # 7 heads doesn't divide by 3
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    ref = _aa.dit_attention(q, k, v, backend="sdpa", split_chunks=0)
    out = _aa.dit_attention(q, k, v, backend="sdpa", split_chunks=3)

    assert ref.shape == out.shape
    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5)
    print("PASS: split_attn handles non-divisible head counts")


def test_split_attn_disables_when_chunks_le_1():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 4, 4)
    k = torch.randn(1, 4, 4, 4)
    v = torch.randn(1, 4, 4, 4)

    a = _aa.dit_attention(q, k, v, backend="sdpa", split_chunks=0)
    b = _aa.dit_attention(q, k, v, backend="sdpa", split_chunks=1)
    c = _aa.dit_attention(q, k, v, backend="sdpa")  # default

    assert torch.allclose(a, b)
    assert torch.allclose(a, c)
    print("PASS: split_chunks <= 1 is a no-op")


def test_resolve_split_chunks_from_config():
    cfg_on = SimpleNamespace(anima_split_attn=True)
    cfg_off = SimpleNamespace(anima_split_attn=False)
    cfg_legacy = SimpleNamespace(split_attn=True)
    cfg_explicit = SimpleNamespace(anima_split_attn=True, anima_split_attn_chunks=4)

    assert _ro._resolve_split_chunks(cfg_on) == 2
    assert _ro._resolve_split_chunks(cfg_off) == 0
    assert _ro._resolve_split_chunks(cfg_legacy) == 2
    assert _ro._resolve_split_chunks(cfg_explicit) == 4
    print("PASS: _resolve_split_chunks reads anima_split_attn / numeric override")


def test_runtime_plan_picks_up_split_attn():
    cfg = SimpleNamespace(attention_backend="sdpa", anima_split_attn=True)
    plan = _ro.build_runtime_optimization_plan(cfg)
    assert plan.attention_split_chunks == 2

    log = list(plan.log_lines())
    assert any("split_chunks=2" in line for line in log)
    print("PASS: RuntimeOptimizationPlan surfaces split_chunks in log")


def test_patch_anima_attention_propagates_split_chunks():
    """patch_anima_attention should set _attention_split_chunks on each module."""

    class _FakeAttn(torch.nn.Module):
        def __init__(self, dim=8, heads=4):
            super().__init__()
            self.q_proj = torch.nn.Linear(dim, dim)
            self.k_proj = torch.nn.Linear(dim, dim)
            self.v_proj = torch.nn.Linear(dim, dim)
            self.output_proj = torch.nn.Linear(dim, dim)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.heads = heads
            self.head_dim = dim // heads

        def _split_heads(self, x):
            B, T, D = x.shape
            return x.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        def _merge_heads(self, x):
            B, H, T, Dh = x.shape
            return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

        def forward(self, x, context=None):
            return x

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = _FakeAttn()
            self.attn2 = _FakeAttn()

    model = _Model()
    patched = _aa.patch_anima_attention(model, backend="sdpa", split_chunks=3)
    assert patched == 2
    assert model.attn1._attention_split_chunks == 3
    assert model.attn2._attention_split_chunks == 3
    assert model.attn1._attention_backend == "sdpa"
    print("PASS: patch_anima_attention propagates split_chunks to all attention modules")


def test_patched_forward_uses_split_chunks():
    """The patched forward should run chunked attention end-to-end."""

    class _FakeAttn(torch.nn.Module):
        def __init__(self, dim=8, heads=4):
            super().__init__()
            self.q_proj = torch.nn.Linear(dim, dim, bias=False)
            self.k_proj = torch.nn.Linear(dim, dim, bias=False)
            self.v_proj = torch.nn.Linear(dim, dim, bias=False)
            self.output_proj = torch.nn.Linear(dim, dim, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.heads = heads
            self.head_dim = dim // heads

        def _split_heads(self, x):
            B, T, D = x.shape
            return x.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        def _merge_heads(self, x):
            B, H, T, Dh = x.shape
            return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

        def forward(self, x, context=None):
            return x

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _FakeAttn()

    torch.manual_seed(7)
    model = _Model()
    x = torch.randn(1, 6, 8)

    # Reference: no split
    _aa.patch_anima_attention(model, backend="sdpa", split_chunks=0)
    out_ref = model.attn(x)

    # With split=2, output should match (modulo float noise)
    model.attn._attention_split_chunks = 2
    out_split = model.attn(x)

    assert out_ref.shape == out_split.shape
    assert torch.allclose(out_ref, out_split, atol=1e-5, rtol=1e-5)
    print("PASS: patched forward with split_chunks=2 matches split_chunks=0 output")


if __name__ == "__main__":
    test_split_attn_equivalent_to_single_pass()
    test_split_attn_handles_non_divisible_heads()
    test_split_attn_disables_when_chunks_le_1()
    test_resolve_split_chunks_from_config()
    test_runtime_plan_picks_up_split_attn()
    test_patch_anima_attention_propagates_split_chunks()
    test_patched_forward_uses_split_chunks()
    print("\nAll split_attn smoke tests passed!")
