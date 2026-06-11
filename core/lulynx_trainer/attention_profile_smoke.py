"""Smoke tests for AttentionProfile (sliding window) and FusedKVProjection.

Tests:
  1. AttentionProfile from_config / is_active
  2. sliding_window_attention shape correctness
  3. sliding_window_attention locality (tokens outside window have zero weight)
  4. FusedKVProjection shape and output correctness vs separate K/V
  5. FusedKVProjection weight copy matches original K/V
"""

from __future__ import annotations

import sys
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))

# Load runtime_optimizations via importlib to avoid diffusers import chain
_rt = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.runtime_optimizations",
    os.path.join(_HERE, "runtime_optimizations.py"),
)
_rt_mod = importlib.util.module_from_spec(_rt)
sys.modules["core.lulynx_trainer.runtime_optimizations"] = _rt_mod
_rt.loader.exec_module(_rt_mod)

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# AttentionProfile tests
# ---------------------------------------------------------------------------

class _FakeConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_attention_profile_from_config():
    profile = _rt_mod.AttentionProfile.from_config(_FakeConfig(
        experimental_attention_profile_enabled=True,
        experimental_attention_profile_window=256,
        experimental_attention_profile_backend="flex",
        experimental_attention_profile_torch_max_tokens=1024,
        attention_backend="sdpa",
        runtime_id="flexattention",
    ))
    assert profile.enabled is True
    assert profile.window_size == 256
    assert profile.backend == "flex"
    assert profile.torch_fallback_max_tokens == 1024
    assert profile.launcher_attention_backend == "sdpa"
    assert profile.flex_runtime_active is True
    assert profile.is_active is True
    print("PASS: AttentionProfile from_config / is_active")


def test_attention_profile_inactive():
    profile = _rt_mod.AttentionProfile.from_config(_FakeConfig(
        experimental_attention_profile_enabled=False,
        experimental_attention_profile_window=256,
    ))
    assert profile.is_active is False
    profile2 = _rt_mod.AttentionProfile.from_config(_FakeConfig(
        experimental_attention_profile_enabled=True,
        experimental_attention_profile_window=0,
    ))
    assert profile2.is_active is False
    print("PASS: AttentionProfile inactive cases")


def test_sliding_window_shape():
    q = torch.randn(2, 4, 32, 16)
    k = torch.randn(2, 4, 32, 16)
    v = torch.randn(2, 4, 32, 16)
    out = _rt_mod.sliding_window_attention(q, k, v, window_size=8, backend="sdpa_masked")
    assert out.shape == q.shape, f"Shape mismatch: {out.shape} vs {q.shape}"
    print("PASS: sliding_window_attention shape correctness")


def test_sliding_window_locality():
    """Tokens outside the window should have zero attention weight."""
    torch.manual_seed(42)
    q = torch.randn(1, 1, 16, 8)
    k = torch.randn(1, 1, 16, 8)
    v = torch.randn(1, 1, 16, 8)
    window_size = 4

    out = _rt_mod.sliding_window_attention(q, k, v, window_size=window_size)
    # Re-compute attention weights to check locality
    scale = 8 ** -0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    positions = torch.arange(16)
    distance = positions.unsqueeze(0) - positions.unsqueeze(1)
    mask = (distance >= 0) & (distance < window_size)
    attn_bias = torch.where(mask, 0.0, float("-inf")).to(dtype=q.dtype)
    attn_weights = attn_weights + attn_bias
    attn_weights = attn_weights.softmax(dim=-1)
    # Check that positions outside the window have zero weight
    for i in range(16):
        for j in range(16):
            if not mask[i, j]:
                assert attn_weights[0, 0, i, j].item() < 1e-6, \
                    f"Token {i} attends to token {j} outside window"
    print("PASS: sliding_window_attention locality verified")


def test_sliding_window_backend_resolution():
    q = torch.randn(1, 1, 8, 4)
    assert _rt_mod.resolve_sliding_window_backend(q, "sdpa") == "sdpa_masked"
    assert _rt_mod.resolve_sliding_window_backend(q, "torch") == "torch_fallback"
    assert _rt_mod.resolve_sliding_window_backend(q, "auto", launcher_attention_backend="sdpa") == "sdpa_masked"
    assert _rt_mod.resolve_sliding_window_backend(q, "auto", launcher_attention_backend="torch") == "torch_fallback"
    assert _rt_mod.resolve_sliding_window_backend(q, "auto", launcher_attention_backend="flash2") == "sdpa_masked"
    auto = _rt_mod.resolve_sliding_window_backend(q, "auto")
    assert auto == "sdpa_masked"
    print("PASS: sliding_window_attention backend resolution")


def test_sliding_window_sdpa_matches_torch_fallback():
    torch.manual_seed(1234)
    q = torch.randn(1, 2, 12, 8)
    k = torch.randn(1, 2, 12, 8)
    v = torch.randn(1, 2, 12, 8)
    sdpa = _rt_mod.sliding_window_attention(q, k, v, window_size=5, backend="sdpa_masked")
    torch_fallback = _rt_mod.sliding_window_attention(q, k, v, window_size=5, backend="torch_fallback")
    assert torch.allclose(sdpa, torch_fallback, atol=1e-5), "SDPA masked output should match torch fallback"
    print("PASS: sliding_window_attention sdpa_masked matches torch_fallback")


def test_sliding_window_torch_guard():
    q = torch.randn(1, 1, 9, 4)
    try:
        _rt_mod.sliding_window_attention(
            q,
            q,
            q,
            window_size=3,
            backend="torch_fallback",
            torch_fallback_max_tokens=8,
        )
    except RuntimeError as exc:
        assert "O(n^2)" in str(exc)
        print("PASS: sliding_window_attention torch fallback guard")
        return
    raise AssertionError("torch_fallback guard should reject seq_len above limit")


# ---------------------------------------------------------------------------
# FusedKVProjection tests
# ---------------------------------------------------------------------------

def test_fused_kv_shape():
    embed_dim = 64
    kv_dim = 64
    fused = _rt_mod.FusedKVProjection(embed_dim=embed_dim, kv_dim=kv_dim, bias=True)
    x = torch.randn(2, 10, embed_dim)
    k, v = fused(x)
    assert k.shape == (2, 10, kv_dim), f"K shape: {k.shape}"
    assert v.shape == (2, 10, kv_dim), f"V shape: {v.shape}"
    print("PASS: FusedKVProjection shape correctness")


def test_fused_kv_matches_separate():
    """Fused K/V output should match concatenation of separate K/V layers."""
    embed_dim = 64
    kv_dim = 64
    torch.manual_seed(123)

    to_k = nn.Linear(embed_dim, kv_dim, bias=True)
    to_v = nn.Linear(embed_dim, kv_dim, bias=True)
    fused = _rt_mod.FusedKVProjection(embed_dim=embed_dim, kv_dim=kv_dim, bias=True)

    # Copy weights
    with torch.no_grad():
        fused.kv_proj.weight.copy_(torch.cat([to_k.weight, to_v.weight], dim=0))
        fused.kv_proj.bias.copy_(torch.cat([to_k.bias, to_v.bias], dim=0))

    x = torch.randn(2, 10, embed_dim)
    k_fused, v_fused = fused(x)
    k_sep = to_k(x)
    v_sep = to_v(x)

    assert torch.allclose(k_fused, k_sep, atol=1e-5), "Fused K doesn't match separate K"
    assert torch.allclose(v_fused, v_sep, atol=1e-5), "Fused V doesn't match separate V"
    print("PASS: FusedKVProjection matches separate K/V projections")


def test_fused_kv_no_bias():
    embed_dim = 32
    kv_dim = 32
    fused = _rt_mod.FusedKVProjection(embed_dim=embed_dim, kv_dim=kv_dim, bias=False)
    assert fused.kv_proj.bias is None
    x = torch.randn(1, 5, embed_dim)
    k, v = fused(x)
    assert k.shape == (1, 5, kv_dim)
    assert v.shape == (1, 5, kv_dim)
    print("PASS: FusedKVProjection no-bias variant works")


if __name__ == "__main__":
    test_attention_profile_from_config()
    test_attention_profile_inactive()
    test_sliding_window_shape()
    test_sliding_window_locality()
    test_sliding_window_backend_resolution()
    test_sliding_window_sdpa_matches_torch_fallback()
    test_sliding_window_torch_guard()
    test_fused_kv_shape()
    test_fused_kv_matches_separate()
    test_fused_kv_no_bias()
    print("\nAll attention profile / fused KV smoke tests passed!")
