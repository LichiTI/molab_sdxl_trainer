# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for FlashAttention2 detection, fallback, and equivalence (Phase 9.3).

Validates:
- FA2 availability detection and version reporting
- FA2 vs SDPA numerical equivalence (when FA2 is available)
- FA2 fallback to SDPA when flash_attn is unavailable
- FA2 patch count on DiT attention modules
- FA2 detection reasons in RuntimeOptimizationPlan
"""

from __future__ import annotations

import sys
import importlib
import logging
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

logging.basicConfig(level=logging.DEBUG)

# ── import shims ───────────────────────────────────────────────────────

if "core" not in sys.modules:
    _stub = SimpleNamespace(__path__=[], __name__="core", __file__="")
    sys.modules["core"] = _stub
    sys.modules["core.lulynx_trainer"] = SimpleNamespace(
        __path__=["."], __name__="core.lulynx_trainer", __file__=""
    )

_rt_spec = importlib.util.spec_from_file_location(
    "runtime_optimizations", "runtime_optimizations.py"
)
_rt_mod = importlib.util.module_from_spec(_rt_spec)
sys.modules["runtime_optimizations"] = _rt_mod
_rt_spec.loader.exec_module(_rt_mod)

RuntimeOptimizationPlan = _rt_mod.RuntimeOptimizationPlan
build_runtime_optimization_plan = _rt_mod.build_runtime_optimization_plan

_attn_spec = importlib.util.spec_from_file_location(
    "anima_attention", "anima_attention.py"
)
_attn_mod = importlib.util.module_from_spec(_attn_spec)
sys.modules["anima_attention"] = _attn_mod
_attn_spec.loader.exec_module(_attn_mod)

dit_attention = _attn_mod.dit_attention
patch_anima_attention = _attn_mod.patch_anima_attention

# ── Check FA2 availability ─────────────────────────────────────────────

_FA2_AVAILABLE = False
_FA2_VERSION = None
try:
    import flash_attn
    _FA2_AVAILABLE = True
    _FA2_VERSION = getattr(flash_attn, "__version__", "unknown")
except ImportError:
    pass


# ── Fake DiT model ────────────────────────────────────────────────────

DIM = 32
NUM_HEADS = 4
HEAD_DIM = DIM // NUM_HEADS


class _FakeProjectionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(DIM, DIM, bias=False)
        self.k_proj = nn.Linear(DIM, DIM, bias=False)
        self.v_proj = nn.Linear(DIM, DIM, bias=False)
        self.output_proj = nn.Linear(DIM, DIM, bias=False)
        self.num_heads = NUM_HEADS
        self.head_dim = HEAD_DIM
        self.hidden_dim = DIM

    def q_norm(self, x):
        return x

    def k_norm(self, x):
        return x

    def _split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(self, x, context=None):
        source = x if context is None else context
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(source))
        v = self._split_heads(self.v_proj(source))
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.output_proj(self._merge_heads(attn))


class _FakeDiT(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.self_attn_modules = nn.ModuleList([_FakeProjectionAttention() for _ in range(n)])


# ── Tests ──────────────────────────────────────────────────────────────


def test_fa2_available_detection():
    """Report FA2 availability and version."""
    if _FA2_AVAILABLE:
        print(f"[INFO] FlashAttention2 is AVAILABLE, version={_FA2_VERSION}")
    else:
        print("[INFO] FlashAttention2 is NOT available (expected in CPU-only environments)")
    print("[PASS] FA2 availability detection")


def test_fa2_vs_sdpa_equivalence():
    """FA2 and SDPA produce similar results (when FA2 is available)."""
    if not _FA2_AVAILABLE:
        print("[SKIP] FA2 vs SDPA equivalence (flash_attn not installed)")
        return

    if not torch.cuda.is_available():
        print("[SKIP] FA2 vs SDPA equivalence (CUDA not available)")
        return

    B, H, T, D = 1, 4, 16, 32
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

    out_sdpa = dit_attention(q, k, v, backend="sdpa")
    out_fa2 = dit_attention(q, k, v, backend="flash2")

    assert out_sdpa.shape == out_fa2.shape, "Shape mismatch"
    max_diff = (out_sdpa - out_fa2).abs().max().item()
    print(f"[INFO] FA2 vs SDPA max diff: {max_diff:.6f}")
    assert max_diff < 0.05, f"FA2 vs SDPA diff too large: {max_diff}"
    print("[PASS] FA2 vs SDPA equivalence")


def test_fa2_fallback_when_unavailable():
    """When flash_attn is not importable, dit_attention(backend='flash2') falls back to SDPA."""
    B, H, T, D = 1, 2, 8, 16
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    # Force flash_attn to be unimportable
    saved = sys.modules.get("flash_attn")
    sys.modules["flash_attn"] = None
    try:
        out = dit_attention(q, k, v, backend="flash2")
        assert out.shape == (B, H, T, D), f"Unexpected shape {out.shape}"
        assert out.isfinite().all(), "Non-finite output from fallback"
    finally:
        if saved is not None:
            sys.modules["flash_attn"] = saved
        else:
            sys.modules.pop("flash_attn", None)

    print("[PASS] FA2 fallback to SDPA when unavailable")


def test_fa2_patch_count():
    """patch_anima_attention with backend='flash2' patches correct number of modules."""
    model = _FakeDiT(3)
    count = patch_anima_attention(model, backend="flash2")
    assert count == 3, f"Expected 3 patched, got {count}"

    for m in model.self_attn_modules:
        assert getattr(m, "_attention_backend", None) == "flash2"

    print("[PASS] FA2 patch count correct")


def test_fa2_plan_reasons_available():
    """When FA2 is available and auto-selected, plan.reasons includes FA2 info."""
    config = SimpleNamespace(
        attention_backend="auto", torch_compile=False,
        torch_compile_backend="inductor", torch_compile_mode="default",
        torch_compile_dynamic=False, torch_compile_fullgraph=False,
        torch_compile_scope="", anima_compile_scope="",
        anima_split_attn=False, anima_split_attn_chunks=0,
        xformers=False, sdpa=False, use_sdpa=False,
        anima_fixed_text_tokens=0, anima_fixed_visual_tokens=0,
    )
    plan = build_runtime_optimization_plan(config)

    if _FA2_AVAILABLE:
        assert plan.attention_backend == "flash2", f"Expected flash2, got {plan.attention_backend}"
        assert any("flash2" in r for r in plan.reasons), "Should have flash2 reason"
        print("[PASS] FA2 plan reasons (available)")
    else:
        assert plan.attention_backend == "sdpa", f"Expected sdpa fallback, got {plan.attention_backend}"
        assert any("FA2" in r or "unavailable" in r for r in plan.reasons), "Should have fallback reason"
        print("[PASS] FA2 plan reasons (unavailable, fallback)")


def test_fa2_plan_reasons_explicit_request_unavailable():
    """Explicitly requesting flash2 when unavailable produces proper fallback reason."""
    saved = sys.modules.get("flash_attn")
    sys.modules["flash_attn"] = None
    try:
        # Clear importlib cache
        importlib.invalidate_caches()
        config = SimpleNamespace(
            attention_backend="flash2", torch_compile=False,
            torch_compile_backend="inductor", torch_compile_mode="default",
            torch_compile_dynamic=False, torch_compile_fullgraph=False,
            torch_compile_scope="", anima_compile_scope="",
            anima_split_attn=False, anima_split_attn_chunks=0,
            anima_fixed_text_tokens=0, anima_fixed_visual_tokens=0,
        )
        plan = build_runtime_optimization_plan(config)
        assert plan.attention_backend == "sdpa", f"Should fall back to sdpa, got {plan.attention_backend}"
        assert any("flash_attn" in w for w in plan.warnings), "Should warn about missing flash_attn"
        assert any("fa2_unavailable" in r for r in plan.reasons), f"Should have fa2_unavailable reason, got {plan.reasons}"
    finally:
        if saved is not None:
            sys.modules["flash_attn"] = saved
        else:
            sys.modules.pop("flash_attn", None)

    print("[PASS] FA2 explicit request unavailable fallback")


if __name__ == "__main__":
    test_fa2_available_detection()
    test_fa2_vs_sdpa_equivalence()
    test_fa2_fallback_when_unavailable()
    test_fa2_patch_count()
    test_fa2_plan_reasons_available()
    test_fa2_plan_reasons_explicit_request_unavailable()
    print("\n[PASS] All Phase 9.3 FA2 proof smoke tests passed")
