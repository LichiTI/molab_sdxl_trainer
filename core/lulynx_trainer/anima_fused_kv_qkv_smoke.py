# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Anima DiT fused KV and fused QKV projections (Phase 9.1).

Validates:
- FusedKVProjection equivalence for cross-attention
- FusedQKVProjection equivalence for self-attention
- apply_anima_fused_kv targets only cross_attn modules
- apply_anima_fused_qkv targets only self_attn modules
- LoRA-wrapped layers are skipped
- Dispatched forward uses fused projections when available
"""

from __future__ import annotations

import sys
import importlib
from collections import OrderedDict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
from torch import nn


# ── minimal runtime_optimizations import shim ──────────────────────────

TRAINER_ROOT = Path(__file__).resolve().parent
CORE_ROOT = TRAINER_ROOT.parent

if "core" not in sys.modules:
    _stub = ModuleType("core")
    _stub.__path__ = [str(CORE_ROOT)]
    _stub.__file__ = ""
    sys.modules["core"] = _stub
if "core.lulynx_trainer" not in sys.modules:
    _pkg = ModuleType("core.lulynx_trainer")
    _pkg.__path__ = [str(TRAINER_ROOT)]
    _pkg.__file__ = ""
    sys.modules["core.lulynx_trainer"] = _pkg

_rt_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.runtime_optimizations", TRAINER_ROOT / "runtime_optimizations.py"
)
_rt_mod = importlib.util.module_from_spec(_rt_spec)
sys.modules["core.lulynx_trainer.runtime_optimizations"] = _rt_mod
_rt_spec.loader.exec_module(_rt_mod)

FusedKVProjection = _rt_mod.FusedKVProjection
FusedQKVProjection = _rt_mod.FusedQKVProjection
RuntimeOptimizationPlan = _rt_mod.RuntimeOptimizationPlan
apply_anima_fused_kv = _rt_mod.apply_anima_fused_kv
apply_anima_fused_qkv = _rt_mod.apply_anima_fused_qkv
_is_lora_wrapped = _rt_mod._is_lora_wrapped
normalize_fused_projection_memory_mode = _rt_mod.normalize_fused_projection_memory_mode

_attn_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.anima_attention", TRAINER_ROOT / "anima_attention.py"
)
_attn_mod = importlib.util.module_from_spec(_attn_spec)
sys.modules["core.lulynx_trainer.anima_attention"] = _attn_mod
_attn_spec.loader.exec_module(_attn_mod)
patch_anima_attention = _attn_mod.patch_anima_attention


# ── Fake DiT model with self_attn / cross_attn blocks ─────────────────

DIM = 16
COND_DIM = 8


class _FakeAttention(nn.Module):
    def __init__(self, in_dim, kv_dim=None):
        super().__init__()
        kv_in = kv_dim if kv_dim is not None else in_dim
        self.q_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.k_proj = nn.Linear(kv_in, in_dim, bias=False)
        self.v_proj = nn.Linear(kv_in, in_dim, bias=False)
        self.output_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.num_heads = 2
        self.head_dim = in_dim // 2
        self.hidden_dim = in_dim

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
        scale = self.head_dim ** -0.5
        attn = torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1)
        out = attn @ v
        return self.output_proj(self._merge_heads(out))


class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttention(DIM)
        self.cross_attn = _FakeAttention(DIM, kv_dim=COND_DIM)


class _FakeDiT(nn.Module):
    def __init__(self, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlock() for _ in range(n_blocks)])


class _FakeModel:
    def __init__(self, n_blocks=2):
        self.unet = _FakeDiT(n_blocks)


# ── Tests ──────────────────────────────────────────────────────────────


def _make_plan():
    return RuntimeOptimizationPlan(attention_backend="sdpa", requested_attention_backend="sdpa")


def test_fused_kv_equivalence():
    """Fused KV produces the same output as separate K/V projections."""
    model = _FakeModel(2)
    plan = _make_plan()

    x = torch.randn(1, 4, DIM)
    ctx = torch.randn(1, 6, COND_DIM)

    # Get reference outputs from cross_attn before fusion
    refs = []
    for block in model.unet.blocks:
        with torch.no_grad():
            refs.append(block.cross_attn(x, context=ctx).clone())

    count = apply_anima_fused_kv(model, plan)
    assert count == 2, f"Expected 2 fused, got {count}"

    # Verify _fused_kv is set on cross_attn, not self_attn
    for block in model.unet.blocks:
        assert hasattr(block.cross_attn, "_fused_kv")
        assert not hasattr(block.self_attn, "_fused_kv")

    # Patch attention so dispatched forward uses _fused_kv
    patch_anima_attention(model.unet, backend="sdpa")

    # Forward with fused should match reference
    for i, block in enumerate(model.unet.blocks):
        with torch.no_grad():
            out = block.cross_attn(x, context=ctx)
        assert torch.allclose(out, refs[i], atol=1e-5), (
            f"Block {i} cross_attn fused KV output mismatch"
        )

    assert any("anima_fused_kv" in r for r in plan.reasons)
    print("[PASS] Fused KV equivalence")


def test_fused_qkv_equivalence():
    """Fused QKV produces the same output as separate Q/K/V projections."""
    model = _FakeModel(2)
    plan = _make_plan()

    x = torch.randn(1, 4, DIM)

    # Get reference outputs from self_attn before fusion
    refs = []
    for block in model.unet.blocks:
        with torch.no_grad():
            refs.append(block.self_attn(x).clone())

    count = apply_anima_fused_qkv(model, plan)
    assert count == 2, f"Expected 2 fused, got {count}"

    for block in model.unet.blocks:
        assert hasattr(block.self_attn, "_fused_qkv")
        assert not hasattr(block.cross_attn, "_fused_qkv")

    patch_anima_attention(model.unet, backend="sdpa")

    for i, block in enumerate(model.unet.blocks):
        with torch.no_grad():
            out = block.self_attn(x)
        assert torch.allclose(out, refs[i], atol=1e-5), (
            f"Block {i} self_attn fused QKV output mismatch"
        )

    assert any("anima_fused_qkv" in r for r in plan.reasons)
    print("[PASS] Fused QKV equivalence")


def test_lora_wrapped_skipped():
    """LoRA-wrapped projections are skipped by fused apply."""
    model = _FakeModel(1)
    plan = _make_plan()

    # Simulate LoRA wrapping by adding lora_down attribute
    block = model.unet.blocks[0]
    block.self_attn.q_proj.lora_down = nn.Linear(4, 4)

    count = apply_anima_fused_qkv(model, plan)
    assert count == 0, f"Should skip LoRA-wrapped, got {count}"
    assert any("skipped" in w for w in plan.warnings)
    print("[PASS] LoRA-wrapped layers skipped")


def test_fused_kv_does_not_touch_self_attn():
    """apply_anima_fused_kv only touches cross_attn, not self_attn."""
    model = _FakeModel(2)
    plan = _make_plan()

    apply_anima_fused_kv(model, plan)

    for block in model.unet.blocks:
        assert not hasattr(block.self_attn, "_fused_kv"), "self_attn should not get _fused_kv"
    print("[PASS] Fused KV does not touch self_attn")


def test_fused_qkv_does_not_touch_cross_attn():
    """apply_anima_fused_qkv only touches self_attn, not cross_attn."""
    model = _FakeModel(2)
    plan = _make_plan()

    apply_anima_fused_qkv(model, plan)

    for block in model.unet.blocks:
        assert not hasattr(block.cross_attn, "_fused_qkv"), "cross_attn should not get _fused_qkv"
    print("[PASS] Fused QKV does not touch cross_attn")


def test_state_dict_roundtrip():
    """Fused projections survive save/load cycle."""
    model = _FakeModel(1)
    plan = _make_plan()

    apply_anima_fused_kv(model, plan)
    apply_anima_fused_qkv(model, plan)

    sd = model.unet.state_dict()
    fused_kv_keys = [k for k in sd if "_fused_kv" in k]
    fused_qkv_keys = [k for k in sd if "_fused_qkv" in k]
    assert len(fused_kv_keys) > 0, "No _fused_kv keys in state dict"
    assert len(fused_qkv_keys) > 0, "No _fused_qkv keys in state dict"

    # Load into fresh model
    model2 = _FakeModel(1)
    plan2 = _make_plan()
    apply_anima_fused_kv(model2, plan2)
    apply_anima_fused_qkv(model2, plan2)
    model2.unet.load_state_dict(sd)

    # Verify finite forward
    x = torch.randn(1, 4, DIM)
    patch_anima_attention(model2.unet, backend="sdpa")
    with torch.no_grad():
        out = model2.unet.blocks[0].self_attn(x)
    assert out.isfinite().all(), "Non-finite output after state dict load"
    print("[PASS] State dict roundtrip")


def test_fused_projection_memory_mode_normalization():
    assert normalize_fused_projection_memory_mode("auto") == "keep_original"
    assert normalize_fused_projection_memory_mode("drop") == "drop_original"
    assert normalize_fused_projection_memory_mode("materialize") == "materialize_on_save"
    assert normalize_fused_projection_memory_mode("unknown") == "keep_original"
    print("[PASS] Fused projection memory mode normalization")


def test_drop_original_memory_mode_uses_fused_forward():
    model = _FakeModel(1)
    plan = _make_plan()
    x = torch.randn(1, 4, DIM)
    ctx = torch.randn(1, 6, COND_DIM)
    ref_cross = model.unet.blocks[0].cross_attn(x, context=ctx).detach()
    ref_self = model.unet.blocks[0].self_attn(x).detach()

    apply_anima_fused_kv(model, plan, memory_mode="drop_original")
    apply_anima_fused_qkv(model, plan, memory_mode="drop_original")

    block = model.unet.blocks[0]
    assert block.cross_attn.k_proj is None
    assert block.cross_attn.v_proj is None
    assert block.self_attn.q_proj is None
    assert block.self_attn.k_proj is None
    assert block.self_attn.v_proj is None

    patch_anima_attention(model.unet, backend="sdpa")
    with torch.no_grad():
        out_cross = block.cross_attn(x, context=ctx)
        out_self = block.self_attn(x)
    assert torch.allclose(out_cross, ref_cross, atol=1e-5)
    assert torch.allclose(out_self, ref_self, atol=1e-5)
    print("[PASS] Drop-original memory mode uses fused forward")


def test_materialize_on_save_state_dict_adds_original_keys():
    model = _FakeModel(1)
    plan = _make_plan()

    apply_anima_fused_kv(model, plan, memory_mode="materialize_on_save")
    apply_anima_fused_qkv(model, plan, memory_mode="materialize_on_save")

    block = model.unet.blocks[0]
    assert block.cross_attn.k_proj is None
    assert block.self_attn.q_proj is None

    sd = model.unet.state_dict()
    assert "blocks.0.cross_attn.k_proj.weight" in sd
    assert "blocks.0.cross_attn.v_proj.weight" in sd
    assert "blocks.0.self_attn.q_proj.weight" in sd
    assert "blocks.0.self_attn.k_proj.weight" in sd
    assert "blocks.0.self_attn.v_proj.weight" in sd
    assert "blocks.0.cross_attn._fused_kv.kv_proj.weight" in sd
    assert "blocks.0.self_attn._fused_qkv.qkv_proj.weight" in sd
    print("[PASS] Materialize-on-save state dict adds original projection keys")


if __name__ == "__main__":
    test_fused_kv_equivalence()
    test_fused_qkv_equivalence()
    test_lora_wrapped_skipped()
    test_fused_kv_does_not_touch_self_attn()
    test_fused_qkv_does_not_touch_cross_attn()
    test_state_dict_roundtrip()
    test_fused_projection_memory_mode_normalization()
    test_drop_original_memory_mode_uses_fused_forward()
    test_materialize_on_save_state_dict_adds_original_keys()
    print("\n[PASS] All Phase 9.1 fused KV/QKV smoke tests passed")
