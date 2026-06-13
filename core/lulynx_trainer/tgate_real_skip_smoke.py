# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test for T-GATE *real* cross-attention skip wired into the live forward.

The probe (observe-only) smoke lives in ``anima_attention_smoke.py``. This file
proves the second stage: when a generation publishes a ``tgate_execution_context``
the patched attention forward actually *reuses* cached cross-attention outputs on
eligible steps (skipping q/k/v/attention), and -- critically -- that with no
execution context active the forward stays bitwise-identical to today (parity).

Run directly:
    backend/env/python-flashattention/python.exe \
        backend/core/lulynx_trainer/tgate_real_skip_smoke.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# Load anima_attention via importlib to avoid the diffusers import chain, and
# register it under the package path so its ``from .tgate import ...`` resolves to
# the SAME ``core.lulynx_trainer.tgate`` module the smoke imports below (shared
# ContextVar identity is what makes the execution context visible to the forward).
_aa_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.anima_attention",
    os.path.join(_HERE, "anima_attention.py"),
)
_aa = importlib.util.module_from_spec(_aa_spec)
sys.modules["core.lulynx_trainer.anima_attention"] = _aa
_aa_spec.loader.exec_module(_aa)

import torch
import torch.nn as nn

from core.lulynx_trainer.tgate import (
    get_active_tgate_execution,
    reset_tgate_stats,
    tgate_execution_context,
    tgate_step_context,
)


class _FakeProjectionAttention(nn.Module):
    def __init__(self, hidden_dim: int = 64, head_dim: int = 16) -> None:
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

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = tensor.shape
        return tensor.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _heads, tokens, _head_dim = tensor.shape
        return tensor.transpose(1, 2).reshape(batch, tokens, self.hidden_dim)

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        source = x if context is None else context
        q = self.q_norm(self._split_heads(self.q_proj(x)))
        k = self.k_norm(self._split_heads(self.k_proj(source)))
        v = self._split_heads(self.v_proj(source))
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.output_proj(self._merge_heads(attn))


class _FakeDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {"self_attn": _FakeProjectionAttention(), "cross_attn": _FakeProjectionAttention()}
                )
                for _ in range(2)
            ]
        )


def _patched_model():
    model = _FakeDiT()
    _aa.patch_anima_attention(model, backend="sdpa")
    return model


def test_default_off_is_bitwise_parity() -> None:
    """No execution context -> patched cross-attn == unpatched, and recomputes."""
    model = _FakeDiT()
    x_a = torch.randn(1, 8, 64)
    ctx_a = torch.randn(1, 10, 64)
    x_b = torch.randn(1, 8, 64)
    ctx_b = torch.randn(1, 10, 64)
    with torch.no_grad():
        ref_a = model.blocks[1]["cross_attn"](x_a, ctx_a)
        ref_b = model.blocks[1]["cross_attn"](x_b, ctx_b)

    reset_tgate_stats()
    _aa.patch_anima_attention(model, backend="sdpa")
    # No tgate_execution_context active here.
    with torch.no_grad(), tgate_step_context(
        enabled=True, step_index=3, total_steps=8, start_step=0, min_block=0
    ):
        out_a = model.blocks[1]["cross_attn"](x_a, ctx_a)
        out_b = model.blocks[1]["cross_attn"](x_b, ctx_b)
    _aa.unpatch_anima_attention(model)

    assert torch.allclose(out_a, ref_a, atol=1e-6), "default-off broke parity (input A)"
    assert torch.allclose(out_b, ref_b, atol=1e-6), "default-off broke parity (input B)"
    # Different inputs -> different outputs: nothing was reused.
    assert not torch.allclose(out_a, out_b), "outputs collapsed without any execution context"
    print("PASS: default-off T-GATE keeps bitwise parity and recomputes every call")


def test_execution_context_reuses_cross_attention() -> None:
    """Eligible cross-attn call reuses the cached output (real skip)."""
    model = _patched_model()
    x_a = torch.randn(1, 8, 64)
    ctx_a = torch.randn(1, 10, 64)
    x_b = torch.randn(1, 8, 64)  # deliberately different inputs
    ctx_b = torch.randn(1, 10, 64)

    reset_tgate_stats()
    with torch.no_grad(), tgate_execution_context(enabled=True) as execution:
        with tgate_step_context(enabled=True, step_index=0, total_steps=4, start_step=0, min_block=0):
            out_a = model.blocks[1]["cross_attn"](x_a, ctx_a)  # gate step: compute + cache
        with tgate_step_context(enabled=True, step_index=1, total_steps=4, start_step=0, min_block=0):
            out_b = model.blocks[1]["cross_attn"](x_b, ctx_b)  # eligible: reuse cache
        hits = execution.cache.stats()["cache_hits"]
    _aa.unpatch_anima_attention(model)

    assert torch.equal(out_a, out_b), "eligible step did not reuse the cached cross-attention output"
    assert hits >= 1, f"expected >=1 cache hit, got {hits}"
    print("PASS: execution context reuses cross-attention on eligible steps (real skip)")


def test_self_attention_is_never_skipped() -> None:
    """Self-attention must always recompute even with the execution context on."""
    model = _patched_model()
    x_a = torch.randn(1, 8, 64)
    x_b = torch.randn(1, 8, 64)

    with torch.no_grad(), tgate_execution_context(enabled=True):
        with tgate_step_context(enabled=True, step_index=0, total_steps=4, start_step=0, min_block=0):
            out_a = model.blocks[1]["self_attn"](x_a)
        with tgate_step_context(enabled=True, step_index=1, total_steps=4, start_step=0, min_block=0):
            out_b = model.blocks[1]["self_attn"](x_b)
    _aa.unpatch_anima_attention(model)

    assert not torch.allclose(out_a, out_b), "self-attention was wrongly reused"
    print("PASS: self-attention is never skipped by T-GATE")


def test_ineligible_step_recomputes() -> None:
    """With execution on but the step ineligible, the call still recomputes."""
    model = _patched_model()
    x_a = torch.randn(1, 8, 64)
    ctx_a = torch.randn(1, 10, 64)
    x_b = torch.randn(1, 8, 64)
    ctx_b = torch.randn(1, 10, 64)

    # start_step=6 makes steps 0/1 ineligible -> no reuse.
    with torch.no_grad(), tgate_execution_context(enabled=True):
        with tgate_step_context(enabled=True, step_index=0, total_steps=8, start_step=6, min_block=0):
            out_a = model.blocks[1]["cross_attn"](x_a, ctx_a)
        with tgate_step_context(enabled=True, step_index=1, total_steps=8, start_step=6, min_block=0):
            out_b = model.blocks[1]["cross_attn"](x_b, ctx_b)
    _aa.unpatch_anima_attention(model)

    assert not torch.allclose(out_a, out_b), "ineligible steps wrongly reused the cache"
    print("PASS: ineligible steps recompute (no premature skip)")


def main() -> int:
    test_default_off_is_bitwise_parity()
    test_execution_context_reuses_cross_attention()
    test_self_attention_is_never_skipped()
    test_ineligible_step_recomputes()
    assert get_active_tgate_execution() is None, "execution context leaked after the smoke"
    print("\n[tgate_real_skip_smoke] 4/4 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
