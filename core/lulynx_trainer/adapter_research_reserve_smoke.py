# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test: P3 adapter-research reserve seam on the real Anima subset.

Proves the seam is a genuine default-off opt-in reserve by swapping adapters
into an ``AnimaNativeExecutableSubset`` and driving a full model forward:

* ``method="none"`` swaps nothing and is bitwise-parity.
* ``step_expert`` / ``chimera_hydra`` swap target linears, are identity at init,
  really change the output once trained (step routing for step_expert), flow
  gradients, and restore exactly on ``handle.remove()``.
* ``apply_soft_tokens_reserve`` is parity with no bank and really prepends tokens
  (with gradient) when given one.

Run:
    backend/env/python-flashattention/python.exe \
        backend/core/lulynx_trainer/adapter_research_reserve_smoke.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import torch

from core.lulynx_trainer.anima_native_dit import AnimaNativeExecutableSubset
from core.lulynx_trainer.soft_tokens import SoftTokenBank, SoftTokenConfig
from core.lulynx_trainer.adapter_research_reserve_seam import (
    adapter_research_reserve_readiness,
    adapter_research_step_context,
    apply_soft_tokens_reserve,
    install_adapter_research_reserve,
)

HIDDEN = 8
HEAD_DIM = 4


def _subset_shapes(n_blocks):
    shapes = {
        "net.x_embedder.proj.1.weight": (HIDDEN, 64),
        "net.t_embedding_norm.weight": (HIDDEN,),
        "net.t_embedder.1.linear_1.weight": (HIDDEN, HIDDEN),
        "net.t_embedder.1.linear_2.weight": (3 * HIDDEN, HIDDEN),
        "net.final_layer.linear.weight": (64, HIDDEN),
        "net.final_layer.adaln_modulation.1.weight": (HIDDEN, HIDDEN),
        "net.final_layer.adaln_modulation.2.weight": (2 * HIDDEN, HIDDEN),
    }
    for index in range(n_blocks):
        prefix = f"net.blocks.{index}"
        for attn in ("self_attn", "cross_attn"):
            for proj in ("q_proj", "k_proj", "v_proj", "output_proj"):
                shapes[f"{prefix}.{attn}.{proj}.weight"] = (HIDDEN, HIDDEN)
            shapes[f"{prefix}.{attn}.q_norm.weight"] = (HEAD_DIM,)
            shapes[f"{prefix}.{attn}.k_norm.weight"] = (HEAD_DIM,)
        shapes[f"{prefix}.mlp.layer1.weight"] = (2 * HIDDEN, HIDDEN)
        shapes[f"{prefix}.mlp.layer2.weight"] = (HIDDEN, 2 * HIDDEN)
        for branch in ("self_attn", "cross_attn", "mlp"):
            shapes[f"{prefix}.adaln_modulation_{branch}.1.weight"] = (HIDDEN, HIDDEN)
            shapes[f"{prefix}.adaln_modulation_{branch}.2.weight"] = (3 * HIDDEN, HIDDEN)
    return shapes


def _build_subset(n_blocks):
    return AnimaNativeExecutableSubset(
        _subset_shapes(n_blocks),
        block_indices=tuple(range(n_blocks)),
        device="cpu",
        dtype=torch.float32,
    )


def _inputs():
    return torch.randn(1, 15, 4, 4), torch.ones(1), torch.randn(1, 6, HIDDEN)


def _rel_drift(a, b):
    return float((a - b).norm() / (b.norm() + 1e-8))


def _randomize(handle):
    with torch.no_grad():
        for param in handle.trainable_parameters():
            param.copy_(torch.randn_like(param) * 0.3)


def test_none_is_bitwise_parity():
    torch.manual_seed(0)
    model = _build_subset(2)
    sample, ts, text = _inputs()
    with torch.no_grad():
        baseline = model(sample, ts, text).sample
    handle = install_adapter_research_reserve(model, "none")
    assert handle.wrapped_count == 0, handle.wrapped_count
    with torch.no_grad():
        out = model(sample, ts, text).sample
    assert torch.equal(baseline, out), "method=none must be bitwise parity"
    handle.remove()
    print("PASS: method=none swaps nothing and is bitwise parity")


def test_step_expert_identity_routing_grad_restore():
    torch.manual_seed(1)
    model = _build_subset(2)
    sample, ts, text = _inputs()
    with torch.no_grad():
        baseline = model(sample, ts, text).sample

    handle = install_adapter_research_reserve(model, "step_expert")
    try:
        assert handle.wrapped_count > 0, "step_expert swapped nothing"
        with torch.no_grad():
            init_out = model(sample, ts, text).sample
        assert torch.equal(baseline, init_out), "zero-init step_expert must be identity"

        _randomize(handle)
        with torch.no_grad():
            with adapter_research_step_context(0.1):
                early = model(sample, ts, text).sample
            with adapter_research_step_context(0.9):
                late = model(sample, ts, text).sample
        assert _rel_drift(early, baseline) > 1e-4, "trained step_expert should change the output"
        assert _rel_drift(early, late) > 1e-5, "different steps must route to different experts"

        out = model(sample, ts, text).sample
        out.square().mean().backward()
        grads = [p.grad for p in handle.trainable_parameters() if p.grad is not None]
        assert grads and any(float(g.abs().sum()) > 0 for g in grads), "no gradient into step_expert"
    finally:
        handle.remove()
    with torch.no_grad():
        restored = model(sample, ts, text).sample
    assert torch.equal(baseline, restored), "remove() did not restore originals"
    print(f"PASS: step_expert identity@init, routes by step, grads flow, restores ({handle.wrapped_count} swaps)")


def test_chimera_hydra_identity_grad_restore():
    torch.manual_seed(2)
    model = _build_subset(1)
    sample, ts, text = _inputs()
    with torch.no_grad():
        baseline = model(sample, ts, text).sample

    handle = install_adapter_research_reserve(model, "chimera_hydra")
    try:
        assert handle.wrapped_count > 0
        with torch.no_grad():
            init_out = model(sample, ts, text).sample
        assert torch.equal(baseline, init_out), "zero-init chimera_hydra must be identity"

        _randomize(handle)
        out = model(sample, ts, text).sample
        assert _rel_drift(out.detach(), baseline) > 1e-4, "trained chimera_hydra should change the output"
        out.square().mean().backward()
        grads = [p.grad for p in handle.trainable_parameters() if p.grad is not None]
        assert grads and any(float(g.abs().sum()) > 0 for g in grads), "no gradient into chimera_hydra"
    finally:
        handle.remove()
    with torch.no_grad():
        restored = model(sample, ts, text).sample
    assert torch.equal(baseline, restored)
    print(f"PASS: chimera_hydra identity@init, grads flow, restores ({handle.wrapped_count} swaps)")


def test_soft_tokens_reserve_parity_and_prepend():
    torch.manual_seed(3)
    text = torch.randn(1, 6, HIDDEN)
    off = apply_soft_tokens_reserve(None, text, layer_index=0, timestep=0.5)
    assert torch.equal(off.embeddings, text) and off.prepended_tokens == 0, "no-bank soft tokens must be parity"

    bank = SoftTokenBank(SoftTokenConfig(num_tokens=3, hidden_size=HIDDEN, layer_ids=(0,)))
    res = apply_soft_tokens_reserve(bank, text, layer_index=0, timestep=0.5, total_steps=10)
    assert res.prepended_tokens == 3, res.prepended_tokens
    assert res.embeddings.shape[1] == text.shape[1] + 3, res.embeddings.shape
    res.embeddings.square().mean().backward()
    assert bank.tokens.grad is not None and float(bank.tokens.grad.abs().sum()) > 0, "no gradient into soft tokens"
    print("PASS: soft-token reserve is parity with no bank and really prepends (with gradient)")


def test_readiness_flags():
    report = adapter_research_reserve_readiness()
    assert report["wired"] is True
    assert report["default_method"] == "none"
    assert set(report["linear_methods"]) == {"step_expert", "chimera_hydra"}
    assert report["training_step_consumption"] is False
    assert report["promotion_ready"] is False
    assert report["default_behavior_changed"] is False
    print("PASS: readiness reports wired reserves with training/promotion gates held False")


def main():
    test_none_is_bitwise_parity()
    test_step_expert_identity_routing_grad_restore()
    test_chimera_hydra_identity_grad_restore()
    test_soft_tokens_reserve_parity_and_prepend()
    test_readiness_flags()
    print("\n[adapter_research_reserve_smoke] 5/5 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
