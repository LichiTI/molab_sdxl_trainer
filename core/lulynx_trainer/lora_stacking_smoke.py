# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke for LoRA + OrthoLoRA + T-LoRA stacking (Phase 8.2 / #110).

Verifies that multiple LoRA variants can co-exist on the same model
without interfering: each maintains its own parameters, gradients, and
projection schedule.

The test does NOT need full LoRA injectors — we use minimal stand-in
adapters that share the same wrapper interface so the OrthoLoRA
projector and T-LoRA scheduler can operate on them.
"""

from __future__ import annotations

import os
import sys
import importlib.util

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))


def _import_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ol = _import_module("core.lulynx_trainer.ortholora", "ortholora.py")


# ---------------------------------------------------------------------------
# Minimal LoRA-like wrappers so we can simulate three coexisting adapter types
# ---------------------------------------------------------------------------

class _StandardLoRA(nn.Module):
    """Standard LoRA: down + up, lora_up zero-init."""
    def __init__(self, dim=16, rank=4, alpha=4.0):
        super().__init__()
        self.lora_down = nn.Linear(dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)
        self.scaling = alpha / rank

    def forward(self, x):
        return self.lora_up(self.lora_down(x)) * self.scaling


class _OrthoLoRA(nn.Module):
    """LoRA whose down/up matrices get re-orthogonalised every K steps."""
    def __init__(self, dim=16, rank=4, alpha=4.0):
        super().__init__()
        self.lora_down = nn.Linear(dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.lora_down.weight, std=0.02)
        nn.init.normal_(self.lora_up.weight, std=0.02)
        self.scaling = alpha / rank

    def forward(self, x):
        return self.lora_up(self.lora_down(x)) * self.scaling


class _TLoRA(nn.Module):
    """T-LoRA: rank schedule grows with global_step."""
    def __init__(self, dim=16, max_rank=8, alpha=8.0):
        super().__init__()
        self.lora_down = nn.Linear(dim, max_rank, bias=False)
        self.lora_up = nn.Linear(max_rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)
        self.max_rank = max_rank
        self.scaling = alpha / max_rank
        self.active_rank = 1  # starts small, grows with steps

    def set_active_rank(self, step: int, total_steps: int = 100) -> None:
        ratio = min(step / max(total_steps, 1), 1.0)
        self.active_rank = max(1, int(round(self.max_rank * ratio)))

    def forward(self, x):
        # Mask out unused rank dimensions
        down_w = self.lora_down.weight[:self.active_rank]
        up_w = self.lora_up.weight[:, :self.active_rank]
        h = x @ down_w.t()
        return (h @ up_w.t()) * self.scaling


class _StackedModel(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.base = nn.Linear(dim, dim)
        for p in self.base.parameters():
            p.requires_grad = False
        self.standard = _StandardLoRA(dim=dim, rank=4)
        self.ortho = _OrthoLoRA(dim=dim, rank=4)
        self.tlora = _TLoRA(dim=dim, max_rank=8)

    def forward(self, x):
        return (
            self.base(x)
            + self.standard(x)
            + self.ortho(x)
            + self.tlora(x)
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_stacked_model_forward_shape():
    model = _StackedModel(dim=16)
    x = torch.randn(2, 16)
    out = model(x)
    assert out.shape == (2, 16)
    print("PASS: stacked LoRA model produces correct output shape")


def test_each_variant_has_independent_params():
    model = _StackedModel(dim=16)
    s_params = list(model.standard.parameters())
    o_params = list(model.ortho.parameters())
    t_params = list(model.tlora.parameters())
    # No parameter is shared between variants
    s_ids = {id(p) for p in s_params}
    o_ids = {id(p) for p in o_params}
    t_ids = {id(p) for p in t_params}
    assert s_ids.isdisjoint(o_ids)
    assert s_ids.isdisjoint(t_ids)
    assert o_ids.isdisjoint(t_ids)
    print("PASS: each LoRA variant maintains independent parameters")


def test_base_weights_remain_frozen():
    model = _StackedModel(dim=16)
    base_params = list(model.base.parameters())
    assert all(not p.requires_grad for p in base_params)
    print("PASS: base layer parameters are frozen")


def test_ortho_projector_only_affects_ortho_lora():
    torch.manual_seed(0)
    model = _StackedModel(dim=16)

    # Snapshot standard + tlora weights before projection
    standard_down_before = model.standard.lora_down.weight.clone()
    tlora_down_before = model.tlora.lora_down.weight.clone()

    proj = _ol.OrthoLoRAProjector(method="gram_schmidt", interval=1)
    proj.register_layer("ortho_block", model.ortho)
    proj.step()

    # Standard / TLora untouched
    assert torch.allclose(model.standard.lora_down.weight, standard_down_before)
    assert torch.allclose(model.tlora.lora_down.weight, tlora_down_before)
    # Ortho rows now orthonormal
    rows = model.ortho.lora_down.weight.float()
    gram = rows @ rows.t()
    eye = torch.eye(rows.shape[0])
    assert torch.allclose(gram, eye, atol=1e-3)
    print("PASS: ortho projector only re-orthogonalises ortho LoRA")


def test_tlora_schedule_changes_active_rank():
    layer = _TLoRA(dim=16, max_rank=8)
    layer.set_active_rank(step=0, total_steps=100)
    early_rank = layer.active_rank
    layer.set_active_rank(step=100, total_steps=100)
    late_rank = layer.active_rank
    assert early_rank < late_rank
    assert late_rank == 8
    print(f"PASS: T-LoRA active_rank grows over training (early={early_rank}, late={late_rank})")


def test_all_variants_receive_gradients():
    torch.manual_seed(0)
    model = _StackedModel(dim=16)
    # Make ortho non-zero by initialising lora_up in standard / tlora too
    nn.init.normal_(model.standard.lora_up.weight, std=0.1)
    nn.init.normal_(model.tlora.lora_up.weight, std=0.1)

    x = torch.randn(2, 16, requires_grad=True)
    loss = model(x).sum()
    loss.backward()

    for variant_name, layer in (("standard", model.standard), ("ortho", model.ortho), ("tlora", model.tlora)):
        # At least one trainable in each variant has non-zero gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in layer.parameters()
        )
        assert has_grad, f"variant {variant_name} got zero gradients"
    print("PASS: every stacked LoRA variant receives gradients")


def test_optimizer_can_iterate_all_variants():
    model = _StackedModel(dim=16)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=1e-3)
    # One step shouldn't raise
    nn.init.normal_(model.standard.lora_up.weight, std=0.1)
    out = model(torch.randn(1, 16)).sum()
    out.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("PASS: optimizer can step over the full stacked-LoRA parameter set")


if __name__ == "__main__":
    test_stacked_model_forward_shape()
    test_each_variant_has_independent_params()
    test_base_weights_remain_frozen()
    test_ortho_projector_only_affects_ortho_lora()
    test_tlora_schedule_changes_active_rank()
    test_all_variants_receive_gradients()
    test_optimizer_can_iterate_all_variants()
    print("\nAll LoRA stacking smoke tests passed!")
