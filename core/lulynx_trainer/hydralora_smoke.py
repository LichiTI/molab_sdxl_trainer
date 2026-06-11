# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for hydralora.py (Phase 8.3 / #111)."""

from __future__ import annotations

import os
import sys
import importlib.util

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.hydralora",
    os.path.join(_HERE, "hydralora.py"),
)
_hl = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.hydralora"] = _hl
_spec.loader.exec_module(_hl)


def test_zero_init_lora_up_is_no_op():
    base = nn.Linear(16, 16)
    layer = _hl.HydraLoRALinear(base, _hl.HydraLoRAConfig(num_experts=4, rank=8, alpha=8.0))
    x = torch.randn(2, 16)
    base_out = base(x)
    layer_out = layer(x)
    # lora_up zero-init means no contribution
    assert torch.allclose(base_out, layer_out, atol=1e-6)
    print("PASS: zero-init lora_up makes HydraLoRA behave like base linear")


def test_nonzero_init_lora_up_changes_output():
    base = nn.Linear(16, 16)
    cfg = _hl.HydraLoRAConfig(num_experts=4, rank=8, alpha=8.0)
    layer = _hl.HydraLoRALinear(base, cfg)
    nn.init.normal_(layer.lora_up, std=0.1)
    x = torch.randn(2, 16)
    out_diff = (layer(x) - base(x)).abs().max().item()
    assert out_diff > 0
    print(f"PASS: non-zero lora_up shifts output (diff={out_diff:.4f})")


def test_dense_routing_sums_all_experts():
    base = nn.Linear(8, 8)
    cfg = _hl.HydraLoRAConfig(num_experts=4, rank=4, routing="dense")
    layer = _hl.HydraLoRALinear(base, cfg)
    nn.init.normal_(layer.lora_up, std=0.1)
    x = torch.randn(2, 8)
    out = layer(x)
    assert out.shape == (2, 8)
    print("PASS: dense routing produces correct output shape")


def test_top_k_routing_selects_subset():
    base = nn.Linear(8, 8)
    cfg = _hl.HydraLoRAConfig(num_experts=8, rank=4, routing="top_k", top_k=2)
    layer = _hl.HydraLoRALinear(base, cfg)

    logits = torch.tensor([[10.0, 1.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    weights = layer._top_k_weights(logits)
    # Top-2 selected: indices 0, 2
    assert weights[0, 0] > 0
    assert weights[0, 2] > 0
    assert weights[0, 1] == 0
    assert weights[0, 3:].sum() == 0
    # Weights sum to 1 across selected experts
    assert abs(weights[0].sum().item() - 1.0) < 1e-5
    print("PASS: top_k routing zeroes non-selected experts and renormalises")


def _dense_top_k_reference(layer, x):
    base_out = layer.original(x)
    x_d = layer.dropout(x)
    logits = layer.gate(x)
    weights = layer._top_k_weights(logits)
    proj = torch.einsum("...i,eri->...er", x_d, layer.lora_down)
    deltas = torch.einsum("...er,eor->...eo", proj, layer.lora_up)
    mixed = (weights.unsqueeze(-1) * deltas * layer.scaling).sum(dim=-2)
    return base_out + mixed


def test_top_k_sparse_matches_dense_reference_forward_backward():
    torch.manual_seed(123)
    base = nn.Linear(8, 6, bias=False)
    cfg = _hl.HydraLoRAConfig(num_experts=5, rank=3, routing="top_k", top_k=2, sparse_top_k=True)
    layer = _hl.HydraLoRALinear(base, cfg).double()
    with torch.no_grad():
        layer.lora_down.normal_(std=0.1)
        layer.lora_up.normal_(std=0.1)
        layer.gate.weight.normal_(std=0.2)

    x_fast = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    x_ref = x_fast.detach().clone().requires_grad_(True)
    out_sparse = layer(x_fast)
    out_ref = _dense_top_k_reference(layer, x_ref)
    torch.testing.assert_close(out_sparse, out_ref, rtol=1e-7, atol=1e-8)

    upstream = torch.randn_like(out_sparse)
    params = (layer.lora_down, layer.lora_up, layer.gate.weight)
    sparse_grads = torch.autograd.grad((out_sparse * upstream).sum(), (x_fast, *params))
    ref_grads = torch.autograd.grad((out_ref * upstream).sum(), (x_ref, *params))
    for sparse_grad, ref_grad in zip(sparse_grads, ref_grads):
        torch.testing.assert_close(sparse_grad, ref_grad, rtol=1e-7, atol=1e-8)

    print("PASS: top_k sparse path matches dense reference forward/backward")


def test_get_trainable_params_excludes_base():
    base = nn.Linear(16, 16)
    layer = _hl.HydraLoRALinear(base, _hl.HydraLoRAConfig(num_experts=2, rank=4))
    params = layer.get_trainable_params()
    # Should be exactly: lora_down, lora_up, gate
    assert len(params) == 3
    # Base linear is frozen
    assert all(not p.requires_grad for p in base.parameters())
    print("PASS: get_trainable_params excludes frozen base linear")


def test_layer_is_differentiable():
    base = nn.Linear(8, 8)
    layer = _hl.HydraLoRALinear(base, _hl.HydraLoRAConfig(num_experts=4, rank=4))
    nn.init.normal_(layer.lora_up, std=0.1)
    x = torch.randn(2, 8, requires_grad=True)
    loss = layer(x).sum()
    loss.backward()
    assert layer.lora_up.grad is not None
    assert layer.lora_down.grad is not None
    assert layer.gate.weight.grad is not None
    print("PASS: HydraLoRALinear receives gradients on all trainables")


def test_invalid_top_k_raises():
    base = nn.Linear(8, 8)
    try:
        _hl.HydraLoRALinear(
            base, _hl.HydraLoRAConfig(num_experts=2, top_k=5, routing="top_k")
        )
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("PASS: top_k > num_experts raises ValueError")


def test_invalid_routing_raises():
    base = nn.Linear(8, 8)
    try:
        _hl.HydraLoRALinear(base, _hl.HydraLoRAConfig(num_experts=2, routing="garbage"))
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("PASS: unknown routing strategy raises")


def test_expert_balance_loss_low_when_balanced():
    base = nn.Linear(8, 8)
    layer = _hl.HydraLoRALinear(base, _hl.HydraLoRAConfig(num_experts=4))
    # Uniform logits → balanced
    logits = torch.zeros(16, 4)
    loss_balanced = layer.expert_balance_loss(logits)
    # Skewed logits → unbalanced
    logits_skew = torch.zeros(16, 4)
    logits_skew[:, 0] = 10.0
    loss_skew = layer.expert_balance_loss(logits_skew)
    assert loss_skew.item() > loss_balanced.item()
    print(f"PASS: balance loss is higher when expert usage is skewed "
          f"(balanced={loss_balanced.item():.6f}, skewed={loss_skew.item():.4f})")


def test_works_with_3d_inputs():
    base = nn.Linear(8, 8)
    layer = _hl.HydraLoRALinear(base, _hl.HydraLoRAConfig(num_experts=4, rank=4))
    nn.init.normal_(layer.lora_up, std=0.1)
    x = torch.randn(2, 16, 8)  # batch, seq, dim
    out = layer(x)
    assert out.shape == (2, 16, 8)
    print("PASS: HydraLoRALinear handles 3D inputs (batch, seq, dim)")


if __name__ == "__main__":
    test_zero_init_lora_up_is_no_op()
    test_nonzero_init_lora_up_changes_output()
    test_dense_routing_sums_all_experts()
    test_top_k_routing_selects_subset()
    test_top_k_sparse_matches_dense_reference_forward_backward()
    test_get_trainable_params_excludes_base()
    test_layer_is_differentiable()
    test_invalid_top_k_raises()
    test_invalid_routing_raises()
    test_expert_balance_loss_low_when_balanced()
    test_works_with_3d_inputs()
    print("\nAll HydraLoRA smoke tests passed!")
