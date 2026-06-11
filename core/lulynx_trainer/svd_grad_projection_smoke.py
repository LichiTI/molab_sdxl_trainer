# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for SVD gradient projection."""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_svd = _import_module(
    "svd_grad_projection",
    os.path.join(_HERE, "svd_grad_projection.py"),
)
apply_svd_gradient_projection = _svd.apply_svd_gradient_projection

import torch
import torch.nn as nn


def test_projection_reduces_rank():
    """After one optimizer step, the projected gradient should have lower effective rank."""
    torch.manual_seed(0)
    layer = nn.Linear(64, 64, bias=False)
    # Set a full-rank random gradient
    layer.weight.grad = torch.randn(64, 64)
    original_grad = layer.weight.grad.clone()
    original_rank = torch.linalg.matrix_rank(original_grad).item()

    base_opt = torch.optim.SGD(layer.parameters(), lr=0.01)
    # update_interval=200: effective_step=1 satisfies 1 % 200 == 1, so basis
    # is computed on the very first step and projection is applied immediately.
    wrapped = _svd.SVDGradientProjectionWrapper(base_opt, rank=4, update_interval=200)
    wrapped.step()

    projected_grad = layer.weight.grad
    projected_rank = torch.linalg.matrix_rank(projected_grad).item()

    assert projected_rank < original_rank, (
        f"Expected projected rank ({projected_rank}) < original rank ({original_rank})"
    )
    print("PASS: projection reduces gradient rank")


def test_warmup_passes_through():
    """During warmup, gradients should be passed through unmodified."""
    torch.manual_seed(1)
    layer = nn.Linear(32, 32, bias=False)
    original_grad = torch.randn(32, 32)
    layer.weight.grad = original_grad.clone()

    base_opt = torch.optim.SGD(layer.parameters(), lr=0.0)  # lr=0 so weights don't move
    wrapped = _svd.SVDGradientProjectionWrapper(base_opt, rank=4, warmup_steps=100)

    # Simulate step 50 (within warmup window)
    wrapped._step_count = 49  # will be incremented to 50 inside step()
    layer.weight.grad = original_grad.clone()
    wrapped.step()

    # Gradient should be unchanged by projection (only base optimizer ran)
    assert torch.allclose(layer.weight.grad, original_grad, atol=1e-6), (
        "Gradient was modified during warmup"
    )
    print("PASS: warmup passes gradients through unchanged")


def test_wrapper_delegation():
    """param_groups, state_dict(), and zero_grad() should delegate to base optimizer."""
    layer = nn.Linear(16, 16)
    base_opt = torch.optim.Adam(layer.parameters(), lr=1e-3)
    wrapped = _svd.SVDGradientProjectionWrapper(base_opt, rank=4)

    # param_groups delegates
    assert wrapped.param_groups is base_opt.param_groups, (
        "param_groups not delegated to base optimizer"
    )

    # state_dict delegates
    sd = wrapped.state_dict()
    base_sd = base_opt.state_dict()
    assert sd == base_sd, "state_dict() not delegated to base optimizer"

    # zero_grad delegates — set some grads, call zero_grad, verify cleared
    layer.weight.grad = torch.ones_like(layer.weight)
    layer.bias.grad = torch.ones_like(layer.bias)
    wrapped.zero_grad(set_to_none=False)
    assert layer.weight.grad is not None and layer.weight.grad.abs().sum().item() == 0.0, (
        "zero_grad() did not clear weight gradient"
    )
    print("PASS: wrapper delegates param_groups, state_dict, zero_grad to base optimizer")


def test_1d_params_untouched():
    """1D bias gradients should not be projected."""
    torch.manual_seed(2)
    layer = nn.Linear(32, 32, bias=True)
    original_bias_grad = torch.randn(32)
    layer.weight.grad = torch.randn(32, 32)
    layer.bias.grad = original_bias_grad.clone()

    base_opt = torch.optim.SGD(layer.parameters(), lr=0.0)
    wrapped = _svd.SVDGradientProjectionWrapper(base_opt, rank=4, update_interval=1)
    wrapped.step()

    assert torch.allclose(layer.bias.grad, original_bias_grad, atol=1e-6), (
        "1D bias gradient was modified by SVD projection"
    )
    print("PASS: 1D bias gradients are not projected")


def test_basis_update_interval():
    """Basis should only update every update_interval steps."""
    torch.manual_seed(3)
    layer = nn.Linear(16, 16, bias=False)

    base_opt = torch.optim.SGD(layer.parameters(), lr=0.0)
    update_interval = 5
    wrapped = _svd.SVDGradientProjectionWrapper(base_opt, rank=4, update_interval=update_interval)

    update_counts = []

    # Patch _GradProjector.update_basis to count calls
    original_update_basis = _svd._GradProjector.update_basis

    def counting_update_basis(self, grad):
        update_counts.append(1)
        original_update_basis(self, grad)

    _svd._GradProjector.update_basis = counting_update_basis

    try:
        num_steps = 20
        for i in range(num_steps):
            layer.weight.grad = torch.randn(16, 16)
            wrapped.step()

        # effective_step = step_count - warmup_steps (warmup=0)
        # update happens when effective_step % update_interval == 1
        # effective_steps 1..20 → updates at 1,6,11,16 → 4 updates
        expected_updates = sum(
            1 for s in range(1, num_steps + 1) if (s - 1) % update_interval == 0
        )
        assert len(update_counts) == expected_updates, (
            f"Expected {expected_updates} basis updates, got {len(update_counts)}"
        )
    finally:
        _svd._GradProjector.update_basis = original_update_basis

    print(f"PASS: basis updated {len(update_counts)} times over {num_steps} steps "
          f"(interval={update_interval})")



def test_scheduler_compatibility():
    """Wrapper should be accepted by PyTorch LR schedulers."""
    layer = nn.Linear(8, 8)
    base_opt = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    wrapped = _svd.SVDGradientProjectionWrapper(base_opt, rank=2, update_interval=1)

    assert isinstance(wrapped, torch.optim.Optimizer), (
        "SVD wrapper must inherit torch.optim.Optimizer for scheduler compatibility"
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(wrapped, factor=1.0, total_iters=1)

    layer(torch.randn(2, 8)).sum().backward()
    wrapped.step()
    scheduler.step()

    assert wrapped.param_groups is base_opt.param_groups, (
        "scheduler must update the base optimizer parameter groups"
    )
    assert wrapped.param_groups[0]["lr"] == base_opt.param_groups[0]["lr"]
    print("PASS: wrapper is compatible with PyTorch LR schedulers")

if __name__ == "__main__":
    test_projection_reduces_rank()
    test_warmup_passes_through()
    test_wrapper_delegation()
    test_1d_params_untouched()
    test_basis_update_interval()
    test_scheduler_compatibility()
    print("\nAll SVD gradient projection smoke tests passed!")

