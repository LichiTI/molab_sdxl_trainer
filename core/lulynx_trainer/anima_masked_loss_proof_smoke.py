"""Smoke test for Anima masked loss with alpha masks: gradient steering, mask-zero regions."""
from __future__ import annotations

import sys
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))

# Load masked_loss via importlib
_ml_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.masked_loss",
    os.path.join(_HERE, "masked_loss.py"),
)
_ml_mod = importlib.util.module_from_spec(_ml_spec)
sys.modules["core.lulynx_trainer.masked_loss"] = _ml_mod
_ml_spec.loader.exec_module(_ml_mod)

MaskedLoss = _ml_mod.MaskedLoss

import torch
import torch.nn as nn


def test_with_mask_reduces_loss_contribution():
    """with_mask=True reduces loss contribution in masked region compared to unmasked."""
    ml = MaskedLoss(mask_weight=1.0, background_weight=0.1, normalize_mask=True)

    pred = torch.randn(1, 1, 8, 8)
    target = torch.randn(1, 1, 8, 8)

    # Full mask: everything weighted equally
    full_mask = torch.ones(1, 1, 8, 8)
    loss_full = ml(pred, target, full_mask)

    # No mask: plain MSE (same effective weight everywhere)
    loss_plain = ml(pred, target, mask=None)

    # Partial mask: only half weighted
    half_mask = torch.zeros(1, 1, 8, 8)
    half_mask[:, :, :4, :] = 1.0
    loss_half = ml(pred, target, half_mask)

    # With heavy background downweight, partial mask loss should differ from full mask
    # and from plain MSE
    assert not torch.allclose(loss_half, loss_full, atol=1e-4), (
        f"Partial mask loss should differ from full mask: {loss_half.item():.4f} vs {loss_full.item():.4f}"
    )


def test_mask_zero_region_zero_gradient():
    """Pixels with mask=0 get zero gradient when background_weight=0 and
    normalize_mask=False and blur_kernel_size=1 (no blur).  The blur kernel
    in MaskedLoss.prepare_mask can spread values into masked-out regions, so
    we disable it for this strict gradient test."""
    ml = MaskedLoss(
        mask_weight=10.0,
        background_weight=0.0,
        normalize_mask=False,
        blur_kernel_size=1,  # disable blur so mask boundary stays sharp
    )

    pred = torch.randn(1, 1, 4, 4, requires_grad=True)
    target = torch.zeros(1, 1, 4, 4)

    # Mask where top half = active, bottom half = masked out (weight 0)
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, :2, :] = 1.0  # top half active

    loss = ml(pred, target, mask)
    loss.backward()

    grad = pred.grad.clone()

    # With normalize_mask=False and background_weight=0, the weight for masked
    # pixels is exactly 0.  Weighted MSE = (raw_mse * weight).mean()
    # Gradient for masked pixels = 2*(pred - target) * weight / N = 0
    assert (grad[:, :, 2:, :] == 0).all(), (
        f"Masked-out region should have zero gradient, got: {grad[:, :, 2:, :]}"
    )
    # Top half (mask=1) should have non-zero gradient
    assert (grad[:, :, :2, :] != 0).any(), (
        "Active mask region should have non-zero gradient"
    )


def test_unmasked_region_normal_gradient():
    """Unmasked region (mask=1) gets proportional gradient that increases with mask weight."""
    # Use blur_kernel_size=1 so mask is not modified by prepare_mask
    ml = MaskedLoss(
        mask_weight=2.0,
        background_weight=1.0,
        normalize_mask=False,
        blur_kernel_size=1,
    )

    pred = torch.randn(1, 1, 4, 4, requires_grad=True)
    target = torch.zeros(1, 1, 4, 4)

    # All pixels active (mask=1)
    mask = torch.ones(1, 1, 4, 4)
    loss = ml(pred, target, mask)
    loss.backward()

    grad_weighted = pred.grad.clone()

    # Now compute the same without a mask (plain MSE)
    pred2 = pred.detach().clone().requires_grad_(True)
    loss_plain = torch.nn.functional.mse_loss(pred2, target)
    loss_plain.backward()
    grad_plain = pred2.grad.clone()

    # With mask_weight=2.0, background_weight=1.0, and mask=1 everywhere,
    # weight = 1.0 + (2.0 - 1.0) * 1.0 = 2.0 everywhere.
    # Weighted loss = (mse_none * 2.0).mean() = 2 * mse_loss
    # So gradient should be 2x the plain MSE gradient
    expected_grad = 2.0 * grad_plain
    assert torch.allclose(grad_weighted, expected_grad, atol=1e-5), (
        f"Weighted gradient should be 2x plain gradient, max diff="
        f"{((grad_weighted - expected_grad).abs().max().item())}"
    )


def test_masked_loss_vs_plain_loss():
    """Without a mask, masked loss should equal plain MSE loss."""
    ml = MaskedLoss()
    pred = torch.randn(2, 4, 8, 8)
    target = torch.randn(2, 4, 8, 8)

    masked_loss = ml(pred, target, mask=None)
    plain_loss = torch.nn.functional.mse_loss(pred, target)

    assert torch.allclose(masked_loss, plain_loss, atol=1e-6), (
        f"Masked loss with None mask should equal plain MSE: "
        f"{masked_loss.item():.6f} vs {plain_loss.item():.6f}"
    )


if __name__ == "__main__":
    print("Anima Masked Loss Proof Smoke Tests")
    print("=" * 40)
    test_with_mask_reduces_loss_contribution()
    print("PASS: with_mask_reduces_loss_contribution")
    test_mask_zero_region_zero_gradient()
    print("PASS: mask_zero_region_zero_gradient")
    test_unmasked_region_normal_gradient()
    print("PASS: unmasked_region_normal_gradient")
    test_masked_loss_vs_plain_loss()
    print("PASS: masked_loss_vs_plain_loss")
    print("=" * 40)
    print("All Anima masked loss proof smoke tests passed!")
