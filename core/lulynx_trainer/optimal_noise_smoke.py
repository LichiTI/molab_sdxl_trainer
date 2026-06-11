# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for optimal noise candidate selection."""

from __future__ import annotations
import sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_on = _import_module("optimal_noise", os.path.join(_HERE, "optimal_noise.py"))
select_optimal_noise = _on.select_optimal_noise

import torch


def _latents(B=2, C=4, H=8, W=8):
    return torch.randn(B, C, H, W)


def test_single_candidate():
    latents = _latents()
    result = select_optimal_noise(latents, loss_fn=lambda n: torch.tensor(0.0), n_candidates=1)
    assert result.shape == latents.shape, (
        f"Output shape {result.shape} should match latents shape {latents.shape}"
    )
    # Should be a randn-like tensor — just check it's not all zeros
    assert result.abs().sum().item() > 0, "Single-candidate result should not be all zeros"


def test_selects_lowest_loss():
    torch.manual_seed(0)
    latents = _latents(B=1, C=4, H=8, W=8)

    # Loss proportional to norm: lower-norm noise wins
    def loss_fn(noise):
        return noise.norm()

    result = select_optimal_noise(latents, loss_fn=loss_fn, n_candidates=10)

    # Estimate expected norm of a single randn candidate
    norms = [torch.randn_like(latents).norm().item() for _ in range(100)]
    avg_norm = sum(norms) / len(norms)

    selected_norm = result.norm().item()
    assert selected_norm < avg_norm, (
        f"Selected noise norm {selected_norm:.4f} should be below average {avg_norm:.4f}"
    )


def test_no_gradient():
    latents = _latents()
    result = select_optimal_noise(latents, loss_fn=lambda n: torch.tensor(0.0), n_candidates=4)
    assert not result.requires_grad, (
        "Returned noise from select_optimal_noise should not require gradients"
    )


def test_shape_preserved():
    latents = _latents(B=3, C=8, H=16, W=16)
    result = select_optimal_noise(latents, loss_fn=lambda n: torch.tensor(0.0), n_candidates=4)
    assert result.shape == latents.shape, (
        f"Output shape {result.shape} must equal latents shape {latents.shape}"
    )


def test_closure_called():
    latents = _latents()
    call_count = [0]

    def counting_loss(noise):
        call_count[0] += 1
        return torch.tensor(0.0)

    n = 7
    select_optimal_noise(latents, loss_fn=counting_loss, n_candidates=n)
    assert call_count[0] == n, (
        f"loss_fn should be called exactly n_candidates={n} times, got {call_count[0]}"
    )


if __name__ == "__main__":
    test_single_candidate()
    test_selects_lowest_loss()
    test_no_gradient()
    test_shape_preserved()
    test_closure_called()
    print("optimal_noise_smoke: all tests passed")
