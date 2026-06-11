# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Differential Output Preservation (DOP)."""

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

_dop = _import_module(
    "dop",
    os.path.join(_HERE, "dop.py"),
)
DifferentialOutputPreservation = _dop.DifferentialOutputPreservation

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal UNet-shaped stub that accepts the kwargs compute_loss passes through
# ---------------------------------------------------------------------------

class _UNetStub(nn.Module):
    """Minimal stub accepting UNet-style kwargs; ignores all but sample shape."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs=None,
    ) -> torch.Tensor:
        # flatten → linear → reshape to match sample
        b = sample.shape[0]
        flat = sample.view(b, -1)
        return self.linear(flat).view_as(sample)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reference_frozen():
    """All reference model parameters must have requires_grad=False after init."""
    ref_linear = nn.Linear(32, 32)
    ref_model = _UNetStub(ref_linear)
    dop = DifferentialOutputPreservation(ref_model)

    for name, p in dop.reference_model.named_parameters():
        assert not p.requires_grad, (
            f"Expected requires_grad=False for '{name}', got True"
        )
    print("PASS: reference model parameters are frozen")


def test_should_compute_logic():
    """start_step=10, interval=5 → correct step gating."""
    ref_model = _UNetStub(nn.Linear(32, 32))
    dop = DifferentialOutputPreservation(ref_model, start_step=10, interval=5)

    assert dop.should_compute(9)  is False, "step 9 should return False"
    assert dop.should_compute(10) is True,  "step 10 should return True"
    assert dop.should_compute(11) is False, "step 11 should return False"
    assert dop.should_compute(15) is True,  "step 15 should return True"
    print("PASS: should_compute logic is correct")


def _make_dop_and_tensors(identical_output: bool):
    """Helper: build a DOP whose reference uses an identity-like linear,
    and return (dop, current_output, noisy_latents, timesteps, enc_hs)."""
    torch.manual_seed(0)
    batch, seq, ch = 1, 4, 32
    flat_dim = seq * ch  # 128

    # Linear(flat_dim, flat_dim) initialised with identity weights for easy control
    ref_linear = nn.Linear(flat_dim, flat_dim, bias=False)
    nn.init.eye_(ref_linear.weight)          # exact identity transform
    ref_model = _UNetStub(ref_linear)

    dop = DifferentialOutputPreservation(ref_model, weight=1.0)

    noisy_latents = torch.randn(batch, seq, ch)
    timesteps = torch.tensor([500])
    encoder_hidden_states = torch.randn(batch, seq, ch)

    # Run reference forward to get "ground truth"
    with torch.no_grad():
        ref_out = ref_model(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        )

    if identical_output:
        current_output = ref_out.clone().requires_grad_(True)
    else:
        current_output = (ref_out + 1.0).detach().requires_grad_(True)

    return dop, current_output, noisy_latents, timesteps, encoder_hidden_states


def test_loss_zero_identical():
    """When current_output matches reference output exactly, DOP loss == 0."""
    dop, current_output, noisy_latents, timesteps, enc_hs = _make_dop_and_tensors(
        identical_output=True
    )
    loss = dop.compute_loss(current_output, noisy_latents, timesteps, enc_hs)
    assert loss.item() == 0.0, f"Expected loss 0.0, got {loss.item()}"
    print("PASS: loss is 0 when outputs are identical")


def test_loss_positive_different():
    """When current_output differs from reference, DOP loss > 0."""
    dop, current_output, noisy_latents, timesteps, enc_hs = _make_dop_and_tensors(
        identical_output=False
    )
    loss = dop.compute_loss(current_output, noisy_latents, timesteps, enc_hs)
    assert loss.item() > 0.0, f"Expected loss > 0, got {loss.item()}"
    print("PASS: loss is positive when outputs differ")


def test_reference_unchanged_after_backward():
    """Reference model weights must not change after loss.backward()."""
    dop, current_output, noisy_latents, timesteps, enc_hs = _make_dop_and_tensors(
        identical_output=False
    )

    # Snapshot reference weights before backward
    ref_weights_before = {
        name: p.data.clone()
        for name, p in dop.reference_model.named_parameters()
    }

    loss = dop.compute_loss(current_output, noisy_latents, timesteps, enc_hs)
    loss.backward()

    for name, p in dop.reference_model.named_parameters():
        before = ref_weights_before[name]
        assert torch.equal(p.data, before), (
            f"Reference weight '{name}' changed after backward()"
        )
    print("PASS: reference model weights unchanged after backward()")


if __name__ == "__main__":
    test_reference_frozen()
    test_should_compute_logic()
    test_loss_zero_identical()
    test_loss_positive_different()
    test_reference_unchanged_after_backward()
    print("\nAll DOP smoke tests passed!")
