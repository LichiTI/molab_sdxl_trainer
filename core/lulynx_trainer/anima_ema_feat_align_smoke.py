# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for the JLT-style EMA feature self-distillation reserve.

Covers (CPU, no model loading):
  * feature-capture seam parity: default context is None; observe only fires
    for requested layers and keeps the autograd graph.
  * EmaLoraShadow: lazy init, monotone move toward live weights, decay effect.
  * analytic latent/noise recovery (velocity path is exact, no division).
  * compute_ema_feat_align_loss end-to-end on a fake unet: teacher swap +
    cosine loss in [0, 2], backprops into student features, length mismatch
    raises, weights restored after the teacher pass.

Run with the python-flashattention env from backend/core/lulynx_trainer/.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import torch

from core.lulynx_trainer.anima_feature_capture import (
    FeatureCapture,
    feature_capture_scope,
    get_active_feature_capture,
    parse_layer_list,
)
from core.lulynx_trainer.anima_ema_feature_align import (
    EmaLoraShadow,
    _recover_latent_and_noise,
    compute_ema_feat_align_loss,
)


def test_capture_seam_parity() -> None:
    assert get_active_feature_capture() is None, "default capture must be None (parity)"
    with feature_capture_scope([1, 3]) as cap:
        assert get_active_feature_capture() is cap
        x0 = torch.randn(2, 5, 4)
        cap.observe(0, x0)  # not requested -> ignored
        x1 = torch.randn(2, 5, 4, requires_grad=True)
        cap.observe(1, x1)
        assert 0 not in cap.features and 1 in cap.features
        # graph preserved (no detach) for the student path
        assert cap.features[1].requires_grad
    assert get_active_feature_capture() is None, "scope must restore None"
    assert parse_layer_list("") == [] and parse_layer_list("4, 9") == [4, 9]
    print("PASS: capture seam parity + scope restore + layer parse")


def test_ema_shadow() -> None:
    p = torch.nn.Parameter(torch.zeros(4))
    p.requires_grad_(True)
    frozen = torch.nn.Parameter(torch.ones(4))
    frozen.requires_grad_(False)
    shadow = EmaLoraShadow()
    assert not shadow.initialized
    shadow.update([("lora", p), ("frozen", frozen)], decay=0.9)
    assert shadow.initialized
    assert "lora" in shadow.state() and "frozen" not in shadow.state(), "only trainable tracked"
    assert torch.allclose(shadow.state()["lora"], torch.zeros(4)), "lazy init = copy"
    # Move live weight, update, shadow should move PART of the way (decay=0.9).
    p.data = torch.ones(4)
    shadow.update([("lora", p)], decay=0.9)
    expected = 0.9 * 0.0 + 0.1 * 1.0
    assert torch.allclose(shadow.state()["lora"], torch.full((4,), expected), atol=1e-6)
    # Larger decay moves slower.
    p2 = torch.nn.Parameter(torch.zeros(4))
    sh2 = EmaLoraShadow()
    sh2.update([("lora", p2)], decay=0.99)
    p2.data = torch.ones(4)
    sh2.update([("lora", p2)], decay=0.99)
    assert sh2.state()["lora"].mean().item() < shadow.state()["lora"].mean().item()
    print("PASS: EMA shadow lazy-init, partial move, decay ordering")


def test_recover_latent_and_noise() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 4, 2, 2)
    e = torch.randn(3, 4, 2, 2)
    sigma = torch.rand(3).view(3, 1, 1, 1) * 0.8 + 0.1  # avoid extremes
    noisy = (1 - sigma) * x + sigma * e
    # velocity: target = e - x
    rx, re = _recover_latent_and_noise(noisy, e - x, sigma, "velocity")
    assert torch.allclose(rx, x, atol=1e-5) and torch.allclose(re, e, atol=1e-5)
    # sample: target = x
    rx, re = _recover_latent_and_noise(noisy, x, sigma, "sample")
    assert torch.allclose(rx, x, atol=1e-4) and torch.allclose(re, e, atol=1e-4)
    # epsilon: target = e
    rx, re = _recover_latent_and_noise(noisy, e, sigma, "epsilon")
    assert torch.allclose(rx, x, atol=1e-4) and torch.allclose(re, e, atol=1e-4)
    print("PASS: analytic latent/noise recovery (velocity/sample/epsilon)")


class _FakeNet(torch.nn.Module):
    """Minimal stand-in: a trainable LoRA-like weight whose forward emits a few
    block features through the active capture seam and returns ``.sample``."""

    def __init__(self) -> None:
        super().__init__()
        self.lora = torch.nn.Parameter(torch.randn(4, 4) * 0.01)
        base = torch.nn.Parameter(torch.randn(4, 4))
        base.requires_grad_(False)
        self.base = base
        self.anima_llm_adapter = None

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        # sample: [B, tokens, 4]; produce per-block "features" and observe them.
        cap = get_active_feature_capture()
        h = sample
        for block_index in range(12):
            h = torch.tanh(h @ (self.base + self.lora))
            if cap is not None:
                cap.observe(block_index, h)
        return SimpleNamespace(sample=h)


def _fake_owner(net: _FakeNet) -> SimpleNamespace:
    shadow = EmaLoraShadow()
    shadow.update(list(net.named_parameters()), decay=0.9)
    return SimpleNamespace(
        unet=net,
        device=torch.device("cpu"),
        dtype=torch.float32,
        anima_faithful_forward=False,
        anima_model_prediction_type="velocity",
        anima_timestep_sampling="sigma",
        anima_sigmoid_scale=1.0,
        anima_discrete_flow_shift=1.0,
        flow_logit_mean=0.0,
        flow_logit_std=1.0,
        anima_ema_feat_align_teacher_layers="9",
        anima_ema_feat_align_student_layers="4",
        _ema_lora_shadow=shadow,
    )


def test_compute_loss_end_to_end() -> None:
    torch.manual_seed(1)
    net = _FakeNet()
    owner = _fake_owner(net)
    B, T, C = 2, 6, 4
    noisy = torch.randn(B, T, C)
    target = torch.randn(B, T, C)
    timesteps = torch.rand(B) * 1000.0
    prompt_embeds = {"encoder_hidden_states": torch.randn(B, 8, C)}

    # Student features (layer 4) must carry a graph to receive gradients.
    student_feat = torch.randn(B, T, C, requires_grad=True)
    student_features = {4: student_feat}

    live_before = net.lora.detach().clone()
    loss = compute_ema_feat_align_loss(
        owner=owner,
        student_features=student_features,
        prompt_embeds=prompt_embeds,
        batch={},
        noisy_latents=noisy,
        target=target,
        timesteps=timesteps,
    )
    assert torch.isfinite(loss) and 0.0 <= loss.item() <= 2.0, f"loss out of range: {loss}"
    # Weights restored after the teacher swap.
    assert torch.allclose(net.lora.detach(), live_before), "live LoRA weights must be restored"
    # Backprop reaches the student features only (teacher is no_grad).
    loss.backward()
    assert student_feat.grad is not None and torch.isfinite(student_feat.grad).all()
    assert net.lora.grad is None, "teacher pass must not accumulate grad on live weights"
    print("PASS: compute_ema_feat_align_loss end-to-end (range, restore, backprop)")


def test_uninitialized_and_mismatch() -> None:
    net = _FakeNet()
    owner = _fake_owner(net)
    # No student features -> zero scalar, no graph.
    z = compute_ema_feat_align_loss(
        owner=owner,
        student_features={},
        prompt_embeds={"encoder_hidden_states": torch.randn(1, 8, 4)},
        batch={},
        noisy_latents=torch.randn(1, 6, 4),
        target=torch.randn(1, 6, 4),
        timesteps=torch.rand(1) * 1000.0,
    )
    assert z.item() == 0.0 and not z.requires_grad
    # Layer-count mismatch raises.
    owner.anima_ema_feat_align_student_layers = "4,5"
    try:
        compute_ema_feat_align_loss(
            owner=owner,
            student_features={4: torch.randn(1, 6, 4)},
            prompt_embeds={"encoder_hidden_states": torch.randn(1, 8, 4)},
            batch={},
            noisy_latents=torch.randn(1, 6, 4),
            target=torch.randn(1, 6, 4),
            timesteps=torch.rand(1) * 1000.0,
        )
    except ValueError:
        print("PASS: empty-features zero + layer-count mismatch raises")
        return
    raise AssertionError("expected ValueError on teacher/student layer length mismatch")


def main() -> int:
    test_capture_seam_parity()
    test_ema_shadow()
    test_recover_latent_and_noise()
    test_compute_loss_end_to_end()
    test_uninitialized_and_mismatch()
    print("\nAll EMA feature-alignment smoke tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
