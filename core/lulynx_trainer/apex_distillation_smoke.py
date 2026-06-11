# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for apex_distillation.py (Phase 8.7 / #115)."""

from __future__ import annotations

import os
import sys
import importlib.util

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.apex_distillation",
    os.path.join(_HERE, "apex_distillation.py"),
)
_ad = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.apex_distillation"] = _ad
_spec.loader.exec_module(_ad)


def test_kl_loss_zero_when_inputs_match():
    pred = torch.randn(2, 16)
    loss = _ad.output_kl_loss(pred, pred.clone(), temperature=1.0)
    assert loss.item() < 1e-5
    print("PASS: KL is ~0 when student equals teacher")


def test_kl_loss_positive_when_inputs_differ():
    torch.manual_seed(0)
    s = torch.randn(2, 16)
    t = torch.randn(2, 16)
    loss = _ad.output_kl_loss(s, t)
    assert loss.item() > 0
    print(f"PASS: KL positive on differing inputs (loss={loss.item():.4f})")


def test_mse_loss_zero_when_inputs_match():
    pred = torch.randn(2, 4, 8, 8)
    loss = _ad.output_mse_loss(pred, pred.clone())
    assert loss.item() < 1e-6
    print("PASS: MSE is 0 on identical inputs")


def test_feature_match_loss_handles_shape_mismatch():
    student_feats = [torch.randn(2, 4, 16, 16)]
    teacher_feats = [torch.randn(2, 4, 32, 32)]  # different spatial size
    loss = _ad.feature_match_loss(student_feats, teacher_feats)
    assert loss.item() >= 0
    print("PASS: feature_match_loss handles spatial-size mismatch via interpolation")


def test_feature_match_length_mismatch_raises():
    s = [torch.randn(2, 4, 8, 8)]
    t = [torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)]
    try:
        _ad.feature_match_loss(s, t)
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("PASS: feature_match_loss raises on length mismatch")


def test_distillation_loss_disabled_returns_zero():
    cfg = _ad.DistillationConfig(enabled=False, output_mse_weight=1.0)
    pred = torch.randn(2, 4)
    loss = _ad.compute_distillation_loss(pred, pred + 1.0, cfg)
    assert loss.item() == 0.0
    print("PASS: disabled distillation returns 0")


def test_distillation_loss_combines_terms():
    cfg = _ad.DistillationConfig(
        enabled=True,
        output_kl_weight=1.0,
        output_mse_weight=0.5,
        feature_match_weight=0.5,
    )
    s_pred = torch.randn(2, 4, 8, 8)
    t_pred = s_pred + 0.1 * torch.randn_like(s_pred)
    s_feats = [torch.randn(2, 4, 8, 8)]
    t_feats = [torch.randn(2, 4, 8, 8)]
    loss = _ad.compute_distillation_loss(
        s_pred, t_pred, cfg,
        student_features=s_feats, teacher_features=t_feats,
    )
    assert loss.item() > 0
    print("PASS: compute_distillation_loss combines all enabled terms")


def test_teacher_detach_blocks_gradient():
    cfg = _ad.DistillationConfig(enabled=True, output_mse_weight=1.0, detach_teacher=True)
    s = torch.randn(2, 4, requires_grad=True)
    t = torch.randn(2, 4, requires_grad=True)
    loss = _ad.compute_distillation_loss(s, t, cfg)
    loss.backward()
    assert s.grad is not None
    assert t.grad is None
    print("PASS: detach_teacher=True blocks gradient flow into teacher")


# ---------------------------------------------------------------------------
# FeatureCapture tests
# ---------------------------------------------------------------------------

class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Linear(16, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 16),
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def test_feature_capture_records_layer_outputs():
    model = _ToyModel()
    capture = _ad.FeatureCapture(model, layer_names=["blocks.0", "blocks.2"])
    x = torch.randn(2, 16)
    model(x)
    feats = capture.features
    assert len(feats) == 2
    assert feats[0].shape == (2, 16)
    capture.remove_hooks()
    print("PASS: FeatureCapture records outputs at named layers")


def test_feature_capture_clear_drops_features():
    model = _ToyModel()
    capture = _ad.FeatureCapture(model, layer_names=["blocks.1"])
    model(torch.randn(2, 16))
    assert len(capture.features) == 1
    capture.clear()
    assert len(capture.features) == 0
    capture.remove_hooks()
    print("PASS: FeatureCapture.clear() drops captured tensors")


def test_feature_capture_unknown_layer_warns_but_does_not_crash():
    model = _ToyModel()
    capture = _ad.FeatureCapture(model, layer_names=["does.not.exist"])
    model(torch.randn(2, 16))
    assert capture.features == []
    capture.remove_hooks()
    print("PASS: FeatureCapture tolerates unknown layer names")


if __name__ == "__main__":
    test_kl_loss_zero_when_inputs_match()
    test_kl_loss_positive_when_inputs_differ()
    test_mse_loss_zero_when_inputs_match()
    test_feature_match_loss_handles_shape_mismatch()
    test_feature_match_length_mismatch_raises()
    test_distillation_loss_disabled_returns_zero()
    test_distillation_loss_combines_terms()
    test_teacher_detach_blocks_gradient()
    test_feature_capture_records_layer_outputs()
    test_feature_capture_clear_drops_features()
    test_feature_capture_unknown_layer_warns_but_does_not_crash()
    print("\nAll APEX distillation smoke tests passed!")
