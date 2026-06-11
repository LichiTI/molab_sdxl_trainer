"""Smoke test for SDXL DreamBooth prior preservation loss."""
from __future__ import annotations

import sys
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))

# Load dreambooth via importlib
_db_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.dreambooth",
    os.path.join(_HERE, "dreambooth.py"),
)
_db_mod = importlib.util.module_from_spec(_db_spec)
sys.modules["core.lulynx_trainer.dreambooth"] = _db_mod
_db_spec.loader.exec_module(_db_mod)

DreamBoothConfig = _db_mod.DreamBoothConfig
PriorPreservationLoss = _db_mod.PriorPreservationLoss

import torch

if __package__ in (None, ""):
    backend_root = os.path.join(_HERE, "..", "..")
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)

from core.configs import UnifiedTrainingConfig


def test_prior_loss_weight_field_on_config():
    """prior_loss_weight field exists on UnifiedTrainingConfig."""
    cfg = UnifiedTrainingConfig()
    assert hasattr(cfg, "prior_loss_weight"), "UnifiedTrainingConfig missing prior_loss_weight"
    assert cfg.prior_loss_weight == 0.0, "Default prior_loss_weight should be 0.0"

    cfg_weighted = UnifiedTrainingConfig(prior_loss_weight=1.0)
    assert cfg_weighted.prior_loss_weight == 1.0, "prior_loss_weight should be settable"


def test_dreambooth_config_has_prior_loss_weight():
    """DreamBoothConfig has prior_loss_weight field."""
    db_cfg = DreamBoothConfig()
    assert hasattr(db_cfg, "prior_loss_weight"), "DreamBoothConfig missing prior_loss_weight"
    assert db_cfg.prior_loss_weight == 1.0, "Default DreamBooth prior_loss_weight should be 1.0"


def test_prior_preservation_adds_loss_term():
    """Prior preservation adds an additional loss term (instance + weight * class)."""
    ppl = PriorPreservationLoss(weight=1.0)

    instance_loss = torch.tensor(0.5)
    class_loss = torch.tensor(0.3)

    total = ppl(instance_loss, class_loss)

    expected = instance_loss + 1.0 * class_loss
    assert torch.allclose(total, expected), (
        f"Expected {expected.item()}, got {total.item()}"
    )
    # Total should be larger than instance loss alone
    assert total > instance_loss, "Prior preservation should increase total loss over instance-only"


def test_prior_loss_weighting_correct():
    """Loss weighting is correct: total = instance + w * class."""
    for w in [0.0, 0.5, 1.0, 2.0]:
        ppl = PriorPreservationLoss(weight=w)
        instance = torch.tensor(1.0)
        cls = torch.tensor(0.8)
        total = ppl(instance, cls)
        expected = instance + w * cls
        assert torch.allclose(total, expected, atol=1e-6), (
            f"weight={w}: expected {expected.item()}, got {total.item()}"
        )


def test_zero_weight_ignores_class_loss():
    """prior_loss_weight=0 means class loss contributes nothing."""
    ppl = PriorPreservationLoss(weight=0.0)
    instance = torch.tensor(1.0)
    cls = torch.tensor(999.0)  # huge class loss, should be ignored
    total = ppl(instance, cls)
    assert torch.allclose(total, instance), (
        f"weight=0: total should equal instance loss, got {total.item()}"
    )


if __name__ == "__main__":
    print("SDXL DreamBooth Smoke Tests")
    print("=" * 40)
    test_prior_loss_weight_field_on_config()
    print("PASS: prior_loss_weight_field_on_config")
    test_dreambooth_config_has_prior_loss_weight()
    print("PASS: dreambooth_config_has_prior_loss_weight")
    test_prior_preservation_adds_loss_term()
    print("PASS: prior_preservation_adds_loss_term")
    test_prior_loss_weighting_correct()
    print("PASS: prior_loss_weighting_correct")
    test_zero_weight_ignores_class_loss()
    print("PASS: zero_weight_ignores_class_loss")
    print("=" * 40)
    print("All SDXL DreamBooth smoke tests passed!")
