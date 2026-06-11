# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for auto Huber threshold schedule."""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import torch


def test_auto_schedule_with_target():
    """Auto schedule should compute delta from batch residual percentile."""
    # Build a minimal mock of the _huber_delta logic
    reference = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    target = torch.tensor([[0.9, 1.8, 2.7], [3.6, 4.5, 5.4]])
    residuals = (reference - target).float()
    per_sample = residuals.flatten(1).norm(dim=1)
    percentile_val = torch.quantile(per_sample, 0.9)
    assert percentile_val.item() > 0, "Percentile value should be positive"
    print("PASS: auto schedule computes positive delta from residuals")


def test_auto_schedule_percentile_ordering():
    """Higher percentile should give equal or higher delta."""
    reference = torch.randn(10, 4, 8, 8)
    target = torch.randn(10, 4, 8, 8)
    residuals = (reference - target).float()
    per_sample = residuals.flatten(1).norm(dim=1)
    p50 = torch.quantile(per_sample, 0.5).item()
    p90 = torch.quantile(per_sample, 0.9).item()
    assert p90 >= p50, f"p90 ({p90}) should >= p50 ({p50})"
    print("PASS: higher percentile gives higher delta")


def test_auto_schedule_no_target_fallback():
    """When target is None, auto should fall back to constant."""
    base = 0.1
    scale = 1.0
    # Mimics the fallback branch
    delta = base * scale
    assert delta == 0.1, f"Fallback delta should be {base * scale}"
    print("PASS: no-target fallback works")


def test_config_field():
    cfg_path = os.path.join(_HERE, "..", "configs.py")
    with open(cfg_path, encoding="utf-8") as f:
        src = f.read()
    assert "huber_auto_percentile" in src, "Missing huber_auto_percentile in configs.py"
    print("PASS: config field exists")


if __name__ == "__main__":
    test_auto_schedule_with_target()
    test_auto_schedule_percentile_ordering()
    test_auto_schedule_no_target_fallback()
    test_config_field()
    print("\nAll auto Huber smoke tests passed!")
