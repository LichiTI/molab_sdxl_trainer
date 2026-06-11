# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for SageAttention drift monitor."""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_attention_drift_monitor = _import_module(
    "attention_drift_monitor",
    os.path.join(_HERE, "attention_drift_monitor.py"),
)
AttentionDriftMonitor = _attention_drift_monitor.AttentionDriftMonitor

import torch
import torch.nn as nn


class _FakeAttnModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)
        self.head_dim = 16
        self.num_heads = 4

    def forward(self, x):
        return self.linear(x)


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _FakeAttnModule()


def test_no_sageattn_returns_zero():
    monitor = AttentionDriftMonitor(threshold=0.01)
    model = _FakeModel()
    # No _attention_backend attr set
    drift = monitor.check_drift(model)
    assert drift == 0.0, f"Expected 0.0, got {drift}"
    print("PASS: no sageattn modules returns 0.0")


def test_monitor_initialization():
    monitor = AttentionDriftMonitor(threshold=0.05, fallback="warn")
    assert monitor.threshold == 0.05
    assert monitor.fallback == "warn"
    assert monitor.last_drift == 0.0
    assert monitor.breach_count == 0
    print("PASS: monitor initialization correct")


def test_config_field():
    cfg_path = os.path.join(_HERE, "..", "configs.py")
    with open(cfg_path, encoding="utf-8") as f:
        src = f.read()
    assert "sageattn_drift_check_interval" in src
    assert "sageattn_drift_threshold" in src
    assert "sageattn_drift_fallback" in src
    print("PASS: config fields exist")


if __name__ == "__main__":
    test_no_sageattn_returns_zero()
    test_monitor_initialization()
    test_config_field()
    print("\nAll attention drift smoke tests passed!")
