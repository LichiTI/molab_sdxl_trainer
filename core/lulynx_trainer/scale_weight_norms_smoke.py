# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for scale_weight_norms.py (#67)."""

from __future__ import annotations

import os
import sys
import importlib.util

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.scale_weight_norms",
    os.path.join(_HERE, "scale_weight_norms.py"),
)
_swn = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.scale_weight_norms"] = _swn
_spec.loader.exec_module(_swn)


class _FakeLora(nn.Module):
    def __init__(self, dim=16, rank=4):
        super().__init__()
        self.lora_down = nn.Linear(dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, dim, bias=False)


class _FakeWrapper(nn.Module):
    def __init__(self, dim=16, rank=4):
        super().__init__()
        self.lora = _FakeLora(dim=dim, rank=rank)


class _FakeInjector:
    def __init__(self):
        self.injected_layers = {}


def _delta_norm(layer):
    up = layer.lora.lora_up.weight.float()
    down = layer.lora.lora_down.weight.float()
    return float(torch.linalg.norm(up @ down).item())


def test_clipper_reduces_norm_above_threshold():
    torch.manual_seed(0)
    inj = _FakeInjector()
    layer = _FakeWrapper(dim=16, rank=4)
    nn.init.normal_(layer.lora.lora_down.weight, std=2.0)
    nn.init.normal_(layer.lora.lora_up.weight, std=2.0)
    inj.injected_layers["block_0"] = layer

    norm_before = _delta_norm(layer)
    assert norm_before > 1.0, f"prereq: {norm_before}"

    clipper = _swn.LoRAWeightNormClipper(max_norm=1.0)
    clipper.register_from_injector(inj)
    count = clipper.step()
    assert count == 1

    norm_after = _delta_norm(layer)
    assert norm_after <= 1.0 + 1e-3
    print(f"PASS: clipper reduces norm {norm_before:.3f} -> {norm_after:.3f}")


def test_clipper_skips_when_below_threshold():
    inj = _FakeInjector()
    layer = _FakeWrapper(dim=16, rank=4)
    nn.init.zeros_(layer.lora.lora_up.weight)
    inj.injected_layers["block_0"] = layer

    clipper = _swn.LoRAWeightNormClipper(max_norm=1.0)
    clipper.register_from_injector(inj)
    count = clipper.step()
    assert count == 0
    print("PASS: clipper skips layers already under threshold")


def test_max_norm_zero_disables():
    inj = _FakeInjector()
    layer = _FakeWrapper()
    nn.init.normal_(layer.lora.lora_up.weight, std=5.0)
    inj.injected_layers["block_0"] = layer

    clipper = _swn.LoRAWeightNormClipper(max_norm=0.0)
    clipper.register_from_injector(inj)
    assert clipper.step() == 0
    print("PASS: max_norm=0 disables clipping")


def test_interval_skips_intermediate_steps():
    inj = _FakeInjector()
    layer = _FakeWrapper()
    nn.init.normal_(layer.lora.lora_up.weight, std=5.0)
    inj.injected_layers["block_0"] = layer

    clipper = _swn.LoRAWeightNormClipper(max_norm=1.0, interval=3)
    clipper.register_from_injector(inj)
    assert clipper.step() == 0  # step 1
    assert clipper.step() == 0  # step 2
    assert clipper.step() == 1  # step 3 -> applies
    print("PASS: interval skips intermediate steps")


def test_clipper_supports_lora_a_b_layout():
    """Adapters using lora_A/lora_B naming should also be clipped."""

    class _DoraLora(nn.Module):
        def __init__(self, dim=16, rank=4):
            super().__init__()
            self.lora_A = nn.Linear(dim, rank, bias=False)
            self.lora_B = nn.Linear(rank, dim, bias=False)
            nn.init.normal_(self.lora_A.weight, std=2.0)
            nn.init.normal_(self.lora_B.weight, std=2.0)

    inj = _FakeInjector()
    layer = nn.Module()
    layer.lora = _DoraLora()
    inj.injected_layers["block_0"] = layer

    norm_before = float(
        torch.linalg.norm(
            layer.lora.lora_B.weight.float() @ layer.lora.lora_A.weight.float()
        ).item()
    )
    assert norm_before > 1.0

    clipper = _swn.LoRAWeightNormClipper(max_norm=1.0)
    clipper.register_from_injector(inj)
    assert clipper.step() == 1
    norm_after = float(
        torch.linalg.norm(
            layer.lora.lora_B.weight.float() @ layer.lora.lora_A.weight.float()
        ).item()
    )
    assert norm_after <= 1.0 + 1e-3
    print(f"PASS: clipper handles lora_A/lora_B layout (norm {norm_before:.3f} -> {norm_after:.3f})")


def test_apply_weight_norm_clipping_oneshot():
    inj = _FakeInjector()
    layer = _FakeWrapper()
    nn.init.normal_(layer.lora.lora_up.weight, std=5.0)
    nn.init.normal_(layer.lora.lora_down.weight, std=5.0)
    inj.injected_layers["block_0"] = layer

    count = _swn.apply_weight_norm_clipping(inj, max_norm=0.5)
    assert count == 1
    assert _delta_norm(layer) <= 0.5 + 1e-3
    print("PASS: apply_weight_norm_clipping helper works one-shot")


if __name__ == "__main__":
    test_clipper_reduces_norm_above_threshold()
    test_clipper_skips_when_below_threshold()
    test_max_norm_zero_disables()
    test_interval_skips_intermediate_steps()
    test_clipper_supports_lora_a_b_layout()
    test_apply_weight_norm_clipping_oneshot()
    print("\nAll scale_weight_norms smoke tests passed!")
