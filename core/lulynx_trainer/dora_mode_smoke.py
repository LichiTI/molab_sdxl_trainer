"""Smoke tests for DoRA full/style/structure lock modes."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx.dora_layer import DoRALinear


def _check_mode(mode: str, normalized: str, expected_direction: bool, expected_magnitude: bool) -> None:
    layer = DoRALinear(nn.Linear(4, 3), rank=2, alpha=2, mode=mode)
    assert layer.mode == normalized
    assert layer.lora_A.requires_grad is expected_direction
    assert layer.lora_B.requires_grad is expected_direction
    assert layer.m.requires_grad is expected_magnitude

    x = torch.randn(5, 4)
    y = layer(x).sum()
    trainable = [param for param in layer.parameters() if param.requires_grad]
    assert trainable, mode
    y.backward()
    for name, param in layer.named_parameters():
        if name.startswith("base_"):
            continue
        if param.requires_grad:
            assert param.grad is not None, (mode, name)
        else:
            assert param.grad is None, (mode, name)


def main() -> int:
    _check_mode("full", "full", True, True)
    _check_mode("style", "style", False, True)
    _check_mode("style_lock", "style", False, True)
    _check_mode("structure", "structure", True, False)
    _check_mode("structure_lock", "structure", True, False)
    print("dora_mode_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
