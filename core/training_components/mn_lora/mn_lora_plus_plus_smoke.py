"""Smoke tests for MN-LoRA++ adaptive delta scaling."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

BACKEND_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.training_components.mn_lora.mn_optimizer import MNLoRAOptimizer


def _step(param: nn.Parameter, optimizer: MNLoRAOptimizer, grad: torch.Tensor) -> float:
    before = param.detach().clone()
    param.grad = grad.clone()
    optimizer.step()
    return float((param.detach() - before).norm())


def test_stable_then_flip_adapts() -> None:
    param = nn.Parameter(torch.ones(4, 2))
    base = torch.optim.SGD([param], lr=0.1)
    optimizer = MNLoRAOptimizer(
        base,
        enable_tgwd=False,
        enable_gsp=False,
        enable_pilot=False,
        plus_plus_config={
            "enabled": True,
            "lr_up": 1.1,
            "lr_down": 0.5,
            "min_mult": 0.25,
            "max_mult": 2.0,
            "update_rms_cap": 0.0,
        },
        param_names={id(param): "block.lora_down.weight"},
    )
    first = _step(param, optimizer, torch.ones_like(param))
    second = _step(param, optimizer, torch.ones_like(param))
    flipped = _step(param, optimizer, -torch.ones_like(param))
    assert second > first, f"Expected stable gradient to increase update, got {first} -> {second}"
    assert flipped < second, f"Expected flipped gradient to reduce update, got {second} -> {flipped}"


def test_clamp_and_state_dict() -> None:
    param = nn.Parameter(torch.ones(3, 2))
    optimizer = MNLoRAOptimizer(
        torch.optim.SGD([param], lr=0.1),
        enable_tgwd=False,
        enable_gsp=False,
        enable_pilot=False,
        plus_plus_config={
            "enabled": True,
            "lr_up": 3.0,
            "lr_down": 0.1,
            "min_mult": 0.5,
            "max_mult": 1.25,
            "update_rms_cap": 0.0,
        },
        param_names={id(param): "block.lora_down.weight"},
    )
    for _ in range(4):
        _step(param, optimizer, torch.ones_like(param))
    state = optimizer.state_dict()
    assert "mn_lora_plus_plus" in state
    controller_state = state["mn_lora_plus_plus"]["state"][str(id(param))]
    assert float(controller_state["module_mult"]) <= 1.25

    restored = MNLoRAOptimizer(
        torch.optim.SGD([nn.Parameter(torch.ones(3, 2))], lr=0.1),
        enable_tgwd=False,
        enable_gsp=False,
        enable_pilot=False,
        plus_plus_config={"enabled": True},
    )
    restored.load_state_dict(state)
    assert restored.plus_plus is not None
    assert restored.plus_plus.state_dict()["state"]


def main() -> int:
    test_stable_then_flip_adapts()
    print("  MN-LoRA++ stable/flip adaptation -- PASS")
    test_clamp_and_state_dict()
    print("  MN-LoRA++ clamp + state_dict -- PASS")
    print("MN-LoRA++ smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
