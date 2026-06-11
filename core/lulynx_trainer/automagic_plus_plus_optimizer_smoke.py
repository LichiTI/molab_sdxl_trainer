# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for the Warehouse Automagic++ optimizer."""

from __future__ import annotations

import sys
from pathlib import Path
import importlib.util

import torch
from torch import nn

optimizer_path = Path(__file__).with_name("automagic_plus_plus_optimizer.py")
spec = importlib.util.spec_from_file_location("automagic_plus_plus_optimizer_local", optimizer_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load {optimizer_path}")
automagic_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = automagic_module
spec.loader.exec_module(automagic_module)
AutomagicPlusPlus = automagic_module.AutomagicPlusPlus


def _manual_step(param: torch.nn.Parameter, optimizer: AutomagicPlusPlus, grad_value: float) -> None:
    param.grad = torch.full_like(param, grad_value)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_lr_mask_sign_rules() -> None:
    param = nn.Parameter(torch.ones(4))
    optimizer = AutomagicPlusPlus([param], lr=1e-4, min_lr=1e-7, max_lr=1e-3, max_update_rms_ratio=None)

    _manual_step(param, optimizer, 1.0)
    lr_after_first = optimizer.state[param]["local_lr"].detach().clone()
    assert torch.allclose(lr_after_first, torch.full_like(lr_after_first, 1e-4))

    _manual_step(param, optimizer, 1.0)
    lr_after_same = optimizer.state[param]["local_lr"].detach().clone()
    assert torch.allclose(lr_after_same, lr_after_first * 1.01)

    _manual_step(param, optimizer, -1.0)
    lr_after_flip = optimizer.state[param]["local_lr"].detach().clone()
    assert torch.allclose(lr_after_flip, lr_after_same * 0.95)


def test_forward_backward_step_and_state_dict() -> None:
    model = nn.Sequential(nn.Linear(3, 5), nn.SiLU(), nn.Linear(5, 2))
    optimizer = AutomagicPlusPlus(model.parameters(), lr=1e-4, beta1=0.0)
    loss = model(torch.randn(4, 3)).square().mean()
    loss.backward()
    optimizer.step()

    saved = optimizer.state_dict()
    restored = AutomagicPlusPlus(model.parameters(), lr=1e-4)
    restored.load_state_dict(saved)
    assert restored.get_avg_learning_rate() > 0


def test_fp32_second_moment_for_low_precision_param() -> None:
    param = nn.Parameter(torch.ones(2, 2, dtype=torch.float16))
    optimizer = AutomagicPlusPlus([param], lr=1e-4)
    _manual_step(param, optimizer, 1.0)
    state = optimizer.state[param]
    assert state["row_var"].dtype == torch.float32
    assert state["col_var"].dtype == torch.float32


def test_parameter_swapping_keeps_at_least_one_param_trainable() -> None:
    large = nn.Parameter(torch.ones(128))
    small = nn.Parameter(torch.ones(1))
    optimizer = AutomagicPlusPlus(
        [large, small],
        lr=1e-4,
        do_parameter_swapping=True,
        parameter_swapping_factor=0.01,
    )
    assert any(param.requires_grad for group in optimizer.param_groups for param in group["params"])


def test_low_overhead_lr_granularity_uses_scalar_lr() -> None:
    param = nn.Parameter(torch.ones(2, 2))
    optimizer = AutomagicPlusPlus(
        [param],
        lr=1e-4,
        min_lr=1e-7,
        max_lr=1e-3,
        lr_granularity="low_overhead",
        max_update_rms_ratio=None,
    )

    _manual_step(param, optimizer, 1.0)
    lr_after_first = optimizer.state[param]["local_lr"].detach().clone()
    assert lr_after_first.shape == torch.Size([])
    assert torch.allclose(lr_after_first, torch.tensor(1e-4))

    _manual_step(param, optimizer, 1.0)
    lr_after_same = optimizer.state[param]["local_lr"].detach().clone()
    assert torch.allclose(lr_after_same, lr_after_first * 1.01)

    _manual_step(param, optimizer, -1.0)
    lr_after_flip = optimizer.state[param]["local_lr"].detach().clone()
    assert torch.allclose(lr_after_flip, lr_after_same * 0.95)


def main() -> int:
    test_lr_mask_sign_rules()
    test_forward_backward_step_and_state_dict()
    test_fp32_second_moment_for_low_precision_param()
    test_parameter_swapping_keeps_at_least_one_param_trainable()
    test_low_overhead_lr_granularity_uses_scalar_lr()
    print("Automagic++ optimizer smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

