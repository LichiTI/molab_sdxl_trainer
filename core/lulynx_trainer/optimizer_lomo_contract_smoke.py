# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""LOMO-family fused-backward/resume contract smoke."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.optimizer_plugin_bridge import create_pytorch_optimizer
from core.lulynx_trainer.optimizer_step_contracts import (
    optimizer_uses_fused_backward,
    run_optimizer_fused_backward,
)


def _make(name: str, value: torch.Tensor) -> tuple[torch.nn.Parameter, torch.optim.Optimizer]:
    param = torch.nn.Parameter(value.detach().clone())
    optimizer = create_pytorch_optimizer(
        [param],
        optimizer_name=name,
        lr=1e-3,
        weight_decay=0.0,
        optimizer_args={"name": name, "loss_scale": 1.0} if name.lower() == "adalomo" else {"name": name},
    )
    return param, optimizer


def _loss(param: torch.nn.Parameter, target: torch.Tensor) -> torch.Tensor:
    return ((param * param) - target).pow(2).mean()


def _step(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer, target: torch.Tensor) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = _loss(param, target)
    assert run_optimizer_fused_backward(optimizer, loss, float(optimizer.param_groups[0]["lr"])) is True
    optimizer.zero_grad(set_to_none=True)


def _assert_resume_parity(name: str) -> None:
    initial = torch.tensor([[0.25, -0.5], [0.75, -1.0]], dtype=torch.float32)
    target1 = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    target2 = torch.tensor([[0.2, -0.1], [0.5, -0.3]], dtype=torch.float32)

    param, optimizer = _make(name, initial)
    assert optimizer_uses_fused_backward(optimizer) is True
    _step(param, optimizer, target1)

    state = copy.deepcopy(optimizer.state_dict())
    restored_param, restored_optimizer = _make(name, param.detach())
    restored_optimizer.load_state_dict(state)

    _step(param, optimizer, target2)
    _step(restored_param, restored_optimizer, target2)

    assert torch.allclose(param.detach(), restored_param.detach(), atol=1e-6, rtol=1e-5), name


def test_lomo_resume_parity() -> None:
    _assert_resume_parity("LOMO")


def test_adalomo_resume_parity() -> None:
    _assert_resume_parity("AdaLOMO")


def main() -> int:
    test_lomo_resume_parity()
    test_adalomo_resume_parity()
    print("optimizer_lomo_contract_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
