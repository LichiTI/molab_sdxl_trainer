# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""AdaHessian create_graph/resume contract smoke."""

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
from core.lulynx_trainer.optimizer_step_contracts import optimizer_requires_create_graph_backward


def _loss(param: torch.nn.Parameter, target: torch.Tensor) -> torch.Tensor:
    return ((param * param) - target).pow(2).mean()


def _step(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer, target: torch.Tensor) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = _loss(param, target)
    loss.backward(create_graph=optimizer_requires_create_graph_backward(optimizer))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _make(value: torch.Tensor) -> tuple[torch.nn.Parameter, torch.optim.Optimizer]:
    param = torch.nn.Parameter(value.detach().clone())
    optimizer = create_pytorch_optimizer(
        [param],
        optimizer_name="AdaHessian",
        lr=1e-2,
        weight_decay=0.0,
        optimizer_args={"name": "AdaHessian", "num_samples": 1, "hessian_distribution": "rademacher"},
    )
    return param, optimizer


def test_adahessian_create_graph_resume_parity() -> None:
    torch.manual_seed(20260531)
    initial = torch.tensor([[0.25, -0.5], [0.75, -1.0]], dtype=torch.float32)
    target1 = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    target2 = torch.tensor([[0.2, -0.1], [0.5, -0.3]], dtype=torch.float32)

    param, optimizer = _make(initial)
    assert optimizer_requires_create_graph_backward(optimizer) is True
    _step(param, optimizer, target1)

    state = copy.deepcopy(optimizer.state_dict())
    restored_param, restored_optimizer = _make(param.detach())
    restored_optimizer.load_state_dict(state)

    torch.manual_seed(20260531)
    _step(param, optimizer, target2)
    torch.manual_seed(20260531)
    _step(restored_param, restored_optimizer, target2)

    assert torch.allclose(param.detach(), restored_param.detach(), atol=1e-6, rtol=1e-5)


def main() -> int:
    test_adahessian_create_graph_resume_parity()
    print("optimizer_adahessian_contract_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
