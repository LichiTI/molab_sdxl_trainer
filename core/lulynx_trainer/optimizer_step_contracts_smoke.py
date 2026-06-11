# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for optimizer step contracts used by TrainingLoop."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.optimizer_plugin_bridge import create_pytorch_optimizer
from core.lulynx_trainer.optimizer_step_contracts import (
    bind_loss_value_closure,
    optimizer_requires_create_graph_backward,
)


class _Wrapper:
    def __init__(self, base) -> None:
        self._base = base


def _make_alig_optimizer() -> tuple[torch.nn.Parameter, torch.optim.Optimizer]:
    param = torch.nn.Parameter(torch.tensor([1.0, -0.5]))
    optimizer = create_pytorch_optimizer(
        [param],
        optimizer_name="AliG",
        lr=1e-3,
        weight_decay=0.0,
        optimizer_args={"name": "AliG"},
    )
    return param, optimizer


def test_loss_value_contract_reaches_wrapped_optimizer() -> None:
    _, optimizer = _make_alig_optimizer()
    wrapped = _Wrapper(_Wrapper(optimizer))
    assert bind_loss_value_closure(wrapped, 0.125) == 1
    assert getattr(optimizer, "_lulynx_loss_value_for_step") == 0.125


def test_alig_requires_and_consumes_bound_loss_value() -> None:
    param, optimizer = _make_alig_optimizer()
    param.grad = torch.tensor([0.25, -0.125])
    try:
        optimizer.step()
    except RuntimeError as exc:
        assert "loss value" in str(exc), exc
    else:
        raise AssertionError("AliG wrapper stepped without a bound loss value.")

    param.grad = torch.tensor([0.25, -0.125])
    bind_loss_value_closure(optimizer, 0.25)
    optimizer.step()
    assert getattr(optimizer, "_lulynx_loss_value_for_step") is None


def test_create_graph_backward_contract_reaches_wrapped_optimizer() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, -0.5]))
    optimizer = create_pytorch_optimizer(
        [param],
        optimizer_name="AdaHessian",
        lr=1e-3,
        weight_decay=0.0,
        optimizer_args={"name": "AdaHessian"},
    )
    assert optimizer_requires_create_graph_backward(_Wrapper(optimizer)) is True


def main() -> int:
    test_loss_value_contract_reaches_wrapped_optimizer()
    test_alig_requires_and_consumes_bound_loss_value()
    test_create_graph_backward_contract_reaches_wrapped_optimizer()
    print("optimizer_step_contracts_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
