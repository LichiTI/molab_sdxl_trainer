"""Smoke tests for the TurboCore stateful optimizer ABI prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_native_abi import validate_native_optimizer_stateful_capability  # noqa: E402
from core.turbocore_optimizer_abi import (  # noqa: E402
    AdamWStatefulOptimizerConfig,
    PyTorchStatefulAdamWBackend,
    build_native_optimizer_stateful_capability_stub,
)
from core.turbocore_parity import check_stateful_native_optimizer_parity  # noqa: E402
from core.turbocore_workspace_pipeline import build_turbocore_native_training_capability_stub  # noqa: E402


def _make_params() -> list[torch.nn.Parameter]:
    torch.manual_seed(123)
    return [
        torch.nn.Parameter(torch.randn(4, 8) * 0.01),
        torch.nn.Parameter(torch.randn(8, 4) * 0.01),
    ]


def _seed_grads(params: list[torch.nn.Parameter], seed: int) -> None:
    torch.manual_seed(seed)
    for param in params:
        param.grad = torch.randn_like(param)


def test_stateful_backend_lifecycle() -> None:
    params = _make_params()
    backend = PyTorchStatefulAdamWBackend(
        params,
        AdamWStatefulOptimizerConfig(lr=1e-3, weight_decay=0.0, max_grad_norm=0.5),
    )
    _seed_grads(params, 900)
    report = backend.step()
    assert report.step_index == 1, report.as_dict()
    assert report.state.state_tensors >= 1, report.as_dict()
    backend.zero_grad()
    assert all(param.grad is None for param in params)
    snapshot = backend.snapshot()
    assert snapshot["training_path_enabled"] is False
    assert snapshot["native_kernel_present"] is False


def test_stateful_parity_suite() -> None:
    result = check_stateful_native_optimizer_parity(
        layers=2,
        in_features=16,
        out_features=16,
        rank=4,
        steps=3,
        device="cpu",
        dtype=torch.float32,
    )
    assert result.ok, result.as_dict()
    assert result.details and result.details["restore_ok"] is True
    assert result.details["nonfinite_skip_ok"] is True


def test_capability_stub_and_validator() -> None:
    feature = build_native_optimizer_stateful_capability_stub()
    assert feature["stateful"] is True
    assert "optimizer_state_dict" in feature["entrypoints"]
    report = build_turbocore_native_training_capability_stub()
    validation = validate_native_optimizer_stateful_capability(report)
    assert validation["ok"] is True, validation
    native_feature = validation["features"]["native_optimizer"]
    assert native_feature["stateful"] is True, native_feature
    assert "AdamW" in native_feature["supported_optimizers"], native_feature


if __name__ == "__main__":
    test_stateful_backend_lifecycle()
    test_stateful_parity_suite()
    test_capability_stub_and_validator()
    print("turbocore_stateful_optimizer_smoke: ok")
