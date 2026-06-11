"""Smoke checks for TurboCore update executor and direct-grad binding."""

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

from core.turbocore_update_executor import (  # noqa: E402
    TurboCoreUpdateExecutor,
    TurboCoreUpdateExecutorConfig,
)


def _make_params() -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(6060)
    values = [
        torch.randn(4, 3, generator=generator) * 0.1,
        torch.randn(5, generator=generator) * 0.1,
    ]
    left = [torch.nn.Parameter(value.detach().clone()) for value in values]
    right = [torch.nn.Parameter(value.detach().clone()) for value in values]
    return left, right


def _loss(params: list[torch.nn.Parameter]) -> torch.Tensor:
    return sum((param.float() * param.float()).sum() for param in params)


def _flatten(params: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([param.detach().float().reshape(-1) for param in params]).contiguous()


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 3e-7) -> None:
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().detach().cpu().item()) if diff.numel() else 0.0
    assert max_abs <= atol, max_abs


def test_executor_matches_torch_adamw_without_direct_grad() -> None:
    ref_params, tc_params = _make_params()
    cfg = TurboCoreUpdateExecutorConfig(lr=1e-3, weight_decay=0.01, max_grad_norm=0.0)
    ref = torch.optim.AdamW(ref_params, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
    executor = TurboCoreUpdateExecutor(tc_params, cfg)

    for step in range(3):
        ref.zero_grad(set_to_none=True)
        executor.zero_grad(set_to_none=True)
        _loss(ref_params).backward()
        _loss(tc_params).backward()
        ref.step()
        report = executor.step()
        assert report.step_index == step + 1, report.as_dict()
        assert report.used_direct_grad is False, report.as_dict()
        assert report.training_path_enabled is False, report.as_dict()
    _assert_close(_flatten(tc_params), _flatten(ref_params))


def test_executor_direct_grad_binding_matches_torch_adamw() -> None:
    ref_params, tc_params = _make_params()
    cfg = TurboCoreUpdateExecutorConfig(
        lr=7e-4,
        weight_decay=0.0,
        max_grad_norm=0.0,
        direct_grad=True,
    )
    ref = torch.optim.AdamW(ref_params, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
    executor = TurboCoreUpdateExecutor(tc_params, cfg)
    try:
        for step in range(4):
            ref.zero_grad(set_to_none=True)
            executor.zero_grad(set_to_none=True)
            _loss(ref_params).backward()
            _loss(tc_params).backward()
            assert float(executor.owner.grad_flat.abs().sum().item()) > 0.0
            ref.step()
            report = executor.step()
            assert report.step_index == step + 1, report.as_dict()
            assert report.used_direct_grad is True, report.as_dict()
            assert report.direct_grad_snapshot["hooks_installed"] == len(tc_params)
            assert report.direct_grad_snapshot["writes"] == len(tc_params)
        _assert_close(_flatten(tc_params), _flatten(ref_params))
        snapshot = executor.snapshot()
        assert snapshot["training_path_enabled"] is False
        assert snapshot["direct_grad"]["direct_grad_to_flat_owner"] is True
    finally:
        executor.close()


def test_nonfinite_skip_preserves_params() -> None:
    _ref, params = _make_params()
    cfg = TurboCoreUpdateExecutorConfig(lr=1e-3, finite_check=True, direct_grad=False)
    executor = TurboCoreUpdateExecutor(params, cfg)
    before = _flatten(params)
    for param in params:
        param.grad = torch.ones_like(param)
    params[0].grad.view(-1)[0] = float("inf")
    report = executor.step()
    assert report.owner_step["skipped"] is True, report.as_dict()
    _assert_close(_flatten(params), before, atol=0.0)


def test_cuda_triton_optional() -> None:
    if not torch.cuda.is_available():
        return
    ref_params, tc_params = _make_params()
    ref_params = [torch.nn.Parameter(param.detach().to("cuda")) for param in ref_params]
    tc_params = [torch.nn.Parameter(param.detach().to("cuda")) for param in tc_params]
    cfg = TurboCoreUpdateExecutorConfig(lr=1e-3, max_grad_norm=0.0, prefer_triton=True)
    ref = torch.optim.AdamW(ref_params, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
    executor = TurboCoreUpdateExecutor(tc_params, cfg)
    ref.zero_grad(set_to_none=True)
    executor.zero_grad(set_to_none=True)
    _loss(ref_params).backward()
    _loss(tc_params).backward()
    ref.step()
    report = executor.step()
    torch.cuda.synchronize()
    assert report.training_path_enabled is False
    _assert_close(_flatten(tc_params).cpu(), _flatten(ref_params).cpu(), atol=5e-5)


def main() -> int:
    test_executor_matches_torch_adamw_without_direct_grad()
    test_executor_direct_grad_binding_matches_torch_adamw()
    test_nonfinite_skip_preserves_params()
    test_cuda_triton_optional()
    print("turbocore_update_executor_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

