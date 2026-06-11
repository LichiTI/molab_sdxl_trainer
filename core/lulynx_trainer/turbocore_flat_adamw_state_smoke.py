"""Smoke checks for the persistent flat AdamW state prototype."""

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

from core.turbocore_flat_adamw_state import (  # noqa: E402
    FlatAdamWConfig,
    PersistentFlatAdamW,
    clone_flat_adamw_state_dict,
)


def _make_values(device: torch.device) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1001)
    return [
        (torch.randn(4, 8, generator=generator) * 0.01).to(device=device),
        (torch.randn(8, 3, generator=generator) * 0.01).to(device=device),
        (torch.randn(5, generator=generator) * 0.01).to(device=device),
    ]


def _make_grads(device: torch.device, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return [
        (torch.randn(4, 8, generator=generator) * 0.001).to(device=device),
        (torch.randn(8, 3, generator=generator) * 0.001).to(device=device),
        (torch.randn(5, generator=generator) * 0.001).to(device=device),
    ]


def _make_params(values: list[torch.Tensor]) -> list[torch.nn.Parameter]:
    return [torch.nn.Parameter(value.detach().clone()) for value in values]


def _flatten(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.detach().float().reshape(-1) for tensor in tensors]).contiguous()


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 2e-7) -> None:
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().detach().cpu().item()) if diff.numel() else 0.0
    assert max_abs <= atol, max_abs


def test_flat_owner_matches_torch_adamw_cpu() -> None:
    device = torch.device("cpu")
    values = _make_values(device)
    ref_params = _make_params(values)
    cfg = FlatAdamWConfig(lr=1e-3, weight_decay=0.01, max_grad_norm=0.0)
    ref = torch.optim.AdamW(ref_params, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
    owner = PersistentFlatAdamW(values, cfg)

    for step in range(3):
        grads = _make_grads(device, 2000 + step)
        for param, grad in zip(ref_params, grads):
            param.grad = grad.detach().clone()
        ref.step()
        owner.set_grads(grads)
        report = owner.step()
        assert report.step_index == step + 1, report.as_dict()
        assert report.backend == "torch_flat_reference", report.as_dict()

    _assert_close(owner.param_flat, _flatten([param.detach() for param in ref_params]))


def test_state_dict_resume_and_zero_grad() -> None:
    device = torch.device("cpu")
    values = _make_values(device)
    cfg = FlatAdamWConfig(lr=5e-4, weight_decay=0.0, max_grad_norm=0.25)
    owner = PersistentFlatAdamW(values, cfg)
    owner.set_grads(_make_grads(device, 3000))
    first_report = owner.step()
    assert first_report.clipped is False
    state = clone_flat_adamw_state_dict(owner.state_dict())

    restored = PersistentFlatAdamW.from_state_dict(state)
    assert restored.config.lr == cfg.lr
    copy_targets = _make_params(values)
    restored.copy_params_to_(copy_targets)
    _assert_close(_flatten([param.detach() for param in copy_targets]), restored.param_flat)
    restored.set_grads(_make_grads(device, 3001))
    restored_report = restored.step()
    owner.set_grads(_make_grads(device, 3001))
    owner_report = owner.step()
    assert restored_report.step_index == owner_report.step_index == 2
    _assert_close(restored.param_flat, owner.param_flat)

    restored.zero_grad()
    assert float(restored.grad_flat.abs().sum().item()) == 0.0
    snapshot = restored.snapshot()
    assert snapshot["training_path_enabled"] is False
    assert "developer_only_layout_prototype" in snapshot["notes"]


def test_nonfinite_gradient_skips_without_param_change() -> None:
    device = torch.device("cpu")
    owner = PersistentFlatAdamW(_make_values(device), FlatAdamWConfig(finite_check=True))
    before = owner.param_flat.detach().clone()
    grads = _make_grads(device, 4000)
    grads[0] = grads[0].detach().clone()
    grads[0].view(-1)[0] = float("nan")
    owner.set_grads(grads)
    report = owner.step()
    assert report.skipped is True, report.as_dict()
    assert report.finite is False, report.as_dict()
    _assert_close(owner.param_flat, before, atol=0.0)


def test_triton_path_when_available() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    cfg = FlatAdamWConfig(lr=1e-3, weight_decay=0.01, prefer_triton=True, block_size=1024)
    owner = PersistentFlatAdamW(_make_values(device), cfg)
    owner.set_grads(_make_grads(device, 5000))
    try:
        report = owner.step()
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        assert "Python.h" in message or "compile" in message or "launch" in message, message
        return
    torch.cuda.synchronize(device)
    assert report.training_path_enabled is False
    assert bool(torch.isfinite(owner.param_flat).all().detach().cpu().item())
    if report.native_kernel_present:
        assert report.backend == "triton_adamw_flat_v0", report.as_dict()


def main() -> int:
    test_flat_owner_matches_torch_adamw_cpu()
    test_state_dict_resume_and_zero_grad()
    test_nonfinite_gradient_skips_without_param_change()
    test_triton_path_when_available()
    print("turbocore_flat_adamw_state_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
