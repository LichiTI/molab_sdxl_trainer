# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for gradient_release per-parameter real optimizer upgrade."""

from __future__ import annotations

import torch
import torch.nn as nn

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = BACKEND_ROOT / "core"
TRAINER_ROOT = CORE_ROOT / "lulynx_trainer"


def _ensure_namespace(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name: str, path: Path):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ensure_namespace("core", CORE_ROOT)
_ensure_namespace("core.lulynx_trainer", TRAINER_ROOT)
gr_mod = _load_module(
    "core.lulynx_trainer.gradient_release",
    TRAINER_ROOT / "gradient_release.py",
)

GradientReleaseManager = gr_mod.GradientReleaseManager
is_gradient_release_available = gr_mod.is_gradient_release_available
_rescale_betas = gr_mod._rescale_betas


def _make_model():
    model = nn.Linear(8, 4, bias=False)
    model.weight.data.fill_(1.0)
    return model


def test_post_step_unchanged() -> None:
    """post_step mode still works: grads freed after explicit call."""
    model = _make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mgr = GradientReleaseManager(mode="post_step")
    mgr.register_parameters(model.parameters(), opt)

    x = torch.randn(2, 8)
    loss = model(x).sum()
    loss.backward()

    assert model.weight.grad is not None, "Grad should exist before release"
    opt.step()
    released = mgr.release_gradients_after_step()
    assert released == 1
    assert model.weight.grad is None, "Grad should be None after release"
    assert mgr.needs_external_optimizer_step is True
    mgr.cleanup()
    print("  PASS: test_post_step_unchanged")


def test_during_backward_real_adamw() -> None:
    """during_backward with real AdamW: params updated, grads freed."""
    if not is_gradient_release_available():
        print("  SKIP: test_during_backward_real_adamw (PyTorch < 2.1)")
        return

    model = _make_model()
    initial_weight = model.weight.data.clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    mgr = GradientReleaseManager(mode="during_backward")
    mgr.register_parameters(model.parameters(), opt)

    x = torch.randn(2, 8)
    loss = model(x).sum()
    loss.backward()

    assert model.weight.grad is None, "Hook should have freed grad"
    assert not torch.equal(model.weight.data, initial_weight), "Weight should be updated"
    assert mgr.needs_external_optimizer_step is False
    mgr.cleanup()
    print("  PASS: test_during_backward_real_adamw")


def test_during_backward_real_sgd() -> None:
    """during_backward with real SGD (no betas): params updated, grads freed."""
    if not is_gradient_release_available():
        print("  SKIP: test_during_backward_real_sgd (PyTorch < 2.1)")
        return

    model = _make_model()
    initial_weight = model.weight.data.clone()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    mgr = GradientReleaseManager(mode="during_backward")
    mgr.register_parameters(model.parameters(), opt)

    x = torch.randn(2, 8)
    loss = model(x).sum()
    loss.backward()

    assert model.weight.grad is None, "Hook should have freed grad"
    assert not torch.equal(model.weight.data, initial_weight), "Weight should be updated"
    mgr.cleanup()
    print("  PASS: test_during_backward_real_sgd")


def test_beta_rescaling() -> None:
    """Beta rescaling: betas^(1/gas) for gradient accumulation."""
    defaults = {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}
    rescaled = _rescale_betas(defaults, gas=4)
    b1, b2 = rescaled["betas"]
    expected_b1 = 0.9 ** 0.25
    expected_b2 = 0.999 ** 0.25
    assert abs(b1 - expected_b1) < 1e-10, f"beta1 {b1} != {expected_b1}"
    assert abs(b2 - expected_b2) < 1e-10, f"beta2 {b2} != {expected_b2}"

    no_change = _rescale_betas(defaults, gas=1)
    assert no_change["betas"] == (0.9, 0.999), "gas=1 should not change betas"

    sgd_defaults = {"lr": 0.01, "momentum": 0.9}
    sgd_rescaled = _rescale_betas(sgd_defaults, gas=4)
    assert "betas" not in sgd_rescaled, "SGD has no betas to rescale"
    print("  PASS: test_beta_rescaling")


def test_lr_sync() -> None:
    """sync_learning_rate propagates LR from main optimizer to per-param instances."""
    if not is_gradient_release_available():
        print("  SKIP: test_lr_sync (PyTorch < 2.1)")
        return

    model = _make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mgr = GradientReleaseManager(mode="during_backward")
    mgr.register_parameters(model.parameters(), opt)

    for pg in opt.param_groups:
        pg["lr"] = 5e-4

    mgr.sync_learning_rate()

    for per_opt in mgr._per_param_optimizers.values():
        for pg in per_opt.param_groups:
            assert pg["lr"] == 5e-4, f"Expected 5e-4, got {pg['lr']}"

    mgr.cleanup()
    print("  PASS: test_lr_sync")


def test_no_boundary_needed() -> None:
    """Hooks fire regardless of step_context boundary flag."""
    if not is_gradient_release_available():
        print("  SKIP: test_no_boundary_needed (PyTorch < 2.1)")
        return

    model = _make_model()
    initial_weight = model.weight.data.clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    mgr = GradientReleaseManager(mode="during_backward")
    mgr.register_parameters(model.parameters(), opt)

    x = torch.randn(2, 8)
    loss = model(x).sum()
    with mgr.step_context(is_accumulation_boundary=False):
        loss.backward()

    assert model.weight.grad is None, "Hook should fire even when boundary=False"
    assert not torch.equal(model.weight.data, initial_weight), "Weight should be updated"
    mgr.cleanup()
    print("  PASS: test_no_boundary_needed")


def test_needs_external_optimizer_step() -> None:
    """post_step needs external step, during_backward does not."""
    mgr_post = GradientReleaseManager(mode="post_step")
    assert mgr_post.needs_external_optimizer_step is True

    if is_gradient_release_available():
        mgr_db = GradientReleaseManager(mode="during_backward")
        assert mgr_db.needs_external_optimizer_step is False
    print("  PASS: test_needs_external_optimizer_step")


def main() -> int:
    print("Gradient Release Smoke Tests")
    print("=" * 40)
    test_post_step_unchanged()
    test_during_backward_real_adamw()
    test_during_backward_real_sgd()
    test_beta_rescaling()
    test_lr_sync()
    test_no_boundary_needed()
    test_needs_external_optimizer_step()
    print("=" * 40)
    print("All gradient release smoke tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
