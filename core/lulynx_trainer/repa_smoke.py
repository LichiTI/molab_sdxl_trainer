# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for REPA alignment primitives."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn

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
repa_mod = _load_module("core.lulynx_trainer.repa", TRAINER_ROOT / "repa.py")

REPALossConfig = repa_mod.REPALossConfig
REPAFeatureProjector = repa_mod.REPAFeatureProjector
REPAFeatureCapture = repa_mod.REPAFeatureCapture
SoftREPAConfig = repa_mod.SoftREPAConfig
repa_alignment_loss = repa_mod.repa_alignment_loss
softrepa_weight = repa_mod.softrepa_weight


class _Block(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def test_repa_loss_cosine_and_l2() -> bool:
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    target = hidden.detach().clone()
    cosine = repa_alignment_loss(hidden, target, REPALossConfig(loss_type="cosine"))
    l2 = repa_alignment_loss(hidden, target, REPALossConfig(loss_type="l2"))
    assert cosine.abs() < 1e-5
    assert l2.abs() < 1e-8
    (cosine + l2).backward()
    assert hidden.grad is not None
    print("PASS: test_repa_loss_cosine_and_l2")
    return True


def test_projector_and_capture() -> bool:
    model = _Model()
    capture = REPAFeatureCapture(model, ["blocks.0", "blocks.1"]).install()
    x = torch.randn(2, 3, 8)
    y = model(x)
    assert y.shape == (2, 3, 8)
    assert set(capture.features) == {"blocks.0", "blocks.1"}
    projector = REPAFeatureProjector(hidden_dim=8, projection_dim=4)
    projected = projector(capture.features["blocks.1"])
    assert projected.shape == (2, 3, 4)
    target = torch.randn_like(projected)
    loss = repa_alignment_loss(capture.features["blocks.1"], target, REPALossConfig(loss_type="l1"), projector)
    loss.backward()
    assert any(p.grad is not None for p in projector.parameters())
    capture.remove()
    assert capture.features == {}
    print("PASS: test_projector_and_capture")
    return True


def test_softrepa_schedule_and_sigma_window() -> bool:
    cfg = SoftREPAConfig(enabled=True, schedule="linear", min_weight=0.1, max_weight=0.5, sigma_window=(0.2, 0.8))
    assert abs(softrepa_weight(0, 11, torch.tensor([0.5]), cfg) - 0.1) < 1e-6
    assert abs(softrepa_weight(10, 11, torch.tensor([0.5]), cfg) - 0.5) < 1e-6
    assert softrepa_weight(5, 11, torch.tensor([0.9]), cfg) == 0.0
    constant = SoftREPAConfig(enabled=True, schedule="constant", min_weight=0.1, max_weight=0.5, sigma_window=(0.0, 1.0))
    assert abs(softrepa_weight(0, 11, None, constant) - 0.5) < 1e-6
    print("PASS: test_softrepa_schedule_and_sigma_window")
    return True


def main() -> int:
    tests = [test_repa_loss_cosine_and_l2, test_projector_and_capture, test_softrepa_schedule_and_sigma_window]
    results = []
    for test_fn in tests:
        try:
            results.append((test_fn.__name__, test_fn()))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"FAIL: {test_fn.__name__} — {exc}")
            results.append((test_fn.__name__, False))
    passed = sum(1 for _, ok in results if ok)
    print("\n" + "=" * 60)
    print("REPA Smoke Test Results")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n{passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
