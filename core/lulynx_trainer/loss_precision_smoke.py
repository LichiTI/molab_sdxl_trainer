# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test loss precision strategy plumbing without starting a trainer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch

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
_load_module("core.constants", CORE_ROOT / "constants.py")
training_loop_mod = _load_module("core.lulynx_trainer.training_loop", TRAINER_ROOT / "training_loop.py")

TrainingLoop = training_loop_mod.TrainingLoop


def _make_loop(mode: str):
    loop = TrainingLoop.__new__(TrainingLoop)
    loop.loss_precision = TrainingLoop._normalize_loss_precision(mode)
    return loop


def _test_mode_normalization() -> None:
    assert TrainingLoop._normalize_loss_precision("fp32") == "fp32_loss"
    assert TrainingLoop._normalize_loss_precision("mixed") == "mixed_loss"
    assert TrainingLoop._normalize_loss_precision("native") == "mixed_loss"
    assert TrainingLoop._normalize_loss_precision("unknown") == "fp32_loss"


def _test_fp32_loss_operands() -> None:
    loop = _make_loop("fp32_loss")
    prediction = torch.ones(2, dtype=torch.bfloat16)
    target = torch.zeros(2, dtype=torch.bfloat16)
    loss_prediction, loss_target = loop._loss_operands(prediction, target)
    assert loss_prediction.dtype == torch.float32
    assert loss_target.dtype == torch.float32


def _test_mixed_loss_operands() -> None:
    loop = _make_loop("mixed_loss")
    prediction = torch.ones(2, dtype=torch.bfloat16)
    target = torch.zeros(2, dtype=torch.bfloat16)
    loss_prediction, loss_target = loop._loss_operands(prediction, target)
    assert loss_prediction is prediction
    assert loss_target is target
    assert loss_prediction.dtype == torch.bfloat16


def main() -> int:
    _test_mode_normalization()
    print("  loss precision mode normalization -- PASS")
    _test_fp32_loss_operands()
    print("  fp32 loss operands -- PASS")
    _test_mixed_loss_operands()
    print("  mixed loss operands -- PASS")
    print("Loss precision smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
