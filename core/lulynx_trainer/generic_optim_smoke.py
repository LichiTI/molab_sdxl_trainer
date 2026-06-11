# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for GenericOptimizer: 3-tier resolution chain."""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import ModuleSpec
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


def _install_xformers_stub() -> None:
    sys.modules.pop("xformers", None)
    sys.modules.pop("xformers.ops", None)
    xformers_module = ModuleType("xformers")
    ops_module = ModuleType("xformers.ops")
    ops_module.__spec__ = ModuleSpec("xformers.ops", loader=None)
    xformers_module.ops = ops_module
    xformers_module.__spec__ = ModuleSpec("xformers", loader=None)
    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = ops_module


_install_xformers_stub()
_ensure_namespace("core", CORE_ROOT)
_ensure_namespace("core.lulynx_trainer", TRAINER_ROOT)
configs_mod = _load_module("core.configs", CORE_ROOT / "configs.py")
bridge_mod = _load_module(
    "core.lulynx_trainer.optimizer_plugin_bridge",
    TRAINER_ROOT / "optimizer_plugin_bridge.py",
)

OptimizerType = configs_mod.OptimizerType
create_generic_optimizer = bridge_mod.create_generic_optimizer


def _dummy_params():
    return [nn.Parameter(torch.randn(4, 4))]


def test_enum_exists() -> None:
    assert OptimizerType.GENERIC.value == "GenericOptimizer"
    assert OptimizerType("GenericOptimizer") == OptimizerType.GENERIC
    print("  PASS: test_enum_exists")


def test_torch_optim_sgd() -> None:
    opt = create_generic_optimizer(
        _dummy_params(), optimizer_name="SGD", lr=0.01,
        weight_decay=0.0, optimizer_args={"name": "SGD", "momentum": 0.9},
    )
    assert isinstance(opt, torch.optim.SGD), f"Expected SGD, got {type(opt)}"
    print("  PASS: test_torch_optim_sgd")


def test_torch_optim_adamw() -> None:
    opt = create_generic_optimizer(
        _dummy_params(), optimizer_name="AdamW", lr=1e-4,
        weight_decay=0.01, optimizer_args={"name": "AdamW"},
    )
    assert isinstance(opt, torch.optim.AdamW), f"Expected AdamW, got {type(opt)}"
    print("  PASS: test_torch_optim_adamw")


def test_case_insensitive() -> None:
    opt = create_generic_optimizer(
        _dummy_params(), optimizer_name="adamw", lr=1e-4,
        weight_decay=0.01, optimizer_args={"name": "adamw"},
    )
    assert isinstance(opt, torch.optim.AdamW), f"Expected AdamW, got {type(opt)}"
    print("  PASS: test_case_insensitive")


def test_dotted_path() -> None:
    opt = create_generic_optimizer(
        _dummy_params(), optimizer_name="torch.optim.Adam", lr=1e-4,
        weight_decay=0.0, optimizer_args={"name": "torch.optim.Adam"},
    )
    assert isinstance(opt, torch.optim.Adam), f"Expected Adam, got {type(opt)}"
    print("  PASS: test_dotted_path")


def test_unknown_raises() -> None:
    try:
        create_generic_optimizer(
            _dummy_params(), optimizer_name="NoSuchOptimizer9999", lr=1e-4,
            weight_decay=0.0, optimizer_args={"name": "NoSuchOptimizer9999"},
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot resolve" in str(e).lower(), f"Unexpected error: {e}"
    print("  PASS: test_unknown_raises")


def test_kwargs_forwarded() -> None:
    opt = create_generic_optimizer(
        _dummy_params(), optimizer_name="SGD", lr=0.01,
        weight_decay=0.0, optimizer_args={"name": "SGD", "momentum": 0.9},
    )
    assert opt.defaults.get("momentum") == 0.9, (
        f"Expected momentum=0.9, got {opt.defaults.get('momentum')}"
    )
    print("  PASS: test_kwargs_forwarded")


def test_empty_name_raises() -> None:
    try:
        create_generic_optimizer(
            _dummy_params(), optimizer_name="", lr=1e-4,
            weight_decay=0.0, optimizer_args={},
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  PASS: test_empty_name_raises")


def main() -> int:
    print("GenericOptimizer Smoke Tests")
    print("=" * 40)
    test_enum_exists()
    test_torch_optim_sgd()
    test_torch_optim_adamw()
    test_case_insensitive()
    test_dotted_path()
    test_unknown_raises()
    test_kwargs_forwarded()
    test_empty_name_raises()
    print("=" * 40)
    print("All GenericOptimizer smoke tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
