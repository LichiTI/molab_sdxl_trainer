"""Smoke tests for the Warehouse AutoProdigy optimizer."""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = BACKEND_ROOT / "core"
TRAINER_ROOT = CORE_ROOT / "lulynx_trainer"


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


_install_xformers_stub()
_ensure_namespace("core", CORE_ROOT)
_ensure_namespace("core.lulynx_trainer", TRAINER_ROOT)
configs_mod = _load_module("core.configs", CORE_ROOT / "configs.py")
optim_mod = _load_module("core.lulynx_trainer.auto_prodigy_optimizer", TRAINER_ROOT / "auto_prodigy_optimizer.py")
trainer_mod = _load_module("core.lulynx_trainer.trainer", TRAINER_ROOT / "trainer.py")

AutoProdigy = optim_mod.AutoProdigy
OptimizerType = configs_mod.OptimizerType
SchedulerType = configs_mod.SchedulerType
LulynxTrainer = trainer_mod.LulynxTrainer


def test_single_param_step() -> None:
    param = nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = AutoProdigy([param], lr=1.0, d0=1e-5, growth_rate=1.01)
    for _ in range(3):
        loss = (param.square()).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    assert torch.isfinite(param).all()
    assert optimizer._ap_global["distance"] >= 1e-5


def test_mlp_and_eval_train() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.SiLU(), nn.Linear(4, 1))
    optimizer = AutoProdigy(model.parameters(), lr=1.0, d0=1e-5)
    x = torch.randn(5, 3)
    y = model(x).sum()
    y.backward()
    optimizer.step()
    shapes = [p.shape for p in model.parameters()]
    dtypes = [p.dtype for p in model.parameters()]
    optimizer.eval()
    assert [p.shape for p in model.parameters()] == shapes
    assert [p.dtype for p in model.parameters()] == dtypes
    optimizer.train()
    assert [p.shape for p in model.parameters()] == shapes


def test_trainer_route_and_scheduler_guard() -> None:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = SimpleNamespace(
        optimizer=OptimizerType.AUTO_PRODIGY,
        optimizer_args="d0=1e-5,unknown=1",
        learning_rate=1.0,
        weight_decay=0.0,
        semantic_tuner_enabled=True,
        scheduler=SchedulerType.COSINE,
        warmup_ratio=0.0,
        lr_scheduler_num_cycles=1,
        lr_scheduler_args="",
        mn_lora_enabled=False,
    )
    trainer._log = lambda _msg: None
    trainer.trainable_params = [nn.Parameter(torch.ones(2))]
    optimizer = trainer._create_optimizer()
    assert optimizer.__class__.__name__ == "AutoProdigy"
    scheduler = trainer._create_scheduler(optimizer, total_steps=10)
    assert scheduler.__class__.__name__ == "ConstantLR"


def main() -> int:
    test_single_param_step()
    print("  AutoProdigy single-param step -- PASS")
    test_mlp_and_eval_train()
    print("  AutoProdigy MLP + train/eval -- PASS")
    test_trainer_route_and_scheduler_guard()
    print("  AutoProdigy trainer route + scheduler guard -- PASS")
    print("AutoProdigy smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

