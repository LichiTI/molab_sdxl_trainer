# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test for Anima ReFT trainer wiring helpers."""
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
_load_module("core.safe_pickle", CORE_ROOT / "safe_pickle.py")
_load_module("core.configs", CORE_ROOT / "configs.py")
_load_module("core.lulynx_trainer.config", TRAINER_ROOT / "config.py")
config_adapter_mod = _load_module("core.lulynx_trainer.config_adapter", TRAINER_ROOT / "config_adapter.py")
reft_mod = _load_module("core.lulynx_trainer.reft", TRAINER_ROOT / "reft.py")

ConfigAdapter = config_adapter_mod.ConfigAdapter
install_reft = reft_mod.install_reft
get_reft_params = reft_mod.get_reft_params
get_reft_state_dict = reft_mod.get_reft_state_dict


class _TinyBlock(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _TinyAnimaDiT(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Module()
        self.net.blocks = nn.ModuleList([_TinyBlock(dim), _TinyBlock(dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.net.blocks:
            x = block(x)
        return x


def test_config_aliases() -> bool:
    cfg = ConfigAdapter.from_frontend_dict({
        "schema_id": "anima-lora",
        "reft_enabled": True,
        "reft_targets": "net.blocks.0, net.blocks.1",
        "reft_rank": 2,
        "reft_init_scale": 0.1,
    })
    assert cfg.reft_enabled is True
    assert cfg.reft_target_modules == "net.blocks.0, net.blocks.1"
    assert cfg.reft_rank == 2
    assert cfg.reft_init_scale == 0.1
    print("PASS: test_config_aliases")
    return True


def test_reft_anima_install_backward_and_state() -> bool:
    torch.manual_seed(0)
    model = _TinyAnimaDiT()
    interventions = install_reft(model, ["net.blocks.0", "net.blocks.1"], rank=2, init_scale=0.1)
    assert len(interventions) == 2
    params = get_reft_params(model)
    assert len(params) == 6
    x = torch.randn(2, 4, 8)
    loss = model(x).sum()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in params)
    state = get_reft_state_dict(model)
    assert state
    assert any(key.startswith("reft.net_blocks_0.") for key in state)
    assert all(torch.isfinite(value).all() for value in state.values())
    print("PASS: test_reft_anima_install_backward_and_state")
    return True


def main() -> int:
    tests = [test_config_aliases, test_reft_anima_install_backward_and_state]
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
    print("Anima ReFT Wiring Smoke Test Results")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n{passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
