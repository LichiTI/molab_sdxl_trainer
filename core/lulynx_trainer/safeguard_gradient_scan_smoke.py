# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test SafeGuard gradient scan modes without starting a trainer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


class _FakeLoraInjector:
    def __init__(self, params):
        self._params = list(params)

    def get_trainable_params(self):
        return list(self._params)


def _make_loop() -> tuple[object, list[torch.nn.Parameter]]:
    params = [
        torch.nn.Parameter(torch.zeros(2, 2)),
        torch.nn.Parameter(torch.zeros(3)),
        torch.nn.Parameter(torch.zeros(1)),
    ]
    params[0].grad = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    params[1].grad = torch.tensor([1.0, 2.0, 2.0])
    params[2].grad = None

    loop = TrainingLoop.__new__(TrainingLoop)
    loop.lora_injector = _FakeLoraInjector(params)
    return loop, params


def _assert_modes_match() -> None:
    loop, _ = _make_loop()
    expected = torch.tensor([5.0, 3.0])

    legacy = loop._collect_gradients_for_safeguard("legacy")
    assert legacy is not None
    assert torch.allclose(legacy, expected), f"legacy norms mismatch: {legacy}"

    batched = loop._collect_gradients_for_safeguard("batched")
    assert batched is not None
    assert torch.allclose(batched, expected), f"batched norms mismatch: {batched}"

    foreach = loop._collect_gradients_for_safeguard("foreach")
    assert foreach is not None
    assert torch.allclose(foreach, expected), f"foreach norms mismatch: {foreach}"

    assert loop._collect_gradients_for_safeguard("off") is None


def _assert_safe_guard_config_wiring() -> None:
    safe_guard_mod = _load_module("core.lulynx_trainer.safe_guard", TRAINER_ROOT / "safe_guard.py")
    config = safe_guard_mod.SafeGuardConfig(gradient_scan_mode="off")
    assert config.gradient_scan_mode == "off"

    config_adapter_mod = _load_module("core.lulynx_trainer.config_adapter", TRAINER_ROOT / "config_adapter.py")
    source = {
        "pretrained_model_name_or_path": "model",
        "train_data_dir": "data",
        "safeguard_gradient_scan_mode": "foreach",
    }
    normalized = config_adapter_mod.ConfigAdapter.from_frontend_dict(source)
    assert getattr(normalized, "so_gradient_scan_mode") == "foreach"


def _assert_device_grouping_guard() -> None:
    source = (TRAINER_ROOT / "training_loop.py").read_text(encoding="utf-8")
    assert "grouped: Dict[torch.device, List[torch.Tensor]] = {}" in source
    assert "grouped.setdefault(grad.device, []).append(grad)" in source
    assert "torch.stack(grad_norms).detach().cpu()" not in source


def main() -> int:
    _assert_modes_match()
    print("  SafeGuard gradient scan modes: legacy/batched/foreach/off -- PASS")

    _assert_safe_guard_config_wiring()
    print("  SafeGuard gradient scan config/adapter wiring -- PASS")

    _assert_device_grouping_guard()
    print("  SafeGuard gradient scan device grouping guard -- PASS")

    print("SafeGuard gradient scan smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
