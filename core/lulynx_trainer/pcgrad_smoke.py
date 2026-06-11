"""Smoke tests for Warehouse PCGrad wiring."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

TRAINER_ROOT = Path(__file__).resolve().parent
CORE_ROOT = TRAINER_ROOT.parent
BACKEND_ROOT = CORE_ROOT.parent
for _path in (BACKEND_ROOT, CORE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    package = type(sys)(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def _import_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("core", CORE_ROOT)
_ensure_package("core.lulynx_trainer", TRAINER_ROOT)

_configs = _import_from_file("core.configs", CORE_ROOT / "configs.py")
_config_adapter = _import_from_file("core.lulynx_trainer.config_adapter", TRAINER_ROOT / "config_adapter.py")
_pcgrad = _import_from_file("core.lulynx_trainer.pcgrad", TRAINER_ROOT / "pcgrad.py")
_training_loop = _import_from_file("core.lulynx_trainer.training_loop", TRAINER_ROOT / "training_loop.py")

ConfigAdapter = _config_adapter.ConfigAdapter
TrainingLoop = _training_loop.TrainingLoop
resolve_pcgrad_gradients = _pcgrad.resolve_pcgrad_gradients


def test_config_aliases() -> None:
    parsed = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "sdxl-lora",
            "pretrained_model_name_or_path": "H:/models/sdxl.safetensors",
            "lulynx_pcgrad_enabled": "true",
            "lulynx_pcgrad_conflict_threshold": "-0.1",
            "lulynx_pcgrad_reduction": "SUM",
        }
    )
    assert parsed.pcgrad_enabled is True
    assert parsed.pcgrad_conflict_threshold == -0.1
    assert parsed.pcgrad_reduction == "sum"
    print("  [PASS] PCGrad config aliases normalize into backend fields")


def test_resolve_pcgrad_gradients() -> None:
    resolved, stats = resolve_pcgrad_gradients(
        [
            {"adapter.weight": torch.tensor([1.0])},
            {"adapter.weight": torch.tensor([-1.0])},
        ],
        conflict_threshold=0.0,
        reduction="mean",
    )
    assert "adapter.weight" in resolved
    assert torch.allclose(resolved["adapter.weight"], torch.zeros(1), atol=1e-6)
    assert stats["input_count"] == 2
    assert stats["conflict_pairs"] == 1
    assert stats["projections"] == 2
    assert stats["reduction"] == "mean"
    print("  [PASS] PCGrad resolver projects conflicting gradients")


def test_training_loop_helper_roundtrip() -> None:
    loop = TrainingLoop.__new__(TrainingLoop)
    param = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    loop.pcgrad_enabled = True
    loop.pcgrad_conflict_threshold = 0.0
    loop.pcgrad_reduction = "mean"
    loop._pcgrad_param_names = {id(param): "adapter.weight"}
    loop._pcgrad_pending_grads = []
    loop._pcgrad_last_stats = {}
    loop._get_trainable_params = lambda: [param]

    param.grad = torch.tensor([0.5], dtype=torch.float32)
    TrainingLoop._capture_pcgrad_microbatch(loop, accumulation_steps=2)
    param.grad = torch.tensor([-0.5], dtype=torch.float32)
    TrainingLoop._capture_pcgrad_microbatch(loop, accumulation_steps=2)
    param.grad = None

    TrainingLoop._apply_pcgrad_pending_grads(loop)

    assert param.grad is not None
    assert torch.allclose(param.grad, torch.zeros_like(param.grad), atol=1e-6)
    assert loop._pcgrad_last_stats["conflict_pairs"] == 1
    assert loop._pcgrad_last_stats["projections"] == 2
    assert loop._pcgrad_pending_grads == []
    runtime_state = TrainingLoop._pcgrad_runtime_state(loop)
    assert runtime_state["enabled"] is True
    assert runtime_state["last_step"]["conflict_pairs"] == 1
    print("  [PASS] TrainingLoop PCGrad helpers capture and replay gradients")


def main() -> int:
    test_config_aliases()
    test_resolve_pcgrad_gradients()
    test_training_loop_helper_roundtrip()
    print("pcgrad_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

