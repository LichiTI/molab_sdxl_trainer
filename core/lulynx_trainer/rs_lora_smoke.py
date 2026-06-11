"""Smoke-test RS-LoRA adapter scaling and export metadata contracts."""

import inspect
import math
import sys
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from backend.core.lulynx_trainer.lora_injector import LoRALayer, LoRAInjector


def _assert_rs_lora_metadata(metadata: dict) -> None:
    assert metadata.get("ss_rs_lora") in {"True", "true", "1", True}
    assert metadata.get("ss_scaling_strategy") in {"rs_lora", "rank_stabilized", "alpha_over_sqrt_rank"}


def test_lora_layer_uses_rank_stabilized_scaling() -> None:
    signature = inspect.signature(LoRALayer)
    assert "rs_lora_enabled" in signature.parameters

    layer = LoRALayer(
        in_features=3,
        out_features=2,
        rank=16,
        alpha=8.0,
        rs_lora_enabled=True,
    )
    expected = 8.0 / math.sqrt(16)
    assert math.isclose(float(layer.scaling), expected, rel_tol=0.0, abs_tol=1e-12)
    assert getattr(layer, "rs_lora_enabled", False) is True
    assert getattr(layer, "scaling_strategy", "") in {"rs_lora", "rank_stabilized", "alpha_over_sqrt_rank"}


def test_lora_injector_propagates_rs_lora_to_layers() -> None:
    signature = inspect.signature(LoRAInjector)
    assert "rs_lora_enabled" in signature.parameters

    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    injector = LoRAInjector(
        rank=9,
        alpha=6.0,
        target_modules=["0"],
        rs_lora_enabled=True,
    )
    injected = injector.inject(model, target_modules=["0"], prefix="tiny")
    assert set(injected) == {"tiny.0"}

    adapter = injected["tiny.0"].lora
    expected = 6.0 / math.sqrt(9)
    assert math.isclose(float(adapter.scaling), expected, rel_tol=0.0, abs_tol=1e-12)
    assert getattr(adapter, "rs_lora_enabled", False) is True
    assert getattr(adapter, "scaling_strategy", "") in {"rs_lora", "rank_stabilized", "alpha_over_sqrt_rank"}


def test_save_lora_records_rs_lora_metadata() -> None:
    captured = {}

    def fake_save_file(state_dict, path, metadata=None):
        captured["state_dict"] = state_dict
        captured["path"] = path
        captured["metadata"] = dict(metadata or {})

    previous_safetensors = sys.modules.get("safetensors")
    previous_safetensors_torch = sys.modules.get("safetensors.torch")
    safetensors_module = ModuleType("safetensors")
    safetensors_torch_module = ModuleType("safetensors.torch")
    safetensors_torch_module.save_file = fake_save_file
    safetensors_module.torch = safetensors_torch_module
    sys.modules["safetensors"] = safetensors_module
    sys.modules["safetensors.torch"] = safetensors_torch_module

    try:
        model = nn.Sequential(nn.Linear(3, 2, bias=False))
        injector = LoRAInjector(
            rank=4,
            alpha=2.0,
            target_modules=["0"],
            rs_lora_enabled=True,
        )
        injector.inject(model, target_modules=["0"], prefix="tiny")
        injector.save_lora("unused.safetensors", metadata={"custom": "kept"})
    finally:
        if previous_safetensors is None:
            sys.modules.pop("safetensors", None)
        else:
            sys.modules["safetensors"] = previous_safetensors
        if previous_safetensors_torch is None:
            sys.modules.pop("safetensors.torch", None)
        else:
            sys.modules["safetensors.torch"] = previous_safetensors_torch

    metadata = captured["metadata"]
    assert metadata["custom"] == "kept"
    assert metadata["ss_network_dim"] == "4"
    assert metadata["ss_network_alpha"] == "2.0"
    _assert_rs_lora_metadata(metadata)


def main() -> None:
    test_lora_layer_uses_rank_stabilized_scaling()
    test_lora_injector_propagates_rs_lora_to_layers()
    test_save_lora_records_rs_lora_metadata()
    print("rs_lora_smoke: ok")


if __name__ == "__main__":
    main()
