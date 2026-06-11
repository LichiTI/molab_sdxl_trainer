"""Anima VRAM swap smoke test.

Validates that:
1. vram_swap_to_ram config field works for Anima model family
2. Enabling it produces a valid Anima config instance
3. AdapterCPUResidency can be constructed and parameters registered (CPU path)
4. ConfigAdapter resolves vram_swap for Anima-orchestrated configs
"""
from __future__ import annotations

import sys
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.join(_HERE, "..", "..")

sys.path.insert(0, os.path.abspath(_BACKEND_ROOT))

# Load configs via importlib to avoid __init__.py pulling in diffusers
_cfgs = importlib.util.spec_from_file_location(
    "core.configs",
    os.path.join(_BACKEND_ROOT, "core", "configs.py"),
)
_cfgs_mod = importlib.util.module_from_spec(_cfgs)
sys.modules["core.configs"] = _cfgs_mod
_const = importlib.util.spec_from_file_location(
    "core.constants",
    os.path.join(_BACKEND_ROOT, "core", "constants.py"),
)
_const_mod = importlib.util.module_from_spec(_const)
sys.modules["core.constants"] = _const_mod
_const.loader.exec_module(_const_mod)
_cfgs.loader.exec_module(_cfgs_mod)

LulynxConfig = _cfgs_mod.UnifiedTrainingConfig
ModelArch = _cfgs_mod.ModelArch

# Load AdapterCPUResidency via importlib
_mo = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.memory_optimizations",
    os.path.join(_HERE, "memory_optimizations.py"),
)
_mo_mod = importlib.util.module_from_spec(_mo)
sys.modules["core.lulynx_trainer.memory_optimizations"] = _mo_mod
_mo.loader.exec_module(_mo_mod)
AdapterCPUResidency = _mo_mod.AdapterCPUResidency

# Load ConfigAdapter via importlib
_ltc = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.config",
    os.path.join(_HERE, "config.py"),
)
_ltc_mod = importlib.util.module_from_spec(_ltc)
sys.modules["core.lulynx_trainer.config"] = _ltc_mod
_ltc.loader.exec_module(_ltc_mod)
_ca = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.config_adapter",
    os.path.join(_HERE, "config_adapter.py"),
)
_ca_mod = importlib.util.module_from_spec(_ca)
sys.modules["core.lulynx_trainer.config_adapter"] = _ca_mod
_ca.loader.exec_module(_ca_mod)
ConfigAdapter = _ca_mod.ConfigAdapter

import torch
import torch.nn as nn


def test_anima_vram_swap_config():
    """Anima config with vram_swap_to_ram=True does not crash."""
    cfg = LulynxConfig(
        model_type=ModelArch.ANIMA,
        vram_swap_to_ram=True,
        anima_model_path="fake_anima.ckpt",
        train_data_dir=_BACKEND_ROOT,
    )
    assert cfg.vram_swap_to_ram is True
    assert cfg.model_type == "anima"
    print("  [PASS] Anima config with vram_swap_to_ram")


def test_anima_vram_swap_residency():
    """AdapterCPUResidency works with Anima-style parameters on CPU."""
    device = torch.device("cpu")
    residency = AdapterCPUResidency(device=device)

    # Simulate Anima DiT adapter parameters
    params = [nn.Parameter(torch.randn(32, 32)) for _ in range(6)]
    count = residency.register_parameters(params)
    assert count == 6, f"Expected 6 registered, got {count}"

    # Move to GPU and back (on CPU, this is a no-op that shouldn't crash)
    residency.to_gpu()
    residency.to_cpu()
    print("  [PASS] Anima VRAM swap residency on CPU")


def test_config_adapter_anima_vram_swap():
    """ConfigAdapter maps vramSwapToRam for Anima schema."""
    data = {
        "schema_id": "anima-lora",
        "vramSwapToRam": True,
        "train_data_dir": _BACKEND_ROOT,
        "animaModelPath": "fake_anima.ckpt",
    }
    cfg = ConfigAdapter.from_frontend_dict(data)
    assert cfg.vram_swap_to_ram is True
    assert cfg.model_type == "anima"
    print("  [PASS] ConfigAdapter Anima vram_swap mapping")


def main() -> int:
    print("Anima VRAM Swap Smoke Tests")
    print("=" * 40)
    test_anima_vram_swap_config()
    test_anima_vram_swap_residency()
    test_config_adapter_anima_vram_swap()
    print("=" * 40)
    print("All Anima VRAM swap smoke tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())