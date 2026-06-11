"""SDXL VRAM swap smoke test.

Validates that:
1. vram_swap_to_ram config field exists and defaults to False
2. Enabling it produces a valid config instance
3. AdapterCPUResidency can be constructed and parameters registered (CPU path)
4. Enum / config resolution does not crash on CPU
"""
from __future__ import annotations

import sys
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.join(_HERE, "..", "..")

# Add backend root to path so `core.configs` is importable
sys.path.insert(0, os.path.abspath(_BACKEND_ROOT))

# Load configs via importlib to avoid __init__.py pulling in diffusers
_cfgs = importlib.util.spec_from_file_location(
    "core.configs",
    os.path.join(_BACKEND_ROOT, "core", "configs.py"),
)
_cfgs_mod = importlib.util.module_from_spec(_cfgs)
sys.modules["core.configs"] = _cfgs_mod
# Pre-load constants (configs.py imports from core.constants)
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

# Load AdapterCPUResidency via importlib (avoids diffusers chain)
_mo = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.memory_optimizations",
    os.path.join(_HERE, "memory_optimizations.py"),
)
_mo_mod = importlib.util.module_from_spec(_mo)
sys.modules["core.lulynx_trainer.memory_optimizations"] = _mo_mod
_mo.loader.exec_module(_mo_mod)
AdapterCPUResidency = _mo_mod.AdapterCPUResidency

# Load ConfigAdapter via importlib
_ca = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.config_adapter",
    os.path.join(_HERE, "config_adapter.py"),
)
_ca_mod = importlib.util.module_from_spec(_ca)
# config_adapter imports from .config which we need to set up first
_ltc = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.config",
    os.path.join(_HERE, "config.py"),
)
_ltc_mod = importlib.util.module_from_spec(_ltc)
sys.modules["core.lulynx_trainer.config"] = _ltc_mod
_ltc.loader.exec_module(_ltc_mod)
sys.modules["core.lulynx_trainer.config_adapter"] = _ca_mod
_ca.loader.exec_module(_ca_mod)
ConfigAdapter = _ca_mod.ConfigAdapter

import torch
import torch.nn as nn


def test_vram_swap_config_field():
    """vram_swap_to_ram field exists, defaults False, and can be toggled."""
    cfg = LulynxConfig()
    assert hasattr(cfg, "vram_swap_to_ram"), "Missing vram_swap_to_ram field"
    assert cfg.vram_swap_to_ram is False, "Default should be False"

    cfg2 = LulynxConfig(vram_swap_to_ram=True)
    assert cfg2.vram_swap_to_ram is True
    print("  [PASS] vram_swap_to_ram config field")


def test_vram_swap_enables_residency():
    """Enabling vram_swap_to_ram config allows AdapterCPUResidency construction."""
    device = torch.device("cpu")
    residency = AdapterCPUResidency(device=device)

    # Register some dummy parameters
    params = [nn.Parameter(torch.randn(16, 16)) for _ in range(2)]
    count = residency.register_parameters(params)
    assert count == 2, f"Expected 2 registered, got {count}"

    savings = residency.estimate_vram_savings_mb()
    assert isinstance(savings, float) and savings >= 0.0
    print("  [PASS] vram_swap_to_ram enables AdapterCPUResidency")


def test_vram_swap_sdxl_config_no_crash():
    """Full SDXL config with vram_swap_to_ram=True does not crash on CPU."""
    cfg = LulynxConfig(
        model_type=ModelArch.SDXL,
        vram_swap_to_ram=True,
        train_data_dir=_BACKEND_ROOT,
    )
    assert cfg.vram_swap_to_ram is True
    assert cfg.model_type == "sdxl"
    print("  [PASS] SDXL config with vram_swap_to_ram does not crash")


def test_config_adapter_maps_vram_swap():
    """ConfigAdapter maps frontend vramSwapToRam to vram_swap_to_ram."""
    data = {
        "model_type": "sdxl",
        "vramSwapToRam": True,
        "train_data_dir": _BACKEND_ROOT,
    }
    cfg = ConfigAdapter.from_frontend_dict(data)
    assert cfg.vram_swap_to_ram is True
    print("  [PASS] ConfigAdapter maps vramSwapToRam -> vram_swap_to_ram")


def main() -> int:
    print("SDXL VRAM Swap Smoke Tests")
    print("=" * 40)
    test_vram_swap_config_field()
    test_vram_swap_enables_residency()
    test_vram_swap_sdxl_config_no_crash()
    test_config_adapter_maps_vram_swap()
    print("=" * 40)
    print("All SDXL VRAM swap smoke tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
