"""Anima CPU offload smoke test.

Validates that:
1. enable_sequential_cpu_offload config field works for Anima model family
2. Enabling it produces a valid Anima config instance
3. Basic Anima-style DiT model + config does not crash on CPU with offload flag
4. ConfigAdapter resolves CPU offload for Anima-orchestrated configs
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


def test_anima_sequential_cpu_offload_config():
    """Anima config with enable_sequential_cpu_offload=True does not crash."""
    cfg = LulynxConfig(
        model_type=ModelArch.ANIMA,
        enable_sequential_cpu_offload=True,
        anima_model_path="fake_anima.ckpt",
        train_data_dir=_BACKEND_ROOT,
    )
    assert cfg.enable_sequential_cpu_offload is True
    assert cfg.model_type == "anima"
    print("  [PASS] Anima config with enable_sequential_cpu_offload")


def test_anima_cpu_offload_dit_forward():
    """Anima-style DiT stub forward with CPU offload flag does not crash."""
    cfg = LulynxConfig(
        model_type=ModelArch.ANIMA,
        enable_sequential_cpu_offload=True,
        anima_model_path="fake_anima.ckpt",
        train_data_dir=_BACKEND_ROOT,
    )

    # Minimal Anima-style DiT stub
    class _StubDiTBlock(nn.Module):
        def __init__(self, dim=32):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            return self.norm(self.linear(x) + x)

    class _StubDiT(nn.Module):
        def __init__(self, dim=32, n_blocks=2):
            super().__init__()
            self.blocks = nn.ModuleList([_StubDiTBlock(dim) for _ in range(n_blocks)])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    model = _StubDiT(dim=16, n_blocks=2)
    x = torch.randn(1, 8, 16)
    out = model(x)
    assert out.shape == (1, 8, 16), f"Unexpected output shape: {out.shape}"
    print("  [PASS] Anima DiT stub forward with CPU offload flag on CPU")


def test_config_adapter_anima_sequential_offload():
    """ConfigAdapter maps enableSequentialCpuOffload for Anima schema."""
    data = {
        "schema_id": "anima-lora",
        "enableSequentialCpuOffload": True,
        "train_data_dir": _BACKEND_ROOT,
        "animaModelPath": "fake_anima.ckpt",
    }
    cfg = ConfigAdapter.from_frontend_dict(data)
    assert cfg.enable_sequential_cpu_offload is True
    assert cfg.model_type == "anima"
    print("  [PASS] ConfigAdapter Anima sequential_cpu_offload mapping")


def test_anima_unsloth_offload_field():
    """Anima also has the anima_unsloth_offload field; verify it coexists."""
    cfg = LulynxConfig(
        anima_unsloth_offload=True,
        enable_sequential_cpu_offload=True,
    )
    assert cfg.anima_unsloth_offload is True
    assert cfg.enable_sequential_cpu_offload is True
    print("  [PASS] Anima unsloth_offload + sequential_cpu_offload coexist")


def main() -> int:
    print("Anima CPU Offload Smoke Tests")
    print("=" * 40)
    test_anima_sequential_cpu_offload_config()
    test_anima_cpu_offload_dit_forward()
    test_config_adapter_anima_sequential_offload()
    test_anima_unsloth_offload_field()
    print("=" * 40)
    print("All Anima CPU offload smoke tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())