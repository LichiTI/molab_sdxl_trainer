# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for SDXL LyCORIS UI aliases and preset targeting."""

from __future__ import annotations

import os
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAINER_ROOT = Path(_HERE)
_CORE_ROOT = _TRAINER_ROOT.parent
_BACKEND_ROOT = _CORE_ROOT.parent
for _path in (str(_BACKEND_ROOT), str(_CORE_ROOT), str(_TRAINER_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _install_xformers_stub() -> None:
    sys.modules.pop("xformers", None)
    sys.modules.pop("xformers.ops", None)
    xformers_module = ModuleType("xformers")
    ops_module = ModuleType("xformers.ops")
    ops_module.__spec__ = ModuleSpec("xformers.ops", loader=None)
    xformers_module.ops = ops_module  # type: ignore[attr-defined]
    xformers_module.__spec__ = ModuleSpec("xformers", loader=None)
    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = ops_module


_install_xformers_stub()

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.lycoris_layers import LyCORISConfig, LyCORISInjector, LyCORISType, LoKrLayer


class _TinySDXLLyCORISBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.q_proj = nn.Linear(8, 8)
        self.attn.k_proj = nn.Linear(8, 8)
        self.attn.v_proj = nn.Linear(8, 8)
        self.attn.out_proj = nn.Linear(8, 8)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(8, 16)
        self.mlp.fc2 = nn.Linear(16, 8)
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(8)


def test_sdxl_lycoris_ui_aliases_normalize_to_native_fields() -> None:
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "sdxl-lora",
            "pretrained_model_name_or_path": "H:/models/sdxl.safetensors",
            "network_module": "lycoris.kohya",
            "lycoris_algo": "lokr",
            "lycoris_preset": "attn-mlp",
            "dropout": 0.21,
            "rank_dropout": 0.13,
            "module_dropout": 0.07,
        }
    )

    assert str(getattr(cfg.network_module, "value", cfg.network_module)) == "lycoris.locon"
    assert str(getattr(cfg.lycoris_algo, "value", cfg.lycoris_algo)) == "lokr"
    assert cfg.lycoris_presets == "attn-mlp"
    assert cfg.network_dropout == 0.21
    assert cfg.lokr_rank_dropout == 0.13
    assert cfg.lokr_module_dropout == 0.07


def test_attn_mlp_preset_expands_to_attention_and_mlp_targets() -> None:
    model = _TinySDXLLyCORISBlock()
    injector = LyCORISInjector(
        LyCORISConfig(
            lycoris_type=LyCORISType.LOKR,
            rank=4,
            alpha=4,
            dropout=0.21,
            lokr_rank_dropout=0.13,
            lokr_module_dropout=0.07,
            presets="attn-mlp",
        )
    )
    injected = injector.inject(model, [], prefix="unet")

    expected = {
        "unet.attn.q_proj",
        "unet.attn.k_proj",
        "unet.attn.v_proj",
        "unet.attn.out_proj",
        "unet.mlp.fc1",
        "unet.mlp.fc2",
    }
    assert expected.issubset(set(injected)), sorted(injected)
    assert "unet.conv" not in injected
    sample_layer = injected["unet.attn.q_proj"]
    assert isinstance(sample_layer, LoKrLayer)
    assert sample_layer.rank_dropout == 0.13
    assert sample_layer.module_dropout == 0.07
    assert isinstance(sample_layer.dropout, nn.Dropout)


def test_full_preset_matches_module_types_and_enables_norm_targets() -> None:
    model = _TinySDXLLyCORISBlock()
    config = LyCORISConfig(
        lycoris_type=LyCORISType.LOKR,
        rank=4,
        alpha=4,
        conv_dim=2,
        conv_alpha=2,
        presets="full",
    )
    injector = LyCORISInjector(config)
    injected = injector.inject(model, [], prefix="unet")

    assert "unet.conv" in injected, sorted(injected)
    assert "unet.norm" in injected, sorted(injected)
    assert config.train_norm is True


def main() -> int:
    test_sdxl_lycoris_ui_aliases_normalize_to_native_fields()
    test_attn_mlp_preset_expands_to_attention_and_mlp_targets()
    test_full_preset_matches_module_types_and_enables_norm_targets()
    print("sdxl_lycoris_preset_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
