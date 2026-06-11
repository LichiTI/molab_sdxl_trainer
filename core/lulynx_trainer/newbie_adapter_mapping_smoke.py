"""Tiny Newbie adapter mapping smoke.

Verifies Newbie adapter aliases without loading the production model bundle.
"""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import torch
from torch import nn


def _install_xformers_stub() -> None:
    sys.modules.pop("xformers", None)
    sys.modules.pop("xformers.ops", None)

    xformers_module = ModuleType("xformers")
    ops_module = ModuleType("xformers.ops")

    def _unavailable(*_: object, **__: object) -> object:
        raise RuntimeError("xFormers is unavailable in the Newbie adapter smoke")

    ops_module.memory_efficient_attention = _unavailable  # type: ignore[attr-defined]
    ops_module.__spec__ = ModuleSpec("xformers.ops", loader=None)
    xformers_module.ops = ops_module  # type: ignore[attr-defined]
    xformers_module.__spec__ = ModuleSpec("xformers", loader=None)
    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = ops_module


_install_xformers_stub()

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.lycoris_layers import LyCORISConfig, LyCORISInjector, LyCORISType


class _TinyNewbieBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.Module()
        self.attention.qkv = nn.Linear(8, 8)
        self.attention.out = nn.Linear(8, 8)
        self.feed_forward = nn.Module()
        self.feed_forward.w2 = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention.out(torch.tanh(self.attention.qkv(x)))
        return self.feed_forward.w2(x)


def main() -> int:
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "newbie-lora",
            "pretrained_model_name_or_path": "H:/tmp/newbie-placeholder",
            "train_data_dir": "H:/tmp/lulynx-newbie-data",
            "adapter_type": "lokr",
            "newbie_target_modules": ["attention.qkv", "feed_forward.w2"],
            "lokr_rank": 2,
            "lokr_alpha": 2,
            "lokr_dropout": 0.1,
            "lokr_factor": 4,
        }
    )
    assert str(getattr(cfg.model_type, "value", cfg.model_type)) == "newbie"
    assert str(getattr(cfg.network_module, "value", cfg.network_module)) == "lycoris.locon"
    assert str(getattr(cfg.lycoris_algo, "value", cfg.lycoris_algo)) == "lokr"
    assert cfg.network_dim == 2
    assert cfg.network_alpha == 2
    assert cfg.network_dropout == 0.1
    assert cfg.lycoris_lokr_factor == 4

    injector = LyCORISInjector(
        LyCORISConfig(
            lycoris_type=LyCORISType.LOKR,
            rank=cfg.network_dim,
            alpha=cfg.network_alpha,
            dropout=cfg.network_dropout,
            lokr_factor=cfg.lycoris_lokr_factor,
        )
    )
    injected = injector.inject(
        _TinyNewbieBlock(),
        [part.strip() for part in cfg.newbie_target_modules.split(",")],
        prefix="unet",
    )
    assert sorted(injected) == ["unet.attention.qkv", "unet.feed_forward.w2"]
    assert injector.get_trainable_params()
    state = injector.get_lora_state_dict()
    assert any(key.endswith(".lokr_w1") for key in state)
    print(
        "Newbie adapter mapping smoke passed: "
        f"adapter={cfg.newbie_adapter_type}, layers={sorted(injected)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
