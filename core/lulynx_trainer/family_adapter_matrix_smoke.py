# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test SDXL/Newbie adapter matrix routing.

This stays tiny on purpose: it validates config aliases, injector selection,
forward/backward, and state dict generation without loading production models.
"""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn


def _install_xformers_stub() -> None:
    sys.modules.pop("xformers", None)
    sys.modules.pop("xformers.ops", None)

    xformers_module = ModuleType("xformers")
    ops_module = ModuleType("xformers.ops")

    def _unavailable(*_: object, **__: object) -> object:
        raise RuntimeError("xFormers is unavailable in family adapter matrix smoke")

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
from core.lulynx_trainer.lora_injector import LoRAInjector, LoRALinear
from core.lulynx_trainer.lycoris_layers import (
    DiagOFTAdapter,
    FullRankAdapter,
    IA3Adapter,
    LoHaLayer,
    LoKrLayer,
    LyCORISConfig,
    LyCORISInjector,
    LyCORISType,
)
from core.lulynx_trainer.lora_fa_layer import LoRAFALinear
from core.lulynx_trainer.vera_layer import VeRALinear
from core.lulynx_trainer.tlora import TLoRALinear
from core.lulynx_trainer.flexrank_lora import FlexRankLoRALinear
from core.lulynx_trainer.hydralora import HydraLoRALinear
from core.lulynx_trainer.fera import FeRALinear


class _TinySDXL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_q(x)


class _TinyNewbie(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.Module()
        self.attention.qkv = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention.qkv(x)


_LORA_EXPECTED = {
    "lora": LoRALinear,
    "lora_plus": LoRALinear,
    "dora": LoRALinear,
    "lora_fa": LoRAFALinear,
    "vera": VeRALinear,
    "tlora": TLoRALinear,
    "flexrank": FlexRankLoRALinear,
    "hydralora": HydraLoRALinear,
    "fera": FeRALinear,
}

_LYCORIS_EXPECTED = {
    "loha": LoHaLayer,
    "locon": object,
    "lokr": LoKrLayer,
    "ia3": IA3Adapter,
    "full": FullRankAdapter,
    "diag_oft": DiagOFTAdapter,
}


def _network_module(cfg) -> str:
    return str(getattr(cfg.network_module, "value", cfg.network_module))


def _lycoris_algo(cfg) -> str:
    return str(getattr(cfg.lycoris_algo, "value", cfg.lycoris_algo))


def _make_cfg(family: str, adapter_type: str):
    key = "adapter_type" if family == "newbie" else "lora_type"
    return ConfigAdapter.from_frontend_dict(
        {
            "schema_id": f"{family}-lora",
            "pretrained_model_name_or_path": f"H:/tmp/{family}.safetensors",
            "train_data_dir": f"H:/tmp/{family}-data",
            key: adapter_type,
            "network_dim": 4,
            "network_alpha": 4,
            "lokr_factor": -1,
            "lokr_no_materialize_forward": True,
            "hydralora_num_experts": 2,
            "hydralora_top_k": 1,
            "hydralora_sparse_top_k": True,
            "flexrank_lora_rank_range_min": 2,
            "tlora_rank_schedule": "linear",
        }
    )


def _inject_lora(cfg, model: nn.Module, target: str, family: str):
    network_module = _network_module(cfg)
    injector = LoRAInjector(
        rank=cfg.network_dim,
        alpha=cfg.network_alpha,
        dropout=cfg.network_dropout,
        model_arch=family,
        dora_enabled=bool(getattr(cfg, "use_dora", False) or getattr(cfg, "dora_enabled", False)),
        tlora_enabled=network_module.endswith("tlora"),
        tlora_min_rank=cfg.tlora_min_rank,
        tlora_rank_schedule=cfg.tlora_rank_schedule,
        tlora_orthogonal_init=cfg.tlora_orthogonal_init,
        tlora_total_steps=1000,
        vera_enabled=network_module == "networks.vera" or bool(getattr(cfg, "vera_enabled", False)),
        vera_d_initial=cfg.vera_d_initial,
        vera_prng_key=cfg.vera_prng_key,
        lora_fa_enabled=network_module == "networks.lora_fa" or bool(getattr(cfg, "lora_fa_enabled", False)),
        flexrank_enabled=network_module == "networks.flexrank_lora" or bool(getattr(cfg, "flexrank_lora_enabled", False)),
        flexrank_rank_range_min=cfg.flexrank_lora_rank_range_min,
        hydralora_enabled=bool(getattr(cfg, "hydralora_enabled", False)),
        hydralora_num_experts=cfg.hydralora_num_experts,
        hydralora_routing=cfg.hydralora_routing,
        hydralora_top_k=cfg.hydralora_top_k,
        hydralora_sparse_top_k=cfg.hydralora_sparse_top_k,
        fera_enabled=bool(getattr(cfg, "fera_enabled", False)),
        fera_gate_init=cfg.fera_gate_init,
    )
    injected = injector.inject(model, [target], prefix="unet")
    assert len(injected) == 1
    return injector, next(iter(injected.values()))


def _inject_lycoris(cfg, model: nn.Module, target: str):
    injector = LyCORISInjector(
        LyCORISConfig(
            lycoris_type=LyCORISType(_lycoris_algo(cfg)),
            rank=cfg.network_dim,
            alpha=cfg.network_alpha,
            dropout=cfg.network_dropout,
            lokr_factor=cfg.lycoris_lokr_factor,
            lokr_rank_dropout=cfg.lokr_rank_dropout,
            lokr_module_dropout=cfg.lokr_module_dropout,
            lokr_no_materialize_forward=cfg.lokr_no_materialize_forward,
            train_norm=cfg.lycoris_train_norm or cfg.lokr_train_norm,
        )
    )
    injected = injector.inject(model, [target], prefix="unet")
    assert len(injected) == 1
    return injector, next(iter(injected.values()))


def _assert_forward_backward(model: nn.Module, injector: object) -> None:
    x = torch.randn(2, 3, 8)
    loss = model(x).sum()
    loss.backward()
    params = injector.get_trainable_params()
    assert params, "Expected trainable adapter parameters"
    assert any(param.grad is not None for param in params), "Expected adapter gradients"
    state = injector.get_lora_state_dict()
    assert state, "Expected adapter state dict"


def _run_family(family: str) -> None:
    model_cls = _TinySDXL if family == "sdxl" else _TinyNewbie
    target = "to_q" if family == "sdxl" else "attention.qkv"

    for adapter_type, expected in _LORA_EXPECTED.items():
        cfg = _make_cfg(family, adapter_type)
        if adapter_type == "lora_plus":
            assert cfg.lora_plus_enabled is True
        model = model_cls()
        injector, layer = _inject_lora(cfg, model, target, family)
        assert isinstance(layer, expected), (family, adapter_type, type(layer), expected)
        if adapter_type == "dora":
            assert getattr(layer, "use_dora", False) is True
        if adapter_type == "flexrank":
            assert layer.min_rank == cfg.flexrank_lora_rank_range_min
        if adapter_type == "hydralora":
            assert layer.config.sparse_top_k is True
        _assert_forward_backward(model, injector)

    for adapter_type, expected in _LYCORIS_EXPECTED.items():
        cfg = _make_cfg(family, adapter_type)
        assert _network_module(cfg) == "lycoris.locon", (family, adapter_type, _network_module(cfg))
        expected_algo = "diag-oft" if adapter_type == "diag_oft" else adapter_type
        assert _lycoris_algo(cfg) == expected_algo, (family, adapter_type, _lycoris_algo(cfg))
        model = model_cls()
        injector, layer = _inject_lycoris(cfg, model, target)
        if adapter_type != "locon":
            assert isinstance(layer, expected), (family, adapter_type, type(layer), expected)
        if adapter_type == "lokr":
            assert layer.no_materialize_forward is True
        _assert_forward_backward(model, injector)


def main() -> int:
    for family in ("sdxl", "newbie"):
        _run_family(family)
        print(f"PASS: {family} adapter matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
