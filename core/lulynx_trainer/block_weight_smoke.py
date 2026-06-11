# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test native Block Weight application against injected LoRA layers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.block_weight import create_block_weight_manager_from_settings
from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.lora_injector import LoRAInjector


class _TinyAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])


class _TinyTransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attentions = nn.ModuleList([_TinyAttention(dim)])
        self.ff = nn.Module()
        self.ff.net = nn.ModuleList([nn.Module(), nn.Module(), nn.Linear(dim, dim)])
        self.ff.net[0].proj = nn.Linear(dim, dim)  # type: ignore[attr-defined]


class _TinyUnet(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.down_blocks = nn.ModuleList([_TinyTransformerBlock(dim)])
        self.mid_block = _TinyTransformerBlock(dim)
        self.up_blocks = nn.ModuleList([_TinyTransformerBlock(dim), _TinyTransformerBlock(dim)])


def main() -> int:
    parsed = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "sdxl-lora",
            "enable_block_weights": "true",
            "lulynx_block_lr_zero_threshold": 0.2,
        }
    )
    if not parsed.bw_enable:
        raise AssertionError("enable_block_weights should map to bw_enable")
    if abs(float(parsed.block_lr_zero_threshold) - 0.2) > 1e-9:
        raise AssertionError(f"block_lr_zero_threshold alias failed: {parsed.block_lr_zero_threshold}")

    unet = _TinyUnet()
    injector = LoRAInjector(rank=2, alpha=2, model_arch="sdxl")
    injected = injector.inject_unet(unet)
    if not injected:
        raise RuntimeError("Expected synthetic UNet to receive LoRA layers")

    manager = create_block_weight_manager_from_settings(
        preset="",
        in_weights="0",
        mid_weight="0.75",
        out_weights="0.1,1",
        te_weight=1.0,
        te2_weight=1.0,
        zero_threshold=0.2,
    )
    layer_weights = manager.apply_to_lora_injector(injector)
    if not layer_weights:
        raise RuntimeError("Block Weight manager produced no layer weights")

    frozen = set(manager.get_frozen_layers())
    if not frozen:
        raise AssertionError("Expected at least one LoRA layer to be frozen by in_weights=0")

    active_scaled = [
        getattr(layer, "_block_weight_lr_scale", None)
        for name, layer in injector.injected_layers.items()
        if getattr(layer, "_block_weight_lr_scale", 1.0) not in (0.0, 1.0)
    ]
    if 0.75 not in active_scaled:
        raise AssertionError(f"Expected a mid-block lr scale of 0.75, got {active_scaled}")

    zero_scaled = [
        name
        for name, layer in injector.injected_layers.items()
        if getattr(layer, "_block_weight_lr_scale", 1.0) == 0.0
    ]
    if not zero_scaled:
        raise AssertionError("Expected block weight application to zero at least one layer scale")

    threshold_frozen = [name for name, weight in layer_weights.items() if weight == 0.0 and "up_blocks" in name]
    if not threshold_frozen:
        raise AssertionError(f"Expected zero_threshold to freeze low non-zero up weight, got {layer_weights}")

    for name in zero_scaled:
        layer = injector.injected_layers[name]
        if any(param.requires_grad for param in layer.lora.parameters()):
            raise AssertionError(f"Frozen layer still has trainable params: {name}")

    param_groups = injector.get_param_groups(base_lr=1e-4, weight_decay=0.0)
    lrs = sorted({round(float(group["lr"]), 8) for group in param_groups})
    if 0.0 in lrs:
        raise AssertionError(f"Frozen layers should not produce optimizer groups: {lrs}")
    if round(1e-4 * 0.75, 8) not in lrs or round(1e-4, 8) not in lrs:
        raise AssertionError(f"Expected scaled lr groups for mid/out weights, got {lrs}")

    print("Block weight smoke passed: alias-mapped weights freeze and scale injected LoRA layers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
