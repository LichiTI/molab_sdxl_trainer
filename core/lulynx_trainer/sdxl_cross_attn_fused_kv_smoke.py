# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test SDXL-style cross-attention fused K/V projection wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.runtime_optimizations import (
    FusedKVProjection,
    apply_cross_attn_fused_kv,
    build_runtime_optimization_plan,
)


class _TinyCrossAttention(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)


class _TinySDXLBlock(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.attn2 = _TinyCrossAttention(dim)


class _TinySDXLUNet(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.down_blocks = nn.ModuleList([_TinySDXLBlock(dim), _TinySDXLBlock(dim)])
        self.mid_block = _TinySDXLBlock(dim)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unet = _TinySDXLUNet()


def test_route_config_alias() -> None:
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "sdxl-lora",
            "cross_attn_fused_kv": "on",
            "fused_projection_memory_mode": "materialize_on_save",
        }
    )
    assert cfg.cross_attn_fused_kv is True
    assert cfg.fused_projection_memory_mode == "materialize_on_save"


def test_sdxl_cross_attn_fused_kv_equivalence() -> None:
    torch.manual_seed(123)
    model = _TinyModel()
    block = model.unet.down_blocks[0]
    x = torch.randn(2, 4, 8)
    expected_k = block.attn2.to_k(x)
    expected_v = block.attn2.to_v(x)

    cfg = ConfigAdapter.from_frontend_dict(
        {"schema_id": "sdxl-lora", "cross_attn_fused_kv": True}
    )
    plan = build_runtime_optimization_plan(cfg)
    apply_cross_attn_fused_kv(cfg, model, plan)

    assert isinstance(block.attn2._fused_kv, FusedKVProjection)
    actual_k, actual_v = block.attn2._fused_kv(x)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)
    assert any("cross_attn_fused_kv: fused 3" in reason for reason in plan.reasons), plan.reasons


def test_materialize_on_save_state_dict_keeps_original_keys() -> None:
    torch.manual_seed(456)
    model = _TinyModel()
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "sdxl-lora",
            "cross_attn_fused_kv": True,
            "fused_projection_memory_mode": "materialize_on_save",
        }
    )
    plan = build_runtime_optimization_plan(cfg)
    apply_cross_attn_fused_kv(cfg, model, plan)

    attn = model.unet.down_blocks[0].attn2
    assert attn.to_k is None
    assert attn.to_v is None
    state = model.state_dict()
    assert "unet.down_blocks.0.attn2.to_k.weight" in state
    assert "unet.down_blocks.0.attn2.to_v.weight" in state
    fused_weight = attn._fused_kv.kv_proj.weight.detach()
    torch.testing.assert_close(state["unet.down_blocks.0.attn2.to_k.weight"], fused_weight[:8])
    torch.testing.assert_close(state["unet.down_blocks.0.attn2.to_v.weight"], fused_weight[8:])


def main() -> int:
    test_route_config_alias()
    test_sdxl_cross_attn_fused_kv_equivalence()
    test_materialize_on_save_state_dict_keeps_original_keys()
    print("SDXL cross-attention fused K/V smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
