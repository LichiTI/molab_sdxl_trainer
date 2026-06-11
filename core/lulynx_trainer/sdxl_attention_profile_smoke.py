# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test SDXL/U-Net experimental attention profile live wiring."""

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

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.diffusers_attention import (
    GenericDiffusersAttnProcessor,
    SlidingWindowDiffusersAttentionKernel,
)
from core.lulynx_trainer.runtime_optimizations import (
    apply_sdxl_attention_profile,
    build_runtime_optimization_plan,
)

try:
    from diffusers.models.attention_processor import Attention
except Exception as exc:  # pragma: no cover - this smoke requires the trainer env
    raise RuntimeError(f"diffusers Attention is required for SDXL attention profile smoke: {exc}") from exc


class _TinySDXLUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = Attention(query_dim=8, heads=2, dim_head=4)
        self.attn2 = Attention(query_dim=8, cross_attention_dim=8, heads=2, dim_head=4)


def _profiled_model(window_size: int, state: dict[str, torch.Tensor]):
    unet = _TinySDXLUNet()
    unet.load_state_dict(state)
    model = SimpleNamespace(unet=unet, vae=None, model_arch="sdxl")
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "sdxl-lora",
            "attention_backend": "torch",
            "experimental_attention_profile_enabled": "yes",
            "experimental_attention_profile_window": window_size,
            "experimental_attention_profile_backend": "torch_fallback",
            "experimental_attention_profile_torch_max_tokens": 32,
        }
    )
    plan = build_runtime_optimization_plan(cfg)
    patched = apply_sdxl_attention_profile(cfg, model, plan)
    assert patched == 2, f"expected both self/cross attention modules patched, got {patched}"
    return model, cfg, plan


def test_route_config_alias() -> None:
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "sdxl-lora",
            "experimental_attention_profile_enabled": "on",
            "experimental_attention_profile_window": 7,
            "experimental_attention_profile_backend": "sdpa",
        }
    )
    assert cfg.experimental_attention_profile_enabled is True
    assert cfg.experimental_attention_profile_window == 7
    assert cfg.experimental_attention_profile_backend == "sdpa"


def test_sdxl_self_attention_window_affects_output() -> None:
    torch.manual_seed(1234)
    base = _TinySDXLUNet()
    state = base.state_dict()
    model_w1, _cfg_w1, plan_w1 = _profiled_model(1, state)
    model_w4, _cfg_w4, plan_w4 = _profiled_model(4, state)

    processor = model_w1.unet.attn1.processor
    assert isinstance(processor, GenericDiffusersAttnProcessor)
    assert isinstance(processor.kernel, SlidingWindowDiffusersAttentionKernel)
    assert any("SDXL sliding-window self-attention" in reason for reason in plan_w1.reasons)

    hidden_states = torch.randn(1, 4, 8)
    out_w1 = model_w1.unet.attn1(hidden_states)
    out_w4 = model_w4.unet.attn1(hidden_states)
    assert out_w1.shape == hidden_states.shape
    assert out_w4.shape == hidden_states.shape
    diff = (out_w1 - out_w4).abs().max().item()
    assert diff > 1e-5, "different SDXL self-attention windows should change the output"
    assert any("window_size=4" in reason for reason in plan_w4.reasons)


def test_sdxl_cross_attention_uses_full_fallback() -> None:
    torch.manual_seed(5678)
    base = _TinySDXLUNet()
    state = base.state_dict()
    model_w1, _cfg_w1, _plan_w1 = _profiled_model(1, state)
    model_w4, _cfg_w4, _plan_w4 = _profiled_model(4, state)

    hidden_states = torch.randn(1, 4, 8)
    encoder_hidden_states = torch.randn(1, 5, 8)
    cross_w1 = model_w1.unet.attn2(hidden_states, encoder_hidden_states=encoder_hidden_states)
    cross_w4 = model_w4.unet.attn2(hidden_states, encoder_hidden_states=encoder_hidden_states)
    torch.testing.assert_close(cross_w1, cross_w4, rtol=1e-5, atol=1e-6)


def main() -> int:
    test_route_config_alias()
    test_sdxl_self_attention_window_affects_output()
    test_sdxl_cross_attention_uses_full_fallback()
    print("SDXL attention profile smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
