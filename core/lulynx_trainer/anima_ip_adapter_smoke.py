# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for anima_ip_adapter.py (Phase 8.10 / #118)."""

from __future__ import annotations

import os
import sys
import importlib.util

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.anima_ip_adapter",
    os.path.join(_HERE, "anima_ip_adapter.py"),
)
_ip = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.anima_ip_adapter"] = _ip
_spec.loader.exec_module(_ip)


def _fake_encoder(batch_size=2, seq=8, dim=1024):
    def _enc(x):
        return torch.randn(batch_size, seq, dim)
    return _enc


def test_projector_shape_2d_input():
    cfg = _ip.IPAdapterConfig(encoder_dim=512, cond_dim=64, num_image_tokens=8, num_layers=1)
    proj = _ip.ImageProjector(cfg)
    x = torch.randn(2, 512)
    out = proj(x)
    assert out.shape == (2, 8, 64)
    print("PASS: projector handles 2D pooled-feature input")


def test_projector_shape_3d_input():
    cfg = _ip.IPAdapterConfig(encoder_dim=768, cond_dim=128, num_image_tokens=16)
    proj = _ip.ImageProjector(cfg)
    x = torch.randn(2, 32, 768)
    out = proj(x)
    assert out.shape == (2, 16, 128)
    print("PASS: projector handles 3D sequence input")


def test_adapter_forward_returns_tokens():
    cfg = _ip.IPAdapterConfig(encoder_dim=1024, cond_dim=128, num_image_tokens=8, scale=1.0)
    adapter = _ip.AnimaIPAdapter(_fake_encoder(2, 16, 1024), cfg)
    image = torch.randn(2, 3, 224, 224)
    out = adapter(image)
    assert out.shape == (2, 8, 128)
    print("PASS: AnimaIPAdapter.forward returns projected tokens")


def test_adapter_scale_zero_yields_zero_tokens():
    cfg = _ip.IPAdapterConfig(encoder_dim=512, cond_dim=64, num_image_tokens=4, scale=0.0)
    adapter = _ip.AnimaIPAdapter(_fake_encoder(2, 8, 512), cfg)
    image = torch.randn(2, 3, 64, 64)
    out = adapter(image)
    assert torch.allclose(out, torch.zeros_like(out))
    print("PASS: scale=0 produces zero tokens")


def test_concat_mode_extends_text_tokens():
    cfg = _ip.IPAdapterConfig(encoder_dim=512, cond_dim=64, num_image_tokens=4, cond_mode="concat")
    adapter = _ip.AnimaIPAdapter(_fake_encoder(1, 4, 512), cfg)

    image_tokens = torch.randn(1, 4, 64)
    text_tokens = torch.randn(1, 16, 64)
    text_mask = torch.ones(1, 16, dtype=torch.long)

    combined, mask = adapter.merge_with_text_cond(image_tokens, text_tokens, text_mask)
    assert combined.shape == (1, 20, 64)
    assert mask.shape == (1, 20)
    assert (mask[:, 16:] == 1).all()
    print("PASS: concat mode extends text tokens and attention mask")


def test_replace_mode_returns_image_tokens_only():
    cfg = _ip.IPAdapterConfig(encoder_dim=512, cond_dim=64, num_image_tokens=4, cond_mode="replace")
    adapter = _ip.AnimaIPAdapter(_fake_encoder(1, 4, 512), cfg)
    image_tokens = torch.randn(1, 4, 64)
    text_tokens = torch.randn(1, 16, 64)

    combined, mask = adapter.merge_with_text_cond(image_tokens, text_tokens)
    assert torch.allclose(combined, image_tokens)
    assert mask is None
    print("PASS: replace mode discards text tokens")


def test_dim_mismatch_falls_back_to_image_only():
    cfg = _ip.IPAdapterConfig(encoder_dim=512, cond_dim=64, num_image_tokens=4)
    adapter = _ip.AnimaIPAdapter(_fake_encoder(1, 4, 512), cfg)

    image_tokens = torch.randn(1, 4, 64)
    text_tokens = torch.randn(1, 16, 128)  # wrong cond dim

    combined, mask = adapter.merge_with_text_cond(image_tokens, text_tokens)
    assert torch.allclose(combined, image_tokens)
    print("PASS: cond_dim mismatch falls back to image tokens only")


def test_adapter_is_differentiable_through_projector():
    cfg = _ip.IPAdapterConfig(encoder_dim=256, cond_dim=64, num_image_tokens=4, num_layers=2)
    adapter = _ip.AnimaIPAdapter(_fake_encoder(2, 8, 256), cfg)

    out = adapter(torch.randn(2, 3, 32, 32))
    out.sum().backward()
    grads = [p.grad for p in adapter.get_trainable_params() if p.grad is not None]
    assert len(grads) > 0
    assert all(g.abs().sum().item() > 0 for g in grads)
    print("PASS: gradient flows through projector while encoder stays frozen")


def test_get_trainable_params_excludes_encoder():
    cfg = _ip.IPAdapterConfig(encoder_dim=256, cond_dim=64, num_image_tokens=4)
    adapter = _ip.AnimaIPAdapter(_fake_encoder(), cfg)
    params = adapter.get_trainable_params()
    # Only projector params: queries, kv_proj, out_proj, mlp, norm
    assert len(params) > 0
    assert all(isinstance(p, nn.Parameter) for p in params)
    print("PASS: get_trainable_params returns only projector parameters")


if __name__ == "__main__":
    test_projector_shape_2d_input()
    test_projector_shape_3d_input()
    test_adapter_forward_returns_tokens()
    test_adapter_scale_zero_yields_zero_tokens()
    test_concat_mode_extends_text_tokens()
    test_replace_mode_returns_image_tokens_only()
    test_dim_mismatch_falls_back_to_image_only()
    test_adapter_is_differentiable_through_projector()
    test_get_trainable_params_excludes_encoder()
    print("\nAll Anima IP-Adapter smoke tests passed!")
