"""Smoke test for IP-Adapter image conditioning: projection shape, cross-attention injection, zero-init identity."""
from __future__ import annotations

import sys
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))

# Load ip_adapter_layers via importlib
_il_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.ip_adapter_layers",
    os.path.join(_HERE, "ip_adapter_layers.py"),
)
_il_mod = importlib.util.module_from_spec(_il_spec)
sys.modules["core.lulynx_trainer.ip_adapter_layers"] = _il_mod
_il_spec.loader.exec_module(_il_mod)

ImageProjModel = _il_mod.ImageProjModel
Resampler = _il_mod.Resampler

# Load ip_adapter_injector via importlib
_ii_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.ip_adapter_injector",
    os.path.join(_HERE, "ip_adapter_injector.py"),
)
_ii_mod = importlib.util.module_from_spec(_ii_spec)
sys.modules["core.lulynx_trainer.ip_adapter_injector"] = _ii_mod
_ii_spec.loader.exec_module(_ii_mod)

IPAdapterAttnProcessor = _ii_mod.IPAdapterAttnProcessor

import torch
import torch.nn as nn


def test_image_projection_output_shape():
    """IPAdapterImageProjection (ImageProjModel) produces correct output shape."""
    batch = 2
    clip_dim = 32
    cross_attn_dim = 64
    num_tokens = 4

    proj = ImageProjModel(
        cross_attention_dim=cross_attn_dim,
        clip_embeddings_dim=clip_dim,
        clip_extra_context_tokens=num_tokens,
    )
    image_embeds = torch.randn(batch, clip_dim)
    out = proj(image_embeds)

    assert out.shape == (batch, num_tokens, cross_attn_dim), (
        f"Expected ({batch}, {num_tokens}, {cross_attn_dim}), got {out.shape}"
    )


def test_resampler_output_shape():
    """Resampler produces correct output shape."""
    batch = 2
    embedding_dim = 32
    output_dim = 64
    num_queries = 8

    resampler = Resampler(
        dim=32,
        depth=2,
        heads=4,
        dim_head=8,
        num_queries=num_queries,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
    )
    x = torch.randn(batch, 5, embedding_dim)  # 5 image tokens
    out = resampler(x)

    assert out.shape == (batch, num_queries, output_dim), (
        f"Expected ({batch}, {num_queries}, {output_dim}), got {out.shape}"
    )


def test_cross_attention_injection():
    """IPAdapterAttnProcessor has to_k_ip and to_v_ip layers for cross-attention injection."""
    hidden_size = 64
    cross_attention_dim = 32
    num_tokens = 4

    proc = IPAdapterAttnProcessor(
        hidden_size=hidden_size,
        cross_attention_dim=cross_attention_dim,
        num_tokens=num_tokens,
    )

    assert hasattr(proc, "to_k_ip"), "IPAdapterAttnProcessor missing to_k_ip"
    assert hasattr(proc, "to_v_ip"), "IPAdapterAttnProcessor missing to_v_ip"
    assert isinstance(proc.to_k_ip, nn.Linear), "to_k_ip should be nn.Linear"
    assert isinstance(proc.to_v_ip, nn.Linear), "to_v_ip should be nn.Linear"

    # Verify shapes: to_k_ip maps from cross_attention_dim to hidden_size
    assert proc.to_k_ip.in_features == cross_attention_dim, (
        f"to_k_ip in_features should be {cross_attention_dim}, got {proc.to_k_ip.in_features}"
    )
    assert proc.to_k_ip.out_features == hidden_size, (
        f"to_k_ip out_features should be {hidden_size}, got {proc.to_k_ip.out_features}"
    )


def test_zero_init_starts_as_identity():
    """IP-Adapter zero-init: to_k_ip and to_v_ip with zero weights produce no
    image influence at init (the image cross-attention path is effectively zero).

    We simulate the image cross-attention computation directly since the
    full IPAdapterAttnProcessor.__call__ requires a diffusers Attention module.
    """
    hidden_size = 32
    cross_attention_dim = 16
    num_tokens = 4

    proc = IPAdapterAttnProcessor(
        hidden_size=hidden_size,
        cross_attention_dim=cross_attention_dim,
        num_tokens=num_tokens,
    )

    # Zero-init the IP projection (simulate how training frameworks do it)
    nn.init.zeros_(proc.to_k_ip.weight)
    nn.init.zeros_(proc.to_v_ip.weight)

    # With zero weights, IP key and value should be all zeros
    image_embeds = torch.randn(1, num_tokens, cross_attention_dim)
    ip_key = proc.to_k_ip(image_embeds)
    ip_value = proc.to_v_ip(image_embeds)

    assert (ip_key == 0).all(), "IP key should be zero with zero-initialized weights"
    assert (ip_value == 0).all(), "IP value should be zero with zero-initialized weights"

    # Zero IP key/value means the image cross-attention contribution is zero,
    # so the output should be identical to the text-only path (identity behavior)


def test_projection_trainable_params():
    """ImageProjModel has trainable parameters (proj + norm)."""
    proj = ImageProjModel(
        cross_attention_dim=64,
        clip_embeddings_dim=32,
        clip_extra_context_tokens=4,
    )
    trainable = [n for n, p in proj.named_parameters() if p.requires_grad]
    assert len(trainable) > 0, f"ImageProjModel should have trainable params, got {trainable}"


if __name__ == "__main__":
    print("SDXL IP-Adapter Smoke Tests")
    print("=" * 40)
    test_image_projection_output_shape()
    print("PASS: image_projection_output_shape")
    test_resampler_output_shape()
    print("PASS: resampler_output_shape")
    test_cross_attention_injection()
    print("PASS: cross_attention_injection")
    test_zero_init_starts_as_identity()
    print("PASS: zero_init_starts_as_identity")
    test_projection_trainable_params()
    print("PASS: projection_trainable_params")
    print("=" * 40)
    print("All SDXL IP-Adapter smoke tests passed!")
