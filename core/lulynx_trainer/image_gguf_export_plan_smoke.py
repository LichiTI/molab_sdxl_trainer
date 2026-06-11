"""Smoke tests for image GGUF export dry-run plans."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch


BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.tools.image_gguf_exporter import plan_image_gguf_export  # noqa: E402


def _save(path: str | Path, state: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    save_file(state, str(path), metadata={"source": "image_gguf_export_plan_smoke"})


def _tiny_vae_state() -> dict[str, torch.Tensor]:
    return {
        "encoder.conv_in.weight": torch.randn(16, 4, 3, 3),
        "encoder.conv_out.weight": torch.randn(8, 16, 3, 3),
        "encoder.down_blocks.0.resnets.0.conv1.weight": torch.randn(16, 16, 3, 3),
        "decoder.conv_in.weight": torch.randn(16, 4, 3, 3),
        "decoder.conv_out.weight": torch.randn(4, 16, 3, 3),
        "decoder.mid_block.attentions.0.to_q.weight": torch.randn(16, 16),
        "decoder.up_blocks.0.resnets.0.conv1.weight": torch.randn(16, 16, 3, 3),
    }


def _tiny_t5_shard0() -> dict[str, torch.Tensor]:
    return {
        "shared.weight": torch.randn(32, 16),
        "encoder.block.0.layer.0.SelfAttention.q.weight": torch.randn(16, 16),
        "encoder.block.0.layer.0.SelfAttention.k.weight": torch.randn(16, 16),
        "encoder.block.0.layer.0.SelfAttention.v.weight": torch.randn(16, 16),
        "encoder.block.0.layer.1.DenseReluDense.wo.weight": torch.randn(16, 32),
    }


def _tiny_t5_shard1() -> dict[str, torch.Tensor]:
    return {
        "encoder.block.1.layer.0.SelfAttention.q.weight": torch.randn(16, 16),
        "encoder.block.1.layer.0.SelfAttention.k.weight": torch.randn(16, 16),
        "encoder.block.1.layer.0.SelfAttention.v.weight": torch.randn(16, 16),
        "encoder.block.1.layer.1.DenseReluDense.wo.weight": torch.randn(16, 32),
        "encoder.final_layer_norm.weight": torch.randn(16),
    }


def _tiny_anima_state() -> dict[str, torch.Tensor]:
    return {
        "net.x_embedder.proj.1.weight": torch.randn(16, 16),
        "net.t_embedder.1.linear_1.weight": torch.randn(16, 16),
        "net.final_layer.linear.weight": torch.randn(16, 16),
        "net.blocks.0.self_attn.q_proj.weight": torch.randn(16, 16),
        "net.blocks.0.self_attn.k_proj.weight": torch.randn(16, 16),
        "net.blocks.0.self_attn.v_proj.weight": torch.randn(16, 16),
        "net.blocks.0.self_attn.output_proj.weight": torch.randn(16, 16),
        "net.blocks.0.cross_attn.q_proj.weight": torch.randn(16, 16),
        "net.blocks.0.mlp.layer1.weight": torch.randn(32, 16),
        "net.blocks.0.mlp.layer2.weight": torch.randn(16, 32),
    }


def _tiny_newbie_state() -> dict[str, torch.Tensor]:
    return {
        "x_embedder.weight": torch.randn(16, 16, 2, 2),
        "t_embedder.mlp.0.weight": torch.randn(16, 16),
        "final_layer.linear.weight": torch.randn(16, 16),
        "layers.0.attention.qkv.weight": torch.randn(48, 16),
        "layers.0.attention.out.weight": torch.randn(16, 16),
        "layers.0.feed_forward.w1.weight": torch.randn(32, 16),
        "layers.0.feed_forward.w2.weight": torch.randn(16, 32),
        "layers.0.feed_forward.w3.weight": torch.randn(32, 16),
        "context_refiner.0.attention.qkv.weight": torch.randn(48, 16),
        "noise_refiner.0.attention.qkv.weight": torch.randn(48, 16),
    }


def _tiny_sd15_unet_state() -> dict[str, torch.Tensor]:
    return {
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(16, 4, 3, 3),
        "model.diffusion_model.time_embed.0.weight": torch.randn(16, 16),
        "model.diffusion_model.middle_block.0.in_layers.0.weight": torch.randn(16),
        "model.diffusion_model.output_blocks.0.0.in_layers.0.weight": torch.randn(16),
        "model.diffusion_model.out.2.weight": torch.randn(4, 16, 3, 3),
        "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": torch.randn(32, 16),
    }


def _tiny_sdxl_unet_state() -> dict[str, torch.Tensor]:
    return {
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(16, 4, 3, 3),
        "model.diffusion_model.time_embed.0.weight": torch.randn(16, 16),
        "model.diffusion_model.middle_block.0.in_layers.0.weight": torch.randn(16),
        "model.diffusion_model.output_blocks.0.0.in_layers.0.weight": torch.randn(16),
        "model.diffusion_model.out.2.weight": torch.randn(4, 16, 3, 3),
        "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": torch.randn(32, 16),
    }


def test_plan_vae_export() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "vae.safetensors")
        _save(src, _tiny_vae_state())
        plan = plan_image_gguf_export(src, family_hint="vae", file_type="f16").to_dict()
        assert plan["ok"] is True, plan
        assert plan["component"] == "vae", plan
        assert plan["family"] == "diffusers_vae", plan
        assert plan["compatibility"] == "container_candidate", plan
        assert plan["unique_tensor_count"] == 7, plan
        assert plan["estimated_output_size_bytes"] > plan["estimated_tensor_bytes"], plan
        print("PASS: image GGUF export plan estimates VAE container")


def test_plan_t5_shards() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        shard0 = os.path.join(tmp, "t5-0.safetensors")
        shard1 = os.path.join(tmp, "t5-1.safetensors")
        _save(shard0, _tiny_t5_shard0())
        _save(shard1, _tiny_t5_shard1())
        plan = plan_image_gguf_export([shard0, shard1], family_hint="t5", file_type="f32").to_dict()
        assert plan["ok"] is True, plan
        assert plan["component"] == "t5", plan
        assert plan["unique_tensor_count"] == 10, plan
        assert plan["duplicate_tensor_count"] == 0, plan
        assert any("source shard has missing required tensors" in item for item in plan["warnings"]), plan
        print("PASS: image GGUF export plan accepts T5 shard set")


def test_plan_accepts_anima_dit_container_candidate() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "anima.safetensors")
        _save(src, _tiny_anima_state())
        plan = plan_image_gguf_export(src, family_hint="anima").to_dict()
        assert plan["ok"] is True, plan
        assert plan["component"] == "anima_dit", plan
        assert plan["compatibility"] == "container_candidate", plan
        assert any("container-compatible only" in item for item in plan["warnings"]), plan
        print("PASS: image GGUF export plan accepts Anima DiT container candidate")


def test_plan_accepts_newbie_dit_container_candidate() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "newbie.safetensors")
        _save(src, _tiny_newbie_state())
        plan = plan_image_gguf_export(src, family_hint="newbie").to_dict()
        assert plan["ok"] is True, plan
        assert plan["component"] == "newbie_dit", plan
        assert plan["compatibility"] == "container_candidate", plan
        assert any("container-compatible only" in item for item in plan["warnings"]), plan
        print("PASS: image GGUF export plan accepts Newbie DiT container candidate")


def test_plan_accepts_sd15_unet_container_candidate() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "sd15.safetensors")
        _save(src, _tiny_sd15_unet_state())
        plan = plan_image_gguf_export(src, family_hint="sd15").to_dict()
        assert plan["ok"] is True, plan
        assert plan["component"] == "sd15_unet", plan
        assert plan["compatibility"] == "container_candidate", plan
        assert any("container-compatible only" in item for item in plan["warnings"]), plan
        print("PASS: image GGUF export plan accepts SD1.5 UNet container candidate")


def test_plan_accepts_sdxl_unet_container_candidate() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "sdxl.safetensors")
        _save(src, _tiny_sdxl_unet_state())
        plan = plan_image_gguf_export(src, family_hint="sdxl").to_dict()
        assert plan["ok"] is True, plan
        assert plan["component"] == "sdxl_unet", plan
        assert plan["compatibility"] == "container_candidate", plan
        assert any("container-compatible only" in item for item in plan["warnings"]), plan
        print("PASS: image GGUF export plan accepts SDXL UNet container candidate")


if __name__ == "__main__":
    test_plan_vae_export()
    test_plan_t5_shards()
    test_plan_accepts_anima_dit_container_candidate()
    test_plan_accepts_newbie_dit_container_candidate()
    test_plan_accepts_sd15_unet_container_candidate()
    test_plan_accepts_sdxl_unet_container_candidate()
    print("\nAll image GGUF export plan smoke tests passed!")
