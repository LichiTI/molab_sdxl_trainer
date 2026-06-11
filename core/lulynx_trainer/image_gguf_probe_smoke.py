"""Smoke tests for image GGUF manifest probes."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.image_gguf_probe import probe_image_gguf_manifest  # noqa: E402


def _save(path: str | Path, state: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    save_file(state, str(path), metadata={"source": "image_gguf_probe_smoke"})


def _tiny_anima_state() -> dict[str, torch.Tensor]:
    return {
        "net.x_embedder.proj.1.weight": torch.randn(16, 16),
        "net.t_embedder.1.linear_1.weight": torch.randn(16, 16),
        "net.t_embedder.1.linear_2.weight": torch.randn(16, 16),
        "net.t_embedding_norm.weight": torch.randn(16),
        "net.final_layer.linear.weight": torch.randn(16, 16),
        "net.blocks.0.self_attn.q_proj.weight": torch.randn(16, 16),
        "net.blocks.0.self_attn.k_proj.weight": torch.randn(16, 16),
        "net.blocks.0.self_attn.v_proj.weight": torch.randn(16, 16),
        "net.blocks.0.self_attn.output_proj.weight": torch.randn(16, 16),
        "net.blocks.0.cross_attn.q_proj.weight": torch.randn(16, 16),
        "net.blocks.0.mlp.layer1.weight": torch.randn(32, 16),
        "net.blocks.0.mlp.layer2.weight": torch.randn(16, 32),
        "net.llm_adapter.proj.weight": torch.randn(16, 16),
    }


def _tiny_newbie_state() -> dict[str, torch.Tensor]:
    return {
        "x_embedder.weight": torch.randn(16, 16, 2, 2),
        "t_embedder.mlp.0.weight": torch.randn(16, 16),
        "final_layer.linear.weight": torch.randn(16, 16),
        "final_layer.adaLN_modulation.1.weight": torch.randn(16, 16),
        "layers.0.attention.qkv.weight": torch.randn(48, 16),
        "layers.0.attention.out.weight": torch.randn(16, 16),
        "layers.0.feed_forward.w1.weight": torch.randn(32, 16),
        "layers.0.feed_forward.w2.weight": torch.randn(16, 32),
        "layers.0.feed_forward.w3.weight": torch.randn(32, 16),
        "context_refiner.0.attention.qkv.weight": torch.randn(48, 16),
        "noise_refiner.0.attention.qkv.weight": torch.randn(48, 16),
    }


def _tiny_diffusers_vae_state() -> dict[str, torch.Tensor]:
    return {
        "encoder.conv_in.weight": torch.randn(16, 4, 3, 3),
        "encoder.conv_out.weight": torch.randn(8, 16, 3, 3),
        "encoder.down_blocks.0.resnets.0.conv1.weight": torch.randn(16, 16, 3, 3),
        "decoder.conv_in.weight": torch.randn(16, 4, 3, 3),
        "decoder.conv_out.weight": torch.randn(4, 16, 3, 3),
        "decoder.mid_block.attentions.0.to_q.weight": torch.randn(16, 16),
        "decoder.up_blocks.0.resnets.0.conv1.weight": torch.randn(16, 16, 3, 3),
    }


def _tiny_qwen_image_vae_state() -> dict[str, torch.Tensor]:
    return {
        "conv1.weight": torch.randn(16, 4, 3, 3),
        "conv2.weight": torch.randn(4, 16, 3, 3),
        "encoder.downsamples.0.residual.2.weight": torch.randn(16, 16, 3, 3),
        "encoder.middle.1.to_qkv.weight": torch.randn(48, 16),
        "decoder.middle.1.to_qkv.weight": torch.randn(48, 16),
        "decoder.upsamples.0.residual.2.weight": torch.randn(16, 16, 3, 3),
    }


def _tiny_clip_text_state() -> dict[str, torch.Tensor]:
    return {
        "text_model.embeddings.token_embedding.weight": torch.randn(32, 16),
        "text_model.embeddings.position_embedding.weight": torch.randn(32, 16),
        "text_model.encoder.layers.0.self_attn.q_proj.weight": torch.randn(16, 16),
        "text_model.encoder.layers.0.self_attn.k_proj.weight": torch.randn(16, 16),
        "text_model.encoder.layers.0.self_attn.v_proj.weight": torch.randn(16, 16),
        "text_model.encoder.layers.0.mlp.fc1.weight": torch.randn(32, 16),
        "text_model.final_layer_norm.weight": torch.randn(16),
    }


def _tiny_jina_clip_text_state() -> dict[str, torch.Tensor]:
    return {
        "model.embeddings.word_embeddings.weight": torch.randn(32, 16),
        "model.embeddings.token_type_embeddings.weight": torch.randn(2, 16),
        "model.encoder.layers.0.mixer.Wqkv.weight": torch.randn(48, 16),
        "model.encoder.layers.0.mixer.out_proj.weight": torch.randn(16, 16),
        "model.encoder.layers.0.mlp.fc1.weight": torch.randn(32, 16),
        "model.emb_ln.weight": torch.randn(16),
        "spiece_model": torch.zeros(8, dtype=torch.uint8),
    }


def _tiny_t5_encoder_state() -> dict[str, torch.Tensor]:
    return {
        "shared.weight": torch.randn(32, 16),
        "encoder.block.0.layer.0.SelfAttention.q.weight": torch.randn(16, 16),
        "encoder.block.0.layer.0.SelfAttention.k.weight": torch.randn(16, 16),
        "encoder.block.0.layer.0.SelfAttention.v.weight": torch.randn(16, 16),
        "encoder.block.0.layer.0.SelfAttention.o.weight": torch.randn(16, 16),
        "encoder.block.0.layer.1.DenseReluDense.wo.weight": torch.randn(16, 32),
        "encoder.final_layer_norm.weight": torch.randn(16),
    }


def _tiny_sdxl_unet_state() -> dict[str, torch.Tensor]:
    return {
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(16, 4, 3, 3),
        "model.diffusion_model.time_embed.0.weight": torch.randn(16, 16),
        "model.diffusion_model.middle_block.0.in_layers.0.weight": torch.randn(16),
        "model.diffusion_model.output_blocks.0.0.in_layers.0.weight": torch.randn(16),
        "model.diffusion_model.out.2.weight": torch.randn(4, 16, 3, 3),
        "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": torch.randn(32, 16),
        "first_stage_model.encoder.conv_in.weight": torch.randn(16, 4, 3, 3),
        "first_stage_model.decoder.conv_in.weight": torch.randn(16, 4, 3, 3),
    }


def _tiny_sd15_unet_state() -> dict[str, torch.Tensor]:
    return {
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(16, 4, 3, 3),
        "model.diffusion_model.time_embed.0.weight": torch.randn(16, 16),
        "model.diffusion_model.middle_block.0.in_layers.0.weight": torch.randn(16),
        "model.diffusion_model.output_blocks.0.0.in_layers.0.weight": torch.randn(16),
        "model.diffusion_model.out.2.weight": torch.randn(4, 16, 3, 3),
        "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": torch.randn(32, 16),
        "first_stage_model.encoder.conv_in.weight": torch.randn(16, 4, 3, 3),
        "first_stage_model.decoder.conv_in.weight": torch.randn(16, 4, 3, 3),
    }


def _tiny_bare_unet_state() -> dict[str, torch.Tensor]:
    return {
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(16, 4, 3, 3),
        "model.diffusion_model.time_embed.0.weight": torch.randn(16, 16),
        "model.diffusion_model.middle_block.0.in_layers.0.weight": torch.randn(16),
        "model.diffusion_model.output_blocks.0.0.in_layers.0.weight": torch.randn(16),
        "model.diffusion_model.out.2.weight": torch.randn(4, 16, 3, 3),
    }


def _tiny_flux_transformer_state() -> dict[str, torch.Tensor]:
    return {
        "context_embedder.weight": torch.randn(8, 8),
        "time_text_embed.timestep_embedder.linear_1.weight": torch.randn(8, 8),
        "transformer_blocks.0.attn.to_q.weight": torch.randn(8, 8),
        "transformer_blocks.0.attn.to_k.weight": torch.randn(8, 8),
        "transformer_blocks.0.ff.net.0.proj.weight": torch.randn(16, 8),
        "single_transformer_blocks.0.attn.to_q.weight": torch.randn(8, 8),
        "single_transformer_blocks.0.proj_mlp.weight": torch.randn(16, 8),
        "proj_out.weight": torch.randn(8, 8),
    }


def test_anima_probe_detects_manifest() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "anima.safetensors")
        _save(path, _tiny_anima_state())
        manifest = probe_image_gguf_manifest(path).to_dict()
        assert manifest["ok"] is True, manifest
        assert manifest["component"] == "anima_dit"
        assert manifest["family"] == "anima"
        assert manifest["compatibility"] == "probe_only"
        assert manifest["matched_tensors"] >= 10
        assert manifest["missing_required_tensors"] == []
        assert manifest["shape_summary"]["total_numel"] > 0
        print("PASS: Anima DiT image GGUF manifest probe detects tiny state")


def test_newbie_probe_detects_manifest() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "newbie.safetensors")
        _save(path, _tiny_newbie_state())
        manifest = probe_image_gguf_manifest(path).to_dict()
        assert manifest["ok"] is True, manifest
        assert manifest["component"] == "newbie_dit"
        assert manifest["family"] == "newbie"
        assert manifest["compatibility"] == "probe_only"
        assert manifest["matched_tensors"] >= 10
        assert manifest["missing_required_tensors"] == []
        print("PASS: Newbie DiT image GGUF manifest probe detects tiny state")


def test_diffusers_vae_probe_detects_manifest() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "vae.safetensors")
        _save(path, _tiny_diffusers_vae_state())
        manifest = probe_image_gguf_manifest(path).to_dict()
        assert manifest["ok"] is True, manifest
        assert manifest["component"] == "vae"
        assert manifest["family"] == "diffusers_vae"
        assert manifest["missing_required_tensors"] == []
        print("PASS: Diffusers VAE image GGUF manifest probe detects tiny state")


def test_qwen_image_vae_probe_detects_manifest() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "qwen-image-vae.safetensors")
        _save(path, _tiny_qwen_image_vae_state())
        manifest = probe_image_gguf_manifest(path).to_dict()
        assert manifest["ok"] is True, manifest
        assert manifest["component"] == "vae"
        assert manifest["family"] == "qwen_image_vae"
        assert manifest["missing_required_tensors"] == []
        print("PASS: Qwen Image VAE image GGUF manifest probe detects tiny state")


def test_clip_text_probe_detects_manifest() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "clip-text.safetensors")
        _save(path, _tiny_clip_text_state())
        manifest = probe_image_gguf_manifest(path).to_dict()
        assert manifest["ok"] is True, manifest
        assert manifest["component"] == "clip"
        assert manifest["family"] == "clip_text"
        assert manifest["missing_required_tensors"] == []
        print("PASS: CLIP text image GGUF manifest probe detects tiny state")


def test_jina_clip_text_probe_detects_manifest() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "jina-clip-text.safetensors")
        _save(path, _tiny_jina_clip_text_state())
        manifest = probe_image_gguf_manifest(path).to_dict()
        assert manifest["ok"] is True, manifest
        assert manifest["component"] == "clip"
        assert manifest["family"] == "jina_clip_text"
        assert manifest["missing_required_tensors"] == []
        print("PASS: Jina CLIP text image GGUF manifest probe detects tiny state")


def test_t5_encoder_probe_detects_manifest() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "t5-encoder.safetensors")
        _save(path, _tiny_t5_encoder_state())
        manifest = probe_image_gguf_manifest(path).to_dict()
        assert manifest["ok"] is True, manifest
        assert manifest["component"] == "t5"
        assert manifest["family"] == "t5_encoder"
        assert manifest["missing_required_tensors"] == []
        print("PASS: T5 encoder image GGUF manifest probe detects tiny state")


def test_sdxl_unet_probe_detects_manifest() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "sdxl.safetensors")
        _save(path, _tiny_sdxl_unet_state())
        manifest = probe_image_gguf_manifest(path).to_dict()
        assert manifest["ok"] is True, manifest
        assert manifest["component"] == "sdxl_unet"
        assert manifest["family"] == "sdxl_unet"
        assert manifest["missing_required_tensors"] == []
        print("PASS: SDXL UNet image GGUF manifest probe detects tiny state")


def test_sd15_unet_probe_detects_manifest() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "sd15.safetensors")
        _save(path, _tiny_sd15_unet_state())
        manifest = probe_image_gguf_manifest(path).to_dict()
        assert manifest["ok"] is True, manifest
        assert manifest["component"] == "sd15_unet"
        assert manifest["family"] == "sd15_unet"
        assert manifest["missing_required_tensors"] == []
        print("PASS: SD1.5 UNet image GGUF manifest probe detects tiny state")


def test_bare_unet_without_family_marker_stays_generic() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "bare-unet.safetensors")
        _save(path, _tiny_bare_unet_state())
        manifest = probe_image_gguf_manifest(path).to_dict()
        assert manifest["ok"] is True, manifest
        assert manifest["component"] == "generic_tensor_bundle"
        assert manifest["family"] == "generic"
        print("PASS: bare UNet namespace stays generic without SD1.5/SDXL family marker")


def test_flux_transformer_guard_stays_generic() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "flux-transformer.safetensors")
        _save(path, _tiny_flux_transformer_state())
        manifest = probe_image_gguf_manifest(path, family_hint="flux_transformer").to_dict()
        assert manifest["ok"] is True, manifest
        assert manifest["adapter_id"] == "flux_transformer_probe_v1", manifest
        assert manifest["component"] == "generic_tensor_bundle", manifest
        assert manifest["family"] == "flux_transformer", manifest
        assert any("not export/runtime compatible" in item for item in manifest["warnings"]), manifest
        print("PASS: FLUX transformer probe is guarded from Newbie DiT misclassification")


def test_missing_required_tensor_is_reported() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "bad-anima.safetensors")
        state = _tiny_anima_state()
        state.pop("net.blocks.0.self_attn.q_proj.weight")
        _save(path, state)
        manifest = probe_image_gguf_manifest(path, family_hint="anima").to_dict()
        assert manifest["component"] == "anima_dit"
        assert manifest["ok"] is False
        assert "net.blocks.0.self_attn.q_proj.weight" in manifest["missing_required_tensors"]
        print("PASS: image GGUF probe reports missing required tensors")


def test_unknown_falls_back_to_generic_bundle() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "unknown.safetensors")
        _save(path, {"linear.weight": torch.randn(4, 4), "linear.bias": torch.randn(4)})
        manifest = probe_image_gguf_manifest(path).to_dict()
        assert manifest["ok"] is True, manifest
        assert manifest["component"] == "generic_tensor_bundle"
        assert manifest["family"] == "generic"
        assert manifest["adapter_id"] == "generic_tensor_bundle_probe_v1"
        assert any("not a runtime-compatible" in item for item in manifest["warnings"])
        print("PASS: image GGUF probe falls back to generic tensor bundle")


if __name__ == "__main__":
    test_anima_probe_detects_manifest()
    test_newbie_probe_detects_manifest()
    test_diffusers_vae_probe_detects_manifest()
    test_qwen_image_vae_probe_detects_manifest()
    test_clip_text_probe_detects_manifest()
    test_jina_clip_text_probe_detects_manifest()
    test_t5_encoder_probe_detects_manifest()
    test_sdxl_unet_probe_detects_manifest()
    test_sd15_unet_probe_detects_manifest()
    test_bare_unet_without_family_marker_stays_generic()
    test_flux_transformer_guard_stays_generic()
    test_missing_required_tensor_is_reported()
    test_unknown_falls_back_to_generic_bundle()
    print("\nAll image GGUF probe smoke tests passed!")
