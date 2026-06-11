"""Smoke tests for the Python reference image GGUF shape loader."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch


BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.tools.image_gguf_exporter import export_image_gguf_component  # noqa: E402
from core.tools.image_gguf_runtime_contract import build_image_gguf_runtime_contract  # noqa: E402
from core.tools.image_gguf_shape_loader import load_image_gguf_shape_contract  # noqa: E402


def _save(path: str | Path, state: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    save_file(state, str(path), metadata={"source": "image_gguf_shape_loader_smoke"})


def _vae_state() -> dict[str, torch.Tensor]:
    return {
        "encoder.conv_in.weight": torch.randn(8, 4, 3, 3),
        "encoder.conv_out.weight": torch.randn(4, 8, 3, 3),
        "encoder.down_blocks.0.resnets.0.conv1.weight": torch.randn(8, 8, 3, 3),
        "decoder.conv_in.weight": torch.randn(8, 4, 3, 3),
        "decoder.conv_out.weight": torch.randn(4, 8, 3, 3),
        "decoder.mid_block.attentions.0.to_q.weight": torch.randn(8, 8),
        "decoder.up_blocks.0.resnets.0.conv1.weight": torch.randn(8, 8, 3, 3),
    }


def _qwen_image_vae_state() -> dict[str, torch.Tensor]:
    return {
        "conv1.weight": torch.randn(32, 32, 1, 1, 1),
        "conv2.weight": torch.randn(16, 16, 1, 1, 1),
        "encoder.downsamples.0.residual.2.weight": torch.randn(96, 96, 3, 3, 3),
        "decoder.upsamples.0.residual.2.weight": torch.randn(384, 384, 3, 3, 3),
        "decoder.middle.1.to_qkv.weight": torch.randn(1152, 384, 1, 1),
    }


def _t5_state() -> dict[str, torch.Tensor]:
    return {
        "shared.weight": torch.randn(32, 16),
        "encoder.block.0.layer.0.SelfAttention.q.weight": torch.randn(16, 16),
        "encoder.block.0.layer.0.SelfAttention.k.weight": torch.randn(16, 16),
        "encoder.block.0.layer.0.SelfAttention.v.weight": torch.randn(16, 16),
        "encoder.block.0.layer.0.SelfAttention.o.weight": torch.randn(16, 16),
        "encoder.block.0.layer.1.DenseReluDense.wo.weight": torch.randn(16, 32),
        "encoder.final_layer_norm.weight": torch.randn(16),
    }


def _clip_state() -> dict[str, torch.Tensor]:
    return {
        "text_model.embeddings.token_embedding.weight": torch.randn(64, 16),
        "text_model.embeddings.position_embedding.weight": torch.randn(32, 16),
        "text_model.encoder.layers.0.self_attn.q_proj.weight": torch.randn(16, 16),
        "text_model.encoder.layers.0.self_attn.k_proj.weight": torch.randn(16, 16),
        "text_model.encoder.layers.0.self_attn.v_proj.weight": torch.randn(16, 16),
        "text_model.encoder.layers.0.self_attn.out_proj.weight": torch.randn(16, 16),
        "text_model.encoder.layers.0.mlp.fc1.weight": torch.randn(32, 16),
        "text_model.encoder.layers.0.mlp.fc2.weight": torch.randn(16, 32),
        "text_model.final_layer_norm.weight": torch.randn(16),
    }


def _sdxl_unet_state() -> dict[str, torch.Tensor]:
    return {
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(16, 4, 3, 3),
        "model.diffusion_model.time_embed.0.weight": torch.randn(16, 16),
        "model.diffusion_model.middle_block.0.in_layers.0.weight": torch.randn(16),
        "model.diffusion_model.output_blocks.0.0.in_layers.0.weight": torch.randn(16),
        "model.diffusion_model.out.2.weight": torch.randn(4, 16, 3, 3),
        "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": torch.randn(32, 16),
    }


def _anima_state() -> dict[str, torch.Tensor]:
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


def _newbie_state() -> dict[str, torch.Tensor]:
    return {
        "x_embedder.weight": torch.randn(16, 4, 2, 2),
        "t_embedder.mlp.0.weight": torch.randn(16, 16),
        "final_layer.linear.weight": torch.randn(16, 16),
        "layers.0.attention.qkv.weight": torch.randn(48, 16),
        "layers.0.attention.out.weight": torch.randn(16, 16),
        "layers.0.feed_forward.w1.weight": torch.randn(32, 16),
        "layers.0.feed_forward.w2.weight": torch.randn(16, 32),
        "layers.0.feed_forward.w3.weight": torch.randn(32, 16),
    }


def _load_contract(state: dict[str, torch.Tensor], family_hint: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, f"{family_hint}.safetensors")
        dst = os.path.join(tmp, f"{family_hint}.gguf")
        _save(src, state)
        export_image_gguf_component(src, dst, family_hint=family_hint, name=f"shape-{family_hint}", overwrite=True)
        return load_image_gguf_shape_contract(dst)


def _tensor(report: dict[str, object], name: str) -> dict[str, object]:
    for item in report["tensor_descriptors_sample"]:
        if item["name"] == name:
            return item
    raise AssertionError(f"tensor not found in sample: {name}")


def _runtime_contract_for(component: str, tensor_type_counts: dict[str, int]) -> dict[str, object]:
    return build_image_gguf_runtime_contract(
        component=component,
        issues=[],
        container_contract={"ok": True},
        shape_contract={"ok": True, "tensor_type_counts": tensor_type_counts},
    )


def test_vae_shape_contract() -> None:
    report = _load_contract(_vae_state(), "vae")
    assert report["ok"] is True, report
    assert report["component"] == "vae", report
    assert report["shape_contract"]["ok"] is True, report
    tensor = _tensor(report, "encoder.conv_in.weight")
    assert tensor["logical_shape"] == [8, 4, 3, 3], tensor
    assert tensor["storage_shape"] == [3, 3, 4, 8], tensor
    assert report["shape_contract"]["derived"]["latent_channels"] == 4, report
    print("PASS: VAE image GGUF shape loader preserves logical/storage shape")


def test_qwen_image_vae_shape_contract() -> None:
    report = _load_contract(_qwen_image_vae_state(), "qwen_image_vae")
    assert report["ok"] is True, report
    assert report["component"] == "vae", report
    assert report["family"] == "qwen_image_vae", report
    assert report["shape_contract"]["ok"] is True, report
    assert report["shape_contract"]["derived"]["temporal_latent_channels"] == 32, report
    assert report["shape_contract"]["derived"]["output_channels"] == 16, report
    assert report["shape_contract"]["derived"]["middle_attention_qkv_dim"] == 1152, report
    print("PASS: Qwen Image VAE image GGUF shape loader validates 5D VAE contract")


def test_clip_shape_contract() -> None:
    report = _load_contract(_clip_state(), "clip")
    assert report["ok"] is True, report
    assert report["component"] == "clip", report
    assert report["shape_contract"]["ok"] is True, report
    assert report["shape_contract"]["derived"]["hidden_dim"] == 16, report
    assert report["shape_contract"]["derived"]["max_positions"] == 32, report
    print("PASS: CLIP image GGUF shape loader validates hidden-dim contract")


def test_t5_shape_contract() -> None:
    report = _load_contract(_t5_state(), "t5")
    assert report["ok"] is True, report
    assert report["component"] == "t5", report
    assert report["shape_contract"]["ok"] is True, report
    assert report["shape_contract"]["rank_counts"].get("2") == 6, report
    assert report["shape_contract"]["derived"]["hidden_dim"] == 16, report
    assert report["shape_contract"]["derived"]["attention_inner_dim"] == 16, report
    print("PASS: T5 image GGUF shape loader validates rank contract")


def test_unet_shape_contract() -> None:
    report = _load_contract(_sdxl_unet_state(), "sdxl")
    assert report["ok"] is True, report
    assert report["component"] == "sdxl_unet", report
    assert report["shape_contract"]["ok"] is True, report
    assert report["shape_contract"]["derived"]["latent_channels"] == 4, report
    assert report["shape_contract"]["derived"]["output_channels"] == 4, report
    assert report["shape_contract"]["derived"]["time_embed_dim"] == 16, report
    print("PASS: SDXL UNet image GGUF shape loader derives channel metadata")


def test_dit_shape_contract() -> None:
    report = _load_contract(_anima_state(), "anima")
    assert report["ok"] is True, report
    assert report["component"] == "anima_dit", report
    assert report["shape_contract"]["ok"] is True, report
    assert report["runtime_loadable"] is False, report
    assert report["runtime_contract"]["loadability"] == "shape_only_reference", report
    assert report["runtime_contract"]["runtime_loader"]["implemented"] is False, report
    assert report["runtime_contract"]["runtime_loader_abi"]["abi"] == "image_gguf_runtime_loader_v1", report
    assert report["runtime_contract"]["runtime_loader_abi"]["report_only"] is True, report
    assert report["runtime_contract"]["runtime_loader_abi"]["reads_tensor_payloads"] is False, report
    assert "dit_module_builder" in report["runtime_contract"]["required_runtime_features"], report
    assert report["runtime_contract"]["tensor_type_policy"]["ok"] is True, report
    assert any("runtime model loader is not implemented" in item for item in report["runtime_blockers"]), report
    assert report["shape_contract"]["derived"]["hidden_dim"] == 16, report
    print("PASS: Anima DiT image GGUF shape loader validates shape-only contract")


def test_newbie_dit_shape_contract() -> None:
    report = _load_contract(_newbie_state(), "newbie")
    assert report["ok"] is True, report
    assert report["component"] == "newbie_dit", report
    assert report["shape_contract"]["ok"] is True, report
    assert report["shape_contract"]["derived"]["hidden_dim"] == 16, report
    assert report["shape_contract"]["derived"]["latent_channels"] == 4, report
    assert report["shape_contract"]["derived"]["patch_size"] == [2, 2], report
    print("PASS: Newbie DiT image GGUF shape loader validates packed-qkv contract")


def test_runtime_contract_blocks_unsupported_tensor_types() -> None:
    contract = _runtime_contract_for("vae", {"f16": 1, "q8_0": 1})
    policy = contract["tensor_type_policy"]
    assert contract["runtime_loadable"] is False, contract
    assert policy["ok"] is False, contract
    assert policy["unsupported"] == ["q8_0"], contract
    assert any("unsupported tensor types" in item for item in contract["blockers"]), contract
    assert any("runtime model loader is not implemented" in item for item in contract["blockers"]), contract
    assert contract["runtime_loader_abi"]["tensor_type_policy"]["unsupported"] == ["q8_0"], contract
    print("PASS: image GGUF runtime contract blocks unsupported tensor types")


def test_runtime_contract_marks_experimental_tensor_types() -> None:
    contract = _runtime_contract_for("clip", {"bf16": 2, "f16": 1})
    policy = contract["tensor_type_policy"]
    assert policy["ok"] is True, contract
    assert policy["requires_explicit_opt_in"] == ["bf16"], contract
    assert policy["unsupported"] == [], contract
    assert "clip_text_encoder_builder" in contract["required_runtime_features"], contract
    assert contract["runtime_loader_abi"]["supported_load_modes"] == ["descriptor_only"], contract
    assert any("runtime model loader is not implemented" in item for item in contract["blockers"]), contract
    print("PASS: image GGUF runtime contract marks experimental tensor types")


def test_runtime_contract_blocks_unknown_component() -> None:
    contract = _runtime_contract_for("unknown_image", {"f16": 1})
    assert contract["runtime_loadable"] is False, contract
    assert "component_runtime_adapter" in contract["required_runtime_features"], contract
    assert contract["runtime_loader_abi"]["component"] == "unknown_image", contract
    assert any("unsupported image GGUF component" in item for item in contract["blockers"]), contract
    assert any("runtime model loader is not implemented" in item for item in contract["blockers"]), contract
    print("PASS: image GGUF runtime contract blocks unknown components")


if __name__ == "__main__":
    test_vae_shape_contract()
    test_qwen_image_vae_shape_contract()
    test_clip_shape_contract()
    test_t5_shape_contract()
    test_unet_shape_contract()
    test_dit_shape_contract()
    test_newbie_dit_shape_contract()
    test_runtime_contract_blocks_unsupported_tensor_types()
    test_runtime_contract_marks_experimental_tensor_types()
    test_runtime_contract_blocks_unknown_component()
    print("\nAll image GGUF shape loader smoke tests passed!")
