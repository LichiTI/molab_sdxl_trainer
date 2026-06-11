"""Smoke tests for image GGUF component container exports."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.tools.image_gguf_exporter import GGUF_ARCH, export_image_gguf_component  # noqa: E402


def _save(path: str | Path, state: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    save_file(state, str(path), metadata={"source": "image_gguf_export_smoke"})


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


def _tiny_t5_shard0_state() -> dict[str, torch.Tensor]:
    return {
        "shared.weight": torch.randn(32, 16),
        "encoder.block.0.layer.0.SelfAttention.q.weight": torch.randn(16, 16),
        "encoder.block.0.layer.0.SelfAttention.k.weight": torch.randn(16, 16),
        "encoder.block.0.layer.0.SelfAttention.v.weight": torch.randn(16, 16),
        "encoder.block.0.layer.0.SelfAttention.o.weight": torch.randn(16, 16),
        "encoder.block.0.layer.1.DenseReluDense.wo.weight": torch.randn(16, 32),
    }


def _tiny_t5_shard1_state() -> dict[str, torch.Tensor]:
    return {
        "encoder.block.1.layer.0.SelfAttention.q.weight": torch.randn(16, 16),
        "encoder.block.1.layer.0.SelfAttention.k.weight": torch.randn(16, 16),
        "encoder.block.1.layer.0.SelfAttention.v.weight": torch.randn(16, 16),
        "encoder.block.1.layer.0.SelfAttention.o.weight": torch.randn(16, 16),
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


def test_vae_export_or_missing_dependency() -> None:
    try:
        import gguf
    except ImportError:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            src = os.path.join(tmp, "vae.safetensors")
            dst = os.path.join(tmp, "vae.gguf")
            _save(src, _tiny_diffusers_vae_state())
            try:
                export_image_gguf_component(src, dst)
            except RuntimeError as exc:
                assert "Image GGUF export requires" in str(exc)
                print("PASS: image GGUF export reports missing gguf dependency")
                return
            raise AssertionError("image GGUF export should require gguf when module is missing")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "vae.safetensors")
        dst = os.path.join(tmp, "vae.gguf")
        _save(src, _tiny_diffusers_vae_state())
        result = export_image_gguf_component(src, dst, family_hint="vae", name="tiny-vae", file_type="f16")
        assert result.ok is True
        assert result.component == "vae"
        assert result.family == "diffusers_vae"
        assert result.tensor_count == 7
        assert result.converted_tensors == 7
        assert result.gguf_arch == GGUF_ARCH
        assert os.path.isfile(result.sidecar_path)

        reader = gguf.GGUFReader(dst)
        tensor_names = {str(tensor.name) for tensor in reader.tensors}
        assert "encoder.conv_in.weight" in tensor_names
        assert "general.name" in reader.fields
        assert "lulynx.image_gguf.component" in reader.fields
        assert "lulynx.image_gguf.compatibility" in reader.fields
        sidecar = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))
        assert sidecar["compatibility"] == "container_compatible"
        assert sidecar["probe_manifests"][0]["component"] == "vae"
        print("PASS: VAE image GGUF export writes readable container and sidecar")


def test_t5_multi_shard_export_or_missing_dependency() -> None:
    try:
        import gguf
    except ImportError:
        print("SKIP: T5 image GGUF export requires gguf")
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        shard0 = os.path.join(tmp, "t5-00001.safetensors")
        shard1 = os.path.join(tmp, "t5-00002.safetensors")
        dst = os.path.join(tmp, "t5.gguf")
        _save(shard0, _tiny_t5_shard0_state())
        _save(shard1, _tiny_t5_shard1_state())
        result = export_image_gguf_component([shard0, shard1], dst, family_hint="t5", name="tiny-t5", file_type="f32")
        assert result.ok is True
        assert result.component == "t5"
        assert result.family == "t5_encoder"
        assert result.tensor_count == 12
        assert result.converted_tensors == 12
        assert len(result.source_paths) == 2

        reader = gguf.GGUFReader(dst)
        tensor_names = {str(tensor.name) for tensor in reader.tensors}
        assert "encoder.block.0.layer.0.SelfAttention.q.weight" in tensor_names
        assert "encoder.block.1.layer.0.SelfAttention.q.weight" in tensor_names
        sidecar = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))
        assert len(sidecar["probe_manifests"]) == 2
        assert sidecar["probe_manifests"][1]["ok"] is False
        print("PASS: T5 image GGUF export aggregates shards into one readable container")


def test_unet_container_export_or_missing_dependency() -> None:
    try:
        import gguf
    except ImportError:
        print("SKIP: UNet image GGUF export requires gguf")
        return

    cases = [
        ("sd15", _tiny_sd15_unet_state(), "sd15_unet"),
        ("sdxl", _tiny_sdxl_unet_state(), "sdxl_unet"),
    ]
    for family_hint, state, expected_component in cases:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            src = os.path.join(tmp, f"{family_hint}.safetensors")
            dst = os.path.join(tmp, f"{family_hint}.gguf")
            _save(src, state)
            result = export_image_gguf_component(src, dst, family_hint=family_hint, name=f"tiny-{family_hint}", file_type="f16")
            assert result.ok is True
            assert result.component == expected_component
            assert result.compatibility == "container_compatible"
            assert any("container-compatible only" in item for item in result.warnings), result

            reader = gguf.GGUFReader(dst)
            tensor_names = {str(tensor.name) for tensor in reader.tensors}
            assert "model.diffusion_model.input_blocks.0.0.weight" in tensor_names
            sidecar = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))
            assert sidecar["component"] == expected_component
            assert sidecar["compatibility"] == "container_compatible"
            assert any("container-compatible only" in item for item in sidecar["warnings"]), sidecar
    print("PASS: SD1.5/SDXL UNet image GGUF export writes container-only GGUF files")


def test_dit_container_export_or_missing_dependency() -> None:
    try:
        import gguf
    except ImportError:
        print("SKIP: DiT image GGUF export requires gguf")
        return

    cases = [
        ("anima", _tiny_anima_state(), "anima_dit", "net.x_embedder.proj.1.weight"),
        ("newbie", _tiny_newbie_state(), "newbie_dit", "x_embedder.weight"),
    ]
    for family_hint, state, expected_component, expected_tensor in cases:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            src = os.path.join(tmp, f"{family_hint}.safetensors")
            dst = os.path.join(tmp, f"{family_hint}.gguf")
            _save(src, state)
            result = export_image_gguf_component(src, dst, family_hint=family_hint, name=f"tiny-{family_hint}", file_type="f16")
            assert result.ok is True
            assert result.component == expected_component
            assert result.compatibility == "container_compatible"
            assert any("container-compatible only" in item for item in result.warnings), result

            reader = gguf.GGUFReader(dst)
            tensor_names = {str(tensor.name) for tensor in reader.tensors}
            assert expected_tensor in tensor_names
            sidecar = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))
            assert sidecar["component"] == expected_component
            assert sidecar["compatibility"] == "container_compatible"
            assert any("container-compatible only" in item for item in sidecar["warnings"]), sidecar
    print("PASS: Anima/Newbie DiT image GGUF export writes container-only GGUF files")


if __name__ == "__main__":
    test_vae_export_or_missing_dependency()
    test_t5_multi_shard_export_or_missing_dependency()
    test_unet_container_export_or_missing_dependency()
    test_dit_container_export_or_missing_dependency()
    print("\nAll image GGUF export smoke tests passed!")
