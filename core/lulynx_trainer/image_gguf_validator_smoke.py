"""Smoke tests for image GGUF container validation."""

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
from core.tools.image_gguf_validator import validate_image_gguf_container  # noqa: E402


def _save(path: str | Path) -> None:
    from safetensors.torch import save_file

    save_file(
        {
            "encoder.conv_in.weight": torch.randn(16, 4, 3, 3),
            "encoder.conv_out.weight": torch.randn(8, 16, 3, 3),
            "encoder.down_blocks.0.resnets.0.conv1.weight": torch.randn(16, 16, 3, 3),
            "decoder.conv_in.weight": torch.randn(16, 4, 3, 3),
            "decoder.conv_out.weight": torch.randn(4, 16, 3, 3),
            "decoder.mid_block.attentions.0.to_q.weight": torch.randn(16, 16),
            "decoder.up_blocks.0.resnets.0.conv1.weight": torch.randn(16, 16, 3, 3),
        },
        str(path),
        metadata={"source": "image_gguf_validator_smoke"},
    )


def _save_dit(path: str | Path) -> None:
    from safetensors.torch import save_file

    save_file(
        {
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
        },
        str(path),
        metadata={"source": "image_gguf_validator_smoke"},
    )


def _save_unet(path: str | Path) -> None:
    from safetensors.torch import save_file

    save_file(
        {
            "model.diffusion_model.input_blocks.0.0.weight": torch.randn(16, 4, 3, 3),
            "model.diffusion_model.time_embed.0.weight": torch.randn(16, 16),
            "model.diffusion_model.middle_block.0.in_layers.0.weight": torch.randn(16),
            "model.diffusion_model.output_blocks.0.0.in_layers.0.weight": torch.randn(16),
            "model.diffusion_model.out.2.weight": torch.randn(4, 16, 3, 3),
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": torch.randn(32, 16),
        },
        str(path),
        metadata={"source": "image_gguf_validator_smoke"},
    )


def test_validate_exported_image_gguf_or_missing_dependency() -> None:
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("SKIP: image GGUF validator requires gguf")
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "vae.safetensors")
        dst = os.path.join(tmp, "vae.gguf")
        _save(src)
        export_image_gguf_component(src, dst, family_hint="vae", name="validator-smoke")
        report = validate_image_gguf_container(dst)
        assert report["ok"] is True, report
        assert report["gguf_arch"] == "lulynx_image", report
        assert report["component"] == "vae", report
        assert report["family"] == "diffusers_vae", report
        assert report["compatibility"] == "container_compatible", report
        assert report["tensor_count"] == 7, report
        assert report["sidecar_present"] is True, report
        assert report["probe_manifest_source"] == "sidecar", report
        assert report["probe_manifest_count"] == 1, report
        assert report["container_contract"]["ok"] is True, report
        assert report["runtime_contract"]["runtime_loader"]["implemented"] is False, report
        assert report["runtime_loadable"] is False, report
        assert any("runtime model loader is not implemented" in item for item in report["runtime_blockers"]), report
        print("PASS: image GGUF validator accepts exported VAE container")


def test_validate_missing_sidecar_still_reports_metadata() -> None:
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("SKIP: image GGUF validator requires gguf")
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "vae.safetensors")
        dst = os.path.join(tmp, "vae.gguf")
        _save(src)
        result = export_image_gguf_component(src, dst, family_hint="vae", name="validator-smoke")
        Path(result.sidecar_path).unlink()
        report = validate_image_gguf_container(dst)
        assert report["ok"] is True, report
        assert report["sidecar_present"] is False, report
        assert report["component"] == "vae", report
        assert report["probe_manifest_source"] == "gguf_metadata", report
        assert report["container_contract"]["ok"] is True, report
        print("PASS: image GGUF validator reads metadata without sidecar")


def test_validate_container_only_dit_and_unet_contracts() -> None:
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("SKIP: image GGUF validator requires gguf")
        return

    cases = [
        ("anima", _save_dit, "anima_dit"),
        ("sdxl", _save_unet, "sdxl_unet"),
    ]
    for family_hint, writer, expected_component in cases:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            src = os.path.join(tmp, f"{family_hint}.safetensors")
            dst = os.path.join(tmp, f"{family_hint}.gguf")
            writer(src)
            export_image_gguf_component(src, dst, family_hint=family_hint, name=f"validator-{family_hint}")
            report = validate_image_gguf_container(dst)
            assert report["ok"] is True, report
            assert report["component"] == expected_component, report
            assert report["container_contract"]["ok"] is True, report
            assert report["runtime_contract"]["runtime_loader"]["implemented"] is False, report
            assert report["runtime_loadable"] is False, report
            assert any(expected_component in item for item in report["runtime_blockers"]), report
    print("PASS: image GGUF validator reports container-only DiT/UNet runtime blockers")


if __name__ == "__main__":
    test_validate_exported_image_gguf_or_missing_dependency()
    test_validate_missing_sidecar_still_reports_metadata()
    test_validate_container_only_dit_and_unet_contracts()
    print("\nAll image GGUF validator smoke tests passed!")
