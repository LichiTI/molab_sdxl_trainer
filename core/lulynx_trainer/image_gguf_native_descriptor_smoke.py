"""Smoke for the native image GGUF descriptor reader.

The native reader is descriptor-only: it must match the Python reference loader
on metadata and shape descriptors without claiming runtime loadability.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.services.native_module_loader import ensure_lulynx_native_artifact_path  # noqa: E402
from core.tools.image_gguf_exporter import export_image_gguf_component  # noqa: E402
from core.tools.image_gguf_shape_loader import load_image_gguf_shape_contract  # noqa: E402


def _save(path: str | Path) -> None:
    from safetensors.torch import save_file

    save_file(
        {
            "encoder.conv_in.weight": torch.randn(8, 4, 3, 3),
            "encoder.conv_out.weight": torch.randn(4, 8, 3, 3),
            "encoder.down_blocks.0.resnets.0.conv1.weight": torch.randn(8, 8, 3, 3),
            "decoder.conv_in.weight": torch.randn(8, 4, 3, 3),
            "decoder.conv_out.weight": torch.randn(4, 8, 3, 3),
            "decoder.mid_block.attentions.0.to_q.weight": torch.randn(8, 8),
            "decoder.up_blocks.0.resnets.0.conv1.weight": torch.randn(8, 8, 3, 3),
        },
        str(path),
        metadata={"source": "image_gguf_native_descriptor_smoke"},
    )


def _save_state(path: str | Path, state: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    save_file(state, str(path), metadata={"source": "image_gguf_native_descriptor_smoke"})


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


def _bad_t5_state() -> dict[str, torch.Tensor]:
    state = _t5_state()
    state["encoder.block.0.layer.0.SelfAttention.k.weight"] = torch.randn(12, 16)
    return state


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


def _native_module() -> Any | None:
    ensure_lulynx_native_artifact_path()
    if importlib.util.find_spec("lulynx_native") is None:
        return None
    import lulynx_native  # type: ignore

    return lulynx_native


def _tensor(report: dict[str, Any], name: str) -> dict[str, Any]:
    for item in report["tensor_descriptors_sample"]:
        if item["name"] == name:
            return item
    raise AssertionError(f"tensor not found in native descriptor sample: {name}")


def test_native_descriptor_matches_python_reference_or_missing_artifact() -> None:
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("SKIP: image GGUF native descriptor smoke requires gguf exporter dependency")
        return

    lulynx_native = _native_module()
    if lulynx_native is None:
        print("SKIP: lulynx_native is not importable; build native artifact first")
        return

    if not hasattr(lulynx_native, "read_image_gguf_descriptor"):
        raise AssertionError("read_image_gguf_descriptor missing from lulynx_native")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "vae.safetensors")
        dst = os.path.join(tmp, "vae.gguf")
        _save(src)
        export_image_gguf_component(src, dst, family_hint="vae", name="native-descriptor-smoke", overwrite=True)
        reference = load_image_gguf_shape_contract(dst)
        native = lulynx_native.read_image_gguf_descriptor(dst, 32, 32)
        assert native["ok"] is True, native
        assert native["provider"] == "native_image_gguf_descriptor_v1", native
        assert native["gguf_arch"] == reference["gguf_arch"] == "lulynx_image", native
        assert native["component"] == reference["component"] == "vae", native
        assert native["family"] == reference["family"] == "diffusers_vae", native
        assert native["compatibility"] == reference["compatibility"] == "container_compatible", native
        assert native["tensor_count"] == reference["tensor_count"] == 7, native
        assert native["sidecar_present"] is True, native
        assert native["probe_manifest_source"] == reference["probe_manifest_source"] == "sidecar", native
        assert native["probe_manifest_count"] == reference["probe_manifest_count"] == 1, native
        assert native["container_contract"]["ok"] is True, native
        assert native["container_contract"]["required_tensor_count"] == reference["container_contract"]["required_tensor_count"], native
        assert native["container_contract"]["required_prefix_count"] == reference["container_contract"]["required_prefix_count"], native
        assert native["container_contract"]["missing_required_tensors"] == [], native
        assert native["container_contract"]["missing_required_prefixes"] == [], native
        assert native["reads_tensor_payloads"] is False, native
        assert native["runtime_loadable"] is False, native
        assert native["training_path_enabled"] is False, native
        assert native["runtime_contract"]["loadability"] == reference["runtime_contract"]["loadability"], native
        assert native["runtime_contract"]["runtime_loader"]["implemented"] is False, native
        assert native["runtime_contract"]["runtime_loader_abi"] == reference["runtime_contract"]["runtime_loader_abi"], native
        assert native["runtime_contract"]["runtime_loader_abi"]["report_only"] is True, native
        assert native["runtime_contract"]["runtime_loader_abi"]["reads_tensor_payloads"] is False, native
        assert native["runtime_contract"]["tensor_type_policy"]["ok"] is True, native
        assert "vae_module_builder" in native["runtime_contract"]["required_runtime_features"], native
        assert any("runtime model loader is not implemented" in item for item in native["runtime_blockers"]), native
        native_tensor = _tensor(native, "encoder.conv_in.weight")
        reference_tensor = _tensor(reference, "encoder.conv_in.weight")
        assert native_tensor["storage_shape"] == reference_tensor["storage_shape"] == [3, 3, 4, 8], native_tensor
        assert native_tensor["logical_shape"] == reference_tensor["logical_shape"] == [8, 4, 3, 3], native_tensor
        assert native_tensor["rank"] == reference_tensor["rank"] == 4, native_tensor
        assert native_tensor["tensor_type"] == reference_tensor["tensor_type"] == "f16", native_tensor
        print("PASS: native image GGUF descriptor matches Python reference shape loader")


def test_native_descriptor_reads_embedded_manifest_without_sidecar() -> None:
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("SKIP: image GGUF native descriptor smoke requires gguf exporter dependency")
        return

    lulynx_native = _native_module()
    if lulynx_native is None:
        print("SKIP: lulynx_native is not importable; build native artifact first")
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "vae.safetensors")
        dst = os.path.join(tmp, "vae.gguf")
        _save(src)
        result = export_image_gguf_component(src, dst, family_hint="vae", name="native-descriptor-no-sidecar", overwrite=True)
        Path(result.sidecar_path).unlink()
        reference = load_image_gguf_shape_contract(dst)
        native = lulynx_native.read_image_gguf_descriptor(dst, 32, 32)
        assert native["ok"] is True, native
        assert native["sidecar_present"] is False, native
        assert native["probe_manifest_source"] == reference["probe_manifest_source"] == "gguf_metadata", native
        assert native["container_contract"]["ok"] is True, native
        assert native["container_contract"]["manifest_components"] == reference["container_contract"]["manifest_components"], native
        assert native["container_contract"]["manifest_families"] == reference["container_contract"]["manifest_families"], native
        print("PASS: native image GGUF descriptor reads embedded probe manifest without sidecar")


def test_native_shape_contract_derived_matches_python_reference() -> None:
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("SKIP: image GGUF native descriptor smoke requires gguf exporter dependency")
        return

    lulynx_native = _native_module()
    if lulynx_native is None:
        print("SKIP: lulynx_native is not importable; build native artifact first")
        return

    cases = [
        ("vae", None, {"latent_channels": 4, "input_channels": 4}),
        ("clip", _clip_state(), {"hidden_dim": 16, "max_positions": 32}),
        ("t5", _t5_state(), {"hidden_dim": 16, "attention_inner_dim": 16}),
        ("sdxl", _sdxl_unet_state(), {"latent_channels": 4, "output_channels": 4, "time_embed_dim": 16}),
        ("anima", _anima_state(), {"hidden_dim": 16}),
        ("newbie", _newbie_state(), {"hidden_dim": 16, "latent_channels": 4, "patch_size": [2, 2]}),
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for family_hint, state, expected in cases:
            src = os.path.join(tmp, f"{family_hint}.safetensors")
            dst = os.path.join(tmp, f"{family_hint}.gguf")
            if state is None:
                _save(src)
            else:
                _save_state(src, state)
            export_image_gguf_component(src, dst, family_hint=family_hint, name=f"native-derived-{family_hint}", overwrite=True)
            reference = load_image_gguf_shape_contract(dst)
            native = lulynx_native.read_image_gguf_descriptor(dst, 128, 64)
            assert native["ok"] is True, native
            assert native["shape_contract"]["ok"] is True, native
            assert native["shape_contract"]["component"] == reference["shape_contract"]["component"], native
            assert native["runtime_contract"]["runtime_loader"]["implemented"] is False, native
            assert native["runtime_contract"]["runtime_loader_abi"] == reference["runtime_contract"]["runtime_loader_abi"], native
            assert native["runtime_contract"]["tensor_type_policy"]["observed"] == reference["runtime_contract"]["tensor_type_policy"]["observed"], native
            for key, value in expected.items():
                assert native["shape_contract"]["derived"].get(key) == value, native
                assert native["shape_contract"]["derived"].get(key) == reference["shape_contract"]["derived"].get(key), native
    print("PASS: native image GGUF shape_contract derived fields match Python reference")


def test_native_shape_contract_reports_shape_failures() -> None:
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("SKIP: image GGUF native descriptor smoke requires gguf exporter dependency")
        return

    lulynx_native = _native_module()
    if lulynx_native is None:
        print("SKIP: lulynx_native is not importable; build native artifact first")
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "bad_t5.safetensors")
        dst = os.path.join(tmp, "bad_t5.gguf")
        _save_state(src, _bad_t5_state())
        export_image_gguf_component(src, dst, family_hint="t5", name="native-bad-t5", overwrite=True)
        reference = load_image_gguf_shape_contract(dst)
        native = lulynx_native.read_image_gguf_descriptor(dst, 128, 64)
        assert reference["ok"] is False, reference
        assert native["ok"] is False, native
        assert reference["shape_contract"]["ok"] is False, reference
        assert native["shape_contract"]["ok"] is False, native
        assert native["runtime_contract"]["shape_contract_ok"] is False, native
        native_issues = "\n".join(native["shape_contract"]["issues"])
        reference_issues = "\n".join(reference["shape_contract"]["issues"])
        assert "t5.self_attention.qkv failed shape rule: q/k/v output dims match" in native_issues, native
        assert "t5.self_attention.qkv failed shape rule: q/k/v output dims match" in reference_issues, reference
        assert any("shape-only reference contract is incomplete" in item for item in native["runtime_blockers"]), native
        assert any("shape-only reference contract is incomplete" in item for item in reference["runtime_blockers"]), reference
    print("PASS: native image GGUF shape_contract reports shape failures")


if __name__ == "__main__":
    test_native_descriptor_matches_python_reference_or_missing_artifact()
    test_native_descriptor_reads_embedded_manifest_without_sidecar()
    test_native_shape_contract_derived_matches_python_reference()
    test_native_shape_contract_reports_shape_failures()
    print("\nAll image GGUF native descriptor smoke tests passed!")
