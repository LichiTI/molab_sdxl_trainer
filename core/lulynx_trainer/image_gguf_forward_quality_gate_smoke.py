"""Smoke tests for image GGUF forward quality gate readiness reports."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import json

import torch


BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.tools.image_gguf_exporter import export_image_gguf_component  # noqa: E402
from core.tools.image_gguf_forward_quality_gate import run_image_gguf_forward_quality_gate  # noqa: E402


def _save_vae(path: str | Path) -> None:
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
        metadata={"source": "image_gguf_forward_quality_gate_smoke"},
    )


def test_forward_quality_gate_reports_missing_dependency() -> None:
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("SKIP: image GGUF forward quality gate smoke requires gguf")
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "vae.safetensors")
        dst = os.path.join(tmp, "vae.gguf")
        _save_vae(src)
        result = export_image_gguf_component(src, dst, family_hint="vae", name="forward-gate-smoke", overwrite=True)
        with patch("core.tools.image_gguf_forward_quality_gate.importlib.util.find_spec", return_value=None):
            report = run_image_gguf_forward_quality_gate(src, result.output_path, sidecar_path=result.sidecar_path)
        assert report["ok"] is False, report
        assert report["status"] == "skipped_missing_dependency", report
        assert report["component"] == "vae", report
        assert report["family"] == "diffusers_vae", report
        assert report["required_dependencies"] == ["diffusers"], report
        assert report["missing_dependencies"] == ["diffusers"], report
        assert report["state_dict_loader_ok"] is True, report
        assert report["reads_tensor_payloads"] is True, report
        assert report["builds_model_modules"] is False, report
        assert report["runs_forward_pass"] is False, report
        assert report["runtime_loadable_enabled"] is False, report
        assert report["training_path_enabled"] is False, report
        print("PASS: image GGUF forward quality gate reports missing VAE dependency")


def test_forward_quality_gate_reports_ready_but_disabled() -> None:
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("SKIP: image GGUF forward quality gate smoke requires gguf")
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        src = root / "vae.safetensors"
        dst = root / "vae.gguf"
        (root / "config.json").write_text(
            json.dumps(
                {
                    "_class_name": "AutoencoderKL",
                    "in_channels": 3,
                    "latent_channels": 4,
                    "block_out_channels": [8],
                    "down_block_types": ["DownEncoderBlock2D"],
                    "up_block_types": ["UpDecoderBlock2D"],
                    "layers_per_block": 1,
                    "sample_size": 32,
                    "norm_num_groups": 4,
                    "act_fn": "silu",
                    "out_channels": 3,
                }
            ),
            encoding="utf-8",
        )
        _save_vae(src)
        result = export_image_gguf_component(src, dst, family_hint="vae", name="forward-gate-ready-smoke", overwrite=True)
        with patch("core.tools.image_gguf_forward_quality_gate.importlib.util.find_spec", return_value=object()):
            report = run_image_gguf_forward_quality_gate(src, result.output_path, sidecar_path=result.sidecar_path)
        assert report["ok"] is False, report
        assert report["status"] == "skipped_reference_forward_disabled", report
        assert report["missing_dependencies"] == [], report
        assert report["config_probe"]["has_model_config"] is True, report
        assert report["forward_plan"]["supported"] is True, report
        assert report["forward_plan"]["adapter"] == "diffusers_autoencoder_kl", report
        assert report["reference_forward_enabled"] is False, report
        assert report["builds_model_modules"] is False, report
        assert report["runs_forward_pass"] is False, report
        assert report["runtime_loadable_enabled"] is False, report
        print("PASS: image GGUF forward quality gate reports ready adapter with forward disabled")


def test_forward_quality_gate_blocks_trust_remote_code_by_default() -> None:
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("SKIP: image GGUF forward quality gate smoke requires gguf")
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        src = root / "vae.safetensors"
        dst = root / "vae.gguf"
        (root / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["JinaCLIPModel"],
                    "auto_map": {"AutoConfig": "configuration_clip.JinaCLIPConfig", "AutoModel": "modeling_clip.JinaCLIPModel"},
                    "model_type": "jina_clip",
                }
            ),
            encoding="utf-8",
        )
        _save_vae(src)
        result = export_image_gguf_component(src, dst, family_hint="vae", name="forward-gate-trust-smoke", overwrite=True)
        with patch("core.tools.image_gguf_forward_quality_gate.importlib.util.find_spec", return_value=object()):
            report = run_image_gguf_forward_quality_gate(
                src,
                result.output_path,
                sidecar_path=result.sidecar_path,
                component="clip",
                family="jina_clip_text",
            )
        assert report["ok"] is False, report
        assert report["status"] == "skipped_requires_trust_remote_code", report
        assert report["forward_plan"]["requires_trust_remote_code"] is True, report
        assert report["forward_plan"]["trust_remote_code_allowed"] is False, report
        assert report["builds_model_modules"] is False, report
        assert report["runs_forward_pass"] is False, report
        assert report["runtime_loadable_enabled"] is False, report
        print("PASS: image GGUF forward quality gate blocks trust_remote_code by default")


if __name__ == "__main__":
    test_forward_quality_gate_reports_missing_dependency()
    test_forward_quality_gate_reports_ready_but_disabled()
    test_forward_quality_gate_blocks_trust_remote_code_by_default()
    print("\nAll image GGUF forward quality gate smoke tests passed!")
