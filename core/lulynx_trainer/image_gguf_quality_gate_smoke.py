"""Smoke tests for lightweight image GGUF quality gates."""

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
from core.tools.image_gguf_quality_gate import run_image_gguf_lightweight_quality_gate  # noqa: E402


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
        metadata={"source": "image_gguf_quality_gate_smoke"},
    )


def test_lightweight_quality_gate_or_missing_dependency() -> None:
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("SKIP: image GGUF quality gate requires gguf")
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "vae.safetensors")
        dst = os.path.join(tmp, "vae.gguf")
        _save_vae(src)
        result = export_image_gguf_component(src, dst, family_hint="vae", name="quality-gate-smoke", overwrite=True)
        report = run_image_gguf_lightweight_quality_gate(src, result.output_path, sidecar_path=result.sidecar_path)
        assert report["ok"] is True, report
        assert report["component"] == "vae", report
        assert report["family"] == "diffusers_vae", report
        assert report["quality_stage"] == "lightweight_structural_gate", report
        assert report["reads_tensor_payloads"] is True, report
        assert report["builds_model_modules"] is False, report
        assert report["runs_forward_pass"] is False, report
        assert report["runtime_loadable_enabled"] is False, report
        assert report["payload_parity_ok"] is True, report
        assert report["failed_check_count"] == 0, report
        assert any(item["name"] == "required_key:encoder.conv_in.weight" for item in report["checks"]), report
        print("PASS: image GGUF lightweight quality gate validates VAE state_dict")


if __name__ == "__main__":
    test_lightweight_quality_gate_or_missing_dependency()
    print("\nAll image GGUF quality gate smoke tests passed!")
