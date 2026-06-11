"""Smoke tests for sampled image GGUF payload parity."""

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
from core.tools.image_gguf_payload_parity import check_image_gguf_payload_parity  # noqa: E402


def _save(path: str | Path) -> None:
    from safetensors.torch import save_file

    save_file(
        {
            "encoder.conv_in.weight": torch.linspace(-1.0, 1.0, steps=8 * 4 * 3 * 3, dtype=torch.float32).reshape(8, 4, 3, 3),
            "encoder.conv_out.weight": torch.randn(4, 8, 3, 3),
            "encoder.down_blocks.0.resnets.0.conv1.weight": torch.randn(8, 8, 3, 3),
            "decoder.conv_in.weight": torch.randn(8, 4, 3, 3),
            "decoder.conv_out.weight": torch.randn(4, 8, 3, 3),
            "decoder.mid_block.attentions.0.to_q.weight": torch.randn(8, 8),
            "decoder.up_blocks.0.resnets.0.conv1.weight": torch.randn(8, 8, 3, 3),
        },
        str(path),
        metadata={"source": "image_gguf_payload_parity_smoke"},
    )


def test_payload_parity_or_missing_dependency() -> None:
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("SKIP: image GGUF payload parity requires gguf")
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "vae.safetensors")
        dst = os.path.join(tmp, "vae.gguf")
        _save(src)
        result = export_image_gguf_component(src, dst, family_hint="vae", name="payload-parity-smoke", overwrite=True)
        report = check_image_gguf_payload_parity(src, result.output_path, sidecar_path=result.sidecar_path, max_tensors=4)
        assert report["ok"] is True, report
        assert report["reads_tensor_payloads"] is True, report
        assert report["runs_forward_pass"] is False, report
        assert report["runtime_loadable_enabled"] is False, report
        assert report["sampled_tensor_count"] == 4, report
        assert report["failed_tensor_count"] == 0, report
        assert report["max_abs_error"] <= 1e-3, report
        assert all(item["source_shape"] == item["gguf_shape"] for item in report["records"]), report
        print("PASS: image GGUF payload parity validates sampled tensors")


if __name__ == "__main__":
    test_payload_parity_or_missing_dependency()
    print("\nAll image GGUF payload parity smoke tests passed!")
