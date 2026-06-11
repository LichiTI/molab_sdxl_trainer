"""Smoke test for launcher toolbox image GGUF export runner wiring."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch


BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from lulynx_launcher.services.toolbox_runner import run_action  # noqa: E402


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
        metadata={"source": "image_gguf_toolbox_runner_smoke"},
    )


def test_toolbox_runner_image_gguf_export_or_missing_dependency() -> None:
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("SKIP: toolbox image GGUF export requires gguf")
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "vae.safetensors")
        dst = os.path.join(tmp, "vae.gguf")
        _save(src)
        plan = run_action("image_gguf_plan", {"input_paths": src, "family_hint": "vae", "file_type": "f16"})
        assert plan["ok"] is True, plan
        assert plan["component"] == "vae", plan
        assert plan["unique_tensor_count"] == 7, plan
        result = run_action(
            "image_gguf_export",
            {
                "input_paths": src,
                "output_path": dst,
                "family_hint": "vae",
                "file_type": "f16",
                "name": "toolbox-runner-smoke",
                "overwrite": False,
            },
        )
        assert result["ok"] is True, result
        assert result["component"] == "vae", result
        assert result["family"] == "diffusers_vae", result
        assert result["tensor_count"] == 7, result
        assert Path(result["output_path"]).is_file(), result
        assert Path(result["sidecar_path"]).is_file(), result
        validation = run_action("image_gguf_validate", {"path": result["output_path"]})
        assert validation["ok"] is True, validation
        assert validation["component"] == "vae", validation
        assert validation["tensor_count"] == result["tensor_count"], validation
        assert validation["container_contract"]["ok"] is True, validation
        assert validation["runtime_contract"]["runtime_loader"]["implemented"] is False, validation
        assert validation["runtime_loadable"] is False, validation
        assert any("runtime model loader is not implemented" in item for item in validation["runtime_blockers"]), validation
        print("PASS: toolbox runner exports image GGUF component container")


if __name__ == "__main__":
    test_toolbox_runner_image_gguf_export_or_missing_dependency()
    print("\nAll image GGUF toolbox runner smoke tests passed!")
