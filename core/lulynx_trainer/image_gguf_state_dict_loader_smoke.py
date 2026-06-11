"""Smoke tests for image GGUF state_dict reference loading."""

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
from core.tools.image_gguf_state_dict_loader import (  # noqa: E402
    load_image_gguf_state_dict,
    load_image_gguf_state_dict_with_report,
    summarize_image_gguf_state_dict_load,
)


def _state() -> dict[str, torch.Tensor]:
    return {
        "encoder.conv_in.weight": torch.linspace(-1.0, 1.0, steps=8 * 4 * 3 * 3, dtype=torch.float32).reshape(8, 4, 3, 3),
        "encoder.conv_out.weight": torch.randn(4, 8, 3, 3),
        "encoder.down_blocks.0.resnets.0.conv1.weight": torch.randn(8, 8, 3, 3),
        "decoder.conv_in.weight": torch.randn(8, 4, 3, 3),
        "decoder.conv_out.weight": torch.randn(4, 8, 3, 3),
        "decoder.mid_block.attentions.0.to_q.weight": torch.randn(8, 8),
        "decoder.up_blocks.0.resnets.0.conv1.weight": torch.randn(8, 8, 3, 3),
    }


def _save(path: str | Path, state: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    save_file(state, str(path), metadata={"source": "image_gguf_state_dict_loader_smoke"})


def test_state_dict_loader_or_missing_dependency() -> None:
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("SKIP: image GGUF state_dict loader requires gguf")
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        source_state = _state()
        src = os.path.join(tmp, "vae.safetensors")
        dst = os.path.join(tmp, "vae.gguf")
        _save(src, source_state)
        result = export_image_gguf_component(src, dst, family_hint="vae", name="state-dict-loader-smoke", overwrite=True)
        state_dict = load_image_gguf_state_dict(result.output_path, sidecar_path=result.sidecar_path, max_tensors=0)
        assert set(state_dict) == set(source_state), state_dict.keys()
        assert state_dict["encoder.conv_in.weight"].shape == source_state["encoder.conv_in.weight"].shape
        assert state_dict["encoder.conv_in.weight"].dtype == torch.float16
        assert torch.equal(state_dict["encoder.conv_in.weight"], source_state["encoder.conv_in.weight"].to(torch.float16))

        report = load_image_gguf_state_dict_with_report(result.output_path, sidecar_path=result.sidecar_path, max_tensors=3)
        assert report["ok"] is True, report
        assert report["reads_tensor_payloads"] is True, report
        assert report["builds_model_modules"] is False, report
        assert report["runtime_loadable_enabled"] is False, report
        assert report["state_dict_tensor_count"] == 3, report
        assert report["truncated"] is True, report
        assert report["dtype_counts"] == {"float16": 3}, report

        summary = summarize_image_gguf_state_dict_load(result.output_path, sidecar_path=result.sidecar_path, max_tensors=2)
        assert "state_dict" not in summary, summary
        assert summary["state_dict_tensor_count"] == 2, summary
        print("PASS: image GGUF state_dict reference loader restores sampled tensors")


if __name__ == "__main__":
    test_state_dict_loader_or_missing_dependency()
    print("\nAll image GGUF state_dict loader smoke tests passed!")
