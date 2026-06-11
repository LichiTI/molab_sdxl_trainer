from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from lora_inspector import analyze_lora, block_analyze
from lora_surgery import LoRASurgeon
from diagnostic_card import check_dependencies, create_diagnostic_card_base64
from model_converter import convert_tensor_file
from model_merger import ModelMerger
from qpissa_converter import convert_qpissa


def _make_tiny_lora(path: Path) -> None:
    save_file(
        {
            "lora_unet_down_blocks_0_attn1_to_q.lora_down.weight": torch.randn(2, 4),
            "lora_unet_down_blocks_0_attn1_to_q.lora_up.weight": torch.randn(4, 2),
            "lora_unet_down_blocks_0_attn1_to_q.alpha": torch.tensor(2.0),
            "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": torch.randn(2, 4),
            "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_up.weight": torch.randn(4, 2),
        },
        str(path),
    )


def _make_tiny_model(path: Path, scale: float = 1.0) -> None:
    save_file(
        {
            "transformer.blocks.0.attn.to_q.weight": torch.randn(4, 4) * scale,
            "transformer.blocks.0.mlp.fc1.weight": torch.randn(6, 4) * scale,
            "norm.weight": torch.ones(4) * scale,
        },
        str(path),
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "tiny.safetensors"
        dst = Path(tmp) / "tiny_pruned.safetensors"
        _make_tiny_lora(src)

        report = analyze_lora(str(src))
        assert report["num_layers"] >= 4
        assert report["total_params"] > 0
        assert report["position_analysis"]

        blocks = block_analyze(str(src))["blocks"]
        assert any(item["id"].startswith("IN") for item in blocks)

        pruned = LoRASurgeon(device="cpu").prune_lora(str(src), str(dst), drop_blocks=["TE1"])
        assert pruned["dropped_tensors"] > 0
        assert dst.exists()

        deps = check_dependencies()
        if deps.get("pillow"):
            card = create_diagnostic_card_base64("tiny", 88.0, metrics={"stable_rank": 2})
            assert card

        model_a = Path(tmp) / "model_a.safetensors"
        model_b = Path(tmp) / "model_b.safetensors"
        merged = Path(tmp) / "merged.safetensors"
        converted_pt = Path(tmp) / "model_a.pt"
        qpissa_dir = Path(tmp) / "qpissa"
        _make_tiny_model(model_a, 1.0)
        _make_tiny_model(model_b, 0.5)

        merge_ok = ModelMerger(device="cpu").weighted_sum(str(model_a), str(model_b), str(merged), alpha=0.25)
        assert merge_ok and merged.exists()

        converted = convert_tensor_file(str(model_a), str(converted_pt), "pt")
        assert converted["tensor_count"] == 3 and converted_pt.exists()

        qpissa = convert_qpissa(str(model_a), str(qpissa_dir), rank=2, device="cpu")
        assert qpissa["converted_layers"] == 2
        assert Path(qpissa["residual_path"]).exists()
        assert Path(qpissa["adapter_path"]).exists()

    print("c_tier_toolbox_smoke: ok")


if __name__ == "__main__":
    main()
