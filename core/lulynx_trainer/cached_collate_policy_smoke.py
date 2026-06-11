"""Smoke checks for cached dataset collate policy options."""

from pathlib import Path
import sys

try:
    from .anima_cached_dataset import anima_cached_collate
    from .newbie_cached_dataset import newbie_cached_collate
except ImportError:  # pragma: no cover - direct script usage
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from backend.core.lulynx_trainer.anima_cached_dataset import anima_cached_collate
    from backend.core.lulynx_trainer.newbie_cached_dataset import newbie_cached_collate

import torch


def _assert_same_tensor(left: torch.Tensor, right: torch.Tensor, name: str) -> None:
    assert left.shape == right.shape, f"{name} shape mismatch: {left.shape} != {right.shape}"
    assert left.dtype == right.dtype, f"{name} dtype mismatch: {left.dtype} != {right.dtype}"
    if left.dtype.is_floating_point:
        assert torch.allclose(left, right), f"{name} value mismatch"
    else:
        assert torch.equal(left, right), f"{name} value mismatch"


def test_newbie_collate_modes() -> None:
    batch = [
        {
            "latents": torch.full((4, 8, 8), float(idx)),
            "encoder_hidden_states": torch.arange((idx + 2) * 3, dtype=torch.float32).view(idx + 2, 3),
            "attention_mask": torch.ones((idx + 2,), dtype=torch.bool),
            "pooled_prompt_embeds": torch.full((6,), float(idx)),
            "loss_mask": torch.ones((1, 8, 8), dtype=torch.float32) * float(idx + 1),
            "captions": f"caption {idx}",
            "sample_id": f"sample-{idx}",
        }
        for idx in range(3)
    ]
    legacy = newbie_cached_collate(batch, collate_mode="legacy")
    fast = newbie_cached_collate(batch, collate_mode="pad_sequence")
    for key in ("latents", "encoder_hidden_states", "attention_mask", "pooled_prompt_embeds", "loss_masks"):
        _assert_same_tensor(legacy[key], fast[key], f"newbie:{key}")  # type: ignore[arg-type]
    assert legacy["captions"] == fast["captions"]
    assert legacy["sample_ids"] == fast["sample_ids"]


def test_anima_collate_modes() -> None:
    batch = []
    for idx in range(3):
        token_count = idx + 2
        qwen_count = idx + 3
        batch.append(
            {
                "latents": torch.full((4, 8, 8), float(idx)),
                "encoder_hidden_states": torch.arange(token_count * 5, dtype=torch.float32).view(token_count, 5),
                "attention_mask": torch.ones((token_count,), dtype=torch.bool),
                "t5_input_ids": torch.arange(token_count, dtype=torch.long),
                "t5_attention_mask": torch.ones((token_count,), dtype=torch.bool),
                "qwen3_hidden_states": torch.arange(qwen_count * 4, dtype=torch.float32).view(qwen_count, 4),
                "qwen3_attention_mask": torch.ones((qwen_count,), dtype=torch.bool),
                "caption_weight": 1.0 + idx,
                "loss_mask": torch.ones((8, 8), dtype=torch.float32) * float(idx + 1),
                "captions": f"caption {idx}",
                "sample_id": f"sample-{idx}",
            }
        )
    legacy = anima_cached_collate(batch, fixed_text_tokens=8, collate_mode="legacy")
    fast = anima_cached_collate(batch, fixed_text_tokens=8, collate_mode="pad_sequence")
    for key in (
        "latents",
        "encoder_hidden_states",
        "attention_mask",
        "caption_weights",
        "t5_input_ids",
        "t5_attention_mask",
        "loss_masks",
        "qwen3_hidden_states",
        "qwen3_attention_mask",
    ):
        _assert_same_tensor(legacy[key], fast[key], f"anima:{key}")  # type: ignore[arg-type]
    assert legacy["padding_mask"] is None and fast["padding_mask"] is None
    assert legacy["captions"] == fast["captions"]
    assert legacy["sample_ids"] == fast["sample_ids"]


def main() -> None:
    test_newbie_collate_modes()
    test_anima_collate_modes()
    print("cached_collate_policy_smoke: ok")


if __name__ == "__main__":
    main()
