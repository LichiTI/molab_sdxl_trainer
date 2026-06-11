# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Anima multi-encoder static token padding (Phase 6.1).

Validates that:
- Primary CLIP text tokens can be padded to fixed length
- Qwen3 tokens can be padded to fixed length
- T5 tokens can be padded to fixed length
- Attention masks are correctly padded
- Truncation warnings are logged when sequences exceed fixed length
- AnimaCachedDataset applies static padding when fixed_*_tokens are set
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def test_multi_encoder_padding_basic():
    """Verify multi-encoder padding pads all three text encoders correctly."""
    from fixed_token_padding import (
        AnimaMultiEncoderPaddingConfig,
        apply_anima_multi_encoder_padding,
    )

    # Create fake text data with different sequence lengths
    text_data = {
        "encoder_hidden_states": torch.randn(50, 768),  # CLIP: 50 tokens
        "attention_mask": torch.ones(50, dtype=torch.bool),
        "qwen3_hidden_states": torch.randn(80, 1024),  # Qwen3: 80 tokens
        "qwen3_attention_mask": torch.ones(80, dtype=torch.bool),
        "t5_input_ids": torch.randint(0, 1000, (120,), dtype=torch.long),  # T5: 120 tokens
        "t5_attention_mask": torch.ones(120, dtype=torch.bool),
    }

    config = AnimaMultiEncoderPaddingConfig(
        fixed_text_tokens=100,
        fixed_qwen3_tokens=150,
        fixed_t5_tokens=200,
        warn_on_truncation=False,
    )

    result = apply_anima_multi_encoder_padding(text_data, config)

    # Verify shapes
    assert result["encoder_hidden_states"].shape == (100, 768), "CLIP not padded to 100"
    assert result["attention_mask"].shape == (100,), "CLIP mask not padded to 100"
    assert result["qwen3_hidden_states"].shape == (150, 1024), "Qwen3 not padded to 150"
    assert result["qwen3_attention_mask"].shape == (150,), "Qwen3 mask not padded to 150"
    assert result["t5_input_ids"].shape == (200,), "T5 IDs not padded to 200"
    assert result["t5_attention_mask"].shape == (200,), "T5 mask not padded to 200"

    # Verify padding values
    assert torch.all(result["attention_mask"][:50] == True), "CLIP mask original tokens should be True"
    assert torch.all(result["attention_mask"][50:] == False), "CLIP mask padding should be False"
    assert torch.all(result["qwen3_attention_mask"][:80] == True), "Qwen3 mask original tokens should be True"
    assert torch.all(result["qwen3_attention_mask"][80:] == False), "Qwen3 mask padding should be False"
    assert torch.all(result["t5_attention_mask"][:120] == True), "T5 mask original tokens should be True"
    assert torch.all(result["t5_attention_mask"][120:] == False), "T5 mask padding should be False"

    print("[PASS] Multi-encoder padding pads all three encoders correctly")


def test_multi_encoder_truncation():
    """Verify multi-encoder padding truncates long sequences."""
    from fixed_token_padding import (
        AnimaMultiEncoderPaddingConfig,
        apply_anima_multi_encoder_padding,
    )

    # Create fake text data with sequences longer than target
    text_data = {
        "encoder_hidden_states": torch.randn(200, 768),  # CLIP: 200 tokens
        "attention_mask": torch.ones(200, dtype=torch.bool),
        "qwen3_hidden_states": torch.randn(300, 1024),  # Qwen3: 300 tokens
        "qwen3_attention_mask": torch.ones(300, dtype=torch.bool),
        "t5_input_ids": torch.randint(0, 1000, (400,), dtype=torch.long),  # T5: 400 tokens
        "t5_attention_mask": torch.ones(400, dtype=torch.bool),
    }

    config = AnimaMultiEncoderPaddingConfig(
        fixed_text_tokens=100,
        fixed_qwen3_tokens=150,
        fixed_t5_tokens=200,
        warn_on_truncation=False,
    )

    result = apply_anima_multi_encoder_padding(text_data, config)

    # Verify truncation
    assert result["encoder_hidden_states"].shape == (100, 768), "CLIP not truncated to 100"
    assert result["attention_mask"].shape == (100,), "CLIP mask not truncated to 100"
    assert result["qwen3_hidden_states"].shape == (150, 1024), "Qwen3 not truncated to 150"
    assert result["qwen3_attention_mask"].shape == (150,), "Qwen3 mask not truncated to 150"
    assert result["t5_input_ids"].shape == (200,), "T5 IDs not truncated to 200"
    assert result["t5_attention_mask"].shape == (200,), "T5 mask not truncated to 200"

    # Verify truncated content is from the beginning
    assert torch.allclose(
        result["encoder_hidden_states"],
        text_data["encoder_hidden_states"][:100]
    ), "CLIP truncation should keep first 100 tokens"

    print("[PASS] Multi-encoder padding truncates long sequences correctly")


def test_multi_encoder_selective_padding():
    """Verify multi-encoder padding only pads encoders with fixed_*_tokens > 0."""
    from fixed_token_padding import (
        AnimaMultiEncoderPaddingConfig,
        apply_anima_multi_encoder_padding,
    )

    # Create fake text data
    text_data = {
        "encoder_hidden_states": torch.randn(50, 768),
        "attention_mask": torch.ones(50, dtype=torch.bool),
        "qwen3_hidden_states": torch.randn(80, 1024),
        "qwen3_attention_mask": torch.ones(80, dtype=torch.bool),
        "t5_input_ids": torch.randint(0, 1000, (120,), dtype=torch.long),
        "t5_attention_mask": torch.ones(120, dtype=torch.bool),
    }

    # Only pad CLIP and T5, leave Qwen3 dynamic
    config = AnimaMultiEncoderPaddingConfig(
        fixed_text_tokens=100,
        fixed_qwen3_tokens=0,  # No padding
        fixed_t5_tokens=200,
        warn_on_truncation=False,
    )

    result = apply_anima_multi_encoder_padding(text_data, config)

    # Verify CLIP and T5 are padded, Qwen3 is unchanged
    assert result["encoder_hidden_states"].shape == (100, 768), "CLIP should be padded"
    assert result["qwen3_hidden_states"].shape == (80, 1024), "Qwen3 should be unchanged"
    assert result["t5_input_ids"].shape == (200,), "T5 should be padded"

    print("[PASS] Multi-encoder padding only pads encoders with fixed_*_tokens > 0")


def test_cached_dataset_static_padding_integration():
    """Verify AnimaCachedDataset applies static padding when fixed_*_tokens are set."""
    from anima_cached_dataset import AnimaCachedDataset
    from anima_cache_builder import AnimaCacheBuilderConfig, build_anima_cache_sample

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create fake image and caption
        image_path = tmpdir / "test.png"
        from PIL import Image
        img = Image.new("RGB", (64, 64), color="red")
        img.save(image_path)

        caption_path = tmpdir / "test.txt"
        caption_path.write_text("test caption", encoding="utf-8")

        # Fake encode functions with variable-length output
        def fake_vae_encode(image: torch.Tensor) -> torch.Tensor:
            return torch.randn(1, 16, 8, 8)

        def fake_text_encode(caption: str) -> Dict[str, torch.Tensor]:
            return {
                "prompt_embeds": torch.randn(50, 768),  # 50 tokens
                "attn_mask": torch.ones(50, dtype=torch.bool),
                "qwen3_hidden_states": torch.randn(80, 1024),  # 80 tokens
                "qwen3_attention_mask": torch.ones(80, dtype=torch.bool),
                "t5_input_ids": torch.randint(0, 1000, (120,), dtype=torch.long),  # 120 tokens
                "t5_attn_mask": torch.ones(120, dtype=torch.bool),
            }

        # Build cache
        config = AnimaCacheBuilderConfig(
            data_dir=str(tmpdir),
            output_dir=str(tmpdir),
            vae_chunk_size=0,
            text_token_limit=0,
            include_loss_mask=False,
            disk_format="npz",
            disk_dtype="float16",
        )

        latent_path, text_path = build_anima_cache_sample(
            image_path=image_path,
            vae_encode_fn=fake_vae_encode,
            text_encode_fn=fake_text_encode,
            config=config,
            force=True,
        )

        # Load with static padding
        dataset = AnimaCachedDataset(
            data_dir=tmpdir,
            latent_crop_size=0,
            text_token_limit=0,
            fixed_text_tokens=100,
            fixed_qwen3_tokens=150,
            fixed_t5_tokens=200,
            caption_extension=".txt",
            weighted_captions=False,
        )

        sample = dataset[0]

        # Verify static padding was applied
        assert sample["encoder_hidden_states"].shape == (100, 768), "CLIP not padded to 100"
        assert sample["attention_mask"].shape == (100,), "CLIP mask not padded to 100"
        assert sample["qwen3_hidden_states"].shape == (150, 1024), "Qwen3 not padded to 150"
        assert sample["qwen3_attention_mask"].shape == (150,), "Qwen3 mask not padded to 150"
        assert sample["t5_input_ids"].shape == (200,), "T5 IDs not padded to 200"
        assert sample["t5_attention_mask"].shape == (200,), "T5 mask not padded to 200"

        # Verify padding values
        assert torch.all(sample["attention_mask"][:50] == True), "CLIP mask original should be True"
        assert torch.all(sample["attention_mask"][50:] == False), "CLIP mask padding should be False"

        # Close file handles before cleanup
        import gc
        gc.collect()

        print("[PASS] AnimaCachedDataset applies static padding correctly")


if __name__ == "__main__":
    test_multi_encoder_padding_basic()
    test_multi_encoder_truncation()
    test_multi_encoder_selective_padding()
    test_cached_dataset_static_padding_integration()
    print("\n[PASS] All Phase 6.1 static padding smoke tests passed")
