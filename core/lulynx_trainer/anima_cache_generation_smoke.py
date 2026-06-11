# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Anima cache generation with Qwen3/T5/LLM adapter support.

Validates that:
- Cache builder accepts and writes Qwen3/T5 fields
- Token limits are applied correctly
- Online cache dataset generates cache on-demand
- Cache metadata includes schema version and encoder identity
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def test_cache_builder_qwen3_t5_fields():
    """Verify cache builder writes Qwen3/T5 fields when provided."""
    from .anima_cache_builder import AnimaCacheBuilderConfig, build_anima_cache_sample

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create fake image
        image_path = tmpdir / "test.png"
        from PIL import Image
        img = Image.new("RGB", (64, 64), color="red")
        img.save(image_path)

        # Create caption
        caption_path = tmpdir / "test.txt"
        caption_path.write_text("test caption", encoding="utf-8")

        # Fake VAE encode
        def fake_vae_encode(image: torch.Tensor) -> torch.Tensor:
            # Input: [1, 3, H, W], Output: [1, 16, h, w]
            return torch.randn(1, 16, 8, 8)

        # Fake text encode with Qwen3/T5 fields
        def fake_text_encode(caption: str) -> Dict[str, torch.Tensor]:
            return {
                "prompt_embeds": torch.randn(77, 768),
                "attn_mask": torch.ones(77, dtype=torch.bool),
                "qwen3_hidden_states": torch.randn(128, 1024),
                "qwen3_attention_mask": torch.ones(128, dtype=torch.bool),
                "t5_input_ids": torch.randint(0, 1000, (256,), dtype=torch.long),
                "t5_attn_mask": torch.ones(256, dtype=torch.bool),
            }

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

        # Verify latent cache exists
        assert latent_path.exists(), f"Latent cache not created: {latent_path}"
        latent_data = np.load(str(latent_path))
        assert "latents_8x8" in latent_data, "Missing latents key"
        assert "schema_version" in latent_data, "Missing schema_version"

        # Verify text cache exists with Qwen3/T5 fields
        assert text_path.exists(), f"Text cache not created: {text_path}"
        text_data = np.load(str(text_path))
        assert "prompt_embeds" in text_data, "Missing prompt_embeds"
        assert "attn_mask" in text_data, "Missing attn_mask"
        assert "qwen3_hidden_states" in text_data, "Missing qwen3_hidden_states"
        assert "qwen3_attention_mask" in text_data, "Missing qwen3_attention_mask"
        assert "t5_input_ids" in text_data, "Missing t5_input_ids"
        assert "t5_attn_mask" in text_data, "Missing t5_attn_mask"
        assert "schema_version" in text_data, "Missing schema_version in text cache"

        print("✓ Cache builder writes Qwen3/T5 fields correctly")


def test_cache_builder_token_limits():
    """Verify token limits are applied during cache generation."""
    from .anima_cache_builder import AnimaCacheBuilderConfig, build_anima_cache_sample

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create fake image
        image_path = tmpdir / "test.png"
        from PIL import Image
        img = Image.new("RGB", (64, 64), color="blue")
        img.save(image_path)

        # Create caption
        caption_path = tmpdir / "test.txt"
        caption_path.write_text("test caption with token limit", encoding="utf-8")

        # Fake VAE encode
        def fake_vae_encode(image: torch.Tensor) -> torch.Tensor:
            return torch.randn(1, 16, 8, 8)

        # Fake text encode with long sequences
        def fake_text_encode(caption: str) -> Dict[str, torch.Tensor]:
            return {
                "prompt_embeds": torch.randn(200, 768),  # Long sequence
                "attn_mask": torch.ones(200, dtype=torch.bool),
                "qwen3_hidden_states": torch.randn(300, 1024),  # Long sequence
                "qwen3_attention_mask": torch.ones(300, dtype=torch.bool),
            }

        config = AnimaCacheBuilderConfig(
            data_dir=str(tmpdir),
            output_dir=str(tmpdir),
            vae_chunk_size=0,
            text_token_limit=100,  # Limit to 100 tokens
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

        # Verify token limit was applied
        text_data = np.load(str(text_path))
        prompt_embeds_shape = text_data["prompt_embeds"].shape
        qwen3_shape = text_data["qwen3_hidden_states"].shape

        assert prompt_embeds_shape[0] == 100, f"Expected 100 tokens, got {prompt_embeds_shape[0]}"
        assert qwen3_shape[0] == 100, f"Expected 100 Qwen3 tokens, got {qwen3_shape[0]}"

        print("✓ Token limits applied correctly during cache generation")


def test_online_cache_dataset_generation():
    """Verify online cache dataset generates cache on-demand."""
    from .anima_online_cache_dataset import AnimaOnlineCacheDataset
    from .anima_cache_builder import AnimaCacheBuilderConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create fake images
        for i in range(3):
            image_path = tmpdir / f"image_{i}.png"
            from PIL import Image
            img = Image.new("RGB", (64, 64), color="green")
            img.save(image_path)

            caption_path = tmpdir / f"image_{i}.txt"
            caption_path.write_text(f"caption {i}", encoding="utf-8")

        # Fake encode functions
        def fake_vae_encode(image: torch.Tensor) -> torch.Tensor:
            return torch.randn(1, 16, 8, 8)

        def fake_text_encode(caption: str) -> Dict[str, torch.Tensor]:
            return {
                "prompt_embeds": torch.randn(77, 768),
                "attn_mask": torch.ones(77, dtype=torch.bool),
                "qwen3_hidden_states": torch.randn(128, 1024),
                "qwen3_attention_mask": torch.ones(128, dtype=torch.bool),
            }

        cache_config = AnimaCacheBuilderConfig(
            data_dir=str(tmpdir),
            output_dir=str(tmpdir),
            vae_chunk_size=0,
            text_token_limit=0,
            include_loss_mask=False,
            disk_format="npz",
            disk_dtype="float16",
        )

        dataset = AnimaOnlineCacheDataset(
            data_dir=tmpdir,
            vae_encode_fn=fake_vae_encode,
            text_encode_fn=fake_text_encode,
            cache_config=cache_config,
            latent_crop_size=0,
            text_token_limit=0,
            fixed_text_tokens=0,
            fixed_visual_tokens=0,
            caption_extension=".txt",
            weighted_captions=False,
        )

        assert len(dataset) == 3, f"Expected 3 images, got {len(dataset)}"

        # Access first sample - should generate cache
        sample = dataset[0]
        assert "latents" in sample, "Missing latents in sample"
        assert "encoder_hidden_states" in sample, "Missing encoder_hidden_states"
        assert "qwen3_hidden_states" in sample, "Missing qwen3_hidden_states"

        # Verify cache files were created
        cache_files = list(tmpdir.glob("*_anima.npz"))
        text_cache_files = list(tmpdir.glob("*_anima_te.npz"))
        assert len(cache_files) >= 1, "No latent cache files created"
        assert len(text_cache_files) >= 1, "No text cache files created"

        # Access same sample again - should use cache
        sample2 = dataset[0]
        assert torch.allclose(sample["latents"], sample2["latents"]), "Cache not reused"

        print("✓ Online cache dataset generates and reuses cache correctly")


def test_cache_metadata_schema_version():
    """Verify cache files include schema version metadata."""
    from .anima_cache_builder import AnimaCacheBuilderConfig, build_anima_cache_sample, ANIMA_CACHE_VERSION

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create fake image
        image_path = tmpdir / "test.png"
        from PIL import Image
        img = Image.new("RGB", (64, 64), color="yellow")
        img.save(image_path)

        caption_path = tmpdir / "test.txt"
        caption_path.write_text("test", encoding="utf-8")

        def fake_vae_encode(image: torch.Tensor) -> torch.Tensor:
            return torch.randn(1, 16, 8, 8)

        def fake_text_encode(caption: str) -> Dict[str, torch.Tensor]:
            return {
                "prompt_embeds": torch.randn(77, 768),
                "attn_mask": torch.ones(77, dtype=torch.bool),
            }

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

        # Verify schema version in both caches
        latent_data = np.load(str(latent_path))
        text_data = np.load(str(text_path))

        assert "schema_version" in latent_data, "Missing schema_version in latent cache"
        assert "schema_version" in text_data, "Missing schema_version in text cache"
        assert int(latent_data["schema_version"]) == ANIMA_CACHE_VERSION, "Wrong latent schema version"
        assert int(text_data["schema_version"]) == ANIMA_CACHE_VERSION, "Wrong text schema version"

        # Verify has_loss_mask metadata
        assert "has_loss_mask" in text_data, "Missing has_loss_mask metadata"

        print("✓ Cache metadata includes schema version and has_loss_mask")


if __name__ == "__main__":
    test_cache_builder_qwen3_t5_fields()
    test_cache_builder_token_limits()
    test_online_cache_dataset_generation()
    test_cache_metadata_schema_version()
    print("\n✓ All Phase 5 cache generation smoke tests passed")
