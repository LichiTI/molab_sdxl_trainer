# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Anima cached dataloader pinned memory and prefetch (Phase 6.2).

Validates that:
- create_anima_cached_dataloader accepts pin_memory parameter
- create_anima_cached_dataloader accepts prefetch_factor parameter
- create_anima_cached_dataloader accepts persistent_workers parameter
- DataLoader is created with correct configuration
- create_newbie_cached_dataloader accepts the same parameters
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def test_anima_dataloader_pin_memory():
    """Verify create_anima_cached_dataloader accepts pin_memory parameter."""
    from anima_cached_dataset import AnimaCachedDataset, create_anima_cached_dataloader
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

        # Fake encode functions
        def fake_vae_encode(image: torch.Tensor) -> torch.Tensor:
            return torch.randn(1, 16, 8, 8)

        def fake_text_encode(caption: str) -> Dict[str, torch.Tensor]:
            return {
                "prompt_embeds": torch.randn(77, 768),
                "attn_mask": torch.ones(77, dtype=torch.bool),
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

        build_anima_cache_sample(
            image_path=image_path,
            vae_encode_fn=fake_vae_encode,
            text_encode_fn=fake_text_encode,
            config=config,
            force=True,
        )

        # Create dataset
        dataset = AnimaCachedDataset(
            data_dir=tmpdir,
            latent_crop_size=0,
            text_token_limit=0,
            fixed_text_tokens=0,
            fixed_visual_tokens=0,
            caption_extension=".txt",
            weighted_captions=False,
        )

        # Create dataloader with pin_memory=True
        dataloader = create_anima_cached_dataloader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        assert dataloader.pin_memory is True, "pin_memory should be True"

        # Create dataloader with pin_memory=False
        dataloader_no_pin = create_anima_cached_dataloader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        assert dataloader_no_pin.pin_memory is False, "pin_memory should be False"

        print("[PASS] create_anima_cached_dataloader accepts pin_memory parameter")


def test_anima_dataloader_prefetch_factor():
    """Verify create_anima_cached_dataloader accepts prefetch_factor parameter."""
    from anima_cached_dataset import AnimaCachedDataset, create_anima_cached_dataloader
    from anima_cache_builder import AnimaCacheBuilderConfig, build_anima_cache_sample

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create fake image and caption
        image_path = tmpdir / "test.png"
        from PIL import Image
        img = Image.new("RGB", (64, 64), color="blue")
        img.save(image_path)

        caption_path = tmpdir / "test.txt"
        caption_path.write_text("test caption", encoding="utf-8")

        # Fake encode functions
        def fake_vae_encode(image: torch.Tensor) -> torch.Tensor:
            return torch.randn(1, 16, 8, 8)

        def fake_text_encode(caption: str) -> Dict[str, torch.Tensor]:
            return {
                "prompt_embeds": torch.randn(77, 768),
                "attn_mask": torch.ones(77, dtype=torch.bool),
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

        build_anima_cache_sample(
            image_path=image_path,
            vae_encode_fn=fake_vae_encode,
            text_encode_fn=fake_text_encode,
            config=config,
            force=True,
        )

        # Create dataset
        dataset = AnimaCachedDataset(
            data_dir=tmpdir,
            latent_crop_size=0,
            text_token_limit=0,
            fixed_text_tokens=0,
            fixed_visual_tokens=0,
            caption_extension=".txt",
            weighted_captions=False,
        )

        # Create dataloader with prefetch_factor (requires num_workers > 0)
        dataloader = create_anima_cached_dataloader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            prefetch_factor=4,
        )

        # prefetch_factor is only set when num_workers > 0
        assert dataloader.prefetch_factor == 4, f"Expected prefetch_factor=4, got {dataloader.prefetch_factor}"

        # Create dataloader with num_workers=0 (prefetch_factor should not be set)
        dataloader_no_workers = create_anima_cached_dataloader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            prefetch_factor=4,
        )

        # When num_workers=0, prefetch_factor is not used
        # DataLoader defaults to None or 2 depending on PyTorch version
        print(f"[INFO] With num_workers=0, prefetch_factor is: {dataloader_no_workers.prefetch_factor}")

        print("[PASS] create_anima_cached_dataloader accepts prefetch_factor parameter")


def test_anima_dataloader_persistent_workers():
    """Verify create_anima_cached_dataloader accepts persistent_workers parameter."""
    from anima_cached_dataset import AnimaCachedDataset, create_anima_cached_dataloader
    from anima_cache_builder import AnimaCacheBuilderConfig, build_anima_cache_sample

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create fake image and caption
        image_path = tmpdir / "test.png"
        from PIL import Image
        img = Image.new("RGB", (64, 64), color="green")
        img.save(image_path)

        caption_path = tmpdir / "test.txt"
        caption_path.write_text("test caption", encoding="utf-8")

        # Fake encode functions
        def fake_vae_encode(image: torch.Tensor) -> torch.Tensor:
            return torch.randn(1, 16, 8, 8)

        def fake_text_encode(caption: str) -> Dict[str, torch.Tensor]:
            return {
                "prompt_embeds": torch.randn(77, 768),
                "attn_mask": torch.ones(77, dtype=torch.bool),
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

        build_anima_cache_sample(
            image_path=image_path,
            vae_encode_fn=fake_vae_encode,
            text_encode_fn=fake_text_encode,
            config=config,
            force=True,
        )

        # Create dataset
        dataset = AnimaCachedDataset(
            data_dir=tmpdir,
            latent_crop_size=0,
            text_token_limit=0,
            fixed_text_tokens=0,
            fixed_visual_tokens=0,
            caption_extension=".txt",
            weighted_captions=False,
        )

        # Create dataloader with persistent_workers (requires num_workers > 0)
        dataloader = create_anima_cached_dataloader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

        assert dataloader.persistent_workers is True, "persistent_workers should be True"

        # Create dataloader with num_workers=0 (persistent_workers should not be set)
        dataloader_no_workers = create_anima_cached_dataloader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            persistent_workers=True,
        )

        # When num_workers=0, persistent_workers is not used
        assert dataloader_no_workers.persistent_workers is False, "persistent_workers should be False when num_workers=0"

        print("[PASS] create_anima_cached_dataloader accepts persistent_workers parameter")


def test_newbie_dataloader_memory_params():
    """Verify create_newbie_cached_dataloader accepts memory optimization parameters."""
    from newbie_cached_dataset import NewbieCachedDataset, create_newbie_cached_dataloader

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create fake Newbie cache
        cache_path = tmpdir / "test_newbie.npz"
        np.savez(
            str(cache_path),
            latents=np.random.randn(1, 16, 8, 8).astype(np.float16),
            encoder_hidden_states=np.random.randn(77, 768).astype(np.float16),
            attention_mask=np.ones(77, dtype=bool),
            pooled_prompt_embeds=np.random.randn(768).astype(np.float16),
            newbie_cache_schema_version=np.array(2, dtype=np.int32),
        )

        # Create dataset
        dataset = NewbieCachedDataset(
            data_dir=tmpdir,
            latent_crop_size=0,
            text_token_limit=0,
        )

        # Create dataloader with all memory optimization parameters
        dataloader = create_newbie_cached_dataloader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
        )

        assert dataloader.pin_memory is True, "pin_memory should be True"
        assert dataloader.persistent_workers is True, "persistent_workers should be True"
        assert dataloader.prefetch_factor == 4, f"Expected prefetch_factor=4, got {dataloader.prefetch_factor}"

        # Close file handles before cleanup
        import gc
        gc.collect()

        print("[PASS] create_newbie_cached_dataloader accepts memory optimization parameters")


if __name__ == "__main__":
    test_anima_dataloader_pin_memory()
    test_anima_dataloader_prefetch_factor()
    test_anima_dataloader_persistent_workers()
    test_newbie_dataloader_memory_params()
    print("\n[PASS] All Phase 6.2 pinned memory smoke tests passed")
