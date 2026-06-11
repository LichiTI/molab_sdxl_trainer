# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Anima cached dataset mmap and lazy loading (Phase 6.3).

Validates that:
- AnimaCachedDataset accepts cache_mmap parameter
- AnimaCachedDataset accepts cache_lazy parameter
- AnimaCachedDataset accepts file_handle_cache_size parameter
- mmap mode works for .npz files
- lazy loading caches file handles
- file handle cache eviction works correctly
- NewbieCachedDataset supports the same parameters
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import numpy as np
import torch

if __package__ in (None, ""):
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))


def test_anima_dataset_mmap():
    """Verify AnimaCachedDataset accepts cache_mmap parameter."""
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

        # Create dataset with mmap enabled
        dataset = AnimaCachedDataset(
            data_dir=tmpdir,
            latent_crop_size=0,
            text_token_limit=0,
            fixed_text_tokens=0,
            fixed_visual_tokens=0,
            caption_extension=".txt",
            weighted_captions=False,
            cache_mmap=True,
        )

        assert dataset.cache_mmap is True, "cache_mmap should be True"

        # Load sample - should use mmap
        sample = dataset[0]
        assert "latents" in sample, "Missing latents in sample"
        assert "encoder_hidden_states" in sample, "Missing encoder_hidden_states"

        # Close file handles before cleanup
        dataset.close_file_handles()
        del dataset
        import gc
        gc.collect()

        print("[PASS] AnimaCachedDataset accepts cache_mmap parameter")


def test_anima_dataset_lazy_loading():
    """Verify AnimaCachedDataset lazy loading caches file handles."""
    from anima_cached_dataset import AnimaCachedDataset
    from anima_cache_builder import AnimaCacheBuilderConfig, build_anima_cache_sample

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create multiple fake images
        for i in range(3):
            image_path = tmpdir / f"image_{i}.png"
            from PIL import Image
            img = Image.new("RGB", (64, 64), color="blue")
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
            }

        # Build cache for all images
        config = AnimaCacheBuilderConfig(
            data_dir=str(tmpdir),
            output_dir=str(tmpdir),
            vae_chunk_size=0,
            text_token_limit=0,
            include_loss_mask=False,
            disk_format="npz",
            disk_dtype="float16",
        )

        for i in range(3):
            image_path = tmpdir / f"image_{i}.png"
            build_anima_cache_sample(
                image_path=image_path,
                vae_encode_fn=fake_vae_encode,
                text_encode_fn=fake_text_encode,
                config=config,
                force=True,
            )

        # Create dataset with lazy loading enabled
        dataset = AnimaCachedDataset(
            data_dir=tmpdir,
            latent_crop_size=0,
            text_token_limit=0,
            fixed_text_tokens=0,
            fixed_visual_tokens=0,
            caption_extension=".txt",
            weighted_captions=False,
            cache_lazy=True,
            file_handle_cache_size=4,  # Two files per sample: latent + text conditioning.
        )

        assert dataset.cache_lazy is True, "cache_lazy should be True"
        assert dataset.file_handle_cache_size == 4, "file_handle_cache_size should be 4"

        # Load first sample - should cache file handle
        sample0 = dataset[0]
        assert len(dataset._file_handle_cache) > 0, "File handle cache should not be empty"

        # Load second sample - should cache another file handle
        sample1 = dataset[1]
        assert len(dataset._file_handle_cache) > 0, "File handle cache should not be empty"

        # Touch first sample again so LRU order is refreshed.
        sample0_again = dataset[0]
        assert sample0_again["latents"].shape == sample0["latents"].shape
        first_latent_path = dataset.samples[0].latent_path
        second_latent_path = dataset.samples[1].latent_path

        # Load third sample - should evict oldest due to cache size limit
        sample2 = dataset[2]
        assert len(dataset._file_handle_cache) <= dataset.file_handle_cache_size, \
            f"File handle cache should not exceed size limit: {len(dataset._file_handle_cache)} > {dataset.file_handle_cache_size}"
        cached_paths = set(dataset._file_handle_cache.keys())
        assert first_latent_path in cached_paths, "Recently reused cache handle should stay resident"
        assert second_latent_path not in cached_paths, "Least-recently used cache handle should be evicted"

        # Close file handles before cleanup
        dataset.close_file_handles()
        del dataset
        import gc
        gc.collect()

        print("[PASS] AnimaCachedDataset lazy loading caches and evicts file handles correctly")


def test_anima_dataset_mmap_and_lazy():
    """Verify AnimaCachedDataset can use both mmap and lazy loading."""
    from anima_cached_dataset import AnimaCachedDataset
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

        # Create dataset with both mmap and lazy loading
        dataset = AnimaCachedDataset(
            data_dir=tmpdir,
            latent_crop_size=0,
            text_token_limit=0,
            fixed_text_tokens=0,
            fixed_visual_tokens=0,
            caption_extension=".txt",
            weighted_captions=False,
            cache_mmap=True,
            cache_lazy=True,
        )

        assert dataset.cache_mmap is True, "cache_mmap should be True"
        assert dataset.cache_lazy is True, "cache_lazy should be True"

        # Load sample - should use both mmap and lazy loading
        sample = dataset[0]
        assert "latents" in sample, "Missing latents in sample"
        assert len(dataset._file_handle_cache) > 0, "File handle cache should not be empty"

        # Close file handles before cleanup
        dataset.close_file_handles()
        del dataset
        import gc
        gc.collect()

        print("[PASS] AnimaCachedDataset can use both mmap and lazy loading")


def test_newbie_dataset_mmap_lazy():
    """Verify NewbieCachedDataset accepts mmap and lazy loading parameters."""
    from newbie_cached_dataset import NewbieCachedDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create fake Newbie caches
        for i in range(3):
            cache_path = tmpdir / f"test_{i}_newbie.npz"
            np.savez(
                str(cache_path),
                latents=np.random.randn(1, 16, 8, 8).astype(np.float16),
                encoder_hidden_states=np.random.randn(77, 768).astype(np.float16),
                attention_mask=np.ones(77, dtype=bool),
                pooled_prompt_embeds=np.random.randn(768).astype(np.float16),
                newbie_cache_schema_version=np.array(2, dtype=np.int32),
            )

        # Create dataset with mmap and lazy loading
        dataset = NewbieCachedDataset(
            data_dir=tmpdir,
            latent_crop_size=0,
            text_token_limit=0,
            cache_mmap=True,
            cache_lazy=True,
            file_handle_cache_size=2,
        )

        assert dataset.cache_mmap is True, "cache_mmap should be True"
        assert dataset.cache_lazy is True, "cache_lazy should be True"
        assert dataset.file_handle_cache_size == 2, "file_handle_cache_size should be 2"

        # Load samples and verify LRU refresh/eviction.
        sample = dataset[0]
        assert "latents" in sample, "Missing latents in sample"
        assert "encoder_hidden_states" in sample, "Missing encoder_hidden_states"
        dataset[1]
        dataset[0]
        first_cache_path = dataset.samples[0].cache_path
        second_cache_path = dataset.samples[1].cache_path
        dataset[2]
        cached_paths = set(dataset._file_handle_cache.keys())
        assert len(cached_paths) <= dataset.file_handle_cache_size
        assert first_cache_path in cached_paths, "Recently reused Newbie cache handle should stay resident"
        assert second_cache_path not in cached_paths, "Least-recently used Newbie cache handle should be evicted"

        # Close file handles before cleanup
        dataset.close_file_handles()
        del dataset
        import gc
        gc.collect()

        print("[PASS] NewbieCachedDataset accepts mmap and lazy loading parameters")


def test_newbie_bucket_metadata_avoids_shape_file_scan():
    """Verify Newbie bucket setup can use sidecar metadata for latent shapes."""
    from newbie_cached_dataset import NewbieCachedDataset
    from cache_metadata import write_cache_metadata

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        shapes = [(8, 8), (8, 12)]
        for i, (h, w) in enumerate(shapes):
            cache_path = tmpdir / f"meta_{i}_newbie.npz"
            np.savez(
                str(cache_path),
                latents=np.random.randn(1, 16, h, w).astype(np.float16),
                encoder_hidden_states=np.random.randn(77, 768).astype(np.float16),
                attention_mask=np.ones(77, dtype=bool),
                pooled_prompt_embeds=np.random.randn(768).astype(np.float16),
                newbie_cache_schema_version=np.array(2, dtype=np.int32),
            )
        metadata = write_cache_metadata(tmpdir, family="newbie")
        assert metadata.sample_count == 2, metadata

        with patch.object(NewbieCachedDataset, "_array_shape_from_cache", side_effect=AssertionError("metadata miss")):
            dataset = NewbieCachedDataset(data_dir=tmpdir)
            bucket_indices = dataset.get_bucket_indices()
            summary = dataset.get_token_bucket_summary()

        assert bucket_indices == {"8x8": [0], "8x12": [1]}, bucket_indices
        assert summary["bucket_count"] == 2, summary


if __name__ == "__main__":
    test_anima_dataset_mmap()
    test_anima_dataset_lazy_loading()
    test_anima_dataset_mmap_and_lazy()
    test_newbie_dataset_mmap_lazy()
    test_newbie_bucket_metadata_avoids_shape_file_scan()
    print("\n[PASS] All Phase 6.3 mmap/lazy loading smoke tests passed")
