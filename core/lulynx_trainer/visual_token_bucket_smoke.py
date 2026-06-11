# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for visual_token_bucket.py (#102)."""

from __future__ import annotations

import os
import sys
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.visual_token_bucket",
    os.path.join(_HERE, "visual_token_bucket.py"),
)
_vtb = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.visual_token_bucket"] = _vtb
_spec.loader.exec_module(_vtb)


def test_visual_token_count_basic():
    # 1024x1024 with patch=16, vae_downsample=8 -> latent 128x128, tokens 8x8 = 64
    count = _vtb.visual_token_count(1024, 1024, patch_size=16, vae_downsample=8)
    assert count == 64
    print(f"PASS: 1024x1024 -> {count} tokens (patch=16, vae=8)")


def test_visual_token_count_rectangular():
    # 1024x512 -> latent 128x64 -> tokens 8x4 = 32
    count = _vtb.visual_token_count(1024, 512, patch_size=16, vae_downsample=8)
    assert count == 32
    print(f"PASS: 1024x512 -> {count} tokens")


def test_round_to_step():
    assert _vtb.round_to_step(1023, 64) == 1024
    assert _vtb.round_to_step(1025, 64) == 1024
    assert _vtb.round_to_step(900, 64) == 896
    print("PASS: round_to_step rounds to nearest multiple")


def test_build_buckets_groups_by_token_count():
    cfg = _vtb.VisualTokenBucketConfig(target_buckets=4)
    resolutions = [
        (1024, 1024),  # 64 tokens
        (1024, 1024),
        (512, 512),    # 16 tokens
        (1024, 512),   # 32 tokens
        (768, 768),    # 36 tokens
    ]
    buckets = _vtb.build_buckets(resolutions, cfg)
    counts = [b.token_count for b in buckets]
    assert counts == sorted(counts)  # sorted ascending
    assert len(buckets) <= cfg.target_buckets
    print(f"PASS: build_buckets groups into {len(buckets)} buckets: {counts}")


def test_build_buckets_coalesces_when_too_many():
    cfg = _vtb.VisualTokenBucketConfig(target_buckets=2)
    resolutions = [
        (256, 256), (384, 384), (512, 512), (768, 768),
        (1024, 1024), (1280, 1280), (1536, 1536),
    ]
    buckets = _vtb.build_buckets(resolutions, cfg)
    assert len(buckets) == 2
    print(f"PASS: bucket coalescing reduces to target={cfg.target_buckets}")


def test_assign_to_bucket_finds_correct_index():
    cfg = _vtb.VisualTokenBucketConfig(target_buckets=8)
    resolutions = [(512, 512), (1024, 1024)]
    buckets = _vtb.build_buckets(resolutions, cfg)

    # 512x512 should land in the first (smaller) bucket
    idx = _vtb.assign_to_bucket(512, 512, buckets,
                                  patch_size=cfg.patch_size,
                                  vae_downsample=cfg.vae_downsample)
    assert idx == 0
    # 1024x1024 should land in the second
    idx = _vtb.assign_to_bucket(1024, 1024, buckets,
                                  patch_size=cfg.patch_size,
                                  vae_downsample=cfg.vae_downsample)
    assert idx == 1
    print("PASS: assign_to_bucket maps resolutions to correct bucket")


def test_bucket_sampler_returns_consistent_batches():
    cfg = _vtb.VisualTokenBucketConfig(target_buckets=4)
    resolutions = [
        (512, 512), (512, 512), (512, 512),       # all same bucket
        (1024, 1024), (1024, 1024),               # all same bucket
        (768, 768),                                # different
    ]
    sampler = _vtb.BucketSampler(resolutions, cfg, batch_size=2)
    batches = sampler.make_batches()

    # Every sample must be in exactly one batch
    flat = [idx for batch in batches for idx in batch]
    assert sorted(flat) == sorted(range(len(resolutions)))

    # Within each batch, all samples should share the same bucket
    for batch in batches:
        bucket_ids = {sampler._sample_bucket[i] for i in batch}
        assert len(bucket_ids) == 1, f"batch {batch} mixes buckets {bucket_ids}"

    print(f"PASS: BucketSampler produced {len(batches)} batches with bucket-uniform contents")


def test_bucket_sampler_invalid_batch_size_raises():
    try:
        _vtb.BucketSampler([(512, 512)], _vtb.VisualTokenBucketConfig(), batch_size=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("PASS: batch_size=0 raises ValueError")


def test_min_max_pixel_clamping():
    cfg = _vtb.VisualTokenBucketConfig(target_buckets=4, min_pixel=512, max_pixel=1024)
    resolutions = [(128, 128), (4096, 4096)]
    buckets = _vtb.build_buckets(resolutions, cfg)
    # Both should clamp into the (512..1024) range
    all_resos = [r for b in buckets for r in b.resolutions]
    for h, w in all_resos:
        assert 512 <= h <= 1024
        assert 512 <= w <= 1024
    print("PASS: min_pixel/max_pixel clamping applied to extreme resolutions")


if __name__ == "__main__":
    test_visual_token_count_basic()
    test_visual_token_count_rectangular()
    test_round_to_step()
    test_build_buckets_groups_by_token_count()
    test_build_buckets_coalesces_when_too_many()
    test_assign_to_bucket_finds_correct_index()
    test_bucket_sampler_returns_consistent_batches()
    test_bucket_sampler_invalid_batch_size_raises()
    test_min_max_pixel_clamping()
    print("\nAll visual_token_bucket smoke tests passed!")
