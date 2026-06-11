# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Visual token bucketing (#102).

For DiT-style models, the latent grid produces a fixed number of visual
tokens per image: ``num_tokens = (H_latent // patch) * (W_latent // patch)``.
When training with mixed resolutions, this number varies per batch,
which breaks ``torch.compile`` graphs and CUDAGraph capture.

This module groups images into "token-budget buckets" so that every
batch has the same visual token count.  Resolutions that produce the
same token count (within tolerance) are grouped together; the dataset
loader then samples batches from a single bucket at a time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VisualTokenBucketConfig:
    """Configuration for visual-token bucketing."""

    patch_size: int = 16          # DiT patch size
    vae_downsample: int = 8       # how much VAE shrinks (Anima Qwen=8, SDXL=8)
    target_buckets: int = 8       # how many distinct token counts to keep
    min_pixel: int = 256
    max_pixel: int = 2048
    step: int = 64                # resolution rounding step


@dataclass
class Bucket:
    """A bucket of resolutions that share the same visual token count."""

    token_count: int
    resolutions: List[Tuple[int, int]] = field(default_factory=list)  # (H, W) pairs


# ---------------------------------------------------------------------------
# Token-count math
# ---------------------------------------------------------------------------

def visual_token_count(
    height: int,
    width: int,
    *,
    patch_size: int = 16,
    vae_downsample: int = 8,
) -> int:
    """Compute the visual token count for an image of (H, W) pixels."""
    h_lat = max(1, height // vae_downsample)
    w_lat = max(1, width // vae_downsample)
    h_tokens = max(1, h_lat // patch_size)
    w_tokens = max(1, w_lat // patch_size)
    return h_tokens * w_tokens


def round_to_step(value: int, step: int) -> int:
    if step <= 1:
        return int(value)
    return int(round(value / step) * step)


# ---------------------------------------------------------------------------
# Bucket builder
# ---------------------------------------------------------------------------

def build_buckets(
    resolutions: Iterable[Tuple[int, int]],
    config: VisualTokenBucketConfig,
) -> List[Bucket]:
    """Group resolutions by their visual token count.

    Returns a sorted list of :class:`Bucket` instances.
    """
    by_count: Dict[int, List[Tuple[int, int]]] = {}
    for h, w in resolutions:
        h_r = round_to_step(h, config.step)
        w_r = round_to_step(w, config.step)
        h_r = max(config.min_pixel, min(config.max_pixel, h_r))
        w_r = max(config.min_pixel, min(config.max_pixel, w_r))

        count = visual_token_count(
            h_r, w_r,
            patch_size=config.patch_size,
            vae_downsample=config.vae_downsample,
        )
        by_count.setdefault(count, []).append((h_r, w_r))

    buckets = [Bucket(token_count=c, resolutions=sorted(set(r))) for c, r in by_count.items()]
    buckets.sort(key=lambda b: b.token_count)

    if len(buckets) > config.target_buckets:
        buckets = _coalesce_buckets(buckets, config.target_buckets)
    return buckets


def _coalesce_buckets(buckets: List[Bucket], target: int) -> List[Bucket]:
    """Merge adjacent buckets greedily until we have ``target`` buckets.

    Each merge takes the larger token count as the canonical value (a tiny
    overhead per smaller image is acceptable; downsizing to a smaller bucket
    would lose information).
    """
    while len(buckets) > target:
        # Find the closest pair (smallest difference in token count)
        best_idx = 0
        best_gap = float("inf")
        for i in range(len(buckets) - 1):
            gap = buckets[i + 1].token_count - buckets[i].token_count
            if gap < best_gap:
                best_gap = gap
                best_idx = i

        # Merge buckets[best_idx] into buckets[best_idx + 1]
        merged_resos = sorted(set(buckets[best_idx].resolutions + buckets[best_idx + 1].resolutions))
        new_bucket = Bucket(
            token_count=buckets[best_idx + 1].token_count,
            resolutions=merged_resos,
        )
        buckets = buckets[:best_idx] + [new_bucket] + buckets[best_idx + 2:]

    return buckets


def assign_to_bucket(
    height: int,
    width: int,
    buckets: Sequence[Bucket],
    *,
    patch_size: int = 16,
    vae_downsample: int = 8,
) -> int:
    """Return the index of the bucket whose token count >= this image's count."""
    count = visual_token_count(
        height, width, patch_size=patch_size, vae_downsample=vae_downsample,
    )
    for i, b in enumerate(buckets):
        if b.token_count >= count:
            return i
    return len(buckets) - 1


# ---------------------------------------------------------------------------
# Bucket-aware sampler
# ---------------------------------------------------------------------------

class BucketSampler:
    """Group sample indices by their bucket; iterate one bucket at a time.

    Each call to :meth:`make_batches` returns a list of batches where every
    batch contains samples from a single bucket — guaranteeing a static
    visual token count per batch.
    """

    def __init__(
        self,
        sample_resolutions: Sequence[Tuple[int, int]],
        config: VisualTokenBucketConfig,
        *,
        batch_size: int = 1,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.config = config
        self.batch_size = batch_size

        self.buckets = build_buckets(sample_resolutions, config)
        # Map sample_index -> bucket_index
        self._sample_bucket: List[int] = []
        for h, w in sample_resolutions:
            self._sample_bucket.append(
                assign_to_bucket(
                    h, w, self.buckets,
                    patch_size=config.patch_size,
                    vae_downsample=config.vae_downsample,
                )
            )

    def make_batches(self) -> List[List[int]]:
        """Return a list of batches; each batch is a list of sample indices."""
        per_bucket: Dict[int, List[int]] = {}
        for idx, b in enumerate(self._sample_bucket):
            per_bucket.setdefault(b, []).append(idx)

        batches: List[List[int]] = []
        for bucket_idx, indices in per_bucket.items():
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start: start + self.batch_size])
        return batches
