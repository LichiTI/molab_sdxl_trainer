"""Dataset Cache Preflight — checks latent cache readiness before training.

Behavioral specification derived from a legacy fork cache preflight utility.
This is a Warehouse reimplementation. No original code was copied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import MessageBag, ModelArchitecture

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache profile by model architecture
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LatentCacheProfile:
    """Describes how latent cache files are named and structured."""

    suffix: str
    stride: int
    multi_resolution: bool


_PROFILES: dict[ModelArchitecture, LatentCacheProfile] = {
    ModelArchitecture.SD15:           LatentCacheProfile(suffix="_sd15",           stride=1, multi_resolution=False),
    ModelArchitecture.SDXL:           LatentCacheProfile(suffix="_sdxl",           stride=1, multi_resolution=False),
    ModelArchitecture.SD3:            LatentCacheProfile(suffix="_sd3",            stride=1, multi_resolution=True),
    ModelArchitecture.FLUX:           LatentCacheProfile(suffix="_flux",           stride=1, multi_resolution=True),
    ModelArchitecture.ANIMA:          LatentCacheProfile(suffix="_anima",          stride=1, multi_resolution=True),
    ModelArchitecture.NEWBIE:         LatentCacheProfile(suffix="_newbie",         stride=1, multi_resolution=False),
    ModelArchitecture.LUMINA:         LatentCacheProfile(suffix="_lumina",         stride=1, multi_resolution=True),
    ModelArchitecture.LUMINA2:        LatentCacheProfile(suffix="_lumina2",        stride=1, multi_resolution=True),
    ModelArchitecture.QWEN_IMAGE:     LatentCacheProfile(suffix="_qwen_image",     stride=1, multi_resolution=True),
    ModelArchitecture.HUNYUAN_DIT:    LatentCacheProfile(suffix="_hunyuan_dit",    stride=1, multi_resolution=True),
    ModelArchitecture.HUNYUAN_IMAGE:  LatentCacheProfile(suffix="_hunyuan_image",  stride=1, multi_resolution=True),
}


def resolve_cache_profile(arch: ModelArchitecture) -> LatentCacheProfile:
    return _PROFILES.get(arch, LatentCacheProfile(suffix="", stride=1, multi_resolution=False))


# ---------------------------------------------------------------------------
# Bucket plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BucketEntry:
    """A single resolution bucket."""

    width: int
    height: int
    count: int


@dataclass(frozen=True)
class BucketPlan:
    """Planned resolution buckets for cache generation."""

    entries: list[BucketEntry]
    base_side: int
    enable_upscale: bool = False


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class CachePreflightReport:
    """Result of a cache preflight check."""

    ready: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    cache_hit_rate: float = 0.0
    total_images: int = 0
    cached_count: int = 0
    missing_count: int = 0
    bucket_plan: BucketPlan | None = None
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def _count_cache_files(data_dir: Path, suffix: str) -> int:
    """Count .npz cache files matching the profile suffix."""
    count = 0
    if not data_dir.is_dir():
        return 0
    for p in data_dir.rglob("*.npz"):
        if suffix and suffix in p.stem:
            count += 1
        elif not suffix:
            count += 1
    return count


def _count_images(data_dir: Path) -> int:
    """Count image files recursively."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    return sum(1 for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)


# ---------------------------------------------------------------------------
# Main preflight
# ---------------------------------------------------------------------------

class DatasetCachePreflight:
    """Checks whether latent cache files are present and sufficient.

    Stateless after construction.
    """

    def __init__(self, arch: ModelArchitecture) -> None:
        self._profile = resolve_cache_profile(arch)

    @property
    def profile(self) -> LatentCacheProfile:
        return self._profile

    def analyze(
        self,
        data_dir: Path,
        *,
        expected_count: int | None = None,
    ) -> CachePreflightReport:
        """Analyze cache readiness for *data_dir*.

        Parameters
        ----------
        data_dir:
            Root of the training image directory.
        expected_count:
            If provided, use this instead of scanning for images.
        """
        data_dir = Path(data_dir)
        bag = MessageBag()
        report = CachePreflightReport(ready=False)

        if not data_dir.is_dir():
            bag.add_error(f"Data directory does not exist: {data_dir}")
            report.errors, report.warnings, report.notes = bag.errors, bag.warnings, bag.notes
            return report

        total_images = expected_count if expected_count is not None else _count_images(data_dir)
        cached = _count_cache_files(data_dir, self._profile.suffix)

        report.total_images = total_images
        report.cached_count = cached
        report.missing_count = max(0, total_images - cached)

        if total_images == 0:
            bag.add_warning("No images found in data directory")
            report.cache_hit_rate = 0.0
        else:
            hit_rate = cached / total_images
            report.cache_hit_rate = min(hit_rate, 1.0)
            if hit_rate < 0.9:
                bag.add_warning(
                    f"Low cache hit rate: {cached}/{total_images} "
                    f"({hit_rate:.1%}). Cache will need to be generated."
                )
            if hit_rate >= 0.99:
                bag.add_note("Cache appears complete")
            elif hit_rate >= 0.9:
                bag.add_note(f"Cache mostly complete: {cached}/{total_images}")

        if self._profile.multi_resolution:
            bag.add_note("Multi-resolution cache mode enabled")

        report.ready = bag.is_clean and cached > 0
        report.errors, report.warnings, report.notes = bag.errors, bag.warnings, bag.notes
        report.summary = {
            "architecture_suffix": self._profile.suffix,
            "multi_resolution": self._profile.multi_resolution,
            "total_images": total_images,
            "cached": cached,
            "hit_rate": report.cache_hit_rate,
        }
        return report

