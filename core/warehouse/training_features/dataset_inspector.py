"""Dataset Inspector — scans training data directories and produces statistics.

Behavioral specification derived from a legacy fork dataset analysis utility.
This is a Warehouse reimplementation. No original code was copied.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"})
CAPTION_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "gb18030", "cp932", "latin-1")

# {repeats}_{name} folder pattern — e.g. "5_character", "10_style"
_FOLDER_RE = re.compile(r"^(\d+)_(.+)$")


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FolderStats:
    """Stats for a single dataset subfolder."""

    name: str
    repeats: int
    image_count: int
    caption_count: int
    caption_coverage: float
    missing_captions: list[str] = field(default_factory=list)
    orphan_captions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetReport:
    """Complete dataset inspection report."""

    folder_count: int
    image_count: int
    effective_image_count: int
    caption_coverage: float
    unique_tag_count: int
    average_tags_per_caption: float
    resolution_distribution: dict[str, int]
    orientation_distribution: dict[str, int]
    top_tags: list[tuple[str, int]]
    folder_stats: list[FolderStats]
    warnings: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_caption(path: Path) -> str | None:
    """Try reading a caption file with multiple encodings."""
    for encoding in CAPTION_ENCODINGS:
        try:
            return path.read_text(encoding=encoding).strip()
        except (UnicodeDecodeError, OSError):
            continue
    return None


def _classify_orientation(width: int, height: int) -> str:
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"


def _classify_resolution_bucket(width: int, height: int) -> str:
    """Bucket resolution into a human-readable label."""
    longest = max(width, height)
    if longest <= 512:
        return "<=512"
    if longest <= 768:
        return "<=768"
    if longest <= 1024:
        return "<=1024"
    if longest <= 1536:
        return "<=1536"
    return ">1536"


def _try_read_image_size(path: Path) -> tuple[int, int] | None:
    """Read image dimensions without external deps.

    Supports PNG and JPEG headers only — returns None for other formats
    or if header parsing fails. This avoids requiring Pillow at the
    contract layer; callers who need full format support should use a
    dedicated image library upstream.
    """
    try:
        data = path.read_bytes()[:32]
    except OSError:
        return None

    # PNG: 8-byte signature then IHDR chunk
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        w = int.from_bytes(data[16:20], "big")
        h = int.from_bytes(data[20:24], "big")
        return w, h

    # JPEG: scan for SOF marker
    try:
        raw = path.read_bytes()
        i = 0
        while i < len(raw) - 1:
            if raw[i] == 0xFF:
                marker = raw[i + 1]
                if marker in (0xC0, 0xC1, 0xC2):
                    h = int.from_bytes(raw[i + 5 : i + 7], "big")
                    w = int.from_bytes(raw[i + 7 : i + 9], "big")
                    return w, h
                if marker == 0xD9:
                    break
                if marker in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0x00):
                    i += 2
                else:
                    length = int.from_bytes(raw[i + 2 : i + 4], "big")
                    i += 2 + length
            else:
                i += 1
    except OSError:
        pass
    return None


def _extract_tags(caption_text: str) -> list[str]:
    """Split a caption string into individual tags.

    Tags are comma-separated. Empty/whitespace-only entries are dropped.
    """
    parts = caption_text.split(",")
    return [p.strip().lower() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Main inspector
# ---------------------------------------------------------------------------

class DatasetInspector:
    """Scans a training data directory and produces a DatasetReport.

    The directory is expected to contain subfolders named ``{repeats}_{name}``
    (e.g. ``5_character``).  Each subfolder contains images and optional
    caption files (one per image, same stem, configurable extension).

    This class is stateless after construction — ``inspect()`` is a pure
    function of its arguments.
    """

    def __init__(
        self,
        caption_extension: str = ".txt",
        top_tags: int = 40,
        max_sample_per_folder: int | None = None,
    ) -> None:
        self._caption_ext = caption_extension
        self._top_tags = top_tags
        self._max_sample = max_sample_per_folder

    # ---- public API -------------------------------------------------------

    def inspect(self, path: Path) -> DatasetReport:
        """Inspect the dataset directory at *path* and return a report."""
        path = Path(path)
        if not path.is_dir():
            return self._empty_report(f"Path is not a directory: {path}")

        folders = self._discover_folders(path)
        if not folders:
            return self._empty_report("No {repeats}_{name} subfolders found")

        all_tags: Counter[str] = Counter()
        folder_stats: list[FolderStats] = []
        total_images = 0
        total_effective = 0
        total_captioned = 0
        res_dist: Counter[str] = Counter()
        orient_dist: Counter[str] = Counter()
        warnings: list[str] = []

        for repeats, folder_path in folders:
            fs = self._inspect_folder(folder_path, repeats, all_tags, res_dist, orient_dist)
            folder_stats.append(fs)
            total_images += fs.image_count
            total_effective += fs.image_count * repeats
            total_captioned += fs.caption_count

        total_captionable = sum(fs.image_count for fs in folder_stats)
        caption_coverage = total_captioned / total_captionable if total_captionable > 0 else 0.0
        unique_tags = len(all_tags)
        total_tag_count = sum(all_tags.values())
        avg_tags = total_tag_count / total_captioned if total_captioned > 0 else 0.0
        top_tags_list = all_tags.most_common(self._top_tags)

        if caption_coverage < 0.5:
            warnings.append(f"Low caption coverage: {caption_coverage:.1%}")
        if total_images == 0:
            warnings.append("No images found in any subfolder")

        return DatasetReport(
            folder_count=len(folder_stats),
            image_count=total_images,
            effective_image_count=total_effective,
            caption_coverage=caption_coverage,
            unique_tag_count=unique_tags,
            average_tags_per_caption=avg_tags,
            resolution_distribution=dict(res_dist),
            orientation_distribution=dict(orient_dist),
            top_tags=top_tags_list,
            folder_stats=folder_stats,
            warnings=warnings,
        )

    # ---- internals --------------------------------------------------------

    def _discover_folders(self, root: Path) -> list[tuple[int, Path]]:
        """Discover {repeats}_{name} subfolders."""
        result: list[tuple[int, Path]] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            m = _FOLDER_RE.match(child.name)
            if m:
                result.append((int(m.group(1)), child))
        return result

    def _inspect_folder(
        self,
        folder: Path,
        repeats: int,
        all_tags: Counter[str],
        res_dist: Counter[str],
        orient_dist: Counter[str],
    ) -> FolderStats:
        images = self._list_images(folder)
        captions = self._list_captions(folder)
        image_stems = {p.stem for p in images}
        caption_stems = {p.stem for p in captions}

        missing = sorted(image_stems - caption_stems)
        orphan = sorted(caption_stems - image_stems)
        coverage = (len(image_stems & caption_stems) / len(images)) if images else 0.0

        sampled_images = images
        if self._max_sample and len(images) > self._max_sample:
            sampled_images = images[: self._max_sample]

        for img in sampled_images:
            size = _try_read_image_size(img)
            if size:
                w, h = size
                bucket = _classify_resolution_bucket(w, h)
                res_dist[bucket] += 1
                orient_dist[_classify_orientation(w, h)] += 1

        for cap_path in captions:
            text = _read_caption(cap_path)
            if text:
                for tag in _extract_tags(text):
                    all_tags[tag] += 1

        return FolderStats(
            name=folder.name,
            repeats=repeats,
            image_count=len(images),
            caption_count=len(image_stems & caption_stems),
            caption_coverage=coverage,
            missing_captions=missing,
            orphan_captions=orphan,
        )

    def _list_images(self, folder: Path) -> list[Path]:
        return sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _list_captions(self, folder: Path) -> list[Path]:
        return sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() == self._caption_ext
        )

    def _empty_report(self, warning: str) -> DatasetReport:
        return DatasetReport(
            folder_count=0,
            image_count=0,
            effective_image_count=0,
            caption_coverage=0.0,
            unique_tag_count=0,
            average_tags_per_caption=0.0,
            resolution_distribution={},
            orientation_distribution={},
            top_tags=[],
            folder_stats=[],
            warnings=[warning],
        )

