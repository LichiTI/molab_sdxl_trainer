# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Pre-training dataset analysis pass.

Produces a DatasetReport with image count, caption coverage, tag frequency,
resolution/bucket distribution, and caption length statistics.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DatasetReport:
    total_images: int = 0
    captioned_images: int = 0
    caption_coverage: float = 0.0
    missing_captions: List[str] = field(default_factory=list)
    tag_frequency: Dict[str, int] = field(default_factory=dict)
    top_tags: List[Tuple[str, int]] = field(default_factory=list)
    resolution_distribution: Dict[str, int] = field(default_factory=dict)
    bucket_distribution: Dict[str, int] = field(default_factory=dict)
    avg_tags_per_image: float = 0.0
    caption_length_stats: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_images": self.total_images,
            "captioned_images": self.captioned_images,
            "caption_coverage": round(self.caption_coverage, 4),
            "missing_captions_count": len(self.missing_captions),
            "top_tags": self.top_tags[:20],
            "unique_tags": len(self.tag_frequency),
            "avg_tags_per_image": round(self.avg_tags_per_image, 2),
            "resolution_distribution": self.resolution_distribution,
            "bucket_distribution": self.bucket_distribution,
            "caption_length_stats": self.caption_length_stats,
        }

    def summary_lines(self) -> List[str]:
        lines = [
            f"  Images: {self.total_images}",
            f"  Caption coverage: {self.captioned_images}/{self.total_images}"
            f" ({self.caption_coverage:.1%})",
            f"  Unique tags: {len(self.tag_frequency)}",
            f"  Avg tags/image: {self.avg_tags_per_image:.1f}",
        ]
        if self.top_tags:
            top5 = ", ".join(f"{t}({c})" for t, c in self.top_tags[:5])
            lines.append(f"  Top tags: {top5}")
        if self.bucket_distribution:
            buckets = len(self.bucket_distribution)
            lines.append(f"  Buckets: {buckets} distinct resolutions")
        if self.caption_length_stats:
            s = self.caption_length_stats
            lines.append(
                f"  Caption length: min={s.get('min', 0):.0f} "
                f"max={s.get('max', 0):.0f} "
                f"mean={s.get('mean', 0):.1f} words"
            )
        return lines


class DatasetAnalyzer:
    """Analyze a CaptionDataset's samples for statistics.

    Operates entirely on the already-scanned ``dataset.samples`` list
    and caption sidecar files — no image I/O.
    """

    _TOP_N = 50

    def __init__(self, dataset: Any, caption_extension: str = ".txt"):
        self._samples = dataset.samples
        self._caption_ext = caption_extension
        try:
            from .caption_sidecar import json_caption_to_training_text
            self._json_to_text = json_caption_to_training_text
        except ImportError:
            self._json_to_text = None

    def analyze(self) -> DatasetReport:
        report = DatasetReport()
        report.total_images = len(self._samples)
        if report.total_images == 0:
            return report

        tag_counter: Counter = Counter()
        word_counts: List[int] = []
        total_tags = 0

        for sample in self._samples:
            w, h = sample.original_size
            report.resolution_distribution[f"{w}x{h}"] = (
                report.resolution_distribution.get(f"{w}x{h}", 0) + 1
            )

            tw, th = sample.target_size
            report.bucket_distribution[f"{tw}x{th}"] = (
                report.bucket_distribution.get(f"{tw}x{th}", 0) + 1
            )

            if sample.caption_path is None:
                report.missing_captions.append(sample.image_path)
                continue

            report.captioned_images += 1

            try:
                raw = Path(sample.caption_path).read_text(encoding="utf-8").strip()
            except Exception:
                report.missing_captions.append(sample.image_path)
                report.captioned_images -= 1
                continue

            text = raw
            if self._json_to_text is not None:
                try:
                    text = self._json_to_text(raw)
                except Exception:
                    pass

            tags = [t.strip() for t in text.split(",") if t.strip()]
            tag_counter.update(tags)
            total_tags += len(tags)
            word_counts.append(len(text.split()))

        report.caption_coverage = (
            report.captioned_images / report.total_images
            if report.total_images > 0
            else 0.0
        )
        report.tag_frequency = dict(tag_counter)
        report.top_tags = tag_counter.most_common(self._TOP_N)
        report.avg_tags_per_image = (
            total_tags / report.captioned_images
            if report.captioned_images > 0
            else 0.0
        )

        if word_counts:
            report.caption_length_stats = {
                "min": float(min(word_counts)),
                "max": float(max(word_counts)),
                "mean": float(statistics.mean(word_counts)),
                "median": float(statistics.median(word_counts)),
            }

        return report
