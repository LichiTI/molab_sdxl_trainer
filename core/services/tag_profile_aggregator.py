# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Cross-dataset tag profile aggregation and trend analysis."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence


def aggregate_tag_profile(
    directories: Sequence[str],
    *,
    top_k: int = 50,
    rare_threshold: int = 3,
    tag_editor: Any = None,
) -> Dict[str, Any]:
    """Aggregate tag frequency and trends across multiple datasets.

    Args:
        directories: list of dataset root paths.
        top_k: number of top tags to return.
        rare_threshold: threshold below which a tag is considered rare.
        tag_editor: optional injected tag editor service.

    Returns:
        dict with ``total_images``, ``total_unique_tags``, ``top_tags``,
        ``rare_tags``, ``untagged_images``, ``tag_length_distribution``.
    """
    if tag_editor is None:
        from core.services.tageditor_service_locator import tag_editor_service
        tag_editor = tag_editor_service()

    all_tags: Counter[str] = Counter()
    total_images = 0
    untagged_images: List[str] = []
    tag_counts_per_image: List[int] = []

    for directory in directories:
        items = tag_editor._scan_dataset(
            Path(directory),
            recursive=True,
            load_caption_from_filename=False,
        )
        for item in items:
            total_images += 1
            tags = item.tags
            if not tags:
                untagged_images.append(item.relative_path)
            tag_counts_per_image.append(len(tags))
            for tag in tags:
                normalized = tag.strip().lower()
                if normalized:
                    all_tags[normalized] += 1

    total_unique = len(all_tags)
    top_tags = [
        {"tag": tag, "count": count, "frequency": round(count / total_images, 4) if total_images else 0}
        for tag, count in all_tags.most_common(top_k)
    ]
    rare_tags = [
        {"tag": tag, "count": count}
        for tag, count in all_tags.items()
        if count <= rare_threshold
    ]

    # Tag length distribution
    distribution: Dict[str, int] = {}
    if tag_counts_per_image:
        max_count = max(tag_counts_per_image)
        for i in range(max_count + 1):
            distribution[str(i)] = tag_counts_per_image.count(i)

    return {
        "total_images": total_images,
        "total_unique_tags": total_unique,
        "top_tags": top_tags,
        "rare_tags": rare_tags,
        "untagged_images": untagged_images,
        "tag_length_distribution": distribution,
    }
