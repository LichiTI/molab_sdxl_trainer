"""Queue selection helpers for aesthetic labeling samples."""

from __future__ import annotations

import math
from typing import Any, Iterable

from backend.core.services.aesthetic_labeling_events import canonical_status


def list_visible_samples(
    samples: Iterable[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    *,
    status: str = "all",
    source: str = "",
    order: str = "desc",
    after_id: int = 0,
) -> list[dict[str, Any]]:
    visible = []
    for sample in samples:
        if not _is_available(sample):
            continue
        sample_id = int(sample.get("sample_id") or 0)
        if after_id > 0 and sample_id <= after_id:
            continue
        if source and not _source_matches(sample, source):
            continue
        annotation = annotations.get(str(sample_id))
        if status in {"labeled", "skipped", "unreviewed"} and canonical_status(annotation) != status:
            continue
        visible.append(sample)
    reverse = str(order).lower() != "asc"
    visible.sort(key=lambda item: int(item.get("sample_id") or 0), reverse=reverse)
    return visible


def paginate_samples(items: list[dict[str, Any]], *, page: int = 1, size: int = 24) -> dict[str, Any]:
    total = len(items)
    page = max(1, int(page or 1))
    size = max(1, min(int(size or 24), 500))
    pages = math.ceil(total / size) if total else 0
    offset = (page - 1) * size
    page_items = items[offset:offset + size]
    for index, item in enumerate(page_items, start=offset + 1):
        item["sample_seq"] = index
        item["sample_total"] = total
    return {"items": page_items, "page": page, "size": size, "pages": pages, "total": total}


def pick_next_sample(
    samples: Iterable[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    *,
    avoid_ids: set[int] | None = None,
    after_id: int = 0,
) -> dict[str, Any] | None:
    avoid = avoid_ids or set()
    unreviewed = list_visible_samples(
        samples,
        annotations,
        status="unreviewed",
        order="asc",
        after_id=after_id,
    )
    for sample in unreviewed:
        if int(sample.get("sample_id") or 0) not in avoid:
            return sample
    fallback = list_visible_samples(samples, annotations, status="all", order="asc", after_id=after_id)
    for sample in fallback:
        if int(sample.get("sample_id") or 0) not in avoid:
            return sample
    return None


def pick_last_reviewed(
    samples: Iterable[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    *,
    status: str = "labeled",
) -> dict[str, Any] | None:
    by_id = {str(int(sample.get("sample_id") or 0)): sample for sample in samples if _is_available(sample)}
    candidates: list[tuple[str, dict[str, Any]]] = []
    for sample_id, annotation in annotations.items():
        if canonical_status(annotation) != status:
            continue
        sample = by_id.get(str(sample_id))
        if not sample:
            continue
        candidates.append((str(annotation.get("updated_at") or ""), sample))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1] if candidates else None


def _source_matches(sample: dict[str, Any], source: str) -> bool:
    return source in {
        str(sample.get("source") or ""),
        str(sample.get("source_path") or ""),
        str(sample.get("source_id") or ""),
    }


def _is_available(sample: dict[str, Any]) -> bool:
    return str(sample.get("storage_state") or "available") == "available"

