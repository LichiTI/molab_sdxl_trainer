"""Pure dataset view helpers for TagEditorService."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List, Sequence

from core.services.tag_editor_common import DatasetCaptionItem


TokenCounter = Callable[[str], tuple[int, str]]


def match_tag_logic(tags_lower: set[str], filter_tags: set[str], logic: str) -> bool:
    if not filter_tags:
        return True
    if str(logic or "OR").strip().upper() == "AND":
        return filter_tags.issubset(tags_lower)
    return bool(tags_lower & filter_tags)


def filter_items(
    items: Sequence[DatasetCaptionItem],
    *,
    filename_query: str,
    caption_query: str,
    positive_tags: Sequence[str],
    positive_logic: str,
    negative_tags: Sequence[str],
    negative_logic: str,
    selected_paths: Sequence[str],
    selection_mode: str,
    has_caption: str,
) -> List[DatasetCaptionItem]:
    positive = {tag.strip().lower() for tag in positive_tags if tag and tag.strip()}
    negative = {tag.strip().lower() for tag in negative_tags if tag and tag.strip()}
    selected = {str(path).replace("\\", "/").lower() for path in selected_paths if path}
    filename_query = str(filename_query or "").strip().lower()
    caption_query = str(caption_query or "").strip().lower()
    results: List[DatasetCaptionItem] = []
    for item in items:
        filename_lower = item.relative_path.lower()
        caption_lower = item.caption_text.lower()
        tags_lower = {tag.lower() for tag in item.tags}
        if filename_query and filename_query not in filename_lower:
            continue
        if caption_query and caption_query not in caption_lower:
            continue
        if has_caption == "yes" and not item.caption_text:
            continue
        if has_caption == "no" and item.caption_text:
            continue
        if positive and not match_tag_logic(tags_lower, positive, positive_logic):
            continue
        if negative and match_tag_logic(tags_lower, negative, negative_logic):
            continue
        if selected:
            selected_match = filename_lower in selected or str(item.image_path).replace("\\", "/").lower() in selected
            if selection_mode == "inclusive" and not selected_match:
                continue
            if selection_mode == "exclusive" and selected_match:
                continue
        results.append(item)
    return results


def sort_items(items: Sequence[DatasetCaptionItem], *, sort_by: str, sort_order: str) -> List[DatasetCaptionItem]:
    reverse = str(sort_order or "asc").strip().lower() == "desc"
    mode = str(sort_by or "name").strip().lower()
    if mode == "mtime":
        key = lambda item: (item.mtime, item.relative_path.lower())
    elif mode == "tag_count":
        key = lambda item: (len(item.tags), item.relative_path.lower())
    elif mode == "caption_length":
        key = lambda item: (len(item.caption_text), item.relative_path.lower())
    else:
        key = lambda item: item.relative_path.lower()
    return sorted(items, key=key, reverse=reverse)


def serialize_item(item: DatasetCaptionItem, token_counter: TokenCounter) -> Dict[str, Any]:
    token_count, tokenizer_mode = token_counter(item.caption_text)
    return {
        "image_path": str(item.image_path),
        "relative_path": item.relative_path,
        "caption_path": str(item.caption_path),
        "caption": item.caption_text,
        "tags": item.tags,
        "has_caption": bool(item.caption_text),
        "caption_exists": item.caption_exists,
        "caption_source": item.caption_source,
        "mtime": item.mtime,
        "tag_count": len(item.tags),
        "caption_length": len(item.caption_text),
        "token_count": token_count,
        "tokenizer_mode": tokenizer_mode,
    }


def tag_counts(items: Sequence[DatasetCaptionItem]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in items:
        counter.update(item.tags)
    return counter


def common_tags(items: Sequence[DatasetCaptionItem]) -> List[str]:
    if not items:
        return []
    common = {tag.lower(): tag for tag in items[0].tags}
    current = set(common.keys())
    for item in items[1:]:
        current &= {tag.lower() for tag in item.tags}
    return sorted([common.get(lowered, lowered) for lowered in current], key=str.lower)


def build_sidebar_stats(
    items: Sequence[DatasetCaptionItem],
    *,
    top_tags_limit: int,
    rare_tags_limit: int,
    token_limit: int,
    directory: str,
    token_counter: TokenCounter,
) -> Dict[str, Any]:
    counts = tag_counts(items)
    token_counts: List[int] = []
    tokenizer_modes: Counter[str] = Counter()
    caption_lengths: List[int] = []
    tag_lengths: List[int] = []
    for item in items:
        count, mode = token_counter(item.caption_text)
        token_counts.append(count)
        tokenizer_modes.update([mode])
        caption_lengths.append(len(item.caption_text))
        tag_lengths.append(len(item.tags))
    rare_tags = sorted(counts.items(), key=lambda pair: (pair[1], pair[0].lower()))
    avg = lambda values: (sum(values) / len(values)) if values else 0.0
    tokenizer_mode = tokenizer_modes.most_common(1)[0][0] if tokenizer_modes else token_counter("")[1]
    return {
        "directory": directory,
        "filtered_total": len(items),
        "common_tags": common_tags(items),
        "top_tags": [{"tag": tag, "count": count} for tag, count in counts.most_common(max(0, top_tags_limit))],
        "rare_tags": [{"tag": tag, "count": count} for tag, count in rare_tags[: max(0, rare_tags_limit)]],
        "caption_stats": {
            "captioned_count": sum(1 for item in items if item.caption_text),
            "uncaptioned_count": sum(1 for item in items if not item.caption_text),
            "avg_caption_length": avg(caption_lengths),
            "avg_tag_count": avg(tag_lengths),
            "max_tag_count": max(tag_lengths) if tag_lengths else 0,
        },
        "token_stats": {
            "tokenizer_mode": tokenizer_mode,
            "token_limit": int(token_limit),
            "min_token_count": min(token_counts) if token_counts else 0,
            "max_token_count": max(token_counts) if token_counts else 0,
            "avg_token_count": avg(token_counts),
            "over_limit_count": sum(1 for count in token_counts if count > int(token_limit)),
        },
    }
