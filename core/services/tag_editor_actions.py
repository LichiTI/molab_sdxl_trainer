# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Pure caption/tag transformations for TagEditor batch actions."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Dict, List, Sequence

from core.services.tag_editor_common import dedupe_tags, join_tags, split_tags

TokenCounter = Callable[[str], tuple[int, str]]


DESTRUCTIVE_ACTIONS = frozenset(
    {
        "replace_tags",
        "remove_tags",
        "search_replace_tags",
        "search_replace_caption",
        "set_caption",
        "truncate_tags_by_token_count",
    }
)


def action_is_destructive(action: str) -> bool:
    return str(action or "").strip().lower() in DESTRUCTIVE_ACTIONS


def search_replace_tags(
    tags: Sequence[str],
    *,
    search_text: str,
    replace_text: str,
    use_regex: bool,
) -> List[str]:
    if not search_text:
        return list(tags)
    results: List[str] = []
    for tag in tags:
        if use_regex:
            try:
                results.append(re.sub(search_text, replace_text, tag))
            except re.error:
                return list(tags)
        else:
            results.append(tag.replace(search_text, replace_text))
    return results


def apply_caption_action(
    caption: str,
    *,
    action: str,
    params: Dict[str, Any],
    tag_counts: Counter[str],
    token_counter: TokenCounter,
) -> str:
    current_tags = split_tags(caption)
    action_name = str(action or "").strip().lower()
    if action_name == "set_caption":
        return str(params.get("caption", "") or "").strip()
    if action_name == "append_tags":
        return join_tags(current_tags + split_tags(str(params.get("tags", "") or "")))
    if action_name == "prepend_tags":
        return join_tags(split_tags(str(params.get("tags", "") or "")) + current_tags)
    if action_name == "replace_tags":
        search_tags = split_tags(str(params.get("search_tags", "") or ""))
        replace_tags = split_tags(str(params.get("replace_tags", "") or ""))
        tags_to_remove: set[str] = set()
        mapping: Dict[str, str] = {}
        for idx, search in enumerate(search_tags):
            if idx < len(replace_tags):
                replacement = replace_tags[idx]
                if replacement:
                    mapping[search.lower()] = replacement
                else:
                    tags_to_remove.add(search.lower())
            else:
                tags_to_remove.add(search.lower())
        out: List[str] = []
        for tag in current_tags:
            lowered = tag.lower()
            if lowered in tags_to_remove:
                continue
            out.append(mapping.get(lowered, tag))
        if len(replace_tags) > len(search_tags):
            out.extend(replace_tags[len(search_tags) :])
        return join_tags(out)
    if action_name == "remove_tags":
        remove = {tag.lower() for tag in split_tags(str(params.get("tags", "") or ""))}
        return join_tags([tag for tag in current_tags if tag.lower() not in remove])
    if action_name == "search_replace_tags":
        return join_tags(
            search_replace_tags(
                current_tags,
                search_text=str(params.get("search_text", "") or ""),
                replace_text=str(params.get("replace_text", "") or ""),
                use_regex=bool(params.get("use_regex", False)),
            )
        )
    if action_name == "search_replace_caption":
        text = caption
        search_text = str(params.get("search_text", "") or "")
        replace_text = str(params.get("replace_text", "") or "")
        if not search_text:
            return text
        if bool(params.get("use_regex", False)):
            try:
                return re.sub(search_text, replace_text, text)
            except re.error:
                return text
        return text.replace(search_text, replace_text)
    if action_name == "sort_tags":
        mode = str(params.get("sort_by", "alpha") or "alpha").strip().lower()
        reverse = str(params.get("sort_order", "asc") or "asc").strip().lower() == "desc"
        if mode == "frequency":
            sorted_tags = sorted(current_tags, key=lambda tag: (tag_counts.get(tag, 0), tag.lower()), reverse=reverse)
        elif mode == "length":
            sorted_tags = sorted(current_tags, key=lambda tag: (len(tag), tag.lower()), reverse=reverse)
        else:
            sorted_tags = sorted(current_tags, key=str.lower, reverse=reverse)
        return join_tags(sorted_tags)
    if action_name == "dedupe_tags":
        return join_tags(dedupe_tags(current_tags))
    if action_name == "truncate_tags_by_token_count":
        max_token_count = int(params.get("max_token_count", 75) or 75)
        truncated: List[str] = []
        for tag in current_tags:
            candidate = join_tags(truncated + [tag])
            token_count, _ = token_counter(candidate)
            if token_count <= max_token_count:
                truncated.append(tag)
            else:
                break
        return join_tags(truncated)
    raise ValueError(f"Unsupported batch action: {action}")
