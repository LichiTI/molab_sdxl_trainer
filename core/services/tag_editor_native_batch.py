# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Native fast path wrapper for TagEditor batch caption transforms."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Callable, Dict, Optional, Sequence

from core.services.tag_editor_common import DatasetCaptionItem, native_tag_editor_batch_action_api


NativeApiLoader = Callable[[], Any]


def apply_batch_action_native(
    items: Sequence[DatasetCaptionItem],
    *,
    action: str,
    params: Dict[str, Any],
    tag_counts: Counter[str],
    sample_limit: int,
    include_changes: bool,
    native_api_loader: NativeApiLoader | None = None,
) -> Optional[Dict[str, Any]]:
    action_name = str(action or "").strip().lower()
    if action_name == "truncate_tags_by_token_count":
        return None
    if action_name in {"search_replace_tags", "search_replace_caption"} and bool(params.get("use_regex", False)):
        return None
    native = (native_api_loader or native_tag_editor_batch_action_api)()
    if native is None:
        return None
    records = [
        {
            "image_path": str(item.image_path),
            "caption_path": str(item.caption_path),
            "caption": item.caption_text,
        }
        for item in items
    ]
    try:
        result = native.apply_tag_editor_batch_action(
            json.dumps(records, ensure_ascii=False),
            action_name,
            json.dumps(params, ensure_ascii=False),
            json.dumps(dict(tag_counts), ensure_ascii=False),
            int(sample_limit),
            bool(include_changes),
        )
    except Exception:
        return None
    return result if isinstance(result, dict) else None


def preview_batch_action_for_paths_native(
    *,
    dataset_dir: str,
    image_paths: Sequence[str],
    caption_extension: str,
    action: str,
    params: Dict[str, Any],
    sample_limit: int,
    native_api_loader: NativeApiLoader | None = None,
) -> Optional[Dict[str, Any]]:
    action_name = str(action or "").strip().lower()
    if not image_paths or action_name == "truncate_tags_by_token_count":
        return None
    if action_name in {"search_replace_tags", "search_replace_caption"} and bool(params.get("use_regex", False)):
        return None
    native = (native_api_loader or native_tag_editor_batch_action_api)()
    if native is None or not hasattr(native, "preview_tag_editor_batch_action_for_paths"):
        return None
    try:
        result = native.preview_tag_editor_batch_action_for_paths(
            str(dataset_dir),
            json.dumps([str(path) for path in image_paths], ensure_ascii=False),
            str(caption_extension or ""),
            action_name,
            json.dumps(params, ensure_ascii=False),
            int(sample_limit),
        )
    except Exception:
        return None
    return result if isinstance(result, dict) else None
