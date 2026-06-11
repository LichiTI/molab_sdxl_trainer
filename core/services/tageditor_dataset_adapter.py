"""Tag-editor dataset browser route adapter."""

from __future__ import annotations

from typing import Any, Mapping


def build_tageditor_dataset_payload(params: Mapping[str, Any], *, tag_editor_service: Any) -> dict[str, Any]:
    directory = str(params.get("dir", "") or params.get("path", "") or "")
    if not directory:
        raise ValueError("Missing dir parameter")
    return tag_editor_service.load_dataset(
        directory,
        recursive=bool(params.get("recursive", True)),
        caption_extension=str(params.get("caption_extension", "") or ""),
        load_caption_from_filename=bool(params.get("load_caption_from_filename", False)),
        filename_regex=str(params.get("filename_regex", "") or ""),
        filename_joiner=str(params.get("filename_joiner", ", ") or ", "),
        limit=int(params.get("limit", 200) or 200),
        offset=int(params.get("offset", 0) or 0),
        sort_by=str(params.get("sort_by", "name") or "name"),
        sort_order=str(params.get("sort_order", "asc") or "asc"),
        filename_query=str(params.get("filename_query", "") or ""),
        caption_query=str(params.get("caption_query", "") or ""),
        positive_tags=list(params.get("positive_tags", []) or []),
        positive_logic=str(params.get("positive_logic", "OR") or "OR"),
        negative_tags=list(params.get("negative_tags", []) or []),
        negative_logic=str(params.get("negative_logic", "OR") or "OR"),
        selected_paths=list(params.get("selected_paths", []) or []),
        selection_mode=str(params.get("selection_mode", "all") or "all"),
        has_caption=str(params.get("has_caption", "any") or "any"),
        top_tags_limit=int(params.get("top_tags_limit", 50) or 50),
    )
