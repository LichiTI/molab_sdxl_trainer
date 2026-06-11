"""Small route adapter helpers for tag-editor utility endpoints."""

from __future__ import annotations

from typing import Any

from backend.core.services.tageditor_service_locator import tag_editor_service


def _directory(params: dict[str, Any]) -> str:
    return str(params.get("dir", "") or params.get("path", "") or "")


def get_tageditor_sidebar_stats(params: dict[str, Any], *, tag_editor: Any) -> dict[str, Any]:
    directory = _directory(params)
    if not directory:
        raise ValueError("Missing dir parameter")
    return tag_editor.get_sidebar_stats(
        directory,
        recursive=bool(params.get("recursive", True)),
        caption_extension=str(params.get("caption_extension", "") or ""),
        load_caption_from_filename=bool(params.get("load_caption_from_filename", False)),
        filename_regex=str(params.get("filename_regex", "") or ""),
        filename_joiner=str(params.get("filename_joiner", ", ") or ", "),
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
        rare_tags_limit=int(params.get("rare_tags_limit", 20) or 20),
        token_limit=int(params.get("token_limit", 75) or 75),
    )


def tokenize_tageditor_caption(params: dict[str, Any], *, tag_editor: Any) -> dict[str, Any]:
    return tag_editor.analyze_caption_tokens(
        str(params.get("caption", "") or ""),
        max_token_count=int(params.get("max_token_count", 75) or 75),
    )


def build_tageditor_status_payload(*, tag_editor: Any) -> dict[str, Any]:
    return {
        "status": "cleanroom",
        "mode": "integrated",
        "capabilities": tag_editor.capabilities(),
    }


def build_tageditor_status_route_payload() -> dict[str, Any]:
    return build_tageditor_status_payload(tag_editor=tag_editor_service())


def create_tageditor_history_snapshot(params: dict[str, Any], *, tag_editor: Any) -> dict[str, Any]:
    directory = _directory(params)
    if not directory:
        raise ValueError("Missing dir parameter")
    return tag_editor.create_history_snapshot(
        directory,
        image_paths=list(params.get("image_paths", []) or []),
        recursive=bool(params.get("recursive", True)),
        caption_extension=str(params.get("caption_extension", "") or ""),
        filter_payload=dict(params.get("filters", {}) or {}),
        label=str(params.get("label", "") or ""),
        operation=str(params.get("operation", "manual_snapshot") or "manual_snapshot"),
    )


def list_tageditor_history_snapshots(params: dict[str, Any], *, tag_editor: Any) -> dict[str, Any]:
    directory = _directory(params)
    if not directory:
        raise ValueError("Missing dir parameter")
    return tag_editor.list_history_snapshots(directory, limit=int(params.get("limit", 50) or 50))


def restore_tageditor_history_snapshot(params: dict[str, Any], *, tag_editor: Any) -> dict[str, Any]:
    directory = _directory(params)
    backup_name = str(params.get("backup_name", "") or params.get("archive_name", "") or params.get("name", "") or "")
    if not directory or not backup_name:
        raise ValueError("Missing dir or backup_name")
    return tag_editor.restore_history_snapshot(
        directory,
        backup_name=backup_name,
        image_paths=list(params.get("image_paths", []) or []),
    )
