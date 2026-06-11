"""Thin adapter helpers for tag-editor file mutation routes."""

from __future__ import annotations

from typing import Any, Callable


def build_tageditor_move_files_request(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_path": str(params.get("dir", "") or params.get("path", "") or ""),
        "target_dir": str(params.get("target_dir", "") or params.get("destination", "") or ""),
        "image_paths": list(params.get("image_paths", []) or []),
        "keep_relative_structure": bool(params.get("keep_relative_structure", True)),
    }


def build_tageditor_delete_files_request(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_path": str(params.get("dir", "") or params.get("path", "") or ""),
        "image_paths": list(params.get("image_paths", []) or []),
    }


def move_tageditor_files(
    params: dict[str, Any],
    *,
    tag_editor: Any,
    invalidate_cache: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    request = build_tageditor_move_files_request(params)
    if not request["dataset_path"] or not request["target_dir"]:
        raise ValueError("Missing dir or target_dir")
    result = tag_editor.move_files(
        directory=request["dataset_path"],
        image_paths=request["image_paths"],
        target_dir=request["target_dir"],
        keep_relative_structure=request["keep_relative_structure"],
    )
    if invalidate_cache:
        invalidate_cache(request["dataset_path"])
    return result


def delete_tageditor_files(
    params: dict[str, Any],
    *,
    tag_editor: Any,
    invalidate_cache: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    request = build_tageditor_delete_files_request(params)
    if not request["dataset_path"]:
        raise ValueError("Missing dir")
    result = tag_editor.delete_files(
        directory=request["dataset_path"],
        image_paths=request["image_paths"],
    )
    if invalidate_cache:
        invalidate_cache(request["dataset_path"])
    return result
