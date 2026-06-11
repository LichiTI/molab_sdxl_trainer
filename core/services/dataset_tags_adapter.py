"""Thin adapter helpers for legacy dataset tag endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def list_dataset_tags(directory: str, *, tag_editor: Any) -> dict[str, Any]:
    dataset_path = str(directory or "")
    if not dataset_path:
        raise ValueError("Missing dir parameter")
    payload = tag_editor.load_dataset(
        dataset_path,
        recursive=True,
        limit=1_000_000,
        offset=0,
    )
    items = [
        {"image": item["relative_path"], "tags": item["caption"]}
        for item in payload.get("items", [])
    ]
    items.sort(key=lambda entry: entry["image"])
    return {"dir": dataset_path, "items": items}


def build_dataset_tag_save_request(params: dict[str, Any]) -> dict[str, Any]:
    dir_param = str(params.get("dir", "") or "")
    image_param = str(params.get("image", "") or "")
    image_path = ""
    if dir_param and image_param:
        image_path = str(Path(dir_param) / image_param)
    else:
        image_path = str(params.get("image_path", "") or params.get("path", "") or "")
    return {
        "image_path": image_path,
        "caption": params.get("tags", ""),
        "caption_extension": str(params.get("caption_extension", "") or ""),
        "dataset_dir": str(params.get("dir", "") or params.get("dataset_dir", "") or ""),
        "create_backup": bool(params.get("create_backup", False)),
        "backup_label": str(params.get("backup_label", "") or "dataset_tags_save"),
    }


def save_dataset_tag_caption(params: dict[str, Any], *, tag_editor: Any) -> dict[str, Any]:
    request = build_dataset_tag_save_request(params)
    if not request["image_path"]:
        raise ValueError("Missing dir+image or image_path")
    image = Path(request["image_path"])
    if not image.is_file():
        raise FileNotFoundError(f"Image not found: {image}")
    return tag_editor.save_caption(
        image_path=str(image),
        caption=request["caption"],
        caption_extension=request["caption_extension"],
        dataset_dir=request["dataset_dir"],
        create_backup=request["create_backup"],
        backup_label=request["backup_label"],
    )
