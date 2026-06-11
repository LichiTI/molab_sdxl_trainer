"""Legacy dataset analysis payload builder for compatibility routes."""

from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Any

from backend.core.services.native_module_loader import load_lulynx_native


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
REPEAT_PREFIX_WARNING = (
    "检测到该目录下没有 N_ 命名的重复次数文件夹。通常格式例如：3_ABC、10_style，"
    "表示该文件夹素材重复对应次数。如果这是你想要的结构，可以继续；否则请检查是否选错了数据集根目录。"
)


def native_dataset_analysis_disabled() -> bool:
    return str(os.environ.get("LULYNX_DISABLE_NATIVE_DATASET_ANALYSIS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_native_dataset_analysis_api() -> Any:
    return load_lulynx_native()


load_native_dataset_analysis_api.cache_clear = load_lulynx_native.cache_clear  # type: ignore[attr-defined]


def native_dataset_analysis_api() -> Any:
    if native_dataset_analysis_disabled():
        return None
    native = load_native_dataset_analysis_api()
    if not hasattr(native, "analyze_dataset_captions"):
        return None
    return native


def analyze_dataset_payload(params: dict[str, Any]) -> dict[str, Any]:
    dir_path = str(params.get("dir", "") or params.get("path", "") or "")
    caption_ext = str(params.get("caption_extension", ".txt") or ".txt")
    if not caption_ext.startswith("."):
        caption_ext = "." + caption_ext
    top_tags_limit = int(params.get("top_tags", 50))
    if not dir_path:
        raise ValueError("Missing dir parameter")
    dataset_dir = Path(dir_path)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    native_summary = analyze_dataset_captions_native(dataset_dir, caption_ext, top_tags_limit)
    if native_summary is not None:
        all_images = [Path(path) for path in native_summary.get("image_paths", [])]
        captioned = int(native_summary.get("captioned_images", 0) or 0)
        uncaptioned = int(native_summary.get("uncaptioned_images", 0) or 0)
        empty_captions = int(native_summary.get("empty_caption_count", 0) or 0)
        top_tags_list = list(native_summary.get("top_tags", []) or [])[:top_tags_limit]
        has_root_images = bool(native_summary.get("has_root_images", False))
    else:
        all_images = [entry for entry in dataset_dir.rglob("*") if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS]
        captioned = 0
        uncaptioned = 0
        empty_captions = 0
        tag_counts: dict[str, int] = {}
        for image_path in all_images:
            caption_path = image_path.with_suffix(caption_ext)
            has_caption = False
            if caption_path.is_file():
                tags_text = caption_path.read_text(encoding="utf-8", errors="replace").strip()
                if tags_text:
                    has_caption = True
                    for tag in tags_text.split(","):
                        value = tag.strip()
                        if value:
                            tag_counts[value] = tag_counts.get(value, 0) + 1
                else:
                    empty_captions += 1
            if has_caption:
                captioned += 1
            else:
                uncaptioned += 1
        top_tags = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)[:top_tags_limit]
        top_tags_list = [{"tag": tag, "count": count} for tag, count in top_tags]
        has_root_images = any(image_path.parent == dataset_dir for image_path in all_images)
    total_images = len(all_images)
    if native_summary is not None and bool(native_summary.get("native_image_header_probe_complete", False)):
        broken_images = int(native_summary.get("broken_image_count", 0) or 0)
        alpha_capable = int(native_summary.get("alpha_capable_image_count", 0) or 0)
        resolution_list = [
            {"resolution": str(item.get("resolution") or ""), "count": int(item.get("count", 0) or 0)}
            for item in list(native_summary.get("resolution_distribution", []) or [])
            if item.get("resolution")
        ]
    else:
        broken_images = 0
        alpha_capable = 0
        resolution_counts: dict[str, int] = {}

        for image_path in all_images:
            try:
                from PIL import Image

                with Image.open(image_path) as im:
                    width, height = im.size
                    resolution_counts[f"{width}x{height}"] = resolution_counts.get(f"{width}x{height}", 0) + 1
                    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                        alpha_capable += 1
            except Exception:
                broken_images += 1

        resolution_distribution = sorted(resolution_counts.items(), key=lambda item: item[1], reverse=True)
        resolution_list = [{"resolution": resolution, "count": count} for resolution, count in resolution_distribution]
    top_resolution_list = [
        {"name": str(item.get("resolution") or ""), "count": int(item.get("count", 0) or 0)}
        for item in resolution_list[:20]
    ]

    folders, prefixed_folders, plain_folders = _analyze_repeat_folders(dataset_dir)
    repeat_prefix_warning = ""
    if not prefixed_folders and (plain_folders or has_root_images):
        repeat_prefix_warning = REPEAT_PREFIX_WARNING

    caption_coverage = captioned / total_images if total_images > 0 else 0.0
    effective_image_count = sum(folder["image_count"] * folder["repeats"] for folder in folders) if folders else total_images
    return {
        "total_images": total_images,
        "captioned_images": captioned,
        "uncaptioned_images": uncaptioned,
        "top_tags": top_tags_list,
        "resolution_distribution": resolution_list,
        "summary": {
            "image_count": total_images,
            "effective_image_count": effective_image_count,
            "caption_count": captioned,
            "caption_coverage": round(caption_coverage, 4),
            "images_without_caption_count": uncaptioned,
            "empty_caption_count": empty_captions,
            "broken_image_count": broken_images,
            "alpha_capable_image_count": alpha_capable,
        },
        "folders": folders,
        "top_resolutions": top_resolution_list,
        "repeat_prefix": {
            "has_prefixed_folders": bool(prefixed_folders),
            "prefixed_folders": prefixed_folders,
            "plain_folders": plain_folders,
            "has_root_images": has_root_images,
            "warning": repeat_prefix_warning,
        },
    }


def _analyze_repeat_folders(dataset_dir: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    folders: list[dict[str, Any]] = []
    prefixed_folders: list[str] = []
    plain_folders: list[str] = []
    for subdir in sorted(dataset_dir.iterdir()):
        if not subdir.is_dir():
            continue
        match = re.match(r"^(\d+)_(.+)", subdir.name)
        repeats = 1
        folder_name = subdir.name
        if match:
            repeats = int(match.group(1))
            folder_name = match.group(2)
            prefixed_folders.append(subdir.name)
        else:
            plain_folders.append(subdir.name)
        image_count = sum(1 for file_path in subdir.iterdir() if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS)
        folders.append(
            {
                "name": folder_name,
                "path": str(subdir),
                "repeats": repeats,
                "image_count": image_count,
            }
        )
    return folders, prefixed_folders, plain_folders


def analyze_dataset_captions_native(dataset_dir: Path, caption_ext: str, top_tags_limit: int) -> dict[str, Any] | None:
    native = native_dataset_analysis_api()
    if native is None:
        return None
    try:
        result = native.analyze_dataset_captions(str(dataset_dir), str(caption_ext), int(top_tags_limit))
    except Exception:
        return None
    return result if isinstance(result, dict) else None


# Backwards-compatible private names used by older tests/patch points.
_load_native_dataset_analysis_api = load_native_dataset_analysis_api
_native_dataset_analysis_api = native_dataset_analysis_api
_native_dataset_analysis_disabled = native_dataset_analysis_disabled
