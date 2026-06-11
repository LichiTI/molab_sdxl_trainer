"""Small dataset preview helpers for WebUI compatibility routes."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Any

from backend.core.services.native_module_loader import load_lulynx_native, native_with_entrypoints


DATASET_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
MASKED_LOSS_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def load_native_dataset_preview_api() -> Any:
    return load_lulynx_native()


load_native_dataset_preview_api.cache_clear = load_lulynx_native.cache_clear  # type: ignore[attr-defined]


def native_dataset_preview_api() -> Any:
    return native_with_entrypoints("list_image_files")


def list_dataset_image_preview(folder: str, *, limit: int = 6) -> dict[str, Any]:
    """Return the legacy dataset image preview payload used by the WebUI."""

    if not folder:
        return {"images": [], "total": 0, "first_tag": ""}

    dataset_dir = Path(folder)
    if not dataset_dir.is_dir():
        return {"images": [], "total": 0, "first_tag": ""}

    all_images = list_dataset_preview_images(dataset_dir)
    total = len(all_images)
    max_items = max(0, int(limit or 0))
    images = all_images[:max_items] if max_items else []
    first_tag = _read_first_caption_tag(all_images[0]) if all_images else ""
    return {"images": images, "total": total, "first_tag": first_tag}


def list_dataset_preview_images(dataset_dir: Path) -> list[str]:
    native = native_dataset_preview_api()
    if native is not None:
        try:
            images = native.list_image_files(str(dataset_dir), True)
            if isinstance(images, list):
                return [str(path) for path in images]
        except Exception:
            pass
    return [
        str(entry)
        for entry in sorted(dataset_dir.rglob("*"))
        if entry.is_file() and entry.suffix.lower() in DATASET_IMAGE_EXTENSIONS
    ]


def _read_first_caption_tag(image_path: str) -> str:
    first_img = Path(image_path)
    text = ""
    for suffix in (".txt", ".caption"):
        caption_path = first_img.with_suffix(suffix)
        if caption_path.is_file():
            text = caption_path.read_text(encoding="utf-8", errors="replace").strip()
            break
    if not text:
        return ""
    return re.split(r"[,\n]", text, maxsplit=1)[0].strip()


def audit_masked_loss_images(folder: str, *, sample_limit: int = 20) -> dict[str, Any]:
    """Scan dataset images for alpha channel presence."""

    dataset_dir = Path(folder)
    if not folder:
        raise ValueError("Missing path")
    if not dataset_dir.is_dir():
        raise ValueError(f"Directory not found: {folder}")

    total_images = 0
    with_alpha = 0
    without_alpha = 0
    samples: list[dict[str, Any]] = []
    max_samples = max(0, int(sample_limit or 0))

    for img_path in sorted(dataset_dir.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in MASKED_LOSS_IMAGE_EXTENSIONS:
            continue
        total_images += 1
        try:
            from PIL import Image

            with Image.open(img_path) as im:
                has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
                width, height = im.size
            if has_alpha:
                with_alpha += 1
            else:
                without_alpha += 1
            if len(samples) < max_samples:
                samples.append(
                    {
                        "file": str(img_path),
                        "name": img_path.name,
                        "has_alpha": has_alpha,
                        "width": width,
                        "height": height,
                    }
                )
        except Exception:
            without_alpha += 1

    return {
        "total_images": total_images,
        "with_alpha": with_alpha,
        "without_alpha": without_alpha,
        "samples": samples,
    }


def list_sample_images(sample_dir: str) -> dict[str, Any]:
    """List generated sample images sorted by modification time descending."""

    root = Path(sample_dir)
    if not root.is_dir():
        return {"images": []}

    images: list[dict[str, Any]] = []
    for img_path in root.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in DATASET_IMAGE_EXTENSIONS:
            continue
        stat = img_path.stat()
        images.append({"name": img_path.name, "mtime": stat.st_mtime, "path": str(img_path)})
    images.sort(key=lambda item: item["mtime"], reverse=True)
    return {"images": images}


def build_resized_image_file(path: str, *, allowed_dirs: list[Path], max_size: int = 512) -> dict[str, Any]:
    """Validate a local image path and return an in-memory thumbnail payload."""

    if not path:
        raise ValueError("Missing path")
    target = Path(path).resolve()
    resolved_allowed = [Path(directory).resolve() for directory in allowed_dirs]
    if not any(_is_within(target, directory) for directory in resolved_allowed):
        raise PermissionError("Path not in allowed directory (output/, train/, data/, models/, sd-models/)")
    if not target.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        from PIL import Image

        with Image.open(target) as im:
            im.thumbnail((int(max_size), int(max_size)))
            buffer = BytesIO()
            fmt = im.format or "PNG"
            if fmt.upper() == "JPEG":
                im = im.convert("RGB")
            im.save(buffer, format=fmt)
            buffer.seek(0)
        media = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "WEBP": "image/webp",
            "BMP": "image/bmp",
        }.get(fmt.upper(), "image/png")
        return {"buffer": buffer, "media_type": media}
    except Exception as exc:
        raise RuntimeError(f"Failed to resize: {exc}") from exc


def resolve_sample_file(sample_dir: str, name: str) -> Path:
    """Resolve a generated sample image path without allowing traversal."""

    if not name or "/" in name or "\\" in name or ".." in name:
        raise ValueError("Invalid sample name")
    root = Path(sample_dir).resolve()
    file_path = (root / name).resolve()
    if not _is_within(file_path, root):
        raise PermissionError("Path traversal detected")
    if not file_path.is_file():
        raise FileNotFoundError(f"Sample not found: {name}")
    return file_path


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return path == parent


# Backwards-compatible private names used by older tests/patch points.
_load_native_dataset_preview_api = load_native_dataset_preview_api
_native_dataset_preview_api = native_dataset_preview_api
