from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_image_files(root: Path, *, recursive: bool = True) -> list[Path]:
    if is_image_file(root):
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Dataset path must be a folder or image file: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in iterator if is_image_file(path))


def resolve_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    lower_parts = [part.lower() for part in parts]
    if "images" in lower_parts:
        idx = lower_parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        number = 0.0
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _normalize_box_coordinates(box: dict[str, Any]) -> tuple[float, float, float, float]:
    if {"x1", "y1", "x2", "y2"}.issubset(box):
        x1 = _clamp01(box.get("x1"))
        y1 = _clamp01(box.get("y1"))
        x2 = _clamp01(box.get("x2"))
        y2 = _clamp01(box.get("y2"))
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        return left, top, right, bottom

    xc = _clamp01(box.get("x_center"))
    yc = _clamp01(box.get("y_center"))
    width = _clamp01(box.get("width"))
    height = _clamp01(box.get("height"))
    left = _clamp01(xc - width / 2.0)
    right = _clamp01(xc + width / 2.0)
    top = _clamp01(yc - height / 2.0)
    bottom = _clamp01(yc + height / 2.0)
    return left, top, right, bottom


def _box_record(
    *,
    class_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width_px: int,
    height_px: int,
    confidence: float | None = None,
    class_name: str = "",
    source: str = "",
) -> dict[str, Any]:
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    return {
        "class_id": int(class_id),
        "class_name": class_name or "",
        "confidence": float(confidence) if confidence is not None else None,
        "source": source or "",
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "x_center": _clamp01((x1 + x2) / 2.0),
        "y_center": _clamp01((y1 + y2) / 2.0),
        "width": _clamp01(box_w),
        "height": _clamp01(box_h),
        "pixel_left": round(x1 * width_px, 2),
        "pixel_top": round(y1 * height_px, 2),
        "pixel_width": round(box_w * width_px, 2),
        "pixel_height": round(box_h * height_px, 2),
    }


def read_annotation(image_path: Path) -> dict[str, Any]:
    image_path = image_path.resolve()
    if not is_image_file(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    label_path = resolve_label_path(image_path)
    with Image.open(image_path) as image:
        width_px, height_px = image.size

    boxes: list[dict[str, Any]] = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except Exception:
                continue
            x1 = _clamp01(x_center - width / 2.0)
            y1 = _clamp01(y_center - height / 2.0)
            x2 = _clamp01(x_center + width / 2.0)
            y2 = _clamp01(y_center + height / 2.0)
            boxes.append(
                _box_record(
                    class_id=class_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    width_px=width_px,
                    height_px=height_px,
                )
            )

    return {
        "image_path": str(image_path),
        "label_path": str(label_path),
        "image_name": image_path.name,
        "width": width_px,
        "height": height_px,
        "boxes": boxes,
    }


def save_annotation(image_path: Path, boxes: list[dict[str, Any]]) -> dict[str, Any]:
    image_path = image_path.resolve()
    if not is_image_file(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    label_path = resolve_label_path(image_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for box in boxes or []:
        class_id = int(float(box.get("class_id", 0)))
        x1, y1, x2, y2 = _normalize_box_coordinates(box)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0.0 or height <= 0.0:
            continue
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {
        "image_path": str(image_path),
        "label_path": str(label_path),
        "box_count": len(lines),
    }


def list_dataset_images(root: Path, *, recursive: bool = True) -> list[dict[str, Any]]:
    root = root.resolve()
    images = []
    for image_path in iter_image_files(root, recursive=recursive):
        label_path = resolve_label_path(image_path)
        with Image.open(image_path) as image:
            width_px, height_px = image.size
        try:
            relative_path = str(image_path.relative_to(root))
        except Exception:
            relative_path = image_path.name
        box_count = 0
        if label_path.exists():
            box_count = sum(1 for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
        images.append(
            {
                "image_path": str(image_path),
                "label_path": str(label_path),
                "relative_path": relative_path.replace("\\", "/"),
                "width": width_px,
                "height": height_px,
                "annotated": label_path.exists(),
                "box_count": box_count,
            }
        )
    return images
