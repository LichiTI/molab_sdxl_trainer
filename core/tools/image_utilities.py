"""Small clean-room image utilities used by the web tools facade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class BatchImageResult:
    processed: int
    outputs: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"processed": self.processed, "outputs": self.outputs, "errors": self.errors}


def canny_edges(input_path: str, output_path: str | None = None, low: int = 80, high: int = 160) -> dict[str, Any]:
    """Generate an edge-map image.

    OpenCV is used when available; Pillow's FIND_EDGES filter is the fallback.
    """

    src = Path(input_path)
    dst = Path(output_path) if output_path else src.with_name(f"{src.stem}_canny.png")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        gray = ImageOps.grayscale(ImageOps.exif_transpose(image))
        try:
            import cv2
            import numpy as np

            arr = np.array(gray)
            edges = cv2.Canny(arr, int(low), int(high))
            out = Image.fromarray(edges)
        except Exception:
            out = gray.filter(ImageFilter.FIND_EDGES)
        out.save(dst)
    return {"output_path": str(dst), "method": "opencv_or_pillow", "low": low, "high": high}


def face_crop_rotate(
    input_path: str,
    output_path: str | None = None,
    target_size: int = 512,
    rotate_degrees: float = 0,
) -> dict[str, Any]:
    """Rotate and face-aware center-crop one image.

    If OpenCV face detection is unavailable, this degrades to a centered crop.
    """

    src = Path(input_path)
    dst = Path(output_path) if output_path else src.with_name(f"{src.stem}_face_crop.png")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if rotate_degrees:
            image = image.rotate(float(rotate_degrees), expand=True, resample=Image.Resampling.BICUBIC)
        box = _face_crop_box(image, int(target_size))
        ImageOps.fit(image.crop(box), (int(target_size), int(target_size)), Image.Resampling.LANCZOS).save(dst)
    return {"output_path": str(dst), "target_size": int(target_size), "rotated": float(rotate_degrees)}


def latent_upscale_preview(input_path: str, output_path: str | None = None, scale: float = 2.0) -> dict[str, Any]:
    """Upscale an image or NumPy latent array without model-family coupling.

    Image files are resized with Lanczos. ``.npy``/``.npz`` arrays are resized
    with NumPy repeat as a lightweight utility preview.
    """

    src = Path(input_path)
    factor = max(1.0, float(scale))
    if src.suffix.lower() in IMAGE_EXTS:
        dst = Path(output_path) if output_path else src.with_name(f"{src.stem}_upscaled{src.suffix}")
        with Image.open(src) as image:
            image = ImageOps.exif_transpose(image)
            size = (max(1, int(image.width * factor)), max(1, int(image.height * factor)))
            image.resize(size, Image.Resampling.LANCZOS).save(dst)
        return {"output_path": str(dst), "kind": "image", "scale": factor}

    import numpy as np

    arr = np.load(src)
    data = arr[arr.files[0]] if hasattr(arr, "files") else arr
    repeat = max(1, int(round(factor)))
    upscaled = np.repeat(np.repeat(data, repeat, axis=-2), repeat, axis=-1)
    dst = Path(output_path) if output_path else src.with_name(f"{src.stem}_upscaled.npy")
    np.save(dst, upscaled)
    return {"output_path": str(dst), "kind": "latent_array", "scale": repeat, "shape": list(upscaled.shape)}


def batch_apply(input_dir: str, output_dir: str, operation: str, **kwargs: Any) -> BatchImageResult:
    root = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    errors: list[str] = []
    for src in sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS):
        try:
            dst = out_root / f"{src.stem}_{operation}.png"
            if operation == "canny":
                outputs.append(canny_edges(str(src), str(dst), **kwargs)["output_path"])
            elif operation == "face_crop":
                outputs.append(face_crop_rotate(str(src), str(dst), **kwargs)["output_path"])
            elif operation == "upscale":
                outputs.append(latent_upscale_preview(str(src), str(dst), **kwargs)["output_path"])
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as exc:
            errors.append(f"{src}: {exc}")
    return BatchImageResult(processed=len(outputs), outputs=outputs, errors=errors)


def _face_crop_box(image: Image.Image, target_size: int) -> tuple[int, int, int, int]:
    width, height = image.size
    center_x, center_y = width // 2, height // 2
    try:
        import cv2
        import numpy as np

        gray = np.array(ImageOps.grayscale(image))
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, 1.1, 4) if not cascade.empty() else []
        if len(faces):
            x, y, w, h = sorted(faces, key=lambda item: item[2] * item[3], reverse=True)[0]
            center_x, center_y = int(x + w / 2), int(y + h / 2)
    except Exception:
        pass
    side = min(width, height, max(1, target_size))
    left = min(max(0, center_x - side // 2), max(0, width - side))
    top = min(max(0, center_y - side // 2), max(0, height - side))
    return (left, top, left + side, top + side)

