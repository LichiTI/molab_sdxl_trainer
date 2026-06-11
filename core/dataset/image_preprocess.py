"""Clean-room image preprocessing helpers for dataset tools."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SIDECAR_EXTS = (".txt", ".caption", ".npz")


def _native_image_scan_disabled() -> bool:
    return str(os.environ.get("LULYNX_DISABLE_NATIVE_IMAGE_SCAN", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@lru_cache(maxsize=1)
def _load_native_image_scan_api() -> Any:
    artifact_dir = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    if artifact_dir and artifact_dir not in sys.path and Path(artifact_dir).is_dir():
        sys.path.insert(0, artifact_dir)
    try:
        return importlib.import_module("lulynx_native")
    except Exception:
        return None


def _native_image_scan_api() -> Any:
    if _native_image_scan_disabled():
        return None
    native = _load_native_image_scan_api()
    if not hasattr(native, "list_image_files"):
        return None
    return native


@dataclass
class ResizeOptions:
    input_dir: Path
    output_dir: Path
    quality: int = 95
    output_format: str = ""
    resolutions: list[tuple[int, int]] = field(default_factory=list)
    enable_resize: bool = True
    rename: bool = False
    rename_mode: str = "legacy_suffix"
    exact_size: bool = True
    sync_metadata: bool = True
    recursive: bool = False


@dataclass
class ResizeResult:
    processed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "skipped": self.skipped,
            "errors": self.errors,
            "outputs": self.outputs,
        }


def run_image_resize(
    options: ResizeOptions,
    progress: Callable[[str], None] | None = None,
) -> ResizeResult:
    """Resize/convert images using Pillow only.

    The web API deliberately does not delete source files. If a caller requests
    deletion, the route records that it was ignored so dataset tools stay safe.
    """

    result = ResizeResult()
    _log(progress, f"Scanning {options.input_dir}")
    if not options.input_dir.is_dir():
        raise ValueError(f"Input directory not found: {options.input_dir}")
    options.output_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_images(options.input_dir, options.recursive)
    for index, src in enumerate(files, start=1):
        try:
            with Image.open(src) as image:
                image = ImageOps.exif_transpose(image)
                target_size = _select_target_size(image.size, options.resolutions)
                converted = _prepare_image(image, target_size, options)
                dst = _destination_path(src, index, options)
                save_kwargs: dict[str, Any] = {}
                if dst.suffix.lower() in {".jpg", ".jpeg", ".webp"}:
                    save_kwargs["quality"] = max(1, min(int(options.quality), 100))
                converted.save(dst, **save_kwargs)
                if options.sync_metadata:
                    _copy_sidecars(src, dst)
                result.processed += 1
                result.outputs.append(str(dst))
                _log(progress, f"[{result.processed}] {src.name} -> {dst.name}")
        except Exception as exc:
            result.skipped += 1
            message = f"{src}: {exc}"
            result.errors.append(message)
            _log(progress, f"Error: {message}")

    _log(progress, f"Done. processed={result.processed}, skipped={result.skipped}")
    return result


def parse_resize_options(params: dict[str, Any]) -> ResizeOptions:
    input_dir = Path(params.get("input_dir", "") or params.get("dir", "") or params.get("path", ""))
    output_dir = Path(params.get("output_dir", "") or input_dir / "resized")
    fmt = str(params.get("format", "") or params.get("output_format", "")).strip().lower().lstrip(".")
    return ResizeOptions(
        input_dir=input_dir,
        output_dir=output_dir,
        quality=int(params.get("quality") or 95),
        output_format=fmt,
        resolutions=_parse_resolutions(params.get("resolutions")),
        enable_resize=_truthy(params.get("enable_resize", True)),
        rename=_truthy(params.get("rename", False)),
        rename_mode=str(params.get("rename_mode", "legacy_suffix") or "legacy_suffix").strip().lower(),
        exact_size=_truthy(params.get("exact_size", True)),
        sync_metadata=_truthy(params.get("sync_metadata", True)),
        recursive=_truthy(params.get("recursive", False)),
    )


def _prepare_image(image: Image.Image, target_size: tuple[int, int] | None, options: ResizeOptions) -> Image.Image:
    out = image.convert("RGB") if options.output_format in {"jpg", "jpeg"} else image.copy()
    if not options.enable_resize or target_size is None:
        return out
    if options.exact_size:
        return ImageOps.fit(out, target_size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    out.thumbnail(target_size, Image.Resampling.LANCZOS)
    return out


def _destination_path(src: Path, index: int, options: ResizeOptions) -> Path:
    ext = "." + options.output_format if options.output_format else src.suffix.lower()
    if ext == ".jpeg":
        ext = ".jpg"
    if options.rename:
        mode = options.rename_mode or "legacy_suffix"
        if mode == "folder_sequence":
            folder_name = src.parent.name or options.input_dir.name or "image"
            name = f"{_safe_stem(folder_name)}_{index:05d}{ext}"
        else:
            name = f"{src.stem}_resized{ext}"
    else:
        name = f"{src.stem}{ext}"
    if options.recursive:
        try:
            rel_parent = src.parent.relative_to(options.input_dir)
        except ValueError:
            rel_parent = Path()
        dst_dir = options.output_dir / rel_parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        return _dedupe_path(dst_dir / name)
    return _dedupe_path(options.output_dir / name)


def _copy_sidecars(src: Path, dst: Path) -> None:
    for ext in SIDECAR_EXTS:
        sidecar = src.with_suffix(ext)
        if sidecar.is_file():
            shutil.copy2(sidecar, dst.with_suffix(ext))


def _dedupe_path(path: Path) -> Path:
    """Return a free path by appending _0002, _0003, ... when needed."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 2
    while True:
        candidate = parent / f"{stem}_{index:04d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _iter_images(root: Path, recursive: bool) -> list[Path]:
    native = _native_image_scan_api()
    if native is not None:
        try:
            return [Path(path) for path in native.list_image_files(str(root), bool(recursive))]
        except Exception:
            pass
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def _safe_stem(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value or "").strip())
    return text.strip("._-") or "image"


def _select_target_size(source_size: tuple[int, int], resolutions: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not resolutions:
        return None
    src_w, src_h = source_size
    if src_w <= 0 or src_h <= 0:
        return resolutions[0]
    src_ratio = src_w / src_h

    def score(size: tuple[int, int]) -> tuple[float, int]:
        w, h = size
        ratio = w / h if h else 1.0
        return (abs(ratio - src_ratio), abs((w * h) - (src_w * src_h)))

    return min(resolutions, key=score)


def _parse_resolutions(value: Any) -> list[tuple[int, int]]:
    raw_values = value if isinstance(value, list) else [value] if value else []
    sizes: list[tuple[int, int]] = []
    for raw in raw_values:
        text = str(raw).lower().replace("×", "x").strip()
        if not text:
            continue
        if "x" in text:
            left, right = text.split("x", 1)
        else:
            left = right = text
        try:
            width = int(float(left))
            height = int(float(right))
        except ValueError:
            continue
        if width > 0 and height > 0:
            sizes.append((width, height))
    return sizes


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() not in {"", "0", "false", "no", "off", "disable", "disabled"}


def _log(progress: Callable[[str], None] | None, message: str) -> None:
    if progress:
        progress(f"{time.strftime('%H:%M:%S')} {message}")

