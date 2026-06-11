# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Automatic mask pipeline (Phase 8.12 / #99).

Generate per-image foreground masks for masked_loss training without
requiring users to hand-paint alpha channels.  Strategies, in order of
fidelity:

1. **alpha_channel** — extract existing alpha channel from RGBA images.
2. **rembg**         — use the ``rembg`` library if installed.
3. **threshold**     — fall back to luminance/colour thresholding.

Output is a sidecar PNG named ``<image_stem>_mask.png`` next to each
image, with mode ``L`` (8-bit greyscale, 0=background, 255=foreground).
``anima_cache_builder`` already detects this naming convention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class MaskPipelineConfig:
    """Configuration for ``generate_masks_for_directory``."""

    data_dir: str
    strategy: str = "auto"            # "auto", "alpha_channel", "rembg", "threshold"
    overwrite: bool = False           # regenerate masks even if file exists
    threshold_low: int = 25           # luminance < this == background
    threshold_high: int = 230         # luminance > this == background (e.g. white bg)
    feather_radius: int = 0           # gaussian blur radius for soft edges
    invert: bool = False              # invert background/foreground
    rembg_model: str = "u2net"        # rembg session model when available


@dataclass
class MaskPipelineResult:
    written: int
    skipped: int
    errors: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_masks_for_directory(config: MaskPipelineConfig) -> MaskPipelineResult:
    """Walk ``data_dir`` and write a mask PNG for every image found."""
    from PIL import Image  # imported lazily to keep test surface minimal

    root = Path(config.data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Mask pipeline data_dir not found: {root}")

    written = 0
    skipped = 0
    errors: List[str] = []

    rembg_session = None
    if config.strategy in {"auto", "rembg"}:
        rembg_session = _try_init_rembg(config.rembg_model)

    for path in _iter_images(root):
        # Skip mask sidecars themselves
        if path.stem.endswith("_mask") or path.stem.endswith("_alpha"):
            continue

        out_path = path.with_name(f"{path.stem}_mask.png")
        if out_path.exists() and not config.overwrite:
            skipped += 1
            continue

        try:
            mask = _build_mask(path, config, rembg_session=rembg_session)
            if mask is None:
                errors.append(f"{path.name}: no mask produced")
                continue
            mask.save(str(out_path))
            written += 1
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

    return MaskPipelineResult(written=written, skipped=skipped, errors=tuple(errors))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _try_init_rembg(model: str):
    """Best-effort rembg session init.  Returns None when unavailable."""
    try:
        from rembg import new_session
        return new_session(model)
    except Exception as exc:
        logger.debug("rembg unavailable (%s)", exc)
        return None


def _build_mask(path: Path, config: MaskPipelineConfig, *, rembg_session=None):
    """Dispatch to a single mask strategy and return the resulting PIL image."""
    from PIL import Image, ImageFilter

    image = Image.open(path)
    strategy = config.strategy

    if strategy == "auto":
        # Prefer alpha channel, then rembg, then threshold
        if image.mode == "RGBA":
            mask = _mask_from_alpha(image)
        elif rembg_session is not None:
            mask = _mask_from_rembg(image.convert("RGB"), rembg_session)
        else:
            mask = _mask_from_threshold(image.convert("RGB"), config)
    elif strategy == "alpha_channel":
        mask = _mask_from_alpha(image) if image.mode == "RGBA" else None
    elif strategy == "rembg":
        mask = _mask_from_rembg(image.convert("RGB"), rembg_session) if rembg_session else None
    elif strategy == "threshold":
        mask = _mask_from_threshold(image.convert("RGB"), config)
    else:
        raise ValueError(f"Unknown mask strategy: {strategy}")

    if mask is None:
        return None

    if config.invert:
        from PIL import ImageOps
        mask = ImageOps.invert(mask)

    if config.feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=config.feather_radius))

    return mask


def _mask_from_alpha(image):
    """Extract the alpha channel from an RGBA image."""
    if image.mode != "RGBA":
        return None
    return image.getchannel("A")


def _mask_from_rembg(image, session) -> Optional["Image.Image"]:
    """Use rembg to produce a foreground mask."""
    try:
        from rembg import remove
        result = remove(image, session=session, only_mask=True)
        if hasattr(result, "convert"):
            return result.convert("L")
    except Exception as exc:
        logger.debug("rembg failed: %s", exc)
    return None


def _mask_from_threshold(image, config: MaskPipelineConfig):
    """Heuristic luminance + saturation threshold mask.

    Pixels whose luminance is between ``threshold_low`` and ``threshold_high``
    OR whose saturation is high are considered foreground.  This handles
    the common cases of solid black or white backgrounds.
    """
    from PIL import Image
    import numpy as np

    arr = np.asarray(image.convert("RGB")).astype("float32") / 255.0
    # luminance via ITU-R BT.601
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    # saturation via max-min
    max_c = arr.max(axis=-1)
    min_c = arr.min(axis=-1)
    sat = (max_c - min_c)

    low = config.threshold_low / 255.0
    high = config.threshold_high / 255.0

    foreground = ((lum > low) & (lum < high)) | (sat > 0.15)
    mask_arr = (foreground.astype("uint8") * 255)

    return Image.fromarray(mask_arr, mode="L")
