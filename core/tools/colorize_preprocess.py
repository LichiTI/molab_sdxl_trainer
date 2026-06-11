# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Colorization dataset preprocess: produce EasyControl-v2 control images.

Colorization is an EasyControl v2 task (``task_id=colorize``).  The sidecar
*contract* (``easycontrol_v2_contract``) already defines where the condition
latents / text cache / control images must live and audits them, but nothing
*produces* the control images.  This tool is that producer.

For each color (target) image it writes a deterministic control/condition image
(line-art edge map or grayscale luminance), names it to match
``sidecar_plan_for_target`` exactly, then returns a contract-compatible task
spec plus a readiness audit (which cond/text latents the trainer still has to
cache).  The adapter training itself stays in the trainer (gated, out of scope).

Cleanroom; CPU-only PIL/OpenCV image processing.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageFilter, ImageOps

from core.tools.image_utilities import IMAGE_EXTS
from core.lulynx_trainer.easycontrol_v2_contract import (
    audit_easycontrol_v2_sidecars,
    build_colorization_task_spec,
    sidecar_plan_for_target,
)

logger = logging.getLogger(__name__)

SUPPORTED_MODES = ("lineart", "grayscale")


def _normalize_suffix(suffix: str) -> str:
    """Ensure the control suffix carries an image extension so the written file
    matches the contract's ``sidecar_plan_for_target`` path exactly."""
    value = str(suffix or "").strip()
    if not value:
        return ""
    if Path(value).suffix.lower() not in IMAGE_EXTS:
        value = f"{value}.png"
    return value


def _make_control_image(image: Image.Image, mode: str, *, low: int, high: int) -> Image.Image:
    gray = ImageOps.grayscale(ImageOps.exif_transpose(image))
    if mode == "grayscale":
        return gray.convert("RGB")
    # lineart: prefer OpenCV Canny, fall back to Pillow FIND_EDGES.
    try:
        import cv2
        import numpy as np

        edges = cv2.Canny(np.array(gray), int(low), int(high))
        out = Image.fromarray(edges)
    except Exception:
        out = gray.filter(ImageFilter.FIND_EDGES)
    return out.convert("RGB")


def prepare_colorization_dataset(
    image_dir: str,
    control_image_dir: str,
    *,
    mode: str = "lineart",
    control_suffix: str = "_cond",
    cond_cache_dir: str = "",
    text_cache_dir: str = "",
    target_family: str = "anima",
    edge_low: int = 80,
    edge_high: int = 160,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate colorization control images + a contract-compatible manifest.

    Args:
        image_dir: folder of color (target) images.
        control_image_dir: destination for the generated control images.
        mode: ``"lineart"`` (edge map, default) or ``"grayscale"`` (luminance).
        control_suffix: appended to each stem; ``.png`` is added if no image ext.
        cond_cache_dir / text_cache_dir: recorded into the spec so the audit can
            report which trainer-side latent caches are still missing.
        target_family: ``anima`` or ``newbie``.
    """
    mode_key = str(mode or "lineart").strip().lower()
    if mode_key not in SUPPORTED_MODES:
        raise ValueError(f"unsupported colorize mode: {mode}")

    src_dir = Path(image_dir)
    if not src_dir.is_dir():
        raise ValueError(f"image_dir is not a directory: {image_dir}")
    dst_dir = Path(control_image_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    suffix = _normalize_suffix(control_suffix)
    spec = build_colorization_task_spec(
        target_family=target_family,
        cond_cache_dir=cond_cache_dir,
        text_cache_dir=text_cache_dir,
        control_image_dir=str(dst_dir),
    )
    if suffix:
        spec = replace(spec, control_suffix=suffix).normalized()

    targets = sorted(
        p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    processed = 0
    errors: list[str] = []
    for target in targets:
        plan = sidecar_plan_for_target(target, spec)
        out_path = Path(plan.control_image_path)
        if out_path.exists() and not overwrite:
            processed += 1
            continue
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(target) as image:
                control = _make_control_image(image, mode_key, low=edge_low, high=edge_high)
                control.save(out_path)
            processed += 1
        except Exception as exc:  # one bad file must not abort the batch
            errors.append(f"{target.name}: {exc}")
            logger.warning("colorize_preprocess skipped %s: %s", target.name, exc)

    audit = audit_easycontrol_v2_sidecars(targets, spec, check_exists=True)
    logger.info("colorize_preprocess: mode=%s processed=%s ready=%s", mode_key, processed, audit.ready)
    return {
        "processed": processed,
        "mode": mode_key,
        "control_image_dir": str(dst_dir),
        "errors": errors,
        "task_spec": spec.to_dict(),
        "audit": audit.to_dict(),
    }
