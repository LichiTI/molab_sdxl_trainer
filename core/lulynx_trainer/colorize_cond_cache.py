# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Colorize condition-latent cache producer (EasyControl v2 ``task_id=colorize``).

The colorize pipeline was wired end-to-end EXCEPT one keystone: the dataset loader
expects a per-target ``cond_latent`` sidecar (``sidecar_plan_for_target`` →
``{stem}.latent.safetensors`` under ``cond_cache_dir``), the training-step handler
feeds ``batch["cond_latents"]`` into the two-stream adapter, but nothing *produced*
those latents. ``colorize_preprocess`` writes the control IMAGES (line-art /
grayscale); the anima cache builder only VAE-encodes the *target* images. This
module closes that gap: it VAE-encodes each control image — through the SAME VAE
encode callable the trainer already builds (``build_anima_cache_encode_bundle``) —
into the cond-latent sidecar the contract names, so colorize training consumes a
REAL control-image condition (not the cache-first derived-from-target fallback).

Cleanroom; reuses ``anima_cache_builder._encode_latents_chunked`` (image → latent)
and ``easycontrol_v2_contract.sidecar_plan_for_target`` (path naming) verbatim.

Scope note: ``color_text_embeds`` is loaded by the dataset loader but NOT consumed
by the current two-stream forward/loss, so this producer intentionally emits only
the cond-latent sidecar. The colorize contract's ``requires_text_cache`` advisory
is non-fatal (the loop uses the normal caption context); a text-cache producer is a
separate future hook if/when the adapter consumes a color-only text stream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import torch

from .anima_cache_builder import _encode_latents_chunked
from .easycontrol_v2_contract import EasyControlV2TaskSpec, sidecar_plan_for_target

logger = logging.getLogger(__name__)


@dataclass
class ColorizeCondCacheReport:
    written: int = 0
    skipped: int = 0
    missing_control: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "written": int(self.written),
            "skipped": int(self.skipped),
            "missing_control": int(self.missing_control),
            "error_count": len(self.errors),
            "errors": list(self.errors[:20]),
        }


def _write_cond_sidecar(path: Path, latent: torch.Tensor, disk_dtype: torch.dtype) -> None:
    """Write ``{stem}.latent.safetensors`` with key ``cond_latents`` (the key the
    dataset loader's ``_load_sidecar_tensor`` looks for first).

    Atomic: write to a sibling ``.tmp`` then ``os.replace`` so a partial write can
    never be observed as a valid sidecar (and a force-rebuild overwrites cleanly).
    """
    import os

    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = latent.detach().to("cpu", dtype=disk_dtype).contiguous()
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_file({"cond_latents": tensor}, str(tmp))
    os.replace(str(tmp), str(path))


def build_colorize_cond_cache(
    *,
    target_image_paths: Sequence[str | Path],
    spec: EasyControlV2TaskSpec,
    vae_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    vae_chunk_size: int = 0,
    target_resolution: int = 0,
    disk_dtype: torch.dtype = torch.float16,
    force: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> ColorizeCondCacheReport:
    """VAE-encode each target's colorize control image into its cond-latent sidecar.

    For every target image we resolve the control image + cond-latent paths via the
    SAME ``sidecar_plan_for_target`` the dataset loader uses, so the produced files
    match exactly where the loader looks. Targets whose control image is absent are
    counted (``missing_control``) and skipped — never fabricated. Existing sidecars
    are skipped unless ``force``.
    """
    emit = log or (lambda message: logger.info(message))
    normalized = spec.normalized()
    report = ColorizeCondCacheReport()

    if normalized.task_id != "colorize":
        emit(f"[colorize-cond-cache] task_id={normalized.task_id!r} is not 'colorize'; nothing to do.")
        return report
    if not normalized.cond_cache_dir:
        report.errors.append("cond_cache_dir is empty")
        emit("[colorize-cond-cache] cond_cache_dir is empty; cannot write sidecars.")
        return report

    for target in target_image_paths:
        plan = sidecar_plan_for_target(target, normalized)
        if not plan.cond_latent_path:
            continue
        control_path = Path(plan.control_image_path) if plan.control_image_path else None
        if control_path is None or not control_path.is_file():
            report.missing_control += 1
            continue
        cond_path = Path(plan.cond_latent_path)
        if cond_path.is_file() and not force:
            report.skipped += 1
            continue
        try:
            latent = _encode_latents_chunked(
                vae_encode_fn=vae_encode_fn,
                image_path=control_path,
                chunk_size=int(vae_chunk_size or 0),
                target_resolution=int(target_resolution or 0),
            )  # [16, h, w]
            _write_cond_sidecar(cond_path, latent, disk_dtype)
            report.written += 1
        except Exception as exc:  # pragma: no cover - per-sample resilience
            report.errors.append(f"{plan.stem}: {type(exc).__name__}: {exc}")

    emit(
        "[colorize-cond-cache] finished: "
        f"written={report.written}, skipped={report.skipped}, "
        f"missing_control={report.missing_control}, errors={len(report.errors)}"
    )
    return report


__all__ = ["build_colorize_cond_cache", "ColorizeCondCacheReport"]
