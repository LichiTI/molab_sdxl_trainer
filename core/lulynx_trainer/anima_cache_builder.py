# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Cache builder for native Anima training.

Turns image/caption pairs into the paired cache files consumed by
``AnimaCachedDataset``:

- ``<stem>_<resolution>_anima.{npz,safetensors,pt}`` — Qwen Image VAE
  16-channel latents (key ``latents_<H>x<W>``, plus optional ``loss_mask``)
- ``<stem>_anima_te.{npz,safetensors,pt}`` — text conditioning (keys
  ``prompt_embeds``, ``attn_mask``, ``t5_input_ids``, ``t5_attn_mask``)

Model calls (VAE encode, text encode) are routed through *callable*
parameters so the builder stays decoupled from any specific model class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch
from PIL import Image

try:
    from .dataset_discovery import discover_smart_subsets, iter_images_for_subset, resolve_caption_path
    from .caption_sidecar import json_caption_to_training_parts, json_caption_to_training_text
    from .caption_source_mix import CaptionSourceMixConfig, caption_source_variant_texts
except ImportError:  # pragma: no cover - direct script smoke loading
    from dataset_discovery import discover_smart_subsets, iter_images_for_subset, resolve_caption_path
    from caption_sidecar import json_caption_to_training_parts, json_caption_to_training_text
    from caption_source_mix import CaptionSourceMixConfig, caption_source_variant_texts


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ANIMA_CACHE_VERSION = 2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnimaCacheBuilderConfig:
    """Configuration for ``build_anima_cache``."""

    data_dir: str = ""
    output_dir: str = ""           # defaults to data_dir when empty
    vae_chunk_size: int = 0        # 0 = no chunking; >0 splits tall images
    text_token_limit: int = 0      # 0 = no truncation
    include_loss_mask: bool = False
    disk_format: str = "npz"       # "npz" | "safetensors" | "pt"
    disk_dtype: str = "float16"    # "float16" | "bfloat16" | "float32"
    text_disk_format: str = ""      # empty = use disk_format
    text_disk_dtype: str = ""       # empty = use disk_dtype
    target_resolution: int = 0      # 0 = keep source size; >0 = resize longest edge before VAE
    caption_source_mix: CaptionSourceMixConfig = field(default_factory=CaptionSourceMixConfig)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnimaCacheBuildResult:
    written: int
    skipped: int
    errors: tuple[str, ...]
    manifest_path: str = ""
    cache_trust: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_anima_cache_sample(
    *,
    image_path: str | Path,
    vae_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    text_encode_fn: Callable[[str], dict[str, torch.Tensor]],
    config: AnimaCacheBuilderConfig,
    caption_extension: str = ".txt",
    force: bool = False,
) -> tuple[Path, Path]:
    image_path = Path(image_path)
    out_root = Path(config.output_dir) if config.output_dir else image_path.parent
    out_root.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    latent_disk_format = _normalize_disk_format(config.disk_format)
    latent_disk_dtype = _normalize_disk_dtype(config.disk_dtype)
    text_disk_format = _normalize_disk_format(config.text_disk_format or latent_disk_format)
    text_disk_dtype = _normalize_disk_dtype(config.text_disk_dtype or latent_disk_dtype)

    text_path = _resolve_cache_path(out_root / f"{stem}_anima_te", text_disk_format)
    existing_latents = sorted(out_root.glob(f"{stem}_*_anima.*"))
    if text_path.exists() and existing_latents and not force:
        return existing_latents[0], text_path

    raw_caption = _read_caption_raw(image_path, caption_extension)
    caption = json_caption_to_training_text(raw_caption) if raw_caption.strip() else image_path.stem
    variant_texts = caption_source_variant_texts(
        config.caption_source_mix,
        json_caption_to_training_parts(raw_caption),
    )
    with torch.no_grad():
        latents = _encode_latents_chunked(
            vae_encode_fn=vae_encode_fn,
            image_path=image_path,
            chunk_size=config.vae_chunk_size,
            target_resolution=config.target_resolution,
        )
        text_data = text_encode_fn(caption)
        variant_text_data = {
            name: text_encode_fn(variant_caption)
            for name, variant_caption in variant_texts.items()
        }
    if config.text_token_limit > 0:
        text_data = _trim_text_data(text_data, config.text_token_limit)
        variant_text_data = {
            name: _trim_text_data(data, config.text_token_limit)
            for name, data in variant_text_data.items()
        }

    h, w = latents.shape[-2], latents.shape[-1]
    resolution_tag = f"{h}x{w}"
    has_loss_mask = False
    latent_payload: dict[str, np.ndarray | object] = {
        "schema_version": np.asarray(ANIMA_CACHE_VERSION, dtype=np.int32),
        f"latents_{resolution_tag}": _cpu_np(latents),
    }
    if config.include_loss_mask:
        loss_mask = _load_alpha_mask(image_path, (h, w))
        if loss_mask is not None:
            latent_payload["loss_mask"] = _cpu_np(loss_mask)
            has_loss_mask = True

    latent_path = _resolve_cache_path(out_root / f"{stem}_{resolution_tag}_anima", latent_disk_format)
    _save_cache(latent_path, latent_payload, latent_disk_format, latent_disk_dtype)

    text_payload: dict[str, np.ndarray | object] = {
        "schema_version": np.asarray(ANIMA_CACHE_VERSION, dtype=np.int32),
        "has_loss_mask": np.asarray(int(has_loss_mask), dtype=np.int32),
        "prompt_embeds": _cpu_np(text_data["prompt_embeds"]),
    }
    if "attn_mask" in text_data:
        text_payload["attn_mask"] = _cpu_np(text_data["attn_mask"].to(torch.bool))
    if "t5_input_ids" in text_data:
        text_payload["t5_input_ids"] = _cpu_np(text_data["t5_input_ids"].long())
    if "t5_attn_mask" in text_data:
        text_payload["t5_attn_mask"] = _cpu_np(text_data["t5_attn_mask"].to(torch.bool))
    if "qwen3_hidden_states" in text_data:
        text_payload["qwen3_hidden_states"] = _cpu_np(text_data["qwen3_hidden_states"])
    if "qwen3_attention_mask" in text_data:
        text_payload["qwen3_attention_mask"] = _cpu_np(text_data["qwen3_attention_mask"].to(torch.bool))
    if variant_text_data:
        text_payload["caption_source_variant_count"] = np.asarray(len(variant_text_data), dtype=np.int32)
        _add_text_variants_to_payload(text_payload, variant_text_data)
    _save_cache(text_path, text_payload, text_disk_format, text_disk_dtype)
    return latent_path, text_path


def build_anima_cache(
    *,
    vae_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    text_encode_fn: Callable[[str], dict[str, torch.Tensor]],
    config: AnimaCacheBuilderConfig,
    caption_extension: str = ".txt",
    force: bool = False,
    log: Callable[[str], None] | None = None,
) -> AnimaCacheBuildResult:
    """Build Anima latent + text cache files.

    Parameters
    ----------
    vae_encode_fn:
        Callable accepting a float image tensor ``[1, 3, H, W]`` in
        ``[-1, 1]`` range and returning 16-channel latents ``[1, 16, h, w]``.
    text_encode_fn:
        Callable accepting a caption string and returning a dict with keys
        ``prompt_embeds`` (``[seq, dim]``), ``attn_mask`` (``[seq]`` bool),
        and optionally ``t5_input_ids`` (``[seq]`` long) and
        ``t5_attn_mask`` (``[seq]`` bool).
    config:
        Builder configuration (see ``AnimaCacheBuilderConfig``).
    caption_extension:
        File extension for sidecar caption files.
    force:
        Overwrite existing cache files.
    log:
        Optional progress callback.
    """

    data_root = Path(config.data_dir)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Anima cache data_dir does not exist: {data_root}")

    out_root = Path(config.output_dir) if config.output_dir else data_root
    out_root.mkdir(parents=True, exist_ok=True)

    image_paths = list(_iter_images(data_root))
    errors: list[str] = []
    written = 0
    skipped = 0

    for image_path in image_paths:
        stem = image_path.stem
        # Skip caption or mask sidecars that happen to have image suffixes
        if any(stem.endswith(s) for s in ("_mask", "_alpha")):
            continue

        # --- determine latent output path (will get resolution suffix) ---
        sample_out_root = image_path.parent if not config.output_dir or out_root == data_root else out_root
        latent_prefix = sample_out_root / f"{stem}"
        text_path = _resolve_cache_path(sample_out_root / f"{stem}_anima_te", config.disk_format)

        if text_path.exists() and not force:
            # Also check if any latent file with this stem already exists
            latent_existing = list(sample_out_root.glob(f"{stem}_*_anima.*"))
            if latent_existing:
                skipped += 1
                continue

        try:
            latent_path, text_path = build_anima_cache_sample(
                image_path=image_path,
                vae_encode_fn=vae_encode_fn,
                text_encode_fn=text_encode_fn,
                config=AnimaCacheBuilderConfig(
                    data_dir=config.data_dir,
                    output_dir=str(sample_out_root),
                    vae_chunk_size=config.vae_chunk_size,
                    text_token_limit=config.text_token_limit,
                    include_loss_mask=config.include_loss_mask,
                    disk_format=config.disk_format,
                    disk_dtype=config.disk_dtype,
                    target_resolution=config.target_resolution,
                    caption_source_mix=config.caption_source_mix,
                ),
                caption_extension=caption_extension,
                force=force,
            )

            if log is not None:
                log(f"Anima cache written: {stem} ({latent_path.stem})")

            written += 1

        except Exception as exc:
            errors.append(f"{image_path.name}: {type(exc).__name__}: {exc}")

    manifest_path = ""
    if written > 0 or skipped > 0:
        try:
            try:
                from .cache_manifest import build_cache_trust_report, write_cache_manifest
            except ImportError:
                from cache_manifest import build_cache_trust_report, write_cache_manifest

            manifest = write_cache_manifest(
                out_root,
                family="anima",
                builder="anima_cache_builder",
                include_sha256=True,
                config={
                    "schema_version": ANIMA_CACHE_VERSION,
                    "fingerprint_mode": "strict_sha256",
                    "disk_format": config.disk_format,
                    "disk_dtype": config.disk_dtype,
                    "caption_extension": caption_extension,
                    "text_token_limit": config.text_token_limit,
                    "include_loss_mask": config.include_loss_mask,
                    "target_resolution": config.target_resolution,
                    "caption_source_mix_enabled": config.caption_source_mix.enabled,
                    "caption_source_mix_variants": _caption_source_variant_names(config.caption_source_mix),
                },
            )
            manifest_path = str(manifest.manifest_path)
            if log is not None:
                log(
                    "Anima cache manifest written: "
                    f"{manifest.manifest_path.name} "
                    f"({manifest.sample_count} samples, {manifest.cache_file_count} cache files)"
                )
            cache_trust = build_cache_trust_report(out_root, family="anima", mode="strict").as_dict()
            try:
                from .cache_metadata import write_cache_metadata
            except ImportError:
                from cache_metadata import write_cache_metadata

            metadata = write_cache_metadata(out_root, family="anima")
            if log is not None:
                log(
                    "Anima cache metadata written: "
                    f"{metadata.metadata_path.name} ({metadata.sample_count} samples)"
                )
        except Exception as exc:
            errors.append(f"manifest: {type(exc).__name__}: {exc}")
            cache_trust = {}

    return AnimaCacheBuildResult(
        written=written,
        skipped=skipped,
        errors=tuple(errors),
        manifest_path=manifest_path,
        cache_trust=cache_trust if "cache_trust" in locals() else {},
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_images(root: Path) -> Iterable[Path]:
    for subset in discover_smart_subsets(root):
        for path in iter_images_for_subset(subset):
            yield path


def _read_caption_raw(image_path: Path, caption_extension: str = ".txt") -> str:
    caption_path = resolve_caption_path(image_path.parent, image_path.stem, caption_extension)
    if caption_path is not None and caption_path.is_file():
        return caption_path.read_text(encoding="utf-8", errors="ignore").strip()
    return ""


def _caption_source_variant_names(config: CaptionSourceMixConfig) -> list[str]:
    names: list[str] = []
    if config.nl_ratio > 0:
        names.append("nl")
    if config.tag_ratio > 0:
        names.append("tag")
    if config.trigger_only_ratio > 0:
        names.append("trigger_only")
    if config.empty_ratio > 0:
        names.append("empty")
    return names if config.enabled else []


def _add_text_variants_to_payload(
    payload: dict[str, np.ndarray | object],
    variants: dict[str, dict[str, torch.Tensor]],
) -> None:
    key_map = {
        "prompt_embeds": "prompt_embeds",
        "attn_mask": "attn_mask",
        "t5_input_ids": "t5_input_ids",
        "t5_attn_mask": "t5_attn_mask",
        "qwen3_hidden_states": "qwen3_hidden_states",
        "qwen3_attention_mask": "qwen3_attention_mask",
    }
    for variant_name, text_data in variants.items():
        for source_key, cache_key in key_map.items():
            if source_key not in text_data:
                continue
            tensor = text_data[source_key]
            if source_key.endswith("mask") or source_key.endswith("attn_mask"):
                tensor = tensor.to(torch.bool)
            elif source_key.endswith("input_ids"):
                tensor = tensor.long()
            payload[f"caption_variant_{variant_name}_{cache_key}"] = _cpu_np(tensor)


def _encode_latents_chunked(
    *,
    vae_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    image_path: Path,
    chunk_size: int = 0,
    target_resolution: int = 0,
) -> torch.Tensor:
    """Encode an image through the VAE, optionally in vertical chunks."""
    img = Image.open(image_path).convert("RGB")
    img = _resize_for_target_resolution(img, target_resolution)
    arr = np.asarray(img).astype("float32") / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    if chunk_size <= 0 or tensor.shape[-2] <= chunk_size:
        return vae_encode_fn(tensor)[0]  # [16, h, w]

    # Chunk along the height dimension, encode each, stitch latents back
    _, _, H, W = tensor.shape
    chunks: list[torch.Tensor] = []
    y = 0
    while y < H:
        end = min(y + chunk_size, H)
        chunk = tensor[:, :, y:end, :]
        lat = vae_encode_fn(chunk)  # [1, 16, h_chunk, w]
        chunks.append(lat[0])       # [16, h_chunk, w]
        y = end
    return torch.cat(chunks, dim=-2)  # [16, h_total, w]


def _resize_for_target_resolution(img: Image.Image, target_resolution: int = 0) -> Image.Image:
    """Resize longest edge to target_resolution while preserving aspect ratio.

    The cache-first staged-resolution path needs the latent shape to be decided
    during cache generation, not by mutating trainer config later.  Width and
    height are rounded to multiples of 16 so common VAE/downstream patch paths
    receive aligned tensors.
    """
    target = int(target_resolution or 0)
    if target <= 0:
        return img
    width, height = img.size
    if width <= 0 or height <= 0:
        return img
    scale = float(target) / float(max(width, height))
    new_width = max(16, int(round(width * scale / 16.0)) * 16)
    new_height = max(16, int(round(height * scale / 16.0)) * 16)
    new_width = min(new_width, target)
    new_height = min(new_height, target)
    if (new_width, new_height) == img.size:
        return img
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _trim_text_data(data: dict[str, torch.Tensor], limit: int) -> dict[str, torch.Tensor]:
    """Trim text conditioning arrays to *limit* tokens along dim-0."""
    trimmed: dict[str, torch.Tensor] = {}
    for key, tensor in data.items():
        if tensor.shape[0] > limit:
            trimmed[key] = tensor[:limit]
        else:
            trimmed[key] = tensor
    return trimmed


def _load_alpha_mask(image_path: Path, latent_size: tuple[int, int]) -> torch.Tensor | None:
    """Load an alpha or sidecar mask, downsampled to latent spatial dims."""
    mask: torch.Tensor | None = None

    # Try image alpha channel
    try:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            alpha = img.getchannel("A")
            alpha = alpha.resize((latent_size[1], latent_size[0]), Image.Resampling.BILINEAR)
            arr = np.asarray(alpha).astype("float32") / 255.0
            mask = torch.from_numpy(arr)
    except Exception:
        pass

    # Try sidecar mask file
    if mask is None:
        for suffix in ("_mask.png", "_mask.jpg", "_mask.jpeg", "_alpha.png"):
            candidate = image_path.with_name(f"{image_path.stem}{suffix}")
            if candidate.is_file():
                try:
                    m = Image.open(candidate).convert("L")
                    m = m.resize((latent_size[1], latent_size[0]), Image.Resampling.BILINEAR)
                    arr = np.asarray(m).astype("float32") / 255.0
                    mask = torch.from_numpy(arr)
                    break
                except Exception:
                    continue

    return mask


def _resolve_cache_path(base: Path, disk_format: str) -> Path:
    """Return the final file path for the chosen disk format."""
    if disk_format == "safetensors":
        return base.with_suffix(".safetensors")
    if disk_format == "pt":
        return base.with_suffix(".pt")
    return base.with_suffix(".npz")


def _normalize_disk_format(value: str | None) -> str:
    normalized = str(value or "npz").strip().lower()
    if normalized in {"safetensors", "safe_tensors", "st"}:
        return "safetensors"
    if normalized in {"pt", "pth", "torch"}:
        return "pt"
    return "npz"


def _normalize_disk_dtype(value: str | None) -> str:
    normalized = str(value or "float16").strip().lower()
    aliases = {
        "fp16": "float16",
        "half": "float16",
        "float16": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "float32": "float32",
    }
    return aliases.get(normalized, "float16")


def _save_cache(
    path: Path,
    payload: dict,
    disk_format: str = "npz",
    disk_dtype: str = "float16",
) -> None:
    """Write cache payload to disk in the specified format and dtype."""
    if disk_format == "safetensors":
        try:
            from safetensors.torch import save_file as save_torch
            save_torch(_torch_cache_payload(payload, disk_dtype), str(path))
        except ImportError:
            np.savez(str(path.with_suffix(".npz")), **_numpy_cache_payload(payload, disk_dtype))
    elif disk_format == "pt":
        torch.save(_torch_cache_payload(payload, disk_dtype), str(path))
    else:
        np.savez(str(path), **_numpy_cache_payload(payload, disk_dtype))


def _numpy_cache_payload(payload: dict, disk_dtype: str) -> dict[str, object]:
    cast_payload: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray) and value.dtype.kind == "f":
            # NumPy has no native bfloat16 storage dtype; materialize bf16
            # requests as float32 for npz rather than truncating to fp16.
            target_dtype = np.float32 if disk_dtype in {"float32", "bfloat16"} else np.float16
            cast_payload[key] = value.astype(target_dtype)
        else:
            cast_payload[key] = value
    return cast_payload


def _torch_cache_payload(payload: dict, disk_dtype: str) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    target_dtype = dtype_map.get(disk_dtype, torch.float16)
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(value))
        else:
            tensor = torch.as_tensor(value)
        if tensor.is_floating_point():
            tensor = tensor.to(target_dtype)
        tensors[key] = tensor
    return tensors


def _cpu_np(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().to("cpu").contiguous()
    if value.dtype is torch.bfloat16:
        value = value.float()
    return value.numpy()
