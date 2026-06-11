# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Online cache builder for native Newbie cache-first training.

This module is intentionally small and explicit: it turns image/caption pairs
into the ``*_newbie.npz`` contract consumed by ``NewbieCachedDataset``.  It does
not borrow implementation from legacy trainers; component calls are routed
through the model objects already loaded by ``newbie_loader``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image

try:
    from .caption_sidecar import json_caption_to_training_parts, json_caption_to_training_text
    from .caption_source_mix import CaptionSourceMixConfig, caption_source_variant_texts
    from .dataset_discovery import resolve_caption_path
except ImportError:  # pragma: no cover - direct script smoke loading
    from caption_sidecar import json_caption_to_training_parts, json_caption_to_training_text
    from caption_source_mix import CaptionSourceMixConfig, caption_source_variant_texts
    from dataset_discovery import resolve_caption_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
NEWBIE_CACHE_VERSION = 2  # v2 adds schema_version + optional loss_mask + format/dtype support


def _save_cache(
    path: Path,
    payload: dict,
    disk_format: str = "npz",
    disk_dtype: str = "float16",
) -> None:
    """Write cache payload to disk in the specified format and dtype."""
    # Cast float arrays to the requested dtype
    cast_payload = {}
    for k, v in payload.items():
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            target_dtype = np.float16 if disk_dtype == "float16" else (
                np.float32 if disk_dtype == "float32" else (
                    getattr(np, disk_dtype, np.float16)
                )
            )
            cast_payload[k] = v.astype(target_dtype)
        else:
            cast_payload[k] = v

    if disk_format == "safetensors":
        try:
            from safetensors.numpy import save_file as save_numpy
            save_numpy(cast_payload, str(path.with_suffix(".safetensors")))
        except ImportError:
            # Fallback to npz if safetensors not available
            np.savez(str(path.with_suffix(".npz")), **cast_payload)
    elif disk_format == "pt":
        pt_payload = {k: torch.from_numpy(v) for k, v in cast_payload.items()}
        torch.save(pt_payload, str(path.with_suffix(".pt")))
    else:
        np.savez(str(path), **cast_payload)


@dataclass(frozen=True)
class NewbieCacheBuildResult:
    written: int
    skipped: int
    errors: tuple[str, ...]
    manifest_path: str = ""
    metadata_path: str = ""
    metadata_fast_path: bool = False
    cache_trust: dict[str, object] = field(default_factory=dict)


def build_newbie_cache(
    *,
    loaded_model: object,
    data_dir: str | Path,
    device: str,
    dtype: torch.dtype,
    resolution: tuple[int, int],
    caption_extension: str = ".txt",
    gemma3_prompt: str = "",
    gemma_max_token_length: int = 512,
    clip_max_token_length: int = 2048,
    alpha_mask: bool = False,
    force: bool = False,
    disk_format: str = "npz",
    disk_dtype: str = "float16",
    caption_source_mix: CaptionSourceMixConfig | None = None,
    log: Callable[[str], None] | None = None,
) -> NewbieCacheBuildResult:
    """Build ``*_newbie.npz`` files next to source images.

    The produced arrays match ``NewbieCachedDataset``:
    ``latents``, ``encoder_hidden_states``, optional
    ``pooled_prompt_embeds``, ``attention_mask``, and optional ``loss_mask``.
    """

    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Newbie cache data_dir does not exist: {root}")

    vae = _require_component(loaded_model, "vae")
    gemma = _require_component(loaded_model, "text_encoder_1")
    gemma_tokenizer = _require_component(loaded_model, "tokenizer_1")
    clip_model = getattr(loaded_model, "text_encoder_2", None)
    clip_tokenizer = getattr(loaded_model, "tokenizer_2", None)

    target_w, target_h = resolution
    image_paths = list(_iter_images(root))
    errors: list[str] = []
    metadata_records: list[dict[str, object]] = []
    written = 0
    skipped = 0

    _move_eval(vae, device, dtype)
    _move_eval(gemma, device, dtype)
    if clip_model is not None:
        _move_eval(clip_model, device, dtype)

    for image_path in image_paths:
        cache_path = image_path.with_name(f"{image_path.stem}_newbie.npz")
        if cache_path.exists() and not force:
            skipped += 1
            continue

        try:
            raw_caption = _read_caption_raw(image_path, caption_extension)
            base_caption = json_caption_to_training_text(raw_caption) if raw_caption.strip() else image_path.stem
            caption = _format_gemma3_prompt(base_caption, gemma3_prompt)
            variant_texts = caption_source_variant_texts(
                caption_source_mix or CaptionSourceMixConfig(),
                json_caption_to_training_parts(raw_caption),
            )
            with torch.no_grad():
                latents = _encode_latents(
                    vae=vae,
                    image_path=image_path,
                    device=device,
                    dtype=dtype,
                    size=(target_h, target_w),
                )
                hidden, mask = _encode_gemma(
                    encoder=gemma,
                    tokenizer=gemma_tokenizer,
                    caption=caption,
                    device=device,
                    dtype=dtype,
                    max_token_length=gemma_max_token_length,
                )
                pooled = None
                if clip_model is not None and clip_tokenizer is not None:
                    pooled = _encode_clip_pooled(
                        model=clip_model,
                        tokenizer=clip_tokenizer,
                        caption=caption,
                        device=device,
                        dtype=dtype,
                        max_token_length=clip_max_token_length,
                    )
                variant_text_data = {}
                for variant_name, variant_caption in variant_texts.items():
                    formatted_variant = _format_gemma3_prompt(variant_caption, gemma3_prompt)
                    variant_hidden, variant_mask = _encode_gemma(
                        encoder=gemma,
                        tokenizer=gemma_tokenizer,
                        caption=formatted_variant,
                        device=device,
                        dtype=dtype,
                        max_token_length=gemma_max_token_length,
                    )
                    variant_pooled = None
                    if clip_model is not None and clip_tokenizer is not None:
                        variant_pooled = _encode_clip_pooled(
                            model=clip_model,
                            tokenizer=clip_tokenizer,
                            caption=formatted_variant,
                            device=device,
                            dtype=dtype,
                            max_token_length=clip_max_token_length,
                        )
                    variant_text_data[variant_name] = (variant_hidden, variant_mask, variant_pooled)

            payload: dict[str, np.ndarray | int] = {
                "newbie_cache_schema_version": np.asarray(NEWBIE_CACHE_VERSION, dtype=np.int32),
                "latents": _cpu_np(latents),
                "encoder_hidden_states": _cpu_np(hidden),
                "attention_mask": _cpu_np(mask.to(torch.bool)),
            }
            if pooled is not None:
                payload["pooled_prompt_embeds"] = _cpu_np(pooled)
            if variant_text_data:
                payload["caption_source_variant_count"] = np.asarray(len(variant_text_data), dtype=np.int32)
                for variant_name, (variant_hidden, variant_mask, variant_pooled) in variant_text_data.items():
                    payload[f"caption_variant_{variant_name}_encoder_hidden_states"] = _cpu_np(variant_hidden)
                    payload[f"caption_variant_{variant_name}_attention_mask"] = _cpu_np(variant_mask.to(torch.bool))
                    if variant_pooled is not None:
                        payload[f"caption_variant_{variant_name}_pooled_prompt_embeds"] = _cpu_np(variant_pooled)
            if alpha_mask:
                loss_mask = _load_alpha_mask(image_path, tuple(int(v) for v in latents.shape[-2:]))
                if loss_mask is not None:
                    payload["loss_mask"] = _cpu_np(loss_mask)
            _save_cache(
                cache_path,
                payload,
                disk_format=disk_format,
                disk_dtype=disk_dtype,
            )
            actual_cache_path = _cache_output_path(cache_path, disk_format)
            metadata_records.append(
                {
                    "stem": image_path.stem,
                    "cache_path": _rel(root, actual_cache_path),
                    "latent_shape": [int(v) for v in np.asarray(payload["latents"]).shape],
                }
            )
            written += 1
            if log is not None:
                log(f"Newbie cache written: {actual_cache_path.name}")
        except Exception as exc:  # keep the batch going and report all bad files
            errors.append(f"{image_path.name}: {type(exc).__name__}: {exc}")

    manifest_path = ""
    metadata_path = ""
    metadata_fast_path = False
    if written > 0 or skipped > 0:
        try:
            try:
                from .cache_manifest import build_cache_trust_report, write_cache_manifest
            except ImportError:
                from cache_manifest import build_cache_trust_report, write_cache_manifest

            manifest = write_cache_manifest(
                root,
                family="newbie",
                builder="newbie_cache_builder",
                include_sha256=True,
                config={
                    "schema_version": NEWBIE_CACHE_VERSION,
                    "fingerprint_mode": "strict_sha256",
                    "disk_format": disk_format,
                    "disk_dtype": disk_dtype,
                    "caption_extension": caption_extension,
                    "resolution": [int(target_w), int(target_h)],
                    "gemma3_prompt": str(gemma3_prompt or ""),
                    "gemma_max_token_length": int(gemma_max_token_length),
                    "clip_max_token_length": int(clip_max_token_length),
                    "alpha_mask": bool(alpha_mask),
                    "caption_source_mix_enabled": bool((caption_source_mix or CaptionSourceMixConfig()).enabled),
                },
            )
            manifest_path = str(manifest.manifest_path)
            if log is not None:
                log(
                    "Newbie cache manifest written: "
                    f"{manifest.manifest_path.name} "
                    f"({manifest.sample_count} samples, {manifest.cache_file_count} cache files)"
                )
            cache_trust = build_cache_trust_report(root, family="newbie", mode="strict").as_dict()
            if skipped == 0 and len(metadata_records) == written:
                try:
                    from .cache_metadata import write_cache_metadata_records
                except ImportError:
                    from cache_metadata import write_cache_metadata_records

                metadata = write_cache_metadata_records(root, family="newbie", samples=metadata_records)
                metadata_fast_path = True
            else:
                try:
                    from .cache_metadata import write_cache_metadata
                except ImportError:
                    from cache_metadata import write_cache_metadata

                metadata = write_cache_metadata(root, family="newbie")
            metadata_path = str(metadata.metadata_path)
            if log is not None:
                log(
                    "Newbie cache metadata written: "
                    f"{metadata.metadata_path.name} ({metadata.sample_count} samples, fast_path={metadata_fast_path})"
                )
        except Exception as exc:
            errors.append(f"manifest: {type(exc).__name__}: {exc}")
            cache_trust = {}

    return NewbieCacheBuildResult(
        written=written,
        skipped=skipped,
        errors=tuple(errors),
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        metadata_fast_path=metadata_fast_path,
        cache_trust=cache_trust if "cache_trust" in locals() else {},
    )


def _iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _cache_output_path(path: Path, disk_format: str) -> Path:
    if disk_format == "safetensors":
        return path.with_suffix(".safetensors")
    if disk_format == "pt":
        return path.with_suffix(".pt")
    return path


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _require_component(loaded_model: object, attr: str) -> object:
    value = getattr(loaded_model, attr, None)
    if value is None:
        raise RuntimeError(f"Loaded Newbie model is missing required component: {attr}")
    return value


def _move_eval(module: object, device: str, dtype: torch.dtype) -> None:
    if isinstance(module, torch.nn.Module):
        module.to(device=device)
        if any(p.is_floating_point() for p in module.parameters(recurse=True)):
            module.to(dtype=dtype)
        module.eval()


def _read_caption_raw(image_path: Path, caption_extension: str = ".txt") -> str:
    caption_path = resolve_caption_path(image_path.parent, image_path.stem, caption_extension)
    if caption_path is not None and caption_path.is_file():
        return caption_path.read_text(encoding="utf-8", errors="ignore").strip()
    return ""


def _format_gemma3_prompt(caption: str, prompt_template: str = "") -> str:
    template = (prompt_template or "").strip()
    if not template:
        return caption
    if "{caption}" in template:
        return template.replace("{caption}", caption)
    return f"{template} {caption}".strip()


def _encode_latents(
    *,
    vae: object,
    image_path: Path,
    device: str,
    dtype: torch.dtype,
    size: tuple[int, int],
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((size[1], size[0]), Image.Resampling.LANCZOS)
    arr = np.asarray(image).astype("float32") / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

    encoded = vae.encode(tensor)
    latent_dist = getattr(encoded, "latent_dist", None)
    latents = latent_dist.sample() if latent_dist is not None else getattr(encoded, "latents", encoded)
    scale = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0) or 1.0)
    shift = float(getattr(getattr(vae, "config", None), "shift_factor", 0.0) or 0.0)
    latents = (latents - shift) * scale
    return latents[0].to(dtype=dtype)


def _encode_gemma(
    *,
    encoder: object,
    tokenizer: object,
    caption: str,
    device: str,
    dtype: torch.dtype,
    max_token_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = tokenizer(
        caption,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=int(max_token_length or getattr(tokenizer, "model_max_length", 512) or 512),
    )
    tokens = {key: value.to(device) for key, value in tokens.items() if isinstance(value, torch.Tensor)}
    output = encoder(**tokens, output_hidden_states=True)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and getattr(output, "hidden_states", None):
        hidden = output.hidden_states[-1]
    if hidden is None:
        raise RuntimeError("Gemma encoder did not return hidden states")
    mask = tokens.get("attention_mask")
    if mask is None:
        mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    return hidden[0].to(dtype=dtype), mask[0].to(torch.bool)


def _encode_clip_pooled(
    *,
    model: object,
    tokenizer: object,
    caption: str,
    device: str,
    dtype: torch.dtype,
    max_token_length: int = 2048,
) -> torch.Tensor:
    tokens = tokenizer(
        caption,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=int(max_token_length or getattr(tokenizer, "model_max_length", 2048) or 2048),
    )
    tokens = {key: value.to(device) for key, value in tokens.items() if isinstance(value, torch.Tensor)}

    pooled: torch.Tensor | None = None
    float32_attempt_error: Exception | None = None

    try:
        pooled = _compute_clip_pooled(model=model, tokens=tokens)
    except Exception as exc:
        float32_attempt_error = exc

    if pooled is None or not bool(torch.isfinite(pooled.float()).all()):
        pooled = _compute_clip_pooled(model=model, tokens=tokens, force_float32=True)
        if not bool(torch.isfinite(pooled.float()).all()):
            raise RuntimeError(
                "CLIP pooled text features contain NaN/Inf even after float32 fallback"
            )
    elif float32_attempt_error is not None:
        raise float32_attempt_error

    pooled = pooled[0].float()
    if pooled.ndim != 1:
        pooled = pooled.reshape(-1)
    return pooled


def _compute_clip_pooled(
    *,
    model: object,
    tokens: dict[str, torch.Tensor],
    force_float32: bool = False,
) -> torch.Tensor:
    if force_float32:
        previous_dtype = _module_float_dtype(model)
        if previous_dtype is not None and previous_dtype != torch.float32:
            _move_module_float_dtype(model, torch.float32)
        try:
            pooled = _compute_clip_pooled(model=model, tokens=tokens, force_float32=False)
        finally:
            if previous_dtype is not None and previous_dtype != torch.float32:
                _move_module_float_dtype(model, previous_dtype)
        return pooled

    if hasattr(model, "get_text_features"):
        pooled = model.get_text_features(**tokens)
    else:
        output = model(**tokens, output_hidden_states=True)
        pooled = getattr(output, "text_embeds", None)
        if pooled is None:
            pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                raise RuntimeError("CLIP model did not return pooled or hidden text features")
            mask = tokens.get("attention_mask")
            pooled = _masked_mean(hidden, mask)
    if not isinstance(pooled, torch.Tensor):
        raise RuntimeError(f"CLIP pooled text features must be a tensor, got {type(pooled)!r}")
    return pooled


def _module_float_dtype(module: object) -> torch.dtype | None:
    if not isinstance(module, torch.nn.Module):
        return None
    for buffer in module.buffers(recurse=True):
        if buffer.is_floating_point():
            return buffer.dtype
    for param in module.parameters(recurse=True):
        if param.is_floating_point():
            return param.dtype
    return None


def _move_module_float_dtype(module: object, dtype: torch.dtype) -> None:
    if not isinstance(module, torch.nn.Module):
        return
    module.to(dtype=dtype)


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return hidden.mean(dim=1)
    weights = mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _cpu_np(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().to("cpu").contiguous()
    if value.dtype is torch.bfloat16:
        value = value.float()
    return value.numpy()


def _load_alpha_mask(
    image_path: Path,
    size: tuple[int, int],
) -> torch.Tensor | None:
    """Load an alpha or sidecar mask and resize to latent spatial dims."""
    mask: torch.Tensor | None = None

    # Try image alpha channel first
    try:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            alpha = img.getchannel("A")
            alpha = alpha.resize((size[1], size[0]), Image.Resampling.BILINEAR)
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
                    m = m.resize((size[1], size[0]), Image.Resampling.BILINEAR)
                    arr = np.asarray(m).astype("float32") / 255.0
                    mask = torch.from_numpy(arr)
                    break
                except Exception:
                    continue

    return mask
