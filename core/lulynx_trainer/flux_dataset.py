"""Dataset construction for the Flux LoRA preview trainer."""

from __future__ import annotations

from typing import Any

from .dataset_loader import CaptionDataset, create_dataloader
from .data_backend_resolver import resolve_data_backend


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _int_resolution(value: Any, default: int = 1024) -> int:
    if isinstance(value, int):
        return max(value, 1)
    raw = str(value or "").strip()
    if not raw:
        return default
    first = raw.replace("x", ",").split(",", 1)[0].strip()
    try:
        return max(int(float(first)), 1)
    except ValueError:
        return default


def _caption_dataset_kwargs(config: Any) -> dict[str, Any]:
    return dict(
        resolution=_int_resolution(getattr(config, "resolution", 1024), 1024),
        caption_extension=str(getattr(config, "caption_extension", ".txt") or ".txt"),
        enable_bucket=_truthy(getattr(config, "enable_bucket", True)),
        min_bucket_reso=int(getattr(config, "min_bucket_reso", 256) or 256),
        max_bucket_reso=int(getattr(config, "max_bucket_reso", 2048) or 2048),
        bucket_reso_steps=int(getattr(config, "bucket_reso_steps", 64) or 64),
        bucket_selection_mode=str(getattr(config, "bucket_selection_mode", "aspect") or "aspect"),
        bucket_custom_resos=getattr(config, "bucket_custom_resos", ""),
        shuffle_caption=_truthy(getattr(config, "shuffle_caption", True)),
        shuffle_caption_tags_only=_truthy(getattr(config, "shuffle_caption_tags_only", False)),
        keep_tokens=int(getattr(config, "keep_tokens", 0) or 0),
        keep_tokens_separator=str(getattr(config, "keep_tokens_separator", "") or ""),
        flip_augment=_truthy(getattr(config, "flip_aug", False)),
        color_augment=_truthy(getattr(config, "color_aug", False)),
        caption_dropout_rate=float(getattr(config, "caption_dropout_rate", 0.0) or 0.0),
        tag_dropout_rate=float(getattr(config, "tag_dropout_rate", 0.0) or 0.0),
        caption_tag_dropout_targets=str(getattr(config, "caption_tag_dropout_targets", "") or ""),
        caption_tag_dropout_target_mode=str(getattr(config, "caption_tag_dropout_target_mode", "drop_all") or "drop_all"),
        caption_tag_dropout_target_count=int(getattr(config, "caption_tag_dropout_target_count", 1) or 1),
        weighted_captions=_truthy(getattr(config, "weighted_captions", False)),
        masked_loss=_truthy(getattr(config, "masked_loss", False)),
        alpha_mask=_truthy(getattr(config, "alpha_mask", False)),
        caption_variants_enabled=_truthy(getattr(config, "caption_variants_enabled", False)),
        caption_variants=str(getattr(config, "caption_variants", "") or ""),
        caption_variant_schedule=str(getattr(config, "caption_variant_schedule", "alternate") or "alternate"),
        caption_variant_ratio=str(getattr(config, "caption_variant_ratio", "") or ""),
        caption_variant_custom_sequence=str(getattr(config, "caption_variant_custom_sequence", "") or ""),
        caption_variant_loss_adaptive=_truthy(getattr(config, "caption_variant_loss_adaptive", False)),
        dual_caption_enabled=_truthy(getattr(config, "dual_caption_enabled", False)),
        dual_caption_short_key=str(getattr(config, "dual_caption_short_key", "short") or "short"),
        dual_caption_long_key=str(getattr(config, "dual_caption_long_key", "long") or "long"),
        caption_source_mix_enabled=_truthy(getattr(config, "caption_source_mix_enabled", False)),
        caption_source_nl_ratio=float(getattr(config, "caption_source_nl_ratio", 65.0) or 65.0),
        caption_source_tag_ratio=float(getattr(config, "caption_source_tag_ratio", 20.0) or 20.0),
        caption_source_trigger_only_ratio=float(getattr(config, "caption_source_trigger_only_ratio", 10.0) or 10.0),
        caption_source_empty_ratio=float(getattr(config, "caption_source_empty_ratio", 5.0) or 5.0),
        caption_source_trigger_tokens=str(getattr(config, "caption_source_trigger_tokens", "") or ""),
        image_decode_backend=str(getattr(config, "image_decode_backend", "pil") or "pil"),
        image_decode_cache_size=int(getattr(config, "image_decode_cache_size", 0) or 0),
    )


def _create_flux_dataset(config: Any) -> tuple[Any, dict[str, Any]]:
    data_dir = str(getattr(config, "train_data_dir", "") or "")
    kwargs = _caption_dataset_kwargs(config)
    try:
        decision = resolve_data_backend(getattr(config, "data_backend", "auto"), data_dir=data_dir)
        profile = decision.as_dict()
    except Exception as exc:
        profile = {
            "requested_backend": str(getattr(config, "data_backend", "auto") or "auto"),
            "resolved_backend": "caption",
            "fallback_reason": f"data backend resolver failed: {type(exc).__name__}: {exc}",
        }

    if str(profile.get("requested_backend") or "") == "webdataset" and str(profile.get("resolved_backend") or "") == "webdataset":
        try:
            from .webdataset_materialized_dataset import MaterializedWebDataset

            dataset = MaterializedWebDataset(source_data_dir=data_dir, **kwargs)
            profile["training_integration"] = "materialized_captiondataset"
            profile["materialization"] = dict(getattr(dataset, "webdataset_materialization_summary", {}) or {})
            return dataset, profile
        except Exception as exc:
            warnings = list(profile.get("warnings") or [])
            reason = f"materialized WebDataset integration failed: {type(exc).__name__}: {exc}"
            warnings.append(reason)
            profile.update({"resolved_backend": "caption", "fallback_reason": reason, "warnings": warnings})

    profile.setdefault("training_integration", "captiondataset")
    return CaptionDataset(data_dir=data_dir, **kwargs), profile


def create_flux_lora_dataloader(config: Any, *, pin_memory: bool):
    dataset, profile = _create_flux_dataset(config)
    if len(dataset) <= 0:
        raise RuntimeError("Flux LoRA training dataset is empty.")
    dataloader = create_dataloader(
        dataset,
        batch_size=int(getattr(config, "train_batch_size", 1) or 1),
        shuffle=True,
        num_workers=int(getattr(config, "dataloader_num_workers", 0) or 0),
        pin_memory=pin_memory,
        drop_last=False,
    )
    dataloader.lulynx_data_backend_profile = dict(profile)
    return dataloader


__all__ = ["create_flux_lora_dataloader"]
