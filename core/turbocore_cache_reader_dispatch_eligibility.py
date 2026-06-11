"""Eligibility policy for native cache reader batch dispatch promotion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


SUPPORTED_CACHED_DATASETS = {"NewbieCachedDataset", "AnimaCachedDataset"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _bucket_summary(dataset: Any) -> dict[str, Any]:
    getter = getattr(dataset, "get_bucket_indices", None)
    buckets = getter() if callable(getter) else None
    if not isinstance(buckets, dict):
        return {"enabled": False, "bucket_count": 0, "sample_counts": {}}
    sample_counts = {str(key): len(list(value or [])) for key, value in buckets.items()}
    return {
        "enabled": bool(buckets),
        "bucket_count": len(buckets),
        "sample_counts": sample_counts,
    }


def _sample_suffixes(dataset: Any, *, limit: int = 8) -> list[str]:
    suffixes: list[str] = []
    for sample in list(getattr(dataset, "samples", []) or [])[: max(int(limit), 1)]:
        paths: list[Any] = []
        for attr in ("cache_path", "latent_path", "text_path", "caption_path"):
            value = getattr(sample, attr, None)
            if value:
                paths.append(value)
        for path in paths:
            suffix = Path(path).suffix.lower()
            if suffix:
                suffixes.append(suffix)
    return _dedupe(suffixes)


def _dataset_flags(dataset: Any) -> dict[str, Any]:
    dataset_class = type(dataset).__name__
    concept_geometry = getattr(dataset, "is_concept_geometry_enabled", None)
    concept_geometry_enabled = bool(concept_geometry()) if callable(concept_geometry) else bool(
        getattr(dataset, "concept_geometry_enabled", False)
    )
    return {
        "dataset_class": dataset_class,
        "latent_crop_size": _as_int(getattr(dataset, "latent_crop_size", 0)),
        "text_token_limit": _as_int(getattr(dataset, "text_token_limit", 0)),
        "fixed_text_tokens": _as_int(getattr(dataset, "fixed_text_tokens", 0)),
        "fixed_visual_tokens": _as_int(getattr(dataset, "fixed_visual_tokens", 0)),
        "fixed_qwen3_tokens": _as_int(getattr(dataset, "fixed_qwen3_tokens", 0)),
        "fixed_t5_tokens": _as_int(getattr(dataset, "fixed_t5_tokens", 0)),
        "shuffle_caption": bool(getattr(dataset, "shuffle_caption", False)),
        "shuffle_caption_tags_only": bool(getattr(dataset, "shuffle_caption_tags_only", False)),
        "weighted_captions": bool(getattr(dataset, "weighted_captions", False)),
        "concept_geometry_enabled": concept_geometry_enabled,
        "sample_count": len(list(getattr(dataset, "samples", []) or [])),
        "sample_suffixes": _sample_suffixes(dataset),
    }


def build_cache_reader_dispatch_eligibility_report(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    prefetch_factor: int | None = None,
    strict_fallback: bool = False,
    policy_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return a structured guard report for future native dispatch promotion.

    This report intentionally separates two questions:
    - whether the current debug/shadow parity gate may run;
    - whether real native batch dispatch may replace Python DataLoader output.
    """
    flags = _dataset_flags(dataset)
    overrides = dict(policy_overrides or {})
    bucket_override = overrides.pop("bucket_policy", None)
    for key, value in overrides.items():
        flags[str(key)] = value
    dataset_class = str(flags["dataset_class"])
    worker_count = max(_as_int(num_workers), 0)
    bucket = dict(bucket_override) if isinstance(bucket_override, dict) else _bucket_summary(dataset)
    shadow_blockers: list[str] = []
    dispatch_blockers: list[str] = [
        "native_cache_reader_training_dispatch_not_implemented",
        "python_dataloader_batch_remains_authoritative",
        "cache_reader_dispatch_route_not_promoted",
        "text_conditioning_payload_ownership_not_promoted",
        "auxiliary_batch_fields_ownership_not_promoted",
        "exception_recovery_policy_not_promoted",
        "representative_training_matrix_not_passed",
    ]

    if dataset_class not in SUPPORTED_CACHED_DATASETS:
        shadow_blockers.append("unsupported_cached_dataset_class")
        dispatch_blockers.append("unsupported_cached_dataset_class")
    if bool(shuffle):
        shadow_blockers.append("shuffle_order_parity_not_ready")
        dispatch_blockers.append("sampler_reseed_policy_not_promoted")
    if bool(drop_last):
        shadow_blockers.append("drop_last_parity_not_ready")
        dispatch_blockers.append("drop_last_batch_boundary_not_promoted")
    if worker_count != 0:
        shadow_blockers.append("multi_worker_cache_reader_parity_not_ready")
        dispatch_blockers.append("multi_worker_sample_ownership_not_promoted")
    if int(bucket.get("bucket_count", 0) or 0) > 1:
        shadow_blockers.append("bucket_sampler_cache_reader_parity_not_ready")
        dispatch_blockers.append("bucket_sampler_ownership_not_promoted")
    if bool(flags.get("concept_geometry_enabled", False)):
        shadow_blockers.append("concept_geometry_cache_reader_parity_not_ready")
        dispatch_blockers.append("concept_geometry_sampler_ownership_not_promoted")
    if _as_int(flags.get("latent_crop_size")) > 0 or _as_int(flags.get("fixed_visual_tokens")) > 0:
        shadow_blockers.append("latent_crop_padding_parity_not_ready")
        dispatch_blockers.append("latent_crop_padding_ownership_not_promoted")
    if any(suffix in {".pt", ".pth"} for suffix in list(flags.get("sample_suffixes", []) or [])):
        shadow_blockers.append("pt_cache_reader_shadow_not_supported")
        dispatch_blockers.append("torch_pickle_payload_not_promoted")
    if any(
        bool(flags.get(name, False))
        for name in ("shuffle_caption", "shuffle_caption_tags_only", "weighted_captions")
    ):
        dispatch_blockers.append("caption_runtime_transform_ownership_not_promoted")
    if any(
        _as_int(flags.get(name)) > 0
        for name in ("text_token_limit", "fixed_text_tokens", "fixed_qwen3_tokens", "fixed_t5_tokens")
    ):
        dispatch_blockers.append("text_token_shape_policy_not_promoted")

    shadow_blockers = _dedupe(shadow_blockers)
    dispatch_blockers = _dedupe(dispatch_blockers)
    strict_fallback_passed = (not bool(strict_fallback)) or bool(dispatch_blockers)
    return {
        "schema_version": 1,
        "provider": "native_cache_reader_dispatch_eligibility_policy_v1",
        "ok": True,
        "debug_only": True,
        "shadow_run": True,
        "dataset_class": dataset_class,
        "dataset_supported": dataset_class in SUPPORTED_CACHED_DATASETS,
        "sample_count": _as_int(flags.get("sample_count")),
        "batch_size": max(_as_int(batch_size), 1),
        "shuffle": bool(shuffle),
        "drop_last": bool(drop_last),
        "worker_count": worker_count,
        "prefetch_factor": None if prefetch_factor is None else max(_as_int(prefetch_factor), 1),
        "bucket_policy": bucket,
        "latent_policy": {
            "latent_crop_size": _as_int(flags.get("latent_crop_size")),
            "fixed_visual_tokens": _as_int(flags.get("fixed_visual_tokens")),
            "native_crop_padding_promoted": False,
        },
        "text_payload_policy": {
            "text_payload_present": True,
            "text_token_limit": _as_int(flags.get("text_token_limit")),
            "fixed_text_tokens": _as_int(flags.get("fixed_text_tokens")),
            "fixed_qwen3_tokens": _as_int(flags.get("fixed_qwen3_tokens")),
            "fixed_t5_tokens": _as_int(flags.get("fixed_t5_tokens")),
            "native_text_payload_promoted": False,
        },
        "aux_payload_policy": {
            "caption_fields_python_authoritative": True,
            "loss_mask_python_authoritative": True,
            "sample_id_python_authoritative": True,
            "native_aux_payload_promoted": False,
        },
        "sample_suffixes": list(flags.get("sample_suffixes", []) or []),
        "shadow_gate_ready": not shadow_blockers,
        "shadow_gate_blockers": shadow_blockers,
        "native_dispatch_eligible": False,
        "native_dispatch_blockers": dispatch_blockers,
        "would_allow_native_dispatch": False,
        "fallback_to_python_batch": True,
        "strict_fallback": bool(strict_fallback),
        "strict_fallback_passed": strict_fallback_passed,
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_dispatch": False,
        "training_path_enabled": False,
    }


__all__ = ["build_cache_reader_dispatch_eligibility_report"]
