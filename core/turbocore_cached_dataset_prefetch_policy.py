"""Policy inspection for cached dataset prefetch shadow adapters."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping


def build_cached_dataset_prefetch_policy_details(
    dataset: Any,
    *,
    shuffle: bool,
    seed_provided: bool,
    num_workers: int,
) -> Dict[str, Any]:
    sample_count = len(getattr(dataset, "samples", []) or [])
    supported_dataset = type(dataset).__name__ in {"NewbieCachedDataset", "AnimaCachedDataset"}
    bucket_indices, bucket_error = _safe_bucket_indices(dataset)
    bucket_report = _bucket_report(bucket_indices, sample_count, bucket_error)
    concept_report = _concept_geometry_report(dataset)
    shape_report = _shape_metadata_report(dataset, sample_count)

    fallback_reasons: List[str] = []
    if not supported_dataset:
        fallback_reasons.append("unsupported_cached_dataset_class")
    if concept_report["active"]:
        fallback_reasons.append("concept_geometry_sampler_parity_not_ready")
    if bucket_report["bucket_sampler_detected"]:
        fallback_reasons.append("bucket_sampler_order_parity_not_ready")

    native_supported = supported_dataset and not fallback_reasons
    if native_supported:
        decision = "run_flat_cached_prefetch_shadow"
    elif bucket_report["bucket_sampler_detected"]:
        decision = "skip_python_bucket_sampler_owns_order"
    elif concept_report["active"]:
        decision = "skip_python_concept_geometry_sampler_owns_order"
    else:
        decision = "skip_unsupported_cached_dataset"

    return {
        "policy_version": "p5g_cached_prefetch_policy_v1",
        "native_shadow_supported": native_supported,
        "native_shadow_decision": decision,
        "fallback_reasons": fallback_reasons,
        "fallback_reason": fallback_reasons[0] if fallback_reasons else "",
        "bucket_sampler_detected": bucket_report["bucket_sampler_detected"],
        "concept_geometry_sampler_detected": concept_report["active"],
        "bucket_sampler_report": bucket_report,
        "concept_geometry_report": concept_report,
        "shape_metadata_report": shape_report,
        "sampler_order_report": {
            "training_order_owner": "python_batch_sampler" if bucket_report["bucket_sampler_detected"] or concept_report["active"] else "torch_dataloader_sampler",
            "native_order_shadow_scope": "flat_sampler_only" if native_supported else "diagnostic_skip_only",
            "live_equivalent": native_supported and (not shuffle or seed_provided),
            "requires_seed_for_shuffle_equivalence": bool(shuffle and not seed_provided),
            "native_bucket_sampler_path_enabled": False,
            "native_concept_geometry_sampler_path_enabled": False,
            "worker_count": max(int(num_workers), 0),
            "worker_fetch_timing_equivalent": max(int(num_workers), 0) == 0,
            "training_path_enabled": False,
        },
    }


def _safe_bucket_indices(dataset: Any) -> tuple[Any, str]:
    getter = getattr(dataset, "get_bucket_indices", None)
    if not callable(getter):
        return None, ""
    try:
        return getter(), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _bucket_report(bucket_indices: Any, sample_count: int, error: str) -> Dict[str, Any]:
    if not isinstance(bucket_indices, Mapping) or not bucket_indices:
        return {
            "enabled": False,
            "bucket_sampler_detected": False,
            "bucket_count": 0,
            "covered_sample_count": 0,
            "unknown_sample_count": 0,
            "min_bucket_size": 0,
            "max_bucket_size": 0,
            "bucket_preview": [],
            "error": error,
            "native_bucket_sampler_path_enabled": False,
            "training_path_enabled": False,
        }
    entries: List[tuple[str, int]] = []
    covered = 0
    unknown = 0
    for key, value in bucket_indices.items():
        try:
            size = len(value)  # type: ignore[arg-type]
        except Exception:
            size = 0
        key_text = str(key)
        entries.append((key_text, int(size)))
        covered += int(size)
        if key_text == "unknown":
            unknown += int(size)
    non_empty = [size for _key, size in entries if size > 0]
    entries.sort(key=lambda item: (-item[1], item[0]))
    return {
        "enabled": True,
        "bucket_sampler_detected": len(non_empty) > 1,
        "bucket_count": len(non_empty),
        "covered_sample_count": covered,
        "sample_count": sample_count,
        "coverage_complete": covered == sample_count,
        "unknown_sample_count": unknown,
        "min_bucket_size": min(non_empty) if non_empty else 0,
        "max_bucket_size": max(non_empty) if non_empty else 0,
        "bucket_preview": [{"bucket": key, "sample_count": size} for key, size in entries[:8]],
        "error": error,
        "policy": "python_bucket_batch_sampler_only" if len(non_empty) > 1 else "flat_sampler_order",
        "native_bucket_sampler_path_enabled": False,
        "training_path_enabled": False,
    }


def _concept_geometry_report(dataset: Any) -> Dict[str, Any]:
    active = False
    summary: Dict[str, Any] = {}
    checker = getattr(dataset, "is_concept_geometry_enabled", None)
    if callable(checker):
        try:
            active = bool(checker())
        except Exception:
            active = False
    getter = getattr(dataset, "get_concept_geometry_summary", None)
    if callable(getter):
        try:
            raw = getter()
            summary = raw if isinstance(raw, dict) else {}
        except Exception:
            summary = {}
    return {
        "active": active,
        "sampler_mode": str(getattr(dataset, "concept_geometry_sampler_mode", "") or summary.get("sampler_mode", "")),
        "attached_count": int(summary.get("attached_count", 0) or 0),
        "native_concept_geometry_sampler_path_enabled": False,
        "training_path_enabled": False,
    }


def _shape_metadata_report(dataset: Any, sample_count: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    getter = getattr(dataset, "get_cache_metadata_summary", None)
    if callable(getter):
        try:
            raw = getter()
            summary = raw if isinstance(raw, dict) else {}
        except Exception as exc:
            summary = {"metadata_error": f"{type(exc).__name__}: {exc}"}
    native_records = int(summary.get("native_shape_metadata_records", 0) or 0)
    metadata_records = int(summary.get("metadata_records", 0) or 0)
    fallback_loads = int(summary.get("fallback_shape_loads", 0) or 0)
    fallback_failures = int(summary.get("fallback_shape_failures", 0) or 0)
    metadata_hits = int(summary.get("metadata_shape_hits", 0) or 0)
    metadata_misses = int(summary.get("metadata_shape_misses", 0) or 0)
    shape_cache_entries = int(summary.get("shape_cache_entries", 0) or 0)
    if native_records > 0 and metadata_hits > 0:
        source = "native_shape_metadata_index"
    elif metadata_records > 0 and metadata_hits > 0:
        source = "cache_manifest_metadata"
    elif fallback_loads > 0:
        source = "python_tensor_shape_fallback"
    else:
        source = "unresolved_or_not_needed"
    return {
        "sample_count": sample_count,
        "source": source,
        "native_shape_metadata_records": native_records,
        "cache_manifest_metadata_records": metadata_records,
        "shape_cache_entries": shape_cache_entries,
        "metadata_shape_hits": metadata_hits,
        "metadata_shape_misses": metadata_misses,
        "fallback_shape_loads": fallback_loads,
        "fallback_shape_failures": fallback_failures,
        "native_shape_metadata_used_for_bucket_build": native_records > 0 and metadata_hits > 0,
        "python_tensor_shape_fallback_used": fallback_loads > 0,
        "shape_metadata_training_path_enabled": False,
        "training_path_enabled": False,
    }


__all__ = ["build_cached_dataset_prefetch_policy_details"]
