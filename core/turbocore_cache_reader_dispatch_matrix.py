"""Representative strict-fallback matrix for native cache reader dispatch."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from core.turbocore_cache_reader_dispatch_eligibility import build_cache_reader_dispatch_eligibility_report


ALWAYS_DISPATCH_BLOCKERS = [
    "native_cache_reader_training_dispatch_not_implemented",
    "python_dataloader_batch_remains_authoritative",
    "cache_reader_dispatch_route_not_promoted",
    "text_conditioning_payload_ownership_not_promoted",
    "auxiliary_batch_fields_ownership_not_promoted",
    "exception_recovery_policy_not_promoted",
    "representative_training_matrix_not_passed",
]

MATRIX_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "baseline_supported_single_worker",
        "description": "Supported cached dataset, deterministic single-worker shadow gate.",
        "shuffle": False,
        "drop_last": False,
        "num_workers": 0,
        "expected_shadow_gate_ready": True,
        "expected_shadow_blockers": [],
        "expected_dispatch_blockers": [],
    },
    {
        "case_id": "shuffle_sampler_reseed",
        "description": "Shuffle order still belongs to Python/DataLoader.",
        "shuffle": True,
        "drop_last": False,
        "num_workers": 0,
        "expected_shadow_gate_ready": False,
        "expected_shadow_blockers": ["shuffle_order_parity_not_ready"],
        "expected_dispatch_blockers": ["sampler_reseed_policy_not_promoted"],
    },
    {
        "case_id": "drop_last_batch_boundary",
        "description": "drop_last boundary semantics remain Python authoritative.",
        "shuffle": False,
        "drop_last": True,
        "num_workers": 0,
        "expected_shadow_gate_ready": False,
        "expected_shadow_blockers": ["drop_last_parity_not_ready"],
        "expected_dispatch_blockers": ["drop_last_batch_boundary_not_promoted"],
    },
    {
        "case_id": "multi_worker_sample_ownership",
        "description": "Worker sharding and sample ownership are not promoted.",
        "shuffle": False,
        "drop_last": False,
        "num_workers": 2,
        "expected_shadow_gate_ready": False,
        "expected_shadow_blockers": ["multi_worker_cache_reader_parity_not_ready"],
        "expected_dispatch_blockers": ["multi_worker_sample_ownership_not_promoted"],
    },
    {
        "case_id": "bucket_sampler_ownership",
        "description": "Bucket sampler state is not owned by native cache reader dispatch.",
        "shuffle": False,
        "drop_last": False,
        "num_workers": 0,
        "policy_overrides": {
            "bucket_policy": {"enabled": True, "bucket_count": 2, "sample_counts": {"0": 1, "1": 1}},
        },
        "expected_shadow_gate_ready": False,
        "expected_shadow_blockers": ["bucket_sampler_cache_reader_parity_not_ready"],
        "expected_dispatch_blockers": ["bucket_sampler_ownership_not_promoted"],
    },
    {
        "case_id": "latent_crop_padding_ownership",
        "description": "Latent crop/padding ownership has no native dispatch promotion yet.",
        "shuffle": False,
        "drop_last": False,
        "num_workers": 0,
        "policy_overrides": {"latent_crop_size": 64},
        "expected_shadow_gate_ready": False,
        "expected_shadow_blockers": ["latent_crop_padding_parity_not_ready"],
        "expected_dispatch_blockers": ["latent_crop_padding_ownership_not_promoted"],
    },
    {
        "case_id": "concept_geometry_sampler_ownership",
        "description": "Concept geometry sampler semantics remain Python-owned.",
        "shuffle": False,
        "drop_last": False,
        "num_workers": 0,
        "policy_overrides": {"concept_geometry_enabled": True},
        "expected_shadow_gate_ready": False,
        "expected_shadow_blockers": ["concept_geometry_cache_reader_parity_not_ready"],
        "expected_dispatch_blockers": ["concept_geometry_sampler_ownership_not_promoted"],
    },
    {
        "case_id": "torch_pickle_cache_payload",
        "description": ".pt/.pth cache payloads are not part of the native reader shadow path.",
        "shuffle": False,
        "drop_last": False,
        "num_workers": 0,
        "policy_overrides": {"sample_suffixes": [".pt"]},
        "expected_shadow_gate_ready": False,
        "expected_shadow_blockers": ["pt_cache_reader_shadow_not_supported"],
        "expected_dispatch_blockers": ["torch_pickle_payload_not_promoted"],
    },
    {
        "case_id": "text_token_shape_policy",
        "description": "Variable/fixed T5/Qwen/text token shape policy is not promoted.",
        "shuffle": False,
        "drop_last": False,
        "num_workers": 0,
        "policy_overrides": {"text_token_limit": 77, "fixed_t5_tokens": 256, "fixed_qwen3_tokens": 128},
        "expected_shadow_gate_ready": True,
        "expected_shadow_blockers": [],
        "expected_dispatch_blockers": ["text_token_shape_policy_not_promoted"],
    },
    {
        "case_id": "caption_runtime_transform_ownership",
        "description": "Caption shuffle/weight transforms remain Python authoritative.",
        "shuffle": False,
        "drop_last": False,
        "num_workers": 0,
        "policy_overrides": {"shuffle_caption": True, "weighted_captions": True},
        "expected_shadow_gate_ready": True,
        "expected_shadow_blockers": [],
        "expected_dispatch_blockers": ["caption_runtime_transform_ownership_not_promoted"],
    },
    {
        "case_id": "unsupported_cached_dataset_class",
        "description": "Unknown cached dataset classes must stay on Python fallback.",
        "shuffle": False,
        "drop_last": False,
        "num_workers": 0,
        "policy_overrides": {"dataset_class": "UnsupportedCachedDataset"},
        "expected_shadow_gate_ready": False,
        "expected_shadow_blockers": ["unsupported_cached_dataset_class"],
        "expected_dispatch_blockers": ["unsupported_cached_dataset_class"],
    },
)


def _dedupe(items: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _missing(expected: Iterable[str], actual: Iterable[Any]) -> list[str]:
    actual_set = {str(item) for item in actual}
    return [item for item in expected if item not in actual_set]


def _build_case(
    dataset: Any,
    case: dict[str, Any],
    *,
    batch_size: int,
    prefetch_factor: int | None,
    strict_fallback: bool,
) -> dict[str, Any]:
    expected_dispatch = _dedupe(ALWAYS_DISPATCH_BLOCKERS + list(case.get("expected_dispatch_blockers", []) or []))
    report = build_cache_reader_dispatch_eligibility_report(
        dataset,
        batch_size=batch_size,
        shuffle=bool(case.get("shuffle", False)),
        drop_last=bool(case.get("drop_last", False)),
        num_workers=int(case.get("num_workers", 0) or 0),
        prefetch_factor=prefetch_factor,
        strict_fallback=strict_fallback,
        policy_overrides=dict(case.get("policy_overrides", {}) or {}),
    )
    actual_shadow_blockers = [str(item) for item in list(report.get("shadow_gate_blockers", []) or [])]
    actual_dispatch_blockers = [str(item) for item in list(report.get("native_dispatch_blockers", []) or [])]
    shadow_ready_matches = bool(report.get("shadow_gate_ready", False)) is bool(case.get("expected_shadow_gate_ready", False))
    missing_shadow = _missing(list(case.get("expected_shadow_blockers", []) or []), actual_shadow_blockers)
    missing_dispatch = _missing(expected_dispatch, actual_dispatch_blockers)
    safety_failures: list[str] = []
    if bool(report.get("native_dispatch_eligible", False)):
        safety_failures.append("native_dispatch_unexpectedly_eligible")
    if bool(report.get("would_allow_native_dispatch", False)):
        safety_failures.append("would_allow_native_dispatch_unexpectedly_true")
    if not bool(report.get("fallback_to_python_batch", False)):
        safety_failures.append("python_fallback_not_selected")
    if bool(report.get("returns_tensor_payloads", False)):
        safety_failures.append("returned_tensor_payloads_unexpectedly_enabled")
    if bool(report.get("training_path_enabled", False)):
        safety_failures.append("training_path_unexpectedly_enabled")
    if strict_fallback and not bool(report.get("strict_fallback_passed", False)):
        safety_failures.append("strict_fallback_failed")
    case_ok = shadow_ready_matches and not missing_shadow and not missing_dispatch and not safety_failures
    return {
        "case_id": str(case.get("case_id") or ""),
        "description": str(case.get("description") or ""),
        "ok": case_ok,
        "expected_shadow_gate_ready": bool(case.get("expected_shadow_gate_ready", False)),
        "shadow_gate_ready": bool(report.get("shadow_gate_ready", False)),
        "missing_shadow_blockers": missing_shadow,
        "missing_dispatch_blockers": missing_dispatch,
        "safety_failures": safety_failures,
        "shadow_gate_blockers": actual_shadow_blockers,
        "native_dispatch_blockers": actual_dispatch_blockers,
        "strict_fallback_passed": bool(report.get("strict_fallback_passed", False)),
        "fallback_to_python_batch": bool(report.get("fallback_to_python_batch", False)),
        "native_dispatch_eligible": False,
        "would_allow_native_dispatch": False,
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "dispatch_eligibility": report,
    }


def build_cache_reader_dispatch_fallback_matrix(
    dataset: Any,
    *,
    batch_size: int = 1,
    prefetch_factor: int | None = 2,
    strict_fallback: bool = True,
) -> Dict[str, Any]:
    """Validate representative no-dispatch/fallback decisions for cache reader promotion."""
    resolved_batch_size = max(int(batch_size or 1), 1)
    cases = [
        _build_case(
            dataset,
            case,
            batch_size=resolved_batch_size,
            prefetch_factor=prefetch_factor,
            strict_fallback=strict_fallback,
        )
        for case in MATRIX_CASES
    ]
    passed = [case for case in cases if bool(case.get("ok", False))]
    failed = [case for case in cases if not bool(case.get("ok", False))]
    all_blockers = _dedupe(blocker for case in cases for blocker in list(case.get("native_dispatch_blockers", []) or []))
    return {
        "schema_version": 1,
        "provider": "native_cache_reader_dispatch_fallback_matrix_v1",
        "ok": not failed,
        "debug_only": True,
        "shadow_run": True,
        "strict_fallback": bool(strict_fallback),
        "strict_fallback_matrix_passed": not failed,
        "case_count": len(cases),
        "passed_case_count": len(passed),
        "failed_case_count": len(failed),
        "failed_case_ids": [str(case.get("case_id") or "") for case in failed],
        "dataset_class": type(dataset).__name__,
        "batch_size": resolved_batch_size,
        "prefetch_factor": prefetch_factor,
        "native_dispatch_eligible": False,
        "would_allow_native_dispatch": False,
        "fallback_to_python_batch": True,
        "native_dispatch_blockers": all_blockers,
        "representative_fallback_matrix_passed": not failed,
        "representative_training_matrix_passed": False,
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "cases": cases,
    }


__all__ = ["build_cache_reader_dispatch_fallback_matrix"]
