"""Report-only runtime adapter shadow for the native TurboCore data pipeline."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_native_data_pipeline_observe_scorecard import (
    build_native_data_pipeline_observe_scorecard,
)


FEATURE = "native_data_pipeline"
ADAPTER_KIND = "native_data_pipeline_training_adapter_shadow_v0"
FALLBACK_BACKEND = "standardcore_python_data_path"


def build_native_data_pipeline_adapter_shadow_scorecard(
    *,
    observe_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "observe",
    sample_count: int = 64,
    batch_size: int = 4,
    prefetch_depth: int = 8,
    chunk_size: int = 4,
) -> dict[str, Any]:
    """Build the future training adapter envelope without dispatching it."""

    observe = dict(
        observe_report
        or build_native_data_pipeline_observe_scorecard(
            sample_count=sample_count,
            batch_size=batch_size,
            prefetch_depth=prefetch_depth,
            chunk_size=chunk_size,
            native_training_mode=native_training_mode,
        )
    )
    route = _adapter_route(observe)
    envelope = _adapter_envelope(observe, route)
    validations = _validations(observe, route, envelope)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_native_data_pipeline_adapter_shadow_scorecard_v0",
        "gate": "p6i_native_data_pipeline_adapter_shadow",
        "ok": ready,
        "promotion_ready": ready,
        "adapter_shadow_ready": ready,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "native_shadow_call_allowed": False,
        "fallback_backend_authoritative": True,
        "adapter_kind": ADAPTER_KIND,
        "feature": FEATURE,
        "fallback_backend": FALLBACK_BACKEND,
        "native_training_mode": str(observe.get("native_training_mode") or native_training_mode),
        "adapter_route": route,
        "adapter_envelope": envelope,
        "observe_summary": dict(observe.get("summary") or {}),
        "validations": validations,
        "summary": {
            "adapter_shadow_ready": ready,
            "adapter_decision": route.get("decision"),
            "fallback_backend_authoritative": True,
            "native_shadow_call_allowed": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
        },
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add native data pipeline semantic parity and H2D ownership matrix before canary"
            if ready
            else "fix native data pipeline adapter shadow blockers"
        ),
        "notes": [
            "This adapter shadow is an internal runtime/request contract only.",
            "The StandardCore Python data path remains authoritative.",
            "No native dataset, collate, or H2D transfer dispatch is enabled here.",
        ],
    }


def _adapter_route(observe: Mapping[str, Any]) -> dict[str, Any]:
    observe_ready = bool(observe.get("observe_manifest_ready", False))
    observe_route = observe.get("route_decision")
    observe_route = observe_route if isinstance(observe_route, Mapping) else {}
    if observe_ready:
        decision = "shadow_adapter_prepared_fallback_authoritative"
        reason = "runtime_dispatch_disabled_pending_review"
    else:
        decision = "fallback"
        reason = "native_data_pipeline_observe_manifest_missing"
    return {
        "schema_version": 1,
        "adapter_kind": ADAPTER_KIND,
        "feature": FEATURE,
        "decision": decision,
        "reason": reason,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "native_shadow_call_allowed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "observe_decision": observe_route.get("decision"),
        "missing_before_dispatch": [
            "manual_integration_review",
            "dataset_semantic_parity_matrix",
            "h2d_transfer_ownership_contract",
            "end_to_end_training_shadow",
            "rollback_manifest",
        ],
    }


def _adapter_envelope(observe: Mapping[str, Any], route: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "adapter_kind": ADAPTER_KIND,
        "feature": FEATURE,
        "training_data_authority": FALLBACK_BACKEND,
        "native_data_authority": "none",
        "fallback_backend": route.get("fallback_backend"),
        "runtime_request_fields": [
            "native_training_mode",
            "dataset_kind",
            "cache_first",
            "sample_manifest_uri",
            "batch_size",
            "resolution_bucket",
            "caption_mode",
            "prefetch_depth",
            "chunk_size",
            "h2d_transfer_mode",
            "fallback_backend",
        ],
        "required_evidence": [
            "native_data_pipeline_observe_manifest",
            "dataset_semantic_parity_matrix",
            "h2d_transfer_ownership_contract",
            "end_to_end_training_shadow",
            "rollback_manifest",
        ],
        "observed_manifest_kind": observe.get("manifest_kind"),
        "observed_components": dict((observe.get("manifest") or {}).get("observed_components") or {}),
        "canary_dispatch_armed": False,
        "native_shadow_call_allowed": False,
        "fallback_backend_authoritative": True,
        "training_path_enabled": False,
    }


def _validations(
    observe: Mapping[str, Any],
    route: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p6h_observe_manifest_ready",
            bool(observe.get("observe_manifest_ready", False)),
            "native_data_pipeline_observe_manifest_missing",
        ),
        _validation(
            "adapter_shadow_envelope_ready",
            bool(envelope.get("runtime_request_fields"))
            and envelope.get("training_data_authority") == FALLBACK_BACKEND
            and envelope.get("native_data_authority") == "none",
            "native_data_pipeline_adapter_shadow_envelope_missing",
        ),
        _validation(
            "fallback_backend_authoritative",
            bool(route.get("fallback_backend_authoritative", False))
            and bool(envelope.get("fallback_backend_authoritative", False)),
            "native_data_pipeline_adapter_shadow_non_authoritative_fallback",
        ),
        _validation(
            "native_shadow_call_disabled",
            not bool(route.get("native_shadow_call_allowed", True))
            and not bool(envelope.get("native_shadow_call_allowed", True)),
            "native_data_pipeline_adapter_shadow_enabled_native_call",
        ),
        _validation(
            "runtime_dispatch_still_disabled",
            not bool(route.get("runtime_dispatch_ready", True))
            and not bool(route.get("native_dispatch_allowed", True))
            and not bool(route.get("training_path_enabled", True)),
            "native_data_pipeline_adapter_shadow_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(observe.get("training_path_enabled", True))
            and not bool(observe.get("default_behavior_changed", True)),
            "native_data_pipeline_adapter_shadow_changed_default_behavior",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_native_data_pipeline_adapter_shadow_scorecard"]
