"""Report-only scorecard for the native TurboCore data pipeline candidate."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.services.native_module_loader import ensure_lulynx_native_artifact_path, probe_lulynx_native_loader
from core.turbocore_capabilities import probe_native_training_bridge
from core.turbocore_dataset_staging import (
    plan_dataset_staging,
    run_native_dataset_staging_lazy_fast_bulk_pipeline_probe,
)
from core.turbocore_dataset_staging_session import run_native_dataset_descriptor_session_probe
from core.turbocore_native_abi import validate_workspace_pipeline_native_capabilities
from core.turbocore_workspace_pipeline import run_workspace_pipeline_lifecycle_probe


MANIFEST_KIND = "native_data_pipeline_observe_manifest_v0"
FEATURE = "native_data_pipeline"


def build_native_data_pipeline_observe_scorecard(
    *,
    sample_count: int = 64,
    batch_size: int = 4,
    prefetch_depth: int = 8,
    chunk_size: int = 4,
    native_training_mode: str = "observe",
) -> dict[str, Any]:
    """Validate the native data pipeline as an observe-only runtime candidate."""

    ensure_lulynx_native_artifact_path()
    mode = _normalize_mode(native_training_mode)
    loader = probe_lulynx_native_loader()
    bridge = probe_native_training_bridge()
    bridge_validation = _safe_probe(
        "workspace_pipeline_native_abi_validation",
        lambda: validate_workspace_pipeline_native_capabilities(bridge),
    )
    workspace_lifecycle = _safe_probe(
        "workspace_pipeline_lifecycle",
        lambda: run_workspace_pipeline_lifecycle_probe(
            batches=max(int(sample_count) // max(int(batch_size), 1), 1),
            prefetch_depth=max(int(prefetch_depth), 1),
            chunk_size=max(int(chunk_size), 1),
            prefer_native=True,
        ),
    )
    sequential_plan = _safe_probe(
        "dataset_staging_sequential_plan",
        lambda: plan_dataset_staging(
            sample_count=max(int(sample_count), 0),
            batch_size=max(int(batch_size), 1),
            drop_last=False,
            shuffle=False,
            seed=17,
            prefetch_depth=max(int(prefetch_depth), 1),
            chunk_size=max(int(chunk_size), 1),
            prefer_native=True,
        ),
    )
    shuffled_plan = _safe_probe(
        "dataset_staging_shuffle_plan",
        lambda: plan_dataset_staging(
            sample_count=max(int(sample_count), 0),
            batch_size=max(int(batch_size), 1),
            drop_last=True,
            shuffle=True,
            seed=29,
            prefetch_depth=max(int(prefetch_depth), 1),
            chunk_size=max(int(chunk_size), 1),
            prefer_native=True,
        ),
    )
    lazy_fast_probe = _safe_probe(
        "dataset_staging_lazy_fast_pipeline",
        lambda: run_native_dataset_staging_lazy_fast_bulk_pipeline_probe(
            sample_count=max(int(sample_count), 0),
            batch_size=max(int(batch_size), 1),
            drop_last=True,
            seed=29,
            prefetch_depth=max(int(prefetch_depth), 1),
            chunk_size=max(int(chunk_size), 1),
        ),
    )
    descriptor_probe = _safe_probe(
        "dataset_descriptor_session",
        lambda: run_native_dataset_descriptor_session_probe(
            _descriptor_manifest(),
            batch_size=2,
            drop_last=False,
            prefetch_depth=max(int(prefetch_depth), 1),
            chunk_size=2,
            epochs=2,
        ),
    )
    probes = {
        "bridge_validation": bridge_validation,
        "workspace_lifecycle": workspace_lifecycle,
        "sequential_plan": sequential_plan,
        "shuffled_plan": shuffled_plan,
        "lazy_fast_probe": lazy_fast_probe,
        "descriptor_probe": descriptor_probe,
    }
    decision = _route_decision(probes, mode)
    manifest = _manifest(bridge, probes, decision, mode)
    validations = _validations(loader, bridge, probes, decision, manifest, mode)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_native_data_pipeline_observe_scorecard_v0",
        "gate": "p6h_native_data_pipeline_observe",
        "ok": True,
        "promotion_ready": ready,
        "observe_manifest_ready": ready,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "experimental_only": True,
        "manifest_kind": MANIFEST_KIND,
        "feature": FEATURE,
        "native_training_mode": mode,
        "sample_count": max(int(sample_count), 0),
        "batch_size": max(int(batch_size), 1),
        "prefetch_depth": max(int(prefetch_depth), 1),
        "chunk_size": max(int(chunk_size), 1),
        "loader": loader,
        "capability_summary": _capability_summary(bridge),
        "route_decision": decision,
        "manifest": manifest,
        "probes": probes,
        "validations": validations,
        "summary": {
            "observe_manifest_ready": ready,
            "decision": decision.get("decision"),
            "reason": decision.get("reason"),
            "candidate_recorded": bool(manifest.get("candidate_recorded", False)),
            "workspace_lifecycle_ok": bool(workspace_lifecycle.get("ok", False)),
            "shuffle_native_ok": bool(shuffled_plan.get("native_runtime", False)),
            "lazy_fast_pipeline_ok": bool(lazy_fast_probe.get("ok", False)),
            "descriptor_session_ok": bool(descriptor_probe.get("ok", False)),
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
        },
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add native data pipeline training adapter shadow before real dispatch"
            if ready
            else "fix native data pipeline observe blockers"
        ),
        "notes": [
            "This scorecard validates native workspace, staging, lazy sampler, and descriptor-session evidence.",
            "Observe mode records a future data-pipeline route but never dispatches it.",
            "Real training integration still needs semantic parity, H2D transfer ownership, and rollback review.",
        ],
    }


def _safe_probe(name: str, fn: Any) -> dict[str, Any]:
    try:
        payload = fn()
    except Exception as exc:
        return {
            "schema_version": 1,
            "probe": name,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": [f"{name}_failed:{type(exc).__name__}"],
            "training_path_enabled": False,
        }
    if isinstance(payload, dict):
        return payload
    return {
        "schema_version": 1,
        "probe": name,
        "ok": False,
        "error": f"invalid_probe_payload:{type(payload).__name__}",
        "blocked_reasons": [f"{name}_invalid_payload"],
        "training_path_enabled": False,
    }


def _route_decision(probes: Mapping[str, Mapping[str, Any]], mode: str) -> dict[str, Any]:
    ready = _probes_ready(probes)
    if mode == "off":
        decision = "off"
        reason = "native_training_mode_off"
        candidate = False
    elif not ready:
        decision = "fallback"
        reason = "native_data_pipeline_observe_not_ready"
        candidate = False
    elif mode == "observe":
        decision = "would_select_native_data_pipeline_observe_but_dispatch_disabled"
        reason = "observe_mode_records_candidate_only"
        candidate = True
    else:
        decision = "blocked_before_native_data_pipeline_canary"
        reason = "native_data_pipeline_training_integration_review_required"
        candidate = True
    return {
        "schema_version": 1,
        "manifest_kind": MANIFEST_KIND,
        "feature": FEATURE,
        "native_training_mode": mode,
        "decision": decision,
        "reason": reason,
        "request_supported": bool(ready),
        "candidate_recorded": candidate,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "missing_before_dispatch": [
            "manual_integration_review",
            "training_dataset_adapter_shadow",
            "dataset_semantic_parity_matrix",
            "h2d_transfer_ownership_contract",
            "fallback_rollback_manifest",
        ],
    }


def _manifest(
    bridge: Mapping[str, Any],
    probes: Mapping[str, Mapping[str, Any]],
    decision: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest_kind": MANIFEST_KIND,
        "feature": FEATURE,
        "native_training_mode": mode,
        "candidate_recorded": bool(decision.get("candidate_recorded", False)),
        "capabilities": _capability_summary(bridge),
        "observed_components": {
            "workspace_pool": bool(probes.get("workspace_lifecycle", {}).get("native_runtime", False)),
            "data_pipeline": bool(probes.get("workspace_lifecycle", {}).get("native_runtime", False)),
            "dataset_staging": bool(probes.get("shuffled_plan", {}).get("native_runtime", False)),
            "lazy_affine_sampler": bool(probes.get("lazy_fast_probe", {}).get("native_runtime", False)),
            "descriptor_session": bool(probes.get("descriptor_probe", {}).get("native_runtime", False)),
        },
        "audit_fields": [
            "native_training_mode",
            "dataset_kind",
            "cache_first",
            "batch_size",
            "resolution_bucket",
            "prefetch_depth",
            "chunk_size",
            "route_decision",
            "fallback_reason",
            "queue_empty_stalls",
            "queue_full_stalls",
        ],
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
    }


def _validations(
    loader: Mapping[str, Any],
    bridge: Mapping[str, Any],
    probes: Mapping[str, Mapping[str, Any]],
    decision: Mapping[str, Any],
    manifest: Mapping[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    workspace = probes.get("workspace_lifecycle", {})
    shuffled = probes.get("shuffled_plan", {})
    lazy = probes.get("lazy_fast_probe", {})
    descriptor = probes.get("descriptor_probe", {})
    validation = probes.get("bridge_validation", {})
    return [
        _validation(
            "native_module_importable",
            bool(loader.get("importable", False)),
            "native_data_pipeline_lulynx_native_not_importable",
        ),
        _validation(
            "workspace_pipeline_abi_valid",
            bool(validation.get("ok", False)),
            "native_data_pipeline_workspace_abi_incomplete",
        ),
        _validation(
            "workspace_lifecycle_native_probe",
            bool(workspace.get("ok", False))
            and bool(workspace.get("native_runtime", False))
            and not bool(workspace.get("training_path_enabled", True)),
            "native_data_pipeline_workspace_lifecycle_failed",
        ),
        _validation(
            "dataset_shuffle_native_plan",
            bool(shuffled.get("ok", False))
            and bool(shuffled.get("native_runtime", False))
            and str(shuffled.get("provider", "")) == "native_dataset_staging"
            and not bool(shuffled.get("indices_returned", True)),
            "native_data_pipeline_shuffle_plan_not_native",
        ),
        _validation(
            "lazy_affine_summary_pipeline",
            bool(lazy.get("ok", False))
            and bool(lazy.get("native_runtime", False))
            and bool(lazy.get("runtime_summary_only", False))
            and not bool(lazy.get("native_index_materialized", True)),
            "native_data_pipeline_lazy_affine_summary_failed",
        ),
        _validation(
            "descriptor_session_native_probe",
            bool(descriptor.get("ok", False))
            and bool(descriptor.get("native_runtime", False))
            and bool(descriptor.get("sample_descriptors_owned", False))
            and bool(descriptor.get("descriptor_parity_ok", False)),
            "native_data_pipeline_descriptor_session_failed",
        ),
        _validation(
            "observe_route_decision_ready",
            mode != "off"
            and decision.get("decision")
            in {
                "would_select_native_data_pipeline_observe_but_dispatch_disabled",
                "blocked_before_native_data_pipeline_canary",
            },
            "native_data_pipeline_observe_route_decision_missing",
        ),
        _validation(
            "candidate_manifest_records_contract",
            bool(manifest.get("audit_fields")) and bool(manifest.get("observed_components")),
            "native_data_pipeline_observe_manifest_contract_missing",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(decision.get("runtime_dispatch_ready", True))
            and not bool(manifest.get("native_dispatch_allowed", True))
            and not bool(manifest.get("training_path_enabled", True)),
            "native_data_pipeline_observe_manifest_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(bridge.get("training_path_enabled", True)),
            "native_data_pipeline_changed_default_behavior",
        ),
    ]


def _descriptor_manifest() -> dict[str, Any]:
    return {
        "samples": [
            {
                "id": "sample_0001",
                "path": "samples/sample_0001.png",
                "caption_path": "samples/sample_0001.txt",
                "width": 512,
                "height": 768,
                "bucket": "512x768",
            },
            {
                "id": "sample_0002",
                "path": "samples/sample_0002.png",
                "caption_path": "samples/sample_0002.txt",
                "width": 768,
                "height": 512,
                "bucket": "768x512",
            },
            {
                "id": "sample_0003",
                "path": "samples/sample_0003.png",
                "caption_path": "samples/sample_0003.txt",
                "width": 512,
                "height": 768,
                "bucket": "512x768",
            },
        ]
    }


def _probes_ready(probes: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        bool(probes.get(name, {}).get("ok", False))
        for name in (
            "bridge_validation",
            "workspace_lifecycle",
            "sequential_plan",
            "shuffled_plan",
            "lazy_fast_probe",
            "descriptor_probe",
        )
    )


def _capability_summary(bridge: Mapping[str, Any]) -> dict[str, Any]:
    features = bridge.get("features") if isinstance(bridge.get("features"), Mapping) else {}
    return {
        "training_bridge_status": str(bridge.get("status", "")),
        "training_path_enabled": bool(bridge.get("training_path_enabled", False)),
        "workspace_pool": _feature_summary(features.get("workspace_pool") if isinstance(features, Mapping) else {}),
        "data_pipeline": _feature_summary(features.get("data_pipeline") if isinstance(features, Mapping) else {}),
        "dataset_staging": _feature_summary(features.get("dataset_staging") if isinstance(features, Mapping) else {}),
    }


def _feature_summary(value: Any) -> dict[str, Any]:
    payload = value if isinstance(value, Mapping) else {}
    return {
        "available": bool(payload.get("available", False)),
        "status": str(payload.get("status", "")),
        "reason": str(payload.get("reason", "")),
        "training_path_enabled": bool(payload.get("training_path_enabled", False)),
    }


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _normalize_mode(value: str) -> str:
    normalized = str(value or "observe").strip().lower()
    return normalized if normalized in {"off", "observe", "canary", "auto"} else "observe"


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_native_data_pipeline_observe_scorecard"]
