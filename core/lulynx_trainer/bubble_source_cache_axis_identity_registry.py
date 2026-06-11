"""JSON-only identity registry for GPU-bubble source/cache axes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .bubble_source_axis_freshness_dedupe_audit import (
    ROADMAP,
    build_source_axis_freshness_dedupe_audit,
)


SOURCE_CACHE_AXIS_IDENTITY_REGISTRY_REPORT = "bubble_source_cache_axis_identity_registry_v0"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def build_source_cache_axis_identity_registry(
    *,
    external_input_intake_registry: Mapping[str, Any] | None = None,
    source_axis_scout: Mapping[str, Any] | None = None,
    source_axis_requirement: Mapping[str, Any] | None = None,
    source_cache_axis_admission_preflight: Mapping[str, Any] | None = None,
    source_cache_axis_manual_canary_plan: Mapping[str, Any] | None = None,
    external_input_replay_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the standalone axis identity registry without scanning or GPU work."""

    audit = build_source_axis_freshness_dedupe_audit(
        external_input_intake_registry=external_input_intake_registry,
        source_axis_scout=source_axis_scout,
        source_axis_requirement=source_axis_requirement,
        source_cache_axis_admission_preflight=source_cache_axis_admission_preflight,
        source_cache_axis_manual_canary_plan=source_cache_axis_manual_canary_plan,
        external_input_replay_plan=external_input_replay_plan,
    )
    registry = dict(_mapping(audit.get("source_cache_axis_identity_registry")))
    registry.update(
        {
            "schema_version": 1,
            "report": SOURCE_CACHE_AXIS_IDENTITY_REGISTRY_REPORT,
            "roadmap": ROADMAP,
            "status": str(audit.get("axis_state") or audit.get("status") or ""),
            "axis_state": str(audit.get("axis_state") or ""),
            "source_axis_freshness_status": str(audit.get("status") or ""),
            "candidate_status": str(_mapping(audit.get("candidate_audit")).get("status") or ""),
            "does_not_run_training": True,
            "does_not_run_cuda": True,
            "publishable": False,
            "not_release_evidence": True,
            "safe_to_auto_start": False,
            "release_claim_allowed": False,
            "acceptance_gates": [
                "axis_identity_digest_present",
                "full_axis_identity_requires_family_root_offset_manifest",
                "duplicate_axis_digest_blocks_manual_gpu_readiness",
                "identity_registry_is_not_release_evidence",
            ],
        }
    )
    return registry


__all__ = [
    "ROADMAP",
    "SOURCE_CACHE_AXIS_IDENTITY_REGISTRY_REPORT",
    "build_source_cache_axis_identity_registry",
]
