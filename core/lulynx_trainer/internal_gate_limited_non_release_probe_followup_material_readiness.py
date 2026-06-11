"""Readiness gate for batch1-only follow-up manual probe materials.

This gate consumes the follow-up material package plus current batch1 evidence
and first-release readiness. It verifies that manual-only materials are ready
without enabling the internal gate, starting training, or relaxing release
boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_FOLLOWUP_MATERIAL_READINESS = (
    "lulynx_internal_gate_limited_non_release_probe_followup_material_readiness_v0"
)
READY_MATERIAL_PACKAGE_STATUS = "ready_for_followup_manual_probe_material_package"


def build_lulynx_internal_gate_limited_non_release_probe_followup_material_readiness(
    *,
    internal_gate_limited_non_release_probe_followup_material_package: Mapping[str, Any] | None = None,
    real_gpu_batch1_golden_evidence: Mapping[str, Any] | None = None,
    first_release_readiness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed readiness report for follow-up manual probe materials."""

    material_package = dict(_mapping(internal_gate_limited_non_release_probe_followup_material_package))
    golden = dict(_mapping(real_gpu_batch1_golden_evidence))
    first_release = dict(_mapping(first_release_readiness))
    checks = _checks(
        material_package=material_package,
        golden=golden,
        first_release=first_release,
    )
    blockers = _blockers(
        material_package=material_package,
        golden=golden,
        first_release=first_release,
        checks=checks,
    )
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_FOLLOWUP_MATERIAL_READINESS,
        "status": "ready_for_followup_manual_probe_material_readiness" if ready else "blocked",
        "passed": ready,
        "manual_review_required": True,
        "safe_to_auto_start": False,
        "internal_gate_enablement_allowed": False,
        "release_claim_allowed": False,
        "does_not_add_training_entrypoint": True,
        "does_not_start_gpu_work": True,
        "does_not_start_dataloader_iteration": True,
        "checks": checks,
        "blockers": blockers,
        "followup_material_package_summary": _material_package_summary(material_package),
        "real_gpu_batch1_golden_summary": _golden_summary(golden),
        "first_release_readiness_summary": _first_release_summary(first_release),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    material_package: Mapping[str, Any],
    golden: Mapping[str, Any],
    first_release: Mapping[str, Any],
) -> dict[str, bool]:
    payload = _mapping(material_package.get("followup_material_package"))
    sections = set(_string_list(payload.get("material_sections")))
    forbidden = set(_string_list(payload.get("forbidden_approvals")))
    deferred = set(_string_list(payload.get("deferred_release_probe_blockers")))
    golden_checks = _mapping(golden.get("checks"))
    first_release_summary = _mapping(first_release.get("summary"))
    return {
        "followup_material_package_present": bool(material_package),
        "followup_material_package_ready": bool(material_package.get("passed"))
        and str(material_package.get("status") or "") == READY_MATERIAL_PACKAGE_STATUS,
        "followup_material_package_default_off": (
            not bool(material_package.get("internal_gate_enablement_allowed"))
            and not bool(material_package.get("release_claim_allowed"))
        ),
        "material_sections_complete": {
            "signed_review_context",
            "baseline_before_evidence_inputs",
            "after_probe_evidence_destination_template",
            "stop_condition_visibility_checklist",
            "deferred_release_probe_blocker_notice",
        }.issubset(sections),
        "batch2_release_probe_still_blocked": "approve_batch2_4_8_release_probe" in forbidden,
        "gate_enablement_still_blocked": "turn_internal_gate_on_now" in forbidden,
        "training_start_still_blocked": "start_training_now" in forbidden,
        "release_claim_still_blocked": "approve_release_claim" in forbidden,
        "deferred_multi_batch_blocker_visible": (
            "multi_batch_promotion_gate:not_real_physical_multi_batch" in deferred
        ),
        "real_gpu_batch1_golden_ready": bool(golden.get("passed"))
        and str(golden.get("status") or "") == "ready_for_internal_gate_enablement_review",
        "real_gpu_batch1_release_claim_closed": not bool(golden.get("release_claim_allowed")),
        "real_gpu_batch1_orchestrator_gate_disabled": bool(
            golden_checks.get("manifest_orchestrator_gate_disabled")
        )
        and bool(golden_checks.get("runtime_orchestrator_gate_disabled")),
        "first_release_ready": bool(first_release.get("release_ready")),
        "first_release_experimental_claim_gates_closed": bool(
            first_release_summary.get("experimental_claim_gates_closed")
        ),
    }


def _blockers(
    *,
    material_package: Mapping[str, Any],
    golden: Mapping[str, Any],
    first_release: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not material_package:
        blockers.append("internal_gate_limited_non_release_probe_followup_material_package_missing")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_followup_material_package:{item}"
        for item in _string_list(material_package.get("blockers"))
    )
    if not golden:
        blockers.append("real_gpu_batch1_golden_evidence_missing")
    blockers.extend(
        f"real_gpu_batch1_golden_evidence:{item}"
        for item in _string_list(golden.get("blockers"))
    )
    if not first_release:
        blockers.append("first_release_readiness_missing")
    blockers.extend(
        f"first_release_readiness:{item}" for item in _string_list(first_release.get("release_blockers"))
    )
    return _dedupe(blockers)


def _material_package_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("followup_material_package"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "material_scope": str(payload.get("material_scope") or ""),
        "forbidden_approval_count": len(_string_list(payload.get("forbidden_approvals"))),
        "deferred_release_probe_blockers": _string_list(
            payload.get("deferred_release_probe_blockers")
        ),
    }


def _golden_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "release_claim_allowed": bool(report.get("release_claim_allowed")),
        "multi_batch_promotion_gate_blockers": _string_list(
            report.get("multi_batch_promotion_gate_blockers")
        ),
    }


def _first_release_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    return {
        "present": bool(report),
        "release_ready": bool(report.get("release_ready")),
        "release_blocker_count": len(_string_list(report.get("release_blockers"))),
        "experimental_claim_gates_closed": bool(summary.get("experimental_claim_gates_closed")),
        "core_release_smoke_covered": bool(summary.get("core_release_smoke_covered")),
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "keep_followup_materials_ready_while_internal_gate_stays_disabled",
            "refresh_batch1_baseline_inputs_when_new_manifest_evidence_arrives",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_followup_material_readiness_prerequisites"]
    if any("followup_material_package" in item for item in blockers):
        actions.append("refresh_followup_material_package")
    if any("real_gpu_batch1_golden_evidence" in item for item in blockers):
        actions.append("refresh_real_gpu_batch1_golden_evidence")
    if any("first_release_readiness" in item for item in blockers):
        actions.append("refresh_first_release_readiness")
    return _dedupe(actions)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


__all__ = [
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_FOLLOWUP_MATERIAL_READINESS",
    "build_lulynx_internal_gate_limited_non_release_probe_followup_material_readiness",
]
