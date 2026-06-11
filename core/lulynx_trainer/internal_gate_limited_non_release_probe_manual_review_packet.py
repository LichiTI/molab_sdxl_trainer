"""Report-only manual review packet for a limited non-release probe.

This packet consolidates the batch1-only default-off chain into one manual
review artifact after the manual runbook package is ready. It never enables the
internal gate, starts training, or opens release claims.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_MANUAL_REVIEW_PACKET = (
    "lulynx_internal_gate_limited_non_release_probe_manual_review_packet_v0"
)
READY_RUNBOOK_STATUS = "ready_for_manual_batch1_non_release_probe_runbook"


def build_lulynx_internal_gate_limited_non_release_probe_manual_review_packet(
    *,
    internal_gate_limited_non_release_probe_manual_runbook_package: Mapping[str, Any] | None = None,
    internal_gate_enablement_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed manual review packet for a future batch1 probe."""

    runbook_package = dict(_mapping(internal_gate_limited_non_release_probe_manual_runbook_package))
    gate_review = dict(_mapping(internal_gate_enablement_review))
    runbook_summary = _runbook_summary(runbook_package)
    gate_review_summary = _gate_review_summary(gate_review)
    checks = _checks(runbook_package=runbook_package, gate_review=gate_review)
    blockers = _blockers(
        runbook_package=runbook_package,
        gate_review=gate_review,
        checks=checks,
    )
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_MANUAL_REVIEW_PACKET,
        "status": "ready_for_manual_batch1_non_release_probe_review_packet" if ready else "blocked",
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
        "manual_runbook_package_summary": runbook_summary,
        "internal_gate_enablement_review_summary": gate_review_summary,
        "review_packet": _review_packet(runbook_package, gate_review),
        "review_sections": _review_sections(runbook_package, gate_review),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    runbook_package: Mapping[str, Any],
    gate_review: Mapping[str, Any],
) -> dict[str, bool]:
    runbook = _mapping(runbook_package.get("manual_runbook"))
    evidence_template = _mapping(runbook_package.get("before_after_evidence_template"))
    stop_inventory = _mapping(runbook_package.get("stop_conditions_inventory"))
    gate_checks = _mapping(gate_review.get("checks"))
    return {
        "manual_runbook_package_present": bool(runbook_package),
        "manual_runbook_package_ready": bool(runbook_package.get("passed"))
        and str(runbook_package.get("status") or "") == READY_RUNBOOK_STATUS,
        "internal_gate_enablement_not_allowed": not bool(
            runbook_package.get("internal_gate_enablement_allowed")
        ),
        "release_claim_closed": not bool(runbook_package.get("release_claim_allowed")),
        "forbidden_actions_visible": {
            "turn_internal_gate_on_now",
            "start_training_now",
            "approve_batch2_4_8_release_probe",
            "approve_release_claim",
        }.issubset(set(_string_list(runbook.get("forbidden_actions")))),
        "before_after_template_present": bool(evidence_template),
        "stop_conditions_inventory_present": bool(stop_inventory),
        "gate_review_present": bool(gate_review),
        "gate_review_ready": bool(gate_review.get("passed"))
        and str(gate_review.get("status") or "") == "ready_for_manual_internal_gate_review",
        "gate_review_keeps_batch2_release_probe_blocked": bool(
            gate_checks.get("batch2_release_probe_still_blocked")
        ),
        "gate_review_keeps_execution_path_disabled": bool(
            gate_checks.get("behavior_equivalent_execution_path_not_enabled")
        ),
    }


def _blockers(
    *,
    runbook_package: Mapping[str, Any],
    gate_review: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not runbook_package:
        blockers.append("internal_gate_limited_non_release_probe_manual_runbook_package_missing")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_manual_runbook_package:{item}"
        for item in _string_list(runbook_package.get("blockers"))
    )
    if not gate_review:
        blockers.append("internal_gate_enablement_review_missing")
    blockers.extend(
        f"internal_gate_enablement_review:{item}"
        for item in _string_list(gate_review.get("blockers"))
    )
    return _dedupe(blockers)


def _runbook_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "internal_gate_enablement_allowed": bool(report.get("internal_gate_enablement_allowed")),
        "release_claim_allowed": bool(report.get("release_claim_allowed")),
        "blocker_count": len(_string_list(report.get("blockers"))),
    }


def _gate_review_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "release_claim_allowed": bool(report.get("release_claim_allowed")),
        "deferred_release_probe_blockers": _string_list(report.get("deferred_release_probe_blockers")),
    }


def _review_packet(
    runbook_package: Mapping[str, Any],
    gate_review: Mapping[str, Any],
) -> dict[str, Any]:
    runbook = _mapping(runbook_package.get("manual_runbook"))
    evidence_template = _mapping(runbook_package.get("before_after_evidence_template"))
    stop_inventory = _mapping(runbook_package.get("stop_conditions_inventory"))
    return {
        "packet_kind": "manual_batch1_non_release_probe_review_packet_v0",
        "review_scope": "behavior_equivalent_internal_gate_batch1_non_release_probe",
        "review_mode": "manual_only",
        "release_policy": "non_release_only",
        "required_inputs": [
            "internal_gate_enablement_review",
            "limited_non_release_probe_execution_contract",
            "manual_runbook_package",
            "before_after_evidence_template",
            "stop_conditions_inventory",
        ],
        "forbidden_approvals": _string_list(runbook.get("forbidden_actions")),
        "deferred_release_probe_blockers": _string_list(
            gate_review.get("deferred_release_probe_blockers")
        ),
        "required_before_fields": _string_list(evidence_template.get("required_before_fields")),
        "required_after_fields": _string_list(evidence_template.get("required_after_fields")),
        "required_stop_condition_ids": [
            str(item.get("id") or "")
            for item in _sequence_of_mappings(stop_inventory.get("required_stop_conditions"))
            if item.get("id")
        ],
    }


def _review_sections(
    runbook_package: Mapping[str, Any],
    gate_review: Mapping[str, Any],
) -> list[dict[str, Any]]:
    runbook = _mapping(runbook_package.get("manual_runbook"))
    evidence_template = _mapping(runbook_package.get("before_after_evidence_template"))
    stop_inventory = _mapping(runbook_package.get("stop_conditions_inventory"))
    return [
        {
            "id": "scope_and_boundaries",
            "title": "Batch1 Scope And Boundaries",
            "items": [
                "batch1_only_non_release_probe",
                "internal_gate_stays_default_off",
                "no_new_training_entrypoint",
                "batch2_4_8_release_probe_remains_blocked",
            ],
        },
        {
            "id": "manual_runbook",
            "title": "Manual Runbook",
            "items": _string_list(runbook.get("operator_steps")),
        },
        {
            "id": "stop_conditions",
            "title": "Stop Conditions",
            "items": [
                str(item.get("id") or "")
                for item in _sequence_of_mappings(stop_inventory.get("required_stop_conditions"))
                if item.get("id")
            ],
        },
        {
            "id": "before_after_evidence",
            "title": "Before After Evidence",
            "items": _string_list(evidence_template.get("required_comparisons")),
        },
        {
            "id": "deferred_release_probe_blockers",
            "title": "Deferred Release Probe Blockers",
            "items": _string_list(gate_review.get("deferred_release_probe_blockers")),
        },
    ]


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_signed_manual_probe_review_record_without_enabling_internal_gate",
            "refresh_batch1_before_after_baseline_evidence_inputs",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_manual_probe_review_packet_prerequisites"]
    if any("manual_runbook_package" in item for item in blockers):
        actions.append("refresh_manual_runbook_package")
    if any("internal_gate_enablement_review" in item for item in blockers):
        actions.append("refresh_internal_gate_enablement_review")
    if any("release_claim" in item for item in blockers):
        actions.append("close_release_claim_leaks_before_manual_probe_review_packet")
    return _dedupe(actions)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _sequence_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [item for item in value if isinstance(item, Mapping)]


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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_MANUAL_REVIEW_PACKET",
    "build_lulynx_internal_gate_limited_non_release_probe_manual_review_packet",
]
