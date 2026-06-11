"""Report-only follow-up material package for a limited non-release probe.

This package prepares the next manual-only material bundle after a signed
manual review record. It stays on the planning/documentation side and never
starts a probe or changes runtime behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_FOLLOWUP_MATERIAL_PACKAGE = (
    "lulynx_internal_gate_limited_non_release_probe_followup_material_package_v0"
)
READY_RECORD_STATUS = "decision_record_ready"
APPROVED_RECORD_DECISION = (
    "internal_gate_limited_non_release_probe_manual_review_recorded_default_off"
)


def build_lulynx_internal_gate_limited_non_release_probe_followup_material_package(
    *,
    internal_gate_limited_non_release_probe_manual_review_record: Mapping[str, Any] | None = None,
    internal_gate_limited_non_release_probe_manual_review_packet: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a default-off follow-up material package for later manual prep."""

    record = dict(_mapping(internal_gate_limited_non_release_probe_manual_review_record))
    packet = dict(_mapping(internal_gate_limited_non_release_probe_manual_review_packet))
    checks = _checks(record=record, packet=packet)
    blockers = _blockers(record=record, packet=packet, checks=checks)
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_FOLLOWUP_MATERIAL_PACKAGE,
        "status": "ready_for_followup_manual_probe_material_package" if ready else "blocked",
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
        "manual_review_record_summary": _record_summary(record),
        "manual_review_packet_summary": _packet_summary(packet),
        "followup_material_package": _followup_material_package(packet),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(*, record: Mapping[str, Any], packet: Mapping[str, Any]) -> dict[str, bool]:
    review_packet = _mapping(packet.get("review_packet"))
    required_inputs = set(_string_list(review_packet.get("required_inputs")))
    forbidden = set(_string_list(review_packet.get("forbidden_approvals")))
    return {
        "manual_review_record_present": bool(record),
        "manual_review_record_ready": bool(record.get("ok"))
        and str(record.get("status") or "") == READY_RECORD_STATUS
        and str(record.get("decision") or "") == APPROVED_RECORD_DECISION
        and bool(record.get("approved_for_followup_manual_probe_preparation")),
        "manual_review_record_default_off": (
            not bool(record.get("internal_gate_enablement_allowed"))
            and not bool(record.get("release_claim_allowed"))
        ),
        "manual_review_packet_present": bool(packet),
        "manual_review_packet_ready": bool(packet.get("passed"))
        and str(packet.get("status") or "") == "ready_for_manual_batch1_non_release_probe_review_packet",
        "manual_review_packet_default_off": (
            not bool(packet.get("internal_gate_enablement_allowed"))
            and not bool(packet.get("release_claim_allowed"))
        ),
        "required_inputs_visible": {
            "internal_gate_enablement_review",
            "limited_non_release_probe_execution_contract",
            "manual_runbook_package",
            "before_after_evidence_template",
            "stop_conditions_inventory",
        }.issubset(required_inputs),
        "batch2_release_probe_still_blocked": "approve_batch2_4_8_release_probe" in forbidden,
        "gate_enablement_still_blocked": "turn_internal_gate_on_now" in forbidden,
        "training_start_still_blocked": "start_training_now" in forbidden,
        "release_claim_still_blocked": "approve_release_claim" in forbidden,
    }


def _blockers(
    *,
    record: Mapping[str, Any],
    packet: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not record:
        blockers.append("internal_gate_limited_non_release_probe_manual_review_record_missing")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_manual_review_record:{item}"
        for item in _string_list(record.get("blocked_reasons"))
    )
    if not packet:
        blockers.append("internal_gate_limited_non_release_probe_manual_review_packet_missing")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_manual_review_packet:{item}"
        for item in _string_list(packet.get("blockers"))
    )
    return _dedupe(blockers)


def _record_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "ok": bool(report.get("ok")),
        "decision": str(report.get("decision") or ""),
        "approved_for_followup_manual_probe_preparation": bool(
            report.get("approved_for_followup_manual_probe_preparation")
        ),
        "release_claim_allowed": bool(report.get("release_claim_allowed")),
        "internal_gate_enablement_allowed": bool(report.get("internal_gate_enablement_allowed")),
    }


def _packet_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    review_packet = _mapping(report.get("review_packet"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "review_scope": str(review_packet.get("review_scope") or ""),
        "required_input_count": len(_string_list(review_packet.get("required_inputs"))),
        "deferred_release_probe_blockers": _string_list(
            review_packet.get("deferred_release_probe_blockers")
        ),
    }


def _followup_material_package(packet: Mapping[str, Any]) -> dict[str, Any]:
    review_packet = _mapping(packet.get("review_packet"))
    sections = _sequence_of_mappings(packet.get("review_sections"))
    return {
        "package_kind": "followup_manual_probe_material_package_v0",
        "material_scope": "behavior_equivalent_internal_gate_batch1_non_release_probe",
        "material_policy": "manual_only_default_off_non_release_only",
        "material_sections": [
            "signed_review_context",
            "baseline_before_evidence_inputs",
            "after_probe_evidence_destination_template",
            "stop_condition_visibility_checklist",
            "deferred_release_probe_blocker_notice",
        ],
        "required_inputs": _string_list(review_packet.get("required_inputs")),
        "required_stop_condition_ids": _string_list(review_packet.get("required_stop_condition_ids")),
        "required_before_fields": _string_list(review_packet.get("required_before_fields")),
        "required_after_fields": _string_list(review_packet.get("required_after_fields")),
        "forbidden_approvals": _string_list(review_packet.get("forbidden_approvals")),
        "deferred_release_probe_blockers": _string_list(
            review_packet.get("deferred_release_probe_blockers")
        ),
        "review_section_ids": [str(item.get("id") or "") for item in sections if item.get("id")],
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_baseline_and_after_probe_material_folders_without_starting_probe",
            "refresh_batch1_before_after_evidence_sources",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_followup_manual_probe_material_package_prerequisites"]
    if any("manual_review_record" in item for item in blockers):
        actions.append("refresh_manual_review_record")
    if any("manual_review_packet" in item for item in blockers):
        actions.append("refresh_manual_review_packet")
    if any("release_claim" in item for item in blockers):
        actions.append("close_release_claim_leaks_before_followup_material_package")
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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_FOLLOWUP_MATERIAL_PACKAGE",
    "build_lulynx_internal_gate_limited_non_release_probe_followup_material_package",
]
