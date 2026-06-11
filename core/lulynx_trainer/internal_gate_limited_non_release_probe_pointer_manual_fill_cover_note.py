"""Cover note for manual fill of pointer-only batch1 probe materials.

This note stays report-only. It consolidates operator briefing and readiness
status into a handoff-oriented summary without writing files, enabling the
internal gate, or starting any probe execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_COVER_NOTE = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_cover_note_v0"
)
READY_POINTER_MANUAL_FILL_OPERATOR_BRIEFING_STATUS = (
    "ready_for_followup_pointer_manual_fill_operator_briefing"
)
READY_POINTER_MANUAL_FILL_READINESS_PACKET_STATUS = (
    "ready_for_followup_pointer_manual_fill_readiness_packet"
)


def build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_cover_note(
    *,
    internal_gate_limited_non_release_probe_pointer_manual_fill_operator_briefing: Mapping[str, Any]
    | None = None,
    internal_gate_limited_non_release_probe_pointer_manual_fill_readiness_packet: Mapping[str, Any]
    | None = None,
) -> dict[str, Any]:
    """Build a fail-closed cover note for manual pointer fill handoff."""

    operator_briefing = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_operator_briefing)
    )
    readiness_packet = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_readiness_packet)
    )
    briefing_payload = _mapping(operator_briefing.get("pointer_manual_fill_operator_briefing"))
    readiness_payload = _mapping(readiness_packet.get("pointer_manual_fill_readiness_packet"))
    briefing_items = _sequence_of_mappings(briefing_payload.get("briefing_items"))
    artifact_readiness = _sequence_of_mappings(readiness_payload.get("artifact_readiness"))
    checks = _checks(
        operator_briefing=operator_briefing,
        readiness_packet=readiness_packet,
        briefing_items=briefing_items,
        artifact_readiness=artifact_readiness,
        readiness_payload=readiness_payload,
    )
    blockers = _blockers(
        operator_briefing=operator_briefing,
        readiness_packet=readiness_packet,
        checks=checks,
    )
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_COVER_NOTE,
        "status": "ready_for_followup_pointer_manual_fill_cover_note" if ready else "blocked",
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
        "pointer_manual_fill_operator_briefing_summary": _briefing_summary(operator_briefing),
        "pointer_manual_fill_readiness_packet_summary": _readiness_summary(readiness_packet),
        "pointer_manual_fill_cover_note": _cover_note(
            briefing_items=briefing_items,
            artifact_readiness=artifact_readiness,
            readiness_payload=readiness_payload,
        ),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    operator_briefing: Mapping[str, Any],
    readiness_packet: Mapping[str, Any],
    briefing_items: Sequence[Mapping[str, Any]],
    artifact_readiness: Sequence[Mapping[str, Any]],
    readiness_payload: Mapping[str, Any],
) -> dict[str, bool]:
    briefing_ids = {str(item.get("id") or "") for item in briefing_items if item.get("id")}
    readiness_ids = {str(item.get("id") or "") for item in artifact_readiness if item.get("id")}
    forbidden_actions = set(_string_list(readiness_payload.get("forbidden_actions")))
    required_completion_evidence = _string_list(readiness_payload.get("required_completion_evidence"))
    return {
        "pointer_manual_fill_operator_briefing_present": bool(operator_briefing),
        "pointer_manual_fill_operator_briefing_ready": bool(operator_briefing.get("passed"))
        and str(operator_briefing.get("status") or "")
        == READY_POINTER_MANUAL_FILL_OPERATOR_BRIEFING_STATUS,
        "pointer_manual_fill_operator_briefing_default_off": (
            not bool(operator_briefing.get("internal_gate_enablement_allowed"))
            and not bool(operator_briefing.get("release_claim_allowed"))
        ),
        "pointer_manual_fill_readiness_packet_present": bool(readiness_packet),
        "pointer_manual_fill_readiness_packet_ready": bool(readiness_packet.get("passed"))
        and str(readiness_packet.get("status") or "")
        == READY_POINTER_MANUAL_FILL_READINESS_PACKET_STATUS,
        "pointer_manual_fill_readiness_packet_default_off": (
            not bool(readiness_packet.get("internal_gate_enablement_allowed"))
            and not bool(readiness_packet.get("release_claim_allowed"))
        ),
        "briefing_item_count_is_four": len(briefing_items) == 4,
        "artifact_readiness_count_is_four": len(artifact_readiness) == 4,
        "artifact_roles_aligned": briefing_ids == readiness_ids == {
            "baseline_manifest_pointer",
            "probe_manifest_pointer",
            "before_after_probe_evidence",
            "manual_probe_review_notes",
        },
        "briefing_items_ready_for_preview": all(
            bool(item.get("ready_for_manual_fill_preview")) for item in briefing_items
        ),
        "operator_actions_visible": all(
            _string(item.get("human_operator_action")) == "fill_template_preview_manually"
            for item in briefing_items
        ),
        "forbidden_actions_keep_execution_closed": {
            "write_pointer_files_automatically",
            "enable_internal_gate_now",
            "start_training_now",
            "approve_batch2_4_8_release_probe",
            "approve_release_claim",
        }.issubset(forbidden_actions),
        "required_completion_evidence_visible": len(required_completion_evidence) >= 4,
    }


def _blockers(
    *,
    operator_briefing: Mapping[str, Any],
    readiness_packet: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not operator_briefing:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_operator_briefing_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_operator_briefing:{item}"
        for item in _string_list(operator_briefing.get("blockers"))
    )
    if not readiness_packet:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_readiness_packet_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_readiness_packet:{item}"
        for item in _string_list(readiness_packet.get("blockers"))
    )
    return _dedupe(blockers)


def _briefing_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_manual_fill_operator_briefing"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "briefing_item_count": len(_sequence_of_mappings(payload.get("briefing_items"))),
        "required_completion_evidence_count": len(
            _string_list(payload.get("required_completion_evidence"))
        ),
    }


def _readiness_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_manual_fill_readiness_packet"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "artifact_readiness_count": len(_sequence_of_mappings(payload.get("artifact_readiness"))),
        "required_completion_evidence_count": len(
            _string_list(payload.get("required_completion_evidence"))
        ),
    }


def _cover_note(
    *,
    briefing_items: Sequence[Mapping[str, Any]],
    artifact_readiness: Sequence[Mapping[str, Any]],
    readiness_payload: Mapping[str, Any],
) -> dict[str, Any]:
    readiness_map = {str(item.get("id") or ""): item for item in artifact_readiness}
    return {
        "note_kind": "manual_pointer_fill_cover_note_v0",
        "note_scope": "batch1_non_release_pointer_manual_fill_cover_note",
        "forbidden_actions": _string_list(readiness_payload.get("forbidden_actions")),
        "required_completion_evidence": _string_list(
            readiness_payload.get("required_completion_evidence")
        ),
        "artifact_notes": [
            {
                "id": _string(item.get("id")),
                "target_path": _string(item.get("target_path")),
                "human_operator_action": _string(item.get("human_operator_action")),
                "checklist_steps": _string_list(item.get("checklist_steps")),
                "template_preview_keys": _string_list(item.get("template_preview_keys")),
                "ready_for_manual_fill_preview": bool(
                    _mapping(readiness_map.get(_string(item.get("id")))).get(
                        "ready_for_manual_fill_preview"
                    )
                ),
                "cover_note_line": "manual_preview_only_non_release_batch1",
            }
            for item in briefing_items
        ],
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_pointer_fill_dossier_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_manual_fill_cover_note_prerequisites"]
    if any("pointer_manual_fill_operator_briefing" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_operator_briefing")
    if any("pointer_manual_fill_readiness_packet" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_readiness_packet")
    if any("operator_action" in item or "briefing_item" in item for item in blockers):
        actions.append("repair_pointer_manual_fill_cover_note_inputs")
    return _dedupe(actions)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _string(value: Any) -> str:
    return str(value or "")


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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_COVER_NOTE",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_cover_note",
]
