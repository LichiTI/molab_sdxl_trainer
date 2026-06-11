"""Dossier for manual fill of pointer-only batch1 probe materials.

This dossier stays report-only. It consolidates the cover note and operator
briefing into a more complete human handoff package without writing files,
enabling the internal gate, or starting any probe execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_DOSSIER = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_dossier_v0"
)
READY_POINTER_MANUAL_FILL_COVER_NOTE_STATUS = "ready_for_followup_pointer_manual_fill_cover_note"
READY_POINTER_MANUAL_FILL_OPERATOR_BRIEFING_STATUS = (
    "ready_for_followup_pointer_manual_fill_operator_briefing"
)


def build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_dossier(
    *,
    internal_gate_limited_non_release_probe_pointer_manual_fill_cover_note: Mapping[str, Any]
    | None = None,
    internal_gate_limited_non_release_probe_pointer_manual_fill_operator_briefing: Mapping[str, Any]
    | None = None,
) -> dict[str, Any]:
    """Build a fail-closed dossier for manual pointer fill handoff."""

    cover_note = dict(_mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_cover_note))
    operator_briefing = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_operator_briefing)
    )
    cover_payload = _mapping(cover_note.get("pointer_manual_fill_cover_note"))
    briefing_payload = _mapping(operator_briefing.get("pointer_manual_fill_operator_briefing"))
    artifact_notes = _sequence_of_mappings(cover_payload.get("artifact_notes"))
    briefing_items = _sequence_of_mappings(briefing_payload.get("briefing_items"))
    checks = _checks(
        cover_note=cover_note,
        operator_briefing=operator_briefing,
        artifact_notes=artifact_notes,
        briefing_items=briefing_items,
        cover_payload=cover_payload,
    )
    blockers = _blockers(cover_note=cover_note, operator_briefing=operator_briefing, checks=checks)
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_DOSSIER,
        "status": "ready_for_followup_pointer_manual_fill_dossier" if ready else "blocked",
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
        "pointer_manual_fill_cover_note_summary": _cover_summary(cover_note),
        "pointer_manual_fill_operator_briefing_summary": _briefing_summary(operator_briefing),
        "pointer_manual_fill_dossier": _dossier(
            artifact_notes=artifact_notes,
            briefing_items=briefing_items,
            cover_payload=cover_payload,
        ),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    cover_note: Mapping[str, Any],
    operator_briefing: Mapping[str, Any],
    artifact_notes: Sequence[Mapping[str, Any]],
    briefing_items: Sequence[Mapping[str, Any]],
    cover_payload: Mapping[str, Any],
) -> dict[str, bool]:
    note_ids = {str(item.get("id") or "") for item in artifact_notes if item.get("id")}
    briefing_ids = {str(item.get("id") or "") for item in briefing_items if item.get("id")}
    forbidden_actions = set(_string_list(cover_payload.get("forbidden_actions")))
    required_completion_evidence = _string_list(cover_payload.get("required_completion_evidence"))
    return {
        "pointer_manual_fill_cover_note_present": bool(cover_note),
        "pointer_manual_fill_cover_note_ready": bool(cover_note.get("passed"))
        and str(cover_note.get("status") or "") == READY_POINTER_MANUAL_FILL_COVER_NOTE_STATUS,
        "pointer_manual_fill_cover_note_default_off": (
            not bool(cover_note.get("internal_gate_enablement_allowed"))
            and not bool(cover_note.get("release_claim_allowed"))
        ),
        "pointer_manual_fill_operator_briefing_present": bool(operator_briefing),
        "pointer_manual_fill_operator_briefing_ready": bool(operator_briefing.get("passed"))
        and str(operator_briefing.get("status") or "")
        == READY_POINTER_MANUAL_FILL_OPERATOR_BRIEFING_STATUS,
        "pointer_manual_fill_operator_briefing_default_off": (
            not bool(operator_briefing.get("internal_gate_enablement_allowed"))
            and not bool(operator_briefing.get("release_claim_allowed"))
        ),
        "artifact_note_count_is_four": len(artifact_notes) == 4,
        "briefing_item_count_is_four": len(briefing_items) == 4,
        "artifact_roles_aligned": note_ids == briefing_ids == {
            "baseline_manifest_pointer",
            "probe_manifest_pointer",
            "before_after_probe_evidence",
            "manual_probe_review_notes",
        },
        "all_artifacts_ready_for_preview": all(
            bool(item.get("ready_for_manual_fill_preview")) for item in artifact_notes
        ),
        "operator_actions_visible": all(
            _string(item.get("human_operator_action")) == "fill_template_preview_manually"
            for item in artifact_notes
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
    cover_note: Mapping[str, Any],
    operator_briefing: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not cover_note:
        blockers.append("internal_gate_limited_non_release_probe_pointer_manual_fill_cover_note_missing")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_cover_note:{item}"
        for item in _string_list(cover_note.get("blockers"))
    )
    if not operator_briefing:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_operator_briefing_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_operator_briefing:{item}"
        for item in _string_list(operator_briefing.get("blockers"))
    )
    return _dedupe(blockers)


def _cover_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_manual_fill_cover_note"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "artifact_note_count": len(_sequence_of_mappings(payload.get("artifact_notes"))),
        "required_completion_evidence_count": len(
            _string_list(payload.get("required_completion_evidence"))
        ),
    }


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


def _dossier(
    *,
    artifact_notes: Sequence[Mapping[str, Any]],
    briefing_items: Sequence[Mapping[str, Any]],
    cover_payload: Mapping[str, Any],
) -> dict[str, Any]:
    briefing_map = {str(item.get("id") or ""): item for item in briefing_items}
    return {
        "dossier_kind": "manual_pointer_fill_dossier_v0",
        "dossier_scope": "batch1_non_release_pointer_manual_fill_dossier",
        "forbidden_actions": _string_list(cover_payload.get("forbidden_actions")),
        "required_completion_evidence": _string_list(
            cover_payload.get("required_completion_evidence")
        ),
        "artifact_entries": [
            {
                "id": _string(item.get("id")),
                "target_path": _string(item.get("target_path")),
                "human_operator_action": _string(item.get("human_operator_action")),
                "checklist_steps": _string_list(item.get("checklist_steps")),
                "template_preview_keys": _string_list(item.get("template_preview_keys")),
                "ready_for_manual_fill_preview": bool(item.get("ready_for_manual_fill_preview")),
                "briefing_alignment": bool(briefing_map.get(_string(item.get("id")))),
            }
            for item in artifact_notes
        ],
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_pointer_fill_release_packet_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_manual_fill_dossier_prerequisites"]
    if any("pointer_manual_fill_cover_note" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_cover_note")
    if any("pointer_manual_fill_operator_briefing" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_operator_briefing")
    if any("artifact_note" in item or "operator_action" in item for item in blockers):
        actions.append("repair_pointer_manual_fill_dossier_inputs")
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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_DOSSIER",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_dossier",
]
