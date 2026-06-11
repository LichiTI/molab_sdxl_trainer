"""Release packet for manual fill of pointer-only batch1 probe materials.

This packet remains report-only. It assembles the dossier and cover note into a
single release-handoff artifact without writing files, enabling the internal
gate, or starting any probe execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_PACKET = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_packet_v0"
)
READY_POINTER_MANUAL_FILL_DOSSIER_STATUS = "ready_for_followup_pointer_manual_fill_dossier"
READY_POINTER_MANUAL_FILL_COVER_NOTE_STATUS = "ready_for_followup_pointer_manual_fill_cover_note"


def build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_packet(
    *,
    internal_gate_limited_non_release_probe_pointer_manual_fill_dossier: Mapping[str, Any] | None = None,
    internal_gate_limited_non_release_probe_pointer_manual_fill_cover_note: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed release packet for manual pointer fill handoff."""

    dossier = dict(_mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_dossier))
    cover_note = dict(_mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_cover_note))
    dossier_payload = _mapping(dossier.get("pointer_manual_fill_dossier"))
    cover_payload = _mapping(cover_note.get("pointer_manual_fill_cover_note"))
    artifact_entries = _sequence_of_mappings(dossier_payload.get("artifact_entries"))
    artifact_notes = _sequence_of_mappings(cover_payload.get("artifact_notes"))
    checks = _checks(
        dossier=dossier,
        cover_note=cover_note,
        artifact_entries=artifact_entries,
        artifact_notes=artifact_notes,
        dossier_payload=dossier_payload,
        cover_payload=cover_payload,
    )
    blockers = _blockers(dossier=dossier, cover_note=cover_note, checks=checks)
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_PACKET,
        "status": "ready_for_followup_pointer_manual_fill_release_packet" if ready else "blocked",
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
        "pointer_manual_fill_dossier_summary": _dossier_summary(dossier),
        "pointer_manual_fill_cover_note_summary": _cover_summary(cover_note),
        "pointer_manual_fill_release_packet": _release_packet(
            artifact_entries=artifact_entries,
            artifact_notes=artifact_notes,
            dossier_payload=dossier_payload,
            cover_payload=cover_payload,
        ),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    dossier: Mapping[str, Any],
    cover_note: Mapping[str, Any],
    artifact_entries: Sequence[Mapping[str, Any]],
    artifact_notes: Sequence[Mapping[str, Any]],
    dossier_payload: Mapping[str, Any],
    cover_payload: Mapping[str, Any],
) -> dict[str, bool]:
    entry_ids = {str(item.get("id") or "") for item in artifact_entries if item.get("id")}
    note_ids = {str(item.get("id") or "") for item in artifact_notes if item.get("id")}
    forbidden_actions = set(_string_list(dossier_payload.get("forbidden_actions")))
    required_completion_evidence = _string_list(dossier_payload.get("required_completion_evidence"))
    return {
        "pointer_manual_fill_dossier_present": bool(dossier),
        "pointer_manual_fill_dossier_ready": bool(dossier.get("passed"))
        and str(dossier.get("status") or "") == READY_POINTER_MANUAL_FILL_DOSSIER_STATUS,
        "pointer_manual_fill_dossier_default_off": (
            not bool(dossier.get("internal_gate_enablement_allowed"))
            and not bool(dossier.get("release_claim_allowed"))
        ),
        "pointer_manual_fill_cover_note_present": bool(cover_note),
        "pointer_manual_fill_cover_note_ready": bool(cover_note.get("passed"))
        and str(cover_note.get("status") or "") == READY_POINTER_MANUAL_FILL_COVER_NOTE_STATUS,
        "pointer_manual_fill_cover_note_default_off": (
            not bool(cover_note.get("internal_gate_enablement_allowed"))
            and not bool(cover_note.get("release_claim_allowed"))
        ),
        "artifact_entry_count_is_four": len(artifact_entries) == 4,
        "artifact_note_count_is_four": len(artifact_notes) == 4,
        "artifact_roles_aligned": entry_ids == note_ids == {
            "baseline_manifest_pointer",
            "probe_manifest_pointer",
            "before_after_probe_evidence",
            "manual_probe_review_notes",
        },
        "all_artifacts_ready_for_preview": all(
            bool(item.get("ready_for_manual_fill_preview")) for item in artifact_entries
        ),
        "operator_actions_visible": all(
            _string(item.get("human_operator_action")) == "fill_template_preview_manually"
            for item in artifact_entries
        ),
        "forbidden_actions_keep_execution_closed": {
            "write_pointer_files_automatically",
            "enable_internal_gate_now",
            "start_training_now",
            "approve_batch2_4_8_release_probe",
            "approve_release_claim",
        }.issubset(forbidden_actions),
        "required_completion_evidence_visible": len(required_completion_evidence) >= 4,
        "cover_note_forbidden_actions_match_dossier": set(
            _string_list(cover_payload.get("forbidden_actions"))
        )
        == forbidden_actions,
    }


def _blockers(
    *,
    dossier: Mapping[str, Any],
    cover_note: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not dossier:
        blockers.append("internal_gate_limited_non_release_probe_pointer_manual_fill_dossier_missing")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_dossier:{item}"
        for item in _string_list(dossier.get("blockers"))
    )
    if not cover_note:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_cover_note_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_cover_note:{item}"
        for item in _string_list(cover_note.get("blockers"))
    )
    return _dedupe(blockers)


def _dossier_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_manual_fill_dossier"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "artifact_entry_count": len(_sequence_of_mappings(payload.get("artifact_entries"))),
        "required_completion_evidence_count": len(
            _string_list(payload.get("required_completion_evidence"))
        ),
    }


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


def _release_packet(
    *,
    artifact_entries: Sequence[Mapping[str, Any]],
    artifact_notes: Sequence[Mapping[str, Any]],
    dossier_payload: Mapping[str, Any],
    cover_payload: Mapping[str, Any],
) -> dict[str, Any]:
    note_map = {str(item.get("id") or ""): item for item in artifact_notes}
    return {
        "release_packet_kind": "manual_pointer_fill_release_packet_v0",
        "release_packet_scope": "batch1_non_release_pointer_manual_fill_release_packet",
        "forbidden_actions": _string_list(dossier_payload.get("forbidden_actions")),
        "required_completion_evidence": _string_list(
            dossier_payload.get("required_completion_evidence")
        ),
        "artifact_release_items": [
            {
                "id": _string(item.get("id")),
                "target_path": _string(item.get("target_path")),
                "human_operator_action": _string(item.get("human_operator_action")),
                "checklist_steps": _string_list(item.get("checklist_steps")),
                "template_preview_keys": _string_list(item.get("template_preview_keys")),
                "ready_for_manual_fill_preview": bool(item.get("ready_for_manual_fill_preview")),
                "briefing_alignment": bool(item.get("briefing_alignment")),
                "cover_note_alignment": bool(note_map.get(_string(item.get("id")))),
                "cover_note_line": _string(
                    _mapping(note_map.get(_string(item.get("id")))).get("cover_note_line")
                ),
            }
            for item in artifact_entries
        ],
        "release_packet_note": (
            "manual_pointer_fill_release_packet_is_handoff_only_and_cannot_enable_runtime_execution"
        ),
        "cover_note_forbidden_actions": _string_list(cover_payload.get("forbidden_actions")),
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_pointer_fill_release_index_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_manual_fill_release_packet_prerequisites"]
    if any("pointer_manual_fill_dossier" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_dossier")
    if any("pointer_manual_fill_cover_note" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_cover_note")
    if any("artifact_" in item or "operator_action" in item for item in blockers):
        actions.append("repair_pointer_manual_fill_release_packet_inputs")
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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_PACKET",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_packet",
]
