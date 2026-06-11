"""Release index for manual fill of pointer-only batch1 probe materials.

This index remains report-only. It assembles the release packet and dossier
into a compact operator-facing index without writing files, enabling the
internal gate, or starting any probe execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_INDEX = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_index_v0"
)
READY_POINTER_MANUAL_FILL_RELEASE_PACKET_STATUS = (
    "ready_for_followup_pointer_manual_fill_release_packet"
)
READY_POINTER_MANUAL_FILL_DOSSIER_STATUS = "ready_for_followup_pointer_manual_fill_dossier"


def build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_index(
    *,
    internal_gate_limited_non_release_probe_pointer_manual_fill_release_packet: Mapping[str, Any]
    | None = None,
    internal_gate_limited_non_release_probe_pointer_manual_fill_dossier: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed release index for manual pointer fill handoff."""

    release_packet = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_release_packet)
    )
    dossier = dict(_mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_dossier))
    packet_payload = _mapping(release_packet.get("pointer_manual_fill_release_packet"))
    dossier_payload = _mapping(dossier.get("pointer_manual_fill_dossier"))
    release_items = _sequence_of_mappings(packet_payload.get("artifact_release_items"))
    dossier_items = _sequence_of_mappings(dossier_payload.get("artifact_entries"))
    checks = _checks(
        release_packet=release_packet,
        dossier=dossier,
        release_items=release_items,
        dossier_items=dossier_items,
        packet_payload=packet_payload,
        dossier_payload=dossier_payload,
    )
    blockers = _blockers(release_packet=release_packet, dossier=dossier, checks=checks)
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_INDEX,
        "status": "ready_for_followup_pointer_manual_fill_release_index" if ready else "blocked",
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
        "pointer_manual_fill_release_packet_summary": _packet_summary(release_packet),
        "pointer_manual_fill_dossier_summary": _dossier_summary(dossier),
        "pointer_manual_fill_release_index": _release_index(
            release_items=release_items,
            dossier_items=dossier_items,
            packet_payload=packet_payload,
            dossier_payload=dossier_payload,
        ),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    release_packet: Mapping[str, Any],
    dossier: Mapping[str, Any],
    release_items: Sequence[Mapping[str, Any]],
    dossier_items: Sequence[Mapping[str, Any]],
    packet_payload: Mapping[str, Any],
    dossier_payload: Mapping[str, Any],
) -> dict[str, bool]:
    release_ids = {str(item.get("id") or "") for item in release_items if item.get("id")}
    dossier_ids = {str(item.get("id") or "") for item in dossier_items if item.get("id")}
    forbidden_actions = set(_string_list(packet_payload.get("forbidden_actions")))
    required_completion_evidence = _string_list(packet_payload.get("required_completion_evidence"))
    return {
        "pointer_manual_fill_release_packet_present": bool(release_packet),
        "pointer_manual_fill_release_packet_ready": bool(release_packet.get("passed"))
        and str(release_packet.get("status") or "") == READY_POINTER_MANUAL_FILL_RELEASE_PACKET_STATUS,
        "pointer_manual_fill_release_packet_default_off": (
            not bool(release_packet.get("internal_gate_enablement_allowed"))
            and not bool(release_packet.get("release_claim_allowed"))
        ),
        "pointer_manual_fill_dossier_present": bool(dossier),
        "pointer_manual_fill_dossier_ready": bool(dossier.get("passed"))
        and str(dossier.get("status") or "") == READY_POINTER_MANUAL_FILL_DOSSIER_STATUS,
        "pointer_manual_fill_dossier_default_off": (
            not bool(dossier.get("internal_gate_enablement_allowed"))
            and not bool(dossier.get("release_claim_allowed"))
        ),
        "artifact_release_item_count_is_four": len(release_items) == 4,
        "artifact_dossier_item_count_is_four": len(dossier_items) == 4,
        "artifact_roles_aligned": release_ids == dossier_ids == {
            "baseline_manifest_pointer",
            "probe_manifest_pointer",
            "before_after_probe_evidence",
            "manual_probe_review_notes",
        },
        "all_artifacts_ready_for_preview": all(
            bool(item.get("ready_for_manual_fill_preview")) for item in release_items
        ),
        "operator_actions_visible": all(
            _string(item.get("human_operator_action")) == "fill_template_preview_manually"
            for item in release_items
        ),
        "forbidden_actions_keep_execution_closed": {
            "write_pointer_files_automatically",
            "enable_internal_gate_now",
            "start_training_now",
            "approve_batch2_4_8_release_probe",
            "approve_release_claim",
        }.issubset(forbidden_actions),
        "required_completion_evidence_visible": len(required_completion_evidence) >= 4,
        "release_packet_forbidden_actions_match_dossier": set(
            _string_list(dossier_payload.get("forbidden_actions"))
        )
        == forbidden_actions,
    }


def _blockers(
    *,
    release_packet: Mapping[str, Any],
    dossier: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not release_packet:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_release_packet_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_release_packet:{item}"
        for item in _string_list(release_packet.get("blockers"))
    )
    if not dossier:
        blockers.append("internal_gate_limited_non_release_probe_pointer_manual_fill_dossier_missing")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_dossier:{item}"
        for item in _string_list(dossier.get("blockers"))
    )
    return _dedupe(blockers)


def _packet_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_manual_fill_release_packet"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "artifact_release_item_count": len(_sequence_of_mappings(payload.get("artifact_release_items"))),
        "required_completion_evidence_count": len(
            _string_list(payload.get("required_completion_evidence"))
        ),
    }


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


def _release_index(
    *,
    release_items: Sequence[Mapping[str, Any]],
    dossier_items: Sequence[Mapping[str, Any]],
    packet_payload: Mapping[str, Any],
    dossier_payload: Mapping[str, Any],
) -> dict[str, Any]:
    dossier_map = {str(item.get("id") or ""): item for item in dossier_items}
    return {
        "release_index_kind": "manual_pointer_fill_release_index_v0",
        "release_index_scope": "batch1_non_release_pointer_manual_fill_release_index",
        "forbidden_actions": _string_list(packet_payload.get("forbidden_actions")),
        "required_completion_evidence": _string_list(
            packet_payload.get("required_completion_evidence")
        ),
        "artifact_index_items": [
            {
                "id": _string(item.get("id")),
                "target_path": _string(item.get("target_path")),
                "human_operator_action": _string(item.get("human_operator_action")),
                "checklist_steps": _string_list(item.get("checklist_steps")),
                "template_preview_keys": _string_list(item.get("template_preview_keys")),
                "ready_for_manual_fill_preview": bool(item.get("ready_for_manual_fill_preview")),
                "briefing_alignment": bool(item.get("briefing_alignment")),
                "cover_note_alignment": bool(item.get("cover_note_alignment")),
                "dossier_alignment": bool(dossier_map.get(_string(item.get("id")))),
                "cover_note_line": _string(item.get("cover_note_line")),
            }
            for item in release_items
        ],
        "release_index_note": (
            "manual_pointer_fill_release_index_is_operator_reference_only_and_cannot_enable_runtime_execution"
        ),
        "dossier_forbidden_actions": _string_list(dossier_payload.get("forbidden_actions")),
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_pointer_fill_release_manifest_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_manual_fill_release_index_prerequisites"]
    if any("pointer_manual_fill_release_packet" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_release_packet")
    if any("pointer_manual_fill_dossier" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_dossier")
    if any("artifact_" in item or "operator_action" in item for item in blockers):
        actions.append("repair_pointer_manual_fill_release_index_inputs")
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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_INDEX",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_index",
]
