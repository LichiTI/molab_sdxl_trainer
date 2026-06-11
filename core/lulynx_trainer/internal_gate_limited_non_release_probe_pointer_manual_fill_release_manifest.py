"""Release manifest for manual fill of pointer-only batch1 probe materials.

This manifest remains report-only. It assembles the release index and release
packet into a compact manifest-style handoff without writing files, enabling
the internal gate, or starting any probe execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_MANIFEST = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_manifest_v0"
)
READY_POINTER_MANUAL_FILL_RELEASE_INDEX_STATUS = (
    "ready_for_followup_pointer_manual_fill_release_index"
)
READY_POINTER_MANUAL_FILL_RELEASE_PACKET_STATUS = (
    "ready_for_followup_pointer_manual_fill_release_packet"
)


def build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_manifest(
    *,
    internal_gate_limited_non_release_probe_pointer_manual_fill_release_index: Mapping[str, Any]
    | None = None,
    internal_gate_limited_non_release_probe_pointer_manual_fill_release_packet: Mapping[str, Any]
    | None = None,
) -> dict[str, Any]:
    """Build a fail-closed release manifest for manual pointer fill handoff."""

    release_index = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_release_index)
    )
    release_packet = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_release_packet)
    )
    index_payload = _mapping(release_index.get("pointer_manual_fill_release_index"))
    packet_payload = _mapping(release_packet.get("pointer_manual_fill_release_packet"))
    index_items = _sequence_of_mappings(index_payload.get("artifact_index_items"))
    release_items = _sequence_of_mappings(packet_payload.get("artifact_release_items"))
    checks = _checks(
        release_index=release_index,
        release_packet=release_packet,
        index_items=index_items,
        release_items=release_items,
        index_payload=index_payload,
        packet_payload=packet_payload,
    )
    blockers = _blockers(release_index=release_index, release_packet=release_packet, checks=checks)
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_MANIFEST,
        "status": "ready_for_followup_pointer_manual_fill_release_manifest" if ready else "blocked",
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
        "pointer_manual_fill_release_index_summary": _index_summary(release_index),
        "pointer_manual_fill_release_packet_summary": _packet_summary(release_packet),
        "pointer_manual_fill_release_manifest": _release_manifest(
            index_items=index_items,
            release_items=release_items,
            index_payload=index_payload,
            packet_payload=packet_payload,
        ),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    release_index: Mapping[str, Any],
    release_packet: Mapping[str, Any],
    index_items: Sequence[Mapping[str, Any]],
    release_items: Sequence[Mapping[str, Any]],
    index_payload: Mapping[str, Any],
    packet_payload: Mapping[str, Any],
) -> dict[str, bool]:
    index_ids = {str(item.get("id") or "") for item in index_items if item.get("id")}
    release_ids = {str(item.get("id") or "") for item in release_items if item.get("id")}
    forbidden_actions = set(_string_list(index_payload.get("forbidden_actions")))
    required_completion_evidence = _string_list(index_payload.get("required_completion_evidence"))
    return {
        "pointer_manual_fill_release_index_present": bool(release_index),
        "pointer_manual_fill_release_index_ready": bool(release_index.get("passed"))
        and str(release_index.get("status") or "") == READY_POINTER_MANUAL_FILL_RELEASE_INDEX_STATUS,
        "pointer_manual_fill_release_index_default_off": (
            not bool(release_index.get("internal_gate_enablement_allowed"))
            and not bool(release_index.get("release_claim_allowed"))
        ),
        "pointer_manual_fill_release_packet_present": bool(release_packet),
        "pointer_manual_fill_release_packet_ready": bool(release_packet.get("passed"))
        and str(release_packet.get("status") or "") == READY_POINTER_MANUAL_FILL_RELEASE_PACKET_STATUS,
        "pointer_manual_fill_release_packet_default_off": (
            not bool(release_packet.get("internal_gate_enablement_allowed"))
            and not bool(release_packet.get("release_claim_allowed"))
        ),
        "artifact_index_item_count_is_four": len(index_items) == 4,
        "artifact_release_item_count_is_four": len(release_items) == 4,
        "artifact_roles_aligned": index_ids == release_ids == {
            "baseline_manifest_pointer",
            "probe_manifest_pointer",
            "before_after_probe_evidence",
            "manual_probe_review_notes",
        },
        "all_artifacts_ready_for_preview": all(
            bool(item.get("ready_for_manual_fill_preview")) for item in index_items
        ),
        "operator_actions_visible": all(
            _string(item.get("human_operator_action")) == "fill_template_preview_manually"
            for item in index_items
        ),
        "forbidden_actions_keep_execution_closed": {
            "write_pointer_files_automatically",
            "enable_internal_gate_now",
            "start_training_now",
            "approve_batch2_4_8_release_probe",
            "approve_release_claim",
        }.issubset(forbidden_actions),
        "required_completion_evidence_visible": len(required_completion_evidence) >= 4,
        "release_index_forbidden_actions_match_release_packet": set(
            _string_list(packet_payload.get("forbidden_actions"))
        )
        == forbidden_actions,
    }


def _blockers(
    *,
    release_index: Mapping[str, Any],
    release_packet: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not release_index:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_release_index_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_release_index:{item}"
        for item in _string_list(release_index.get("blockers"))
    )
    if not release_packet:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_release_packet_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_release_packet:{item}"
        for item in _string_list(release_packet.get("blockers"))
    )
    return _dedupe(blockers)


def _index_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_manual_fill_release_index"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "artifact_index_item_count": len(_sequence_of_mappings(payload.get("artifact_index_items"))),
        "required_completion_evidence_count": len(
            _string_list(payload.get("required_completion_evidence"))
        ),
    }


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


def _release_manifest(
    *,
    index_items: Sequence[Mapping[str, Any]],
    release_items: Sequence[Mapping[str, Any]],
    index_payload: Mapping[str, Any],
    packet_payload: Mapping[str, Any],
) -> dict[str, Any]:
    release_map = {str(item.get("id") or ""): item for item in release_items}
    return {
        "release_manifest_kind": "manual_pointer_fill_release_manifest_v0",
        "release_manifest_scope": "batch1_non_release_pointer_manual_fill_release_manifest",
        "forbidden_actions": _string_list(index_payload.get("forbidden_actions")),
        "required_completion_evidence": _string_list(
            index_payload.get("required_completion_evidence")
        ),
        "artifact_manifest_items": [
            {
                "id": _string(item.get("id")),
                "target_path": _string(item.get("target_path")),
                "human_operator_action": _string(item.get("human_operator_action")),
                "checklist_steps": _string_list(item.get("checklist_steps")),
                "template_preview_keys": _string_list(item.get("template_preview_keys")),
                "ready_for_manual_fill_preview": bool(item.get("ready_for_manual_fill_preview")),
                "briefing_alignment": bool(item.get("briefing_alignment")),
                "cover_note_alignment": bool(item.get("cover_note_alignment")),
                "dossier_alignment": bool(item.get("dossier_alignment")),
                "release_packet_alignment": bool(release_map.get(_string(item.get("id")))),
                "cover_note_line": _string(item.get("cover_note_line")),
            }
            for item in index_items
        ],
        "release_manifest_note": (
            "manual_pointer_fill_release_manifest_is_reference_only_and_cannot_enable_runtime_execution"
        ),
        "release_packet_forbidden_actions": _string_list(packet_payload.get("forbidden_actions")),
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_pointer_fill_release_brief_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_manual_fill_release_manifest_prerequisites"]
    if any("pointer_manual_fill_release_index" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_release_index")
    if any("pointer_manual_fill_release_packet" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_release_packet")
    if any("artifact_" in item or "operator_action" in item for item in blockers):
        actions.append("repair_pointer_manual_fill_release_manifest_inputs")
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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_MANIFEST",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_manifest",
]
