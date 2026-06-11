"""Operator briefing for manual fill of pointer-only batch1 probe materials.

This briefing remains report-only. It summarizes the manual-fill example pack
and handoff constraints for a human operator without writing files, enabling
the internal gate, or starting any probe execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_OPERATOR_BRIEFING = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_operator_briefing_v0"
)
READY_POINTER_MANUAL_FILL_EXAMPLE_PREVIEW_PACK_STATUS = (
    "ready_for_followup_pointer_manual_fill_example_preview_pack"
)
READY_POINTER_MANUAL_FILL_HANDOFF_PACKET_STATUS = (
    "ready_for_followup_pointer_manual_fill_handoff_packet"
)


def build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_operator_briefing(
    *,
    internal_gate_limited_non_release_probe_pointer_manual_fill_example_preview_pack: Mapping[str, Any]
    | None = None,
    internal_gate_limited_non_release_probe_pointer_manual_fill_handoff_packet: Mapping[str, Any]
    | None = None,
) -> dict[str, Any]:
    """Build a fail-closed operator briefing for manual pointer fill work."""

    example_pack = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_example_preview_pack)
    )
    handoff_packet = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_handoff_packet)
    )
    example_payload = _mapping(example_pack.get("pointer_manual_fill_example_preview_pack"))
    handoff_payload = _mapping(handoff_packet.get("pointer_manual_fill_handoff_packet"))
    artifact_examples = _sequence_of_mappings(example_payload.get("artifact_examples"))
    artifact_handoffs = _sequence_of_mappings(handoff_payload.get("artifact_handoff_items"))
    checks = _checks(
        example_pack=example_pack,
        handoff_packet=handoff_packet,
        artifact_examples=artifact_examples,
        artifact_handoffs=artifact_handoffs,
        handoff_payload=handoff_payload,
    )
    blockers = _blockers(example_pack=example_pack, handoff_packet=handoff_packet, checks=checks)
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_OPERATOR_BRIEFING,
        "status": "ready_for_followup_pointer_manual_fill_operator_briefing" if ready else "blocked",
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
        "pointer_manual_fill_example_preview_pack_summary": _example_summary(example_pack),
        "pointer_manual_fill_handoff_packet_summary": _handoff_summary(handoff_packet),
        "pointer_manual_fill_operator_briefing": _operator_briefing(
            artifact_examples=artifact_examples,
            artifact_handoffs=artifact_handoffs,
            handoff_payload=handoff_payload,
        ),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    example_pack: Mapping[str, Any],
    handoff_packet: Mapping[str, Any],
    artifact_examples: Sequence[Mapping[str, Any]],
    artifact_handoffs: Sequence[Mapping[str, Any]],
    handoff_payload: Mapping[str, Any],
) -> dict[str, bool]:
    example_ids = {str(item.get("id") or "") for item in artifact_examples if item.get("id")}
    handoff_ids = {str(item.get("id") or "") for item in artifact_handoffs if item.get("id")}
    forbidden_actions = set(_string_list(handoff_payload.get("forbidden_actions")))
    required_completion_evidence = _string_list(handoff_payload.get("required_completion_evidence"))
    return {
        "pointer_manual_fill_example_preview_pack_present": bool(example_pack),
        "pointer_manual_fill_example_preview_pack_ready": bool(example_pack.get("passed"))
        and str(example_pack.get("status") or "") == READY_POINTER_MANUAL_FILL_EXAMPLE_PREVIEW_PACK_STATUS,
        "pointer_manual_fill_example_preview_pack_default_off": (
            not bool(example_pack.get("internal_gate_enablement_allowed"))
            and not bool(example_pack.get("release_claim_allowed"))
        ),
        "pointer_manual_fill_handoff_packet_present": bool(handoff_packet),
        "pointer_manual_fill_handoff_packet_ready": bool(handoff_packet.get("passed"))
        and str(handoff_packet.get("status") or "") == READY_POINTER_MANUAL_FILL_HANDOFF_PACKET_STATUS,
        "pointer_manual_fill_handoff_packet_default_off": (
            not bool(handoff_packet.get("internal_gate_enablement_allowed"))
            and not bool(handoff_packet.get("release_claim_allowed"))
        ),
        "artifact_example_count_is_four": len(artifact_examples) == 4,
        "artifact_handoff_count_is_four": len(artifact_handoffs) == 4,
        "artifact_roles_aligned": example_ids == handoff_ids == {
            "baseline_manifest_pointer",
            "probe_manifest_pointer",
            "before_after_probe_evidence",
            "manual_probe_review_notes",
        },
        "example_previews_present": all(bool(_mapping(item.get("template_preview"))) for item in artifact_examples),
        "operator_actions_visible": all(
            _string(item.get("human_operator_action")) == "fill_template_preview_manually"
            for item in artifact_examples
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
    example_pack: Mapping[str, Any],
    handoff_packet: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not example_pack:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_example_preview_pack_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_example_preview_pack:{item}"
        for item in _string_list(example_pack.get("blockers"))
    )
    if not handoff_packet:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_handoff_packet_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_handoff_packet:{item}"
        for item in _string_list(handoff_packet.get("blockers"))
    )
    return _dedupe(blockers)


def _example_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_manual_fill_example_preview_pack"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "artifact_example_count": len(_sequence_of_mappings(payload.get("artifact_examples"))),
        "required_completion_evidence_count": len(
            _string_list(payload.get("required_completion_evidence"))
        ),
    }


def _handoff_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_manual_fill_handoff_packet"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "artifact_handoff_count": len(_sequence_of_mappings(payload.get("artifact_handoff_items"))),
        "required_completion_evidence_count": len(
            _string_list(payload.get("required_completion_evidence"))
        ),
    }


def _operator_briefing(
    *,
    artifact_examples: Sequence[Mapping[str, Any]],
    artifact_handoffs: Sequence[Mapping[str, Any]],
    handoff_payload: Mapping[str, Any],
) -> dict[str, Any]:
    handoff_map = {str(item.get("id") or ""): item for item in artifact_handoffs}
    return {
        "briefing_kind": "manual_pointer_fill_operator_briefing_v0",
        "briefing_scope": "batch1_non_release_pointer_manual_fill_operator_handoff",
        "forbidden_actions": _string_list(handoff_payload.get("forbidden_actions")),
        "required_completion_evidence": _string_list(
            handoff_payload.get("required_completion_evidence")
        ),
        "briefing_items": [
            {
                "id": _string(item.get("id")),
                "target_path": _string(item.get("target_path")),
                "human_operator_action": _string(item.get("human_operator_action")),
                "checklist_steps": _string_list(item.get("checklist_steps")),
                "template_preview_keys": sorted(_mapping(item.get("template_preview")).keys()),
                "ready_for_manual_fill_preview": bool(
                    _mapping(handoff_map.get(_string(item.get("id")))).get(
                        "ready_for_manual_fill_preview"
                    )
                ),
                "operator_note": "manual_preview_only_non_release_batch1",
            }
            for item in artifact_examples
        ],
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_pointer_fill_cover_note_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_manual_fill_operator_briefing_prerequisites"]
    if any("pointer_manual_fill_example_preview_pack" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_example_preview_pack")
    if any("pointer_manual_fill_handoff_packet" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_handoff_packet")
    if any("operator_action" in item or "template_preview" in item for item in blockers):
        actions.append("repair_pointer_manual_fill_operator_briefing_inputs")
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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_OPERATOR_BRIEFING",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_operator_briefing",
]
