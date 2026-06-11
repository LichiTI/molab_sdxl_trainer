"""Handoff packet for manual fill of pointer-only batch1 probe materials.

This packet stays report-only. It consolidates manual-fill readiness and
template previews into a handoff-oriented artifact without writing files,
enabling the internal gate, or starting any probe execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_HANDOFF_PACKET = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_handoff_packet_v0"
)
READY_POINTER_MANUAL_FILL_READINESS_PACKET_STATUS = (
    "ready_for_followup_pointer_manual_fill_readiness_packet"
)
READY_POINTER_MANUAL_FILL_TEMPLATE_BUNDLE_STATUS = (
    "ready_for_followup_pointer_manual_fill_template_bundle"
)


def build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_handoff_packet(
    *,
    internal_gate_limited_non_release_probe_pointer_manual_fill_readiness_packet: Mapping[str, Any]
    | None = None,
    internal_gate_limited_non_release_probe_pointer_manual_fill_template_bundle: Mapping[str, Any]
    | None = None,
) -> dict[str, Any]:
    """Build a fail-closed handoff packet for manual pointer fill work."""

    readiness_packet = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_readiness_packet)
    )
    template_bundle = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_template_bundle)
    )
    readiness_payload = _mapping(readiness_packet.get("pointer_manual_fill_readiness_packet"))
    bundle_payload = _mapping(template_bundle.get("pointer_manual_fill_template_bundle"))
    artifact_readiness = _sequence_of_mappings(readiness_payload.get("artifact_readiness"))
    artifact_templates = _sequence_of_mappings(bundle_payload.get("artifact_templates"))
    checks = _checks(
        readiness_packet=readiness_packet,
        template_bundle=template_bundle,
        artifact_readiness=artifact_readiness,
        artifact_templates=artifact_templates,
        readiness_payload=readiness_payload,
    )
    blockers = _blockers(
        readiness_packet=readiness_packet,
        template_bundle=template_bundle,
        checks=checks,
    )
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_HANDOFF_PACKET,
        "status": "ready_for_followup_pointer_manual_fill_handoff_packet" if ready else "blocked",
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
        "pointer_manual_fill_readiness_packet_summary": _readiness_summary(readiness_packet),
        "pointer_manual_fill_template_bundle_summary": _bundle_summary(template_bundle),
        "pointer_manual_fill_handoff_packet": _handoff_packet(
            artifact_readiness=artifact_readiness,
            artifact_templates=artifact_templates,
            readiness_payload=readiness_payload,
        ),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    readiness_packet: Mapping[str, Any],
    template_bundle: Mapping[str, Any],
    artifact_readiness: Sequence[Mapping[str, Any]],
    artifact_templates: Sequence[Mapping[str, Any]],
    readiness_payload: Mapping[str, Any],
) -> dict[str, bool]:
    readiness_ids = {str(item.get("id") or "") for item in artifact_readiness if item.get("id")}
    template_ids = {str(item.get("id") or "") for item in artifact_templates if item.get("id")}
    forbidden_actions = set(_string_list(readiness_payload.get("forbidden_actions")))
    return {
        "pointer_manual_fill_readiness_packet_present": bool(readiness_packet),
        "pointer_manual_fill_readiness_packet_ready": bool(readiness_packet.get("passed"))
        and str(readiness_packet.get("status") or "")
        == READY_POINTER_MANUAL_FILL_READINESS_PACKET_STATUS,
        "pointer_manual_fill_readiness_packet_default_off": (
            not bool(readiness_packet.get("internal_gate_enablement_allowed"))
            and not bool(readiness_packet.get("release_claim_allowed"))
        ),
        "pointer_manual_fill_template_bundle_present": bool(template_bundle),
        "pointer_manual_fill_template_bundle_ready": bool(template_bundle.get("passed"))
        and str(template_bundle.get("status") or "")
        == READY_POINTER_MANUAL_FILL_TEMPLATE_BUNDLE_STATUS,
        "pointer_manual_fill_template_bundle_default_off": (
            not bool(template_bundle.get("internal_gate_enablement_allowed"))
            and not bool(template_bundle.get("release_claim_allowed"))
        ),
        "artifact_readiness_count_is_four": len(artifact_readiness) == 4,
        "artifact_template_count_is_four": len(artifact_templates) == 4,
        "artifact_roles_aligned": readiness_ids == template_ids == {
            "baseline_manifest_pointer",
            "probe_manifest_pointer",
            "before_after_probe_evidence",
            "manual_probe_review_notes",
        },
        "all_artifacts_ready_for_preview": all(
            bool(item.get("ready_for_manual_fill_preview")) for item in artifact_readiness
        ),
        "template_preview_keys_visible": all(
            bool(_string_list(item.get("template_preview_keys")))
            or bool(_mapping(item.get("template_preview")))
            for item in artifact_readiness + artifact_templates
        ),
        "forbidden_actions_keep_execution_closed": {
            "write_pointer_files_automatically",
            "enable_internal_gate_now",
            "start_training_now",
            "approve_batch2_4_8_release_probe",
            "approve_release_claim",
        }.issubset(forbidden_actions),
    }


def _blockers(
    *,
    readiness_packet: Mapping[str, Any],
    template_bundle: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not readiness_packet:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_readiness_packet_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_readiness_packet:{item}"
        for item in _string_list(readiness_packet.get("blockers"))
    )
    if not template_bundle:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_template_bundle_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_template_bundle:{item}"
        for item in _string_list(template_bundle.get("blockers"))
    )
    return _dedupe(blockers)


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


def _bundle_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_manual_fill_template_bundle"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "artifact_template_count": len(_sequence_of_mappings(payload.get("artifact_templates"))),
        "bundle_mode": str(payload.get("bundle_mode") or ""),
    }


def _handoff_packet(
    *,
    artifact_readiness: Sequence[Mapping[str, Any]],
    artifact_templates: Sequence[Mapping[str, Any]],
    readiness_payload: Mapping[str, Any],
) -> dict[str, Any]:
    template_map = {str(item.get("id") or ""): item for item in artifact_templates}
    return {
        "packet_kind": "manual_pointer_fill_handoff_packet_v0",
        "handoff_scope": "batch1_non_release_pointer_manual_fill_handoff",
        "forbidden_actions": _string_list(readiness_payload.get("forbidden_actions")),
        "required_completion_evidence": _string_list(
            readiness_payload.get("required_completion_evidence")
        ),
        "artifact_handoff_items": [
            {
                "id": _string(item.get("id")),
                "target_path": _string(item.get("target_path")),
                "ready_for_manual_fill_preview": bool(item.get("ready_for_manual_fill_preview")),
                "template_preview_keys": sorted(
                    set(
                        _string_list(item.get("template_preview_keys"))
                        + list(_mapping(_mapping(template_map.get(_string(item.get("id")))).get("template_preview")).keys())
                    )
                ),
                "checklist_steps": _string_list(item.get("checklist_steps")),
                "human_operator_action": "fill_template_preview_manually",
            }
            for item in artifact_readiness
        ],
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_pointer_fill_examples_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_manual_fill_handoff_packet_prerequisites"]
    if any("pointer_manual_fill_readiness_packet" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_readiness_packet")
    if any("pointer_manual_fill_template_bundle" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_template_bundle")
    if any("artifact_" in item or "template_preview" in item for item in blockers):
        actions.append("repair_pointer_manual_fill_handoff_inputs")
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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_HANDOFF_PACKET",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_handoff_packet",
]
