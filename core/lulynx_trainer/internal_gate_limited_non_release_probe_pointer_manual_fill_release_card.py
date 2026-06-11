"""Release card for manual fill of pointer-only batch1 probe materials.

This card remains report-only. It assembles the release capsule and release
snapshot into a compact operator-facing summary without writing files,
enabling the internal gate, or starting any probe execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_CARD = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_card_v0"
)
READY_POINTER_MANUAL_FILL_RELEASE_CAPSULE_STATUS = (
    "ready_for_followup_pointer_manual_fill_release_capsule"
)
READY_POINTER_MANUAL_FILL_RELEASE_SNAPSHOT_STATUS = (
    "ready_for_followup_pointer_manual_fill_release_snapshot"
)


def build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_card(
    *,
    internal_gate_limited_non_release_probe_pointer_manual_fill_release_capsule: Mapping[str, Any]
    | None = None,
    internal_gate_limited_non_release_probe_pointer_manual_fill_release_snapshot: Mapping[str, Any]
    | None = None,
) -> dict[str, Any]:
    """Build a fail-closed release card for manual pointer fill handoff."""

    release_capsule = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_release_capsule)
    )
    release_snapshot = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_release_snapshot)
    )
    capsule_payload = _mapping(release_capsule.get("pointer_manual_fill_release_capsule"))
    snapshot_payload = _mapping(release_snapshot.get("pointer_manual_fill_release_snapshot"))
    capsule_items = _sequence_of_mappings(capsule_payload.get("artifact_capsule_items"))
    snapshot_items = _sequence_of_mappings(snapshot_payload.get("artifact_snapshot_items"))
    checks = _checks(
        release_capsule=release_capsule,
        release_snapshot=release_snapshot,
        capsule_items=capsule_items,
        snapshot_items=snapshot_items,
        capsule_payload=capsule_payload,
        snapshot_payload=snapshot_payload,
    )
    blockers = _blockers(
        release_capsule=release_capsule,
        release_snapshot=release_snapshot,
        checks=checks,
    )
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_CARD,
        "status": "ready_for_followup_pointer_manual_fill_release_card" if ready else "blocked",
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
        "pointer_manual_fill_release_capsule_summary": _capsule_summary(release_capsule),
        "pointer_manual_fill_release_snapshot_summary": _snapshot_summary(release_snapshot),
        "pointer_manual_fill_release_card": _release_card(
            capsule_items=capsule_items,
            snapshot_items=snapshot_items,
            capsule_payload=capsule_payload,
            snapshot_payload=snapshot_payload,
        ),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    release_capsule: Mapping[str, Any],
    release_snapshot: Mapping[str, Any],
    capsule_items: Sequence[Mapping[str, Any]],
    snapshot_items: Sequence[Mapping[str, Any]],
    capsule_payload: Mapping[str, Any],
    snapshot_payload: Mapping[str, Any],
) -> dict[str, bool]:
    capsule_ids = {str(item.get("id") or "") for item in capsule_items if item.get("id")}
    snapshot_ids = {str(item.get("id") or "") for item in snapshot_items if item.get("id")}
    forbidden_actions = set(_string_list(capsule_payload.get("forbidden_actions")))
    required_completion_evidence = _string_list(
        capsule_payload.get("required_completion_evidence")
    )
    return {
        "pointer_manual_fill_release_capsule_present": bool(release_capsule),
        "pointer_manual_fill_release_capsule_ready": bool(release_capsule.get("passed"))
        and str(release_capsule.get("status") or "") == READY_POINTER_MANUAL_FILL_RELEASE_CAPSULE_STATUS,
        "pointer_manual_fill_release_capsule_default_off": (
            not bool(release_capsule.get("internal_gate_enablement_allowed"))
            and not bool(release_capsule.get("release_claim_allowed"))
        ),
        "pointer_manual_fill_release_snapshot_present": bool(release_snapshot),
        "pointer_manual_fill_release_snapshot_ready": bool(release_snapshot.get("passed"))
        and str(release_snapshot.get("status") or "") == READY_POINTER_MANUAL_FILL_RELEASE_SNAPSHOT_STATUS,
        "pointer_manual_fill_release_snapshot_default_off": (
            not bool(release_snapshot.get("internal_gate_enablement_allowed"))
            and not bool(release_snapshot.get("release_claim_allowed"))
        ),
        "artifact_capsule_item_count_is_four": len(capsule_items) == 4,
        "artifact_snapshot_item_count_is_four": len(snapshot_items) == 4,
        "artifact_roles_aligned": capsule_ids == snapshot_ids == {
            "baseline_manifest_pointer",
            "probe_manifest_pointer",
            "before_after_probe_evidence",
            "manual_probe_review_notes",
        },
        "all_artifacts_ready_for_preview": all(
            bool(item.get("ready_for_manual_fill_preview")) for item in capsule_items
        ),
        "operator_actions_visible": all(
            _string(item.get("human_operator_action")) == "fill_template_preview_manually"
            for item in capsule_items
        ),
        "forbidden_actions_keep_execution_closed": {
            "write_pointer_files_automatically",
            "enable_internal_gate_now",
            "start_training_now",
            "approve_batch2_4_8_release_probe",
            "approve_release_claim",
        }.issubset(forbidden_actions),
        "required_completion_evidence_visible": len(required_completion_evidence) >= 4,
        "release_capsule_forbidden_actions_match_release_snapshot": set(
            _string_list(snapshot_payload.get("forbidden_actions"))
        )
        == forbidden_actions,
    }


def _blockers(
    *,
    release_capsule: Mapping[str, Any],
    release_snapshot: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not release_capsule:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_release_capsule_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_release_capsule:{item}"
        for item in _string_list(release_capsule.get("blockers"))
    )
    if not release_snapshot:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_release_snapshot_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_release_snapshot:{item}"
        for item in _string_list(release_snapshot.get("blockers"))
    )
    return _dedupe(blockers)


def _capsule_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_manual_fill_release_capsule"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "artifact_capsule_item_count": len(_sequence_of_mappings(payload.get("artifact_capsule_items"))),
        "required_completion_evidence_count": len(
            _string_list(payload.get("required_completion_evidence"))
        ),
    }


def _snapshot_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_manual_fill_release_snapshot"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "artifact_snapshot_item_count": len(
            _sequence_of_mappings(payload.get("artifact_snapshot_items"))
        ),
        "required_completion_evidence_count": len(
            _string_list(payload.get("required_completion_evidence"))
        ),
    }


def _release_card(
    *,
    capsule_items: Sequence[Mapping[str, Any]],
    snapshot_items: Sequence[Mapping[str, Any]],
    capsule_payload: Mapping[str, Any],
    snapshot_payload: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot_map = {str(item.get("id") or ""): item for item in snapshot_items}
    return {
        "release_card_kind": "manual_pointer_fill_release_card_v0",
        "release_card_scope": "batch1_non_release_pointer_manual_fill_release_card",
        "forbidden_actions": _string_list(capsule_payload.get("forbidden_actions")),
        "required_completion_evidence": _string_list(
            capsule_payload.get("required_completion_evidence")
        ),
        "artifact_card_items": [
            {
                "id": _string(item.get("id")),
                "target_path": _string(item.get("target_path")),
                "human_operator_action": _string(item.get("human_operator_action")),
                "ready_for_manual_fill_preview": bool(item.get("ready_for_manual_fill_preview")),
                "capsule_alignment": True,
                "snapshot_alignment": bool(snapshot_map.get(_string(item.get("id")))),
                "cover_note_line": _string(item.get("cover_note_line")),
                "template_preview_key_count": int(item.get("template_preview_key_count") or 0),
            }
            for item in capsule_items
        ],
        "release_card_note": (
            "manual_pointer_fill_release_card_is_summary_only_and_cannot_enable_runtime_execution"
        ),
        "release_snapshot_forbidden_actions": _string_list(snapshot_payload.get("forbidden_actions")),
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_pointer_fill_release_tile_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_manual_fill_release_card_prerequisites"]
    if any("pointer_manual_fill_release_capsule" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_release_capsule")
    if any("pointer_manual_fill_release_snapshot" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_release_snapshot")
    if any("artifact_" in item or "operator_action" in item for item in blockers):
        actions.append("repair_pointer_manual_fill_release_card_inputs")
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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_CARD",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_card",
]
