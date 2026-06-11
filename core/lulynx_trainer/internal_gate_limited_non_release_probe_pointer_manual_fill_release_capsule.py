"""Release capsule for manual fill of pointer-only batch1 probe materials.

This capsule remains report-only. It assembles the release snapshot and
release digest into a lighter operator-facing summary without writing files,
enabling the internal gate, or starting any probe execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_CAPSULE = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_capsule_v0"
)
READY_POINTER_MANUAL_FILL_RELEASE_SNAPSHOT_STATUS = (
    "ready_for_followup_pointer_manual_fill_release_snapshot"
)
READY_POINTER_MANUAL_FILL_RELEASE_DIGEST_STATUS = (
    "ready_for_followup_pointer_manual_fill_release_digest"
)


def build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_capsule(
    *,
    internal_gate_limited_non_release_probe_pointer_manual_fill_release_snapshot: Mapping[str, Any]
    | None = None,
    internal_gate_limited_non_release_probe_pointer_manual_fill_release_digest: Mapping[str, Any]
    | None = None,
) -> dict[str, Any]:
    """Build a fail-closed release capsule for manual pointer fill handoff."""

    release_snapshot = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_release_snapshot)
    )
    release_digest = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_release_digest)
    )
    snapshot_payload = _mapping(release_snapshot.get("pointer_manual_fill_release_snapshot"))
    digest_payload = _mapping(release_digest.get("pointer_manual_fill_release_digest"))
    snapshot_items = _sequence_of_mappings(snapshot_payload.get("artifact_snapshot_items"))
    digest_items = _sequence_of_mappings(digest_payload.get("artifact_digest_items"))
    checks = _checks(
        release_snapshot=release_snapshot,
        release_digest=release_digest,
        snapshot_items=snapshot_items,
        digest_items=digest_items,
        snapshot_payload=snapshot_payload,
        digest_payload=digest_payload,
    )
    blockers = _blockers(
        release_snapshot=release_snapshot,
        release_digest=release_digest,
        checks=checks,
    )
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_CAPSULE,
        "status": "ready_for_followup_pointer_manual_fill_release_capsule" if ready else "blocked",
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
        "pointer_manual_fill_release_snapshot_summary": _snapshot_summary(release_snapshot),
        "pointer_manual_fill_release_digest_summary": _digest_summary(release_digest),
        "pointer_manual_fill_release_capsule": _release_capsule(
            snapshot_items=snapshot_items,
            digest_items=digest_items,
            snapshot_payload=snapshot_payload,
            digest_payload=digest_payload,
        ),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    release_snapshot: Mapping[str, Any],
    release_digest: Mapping[str, Any],
    snapshot_items: Sequence[Mapping[str, Any]],
    digest_items: Sequence[Mapping[str, Any]],
    snapshot_payload: Mapping[str, Any],
    digest_payload: Mapping[str, Any],
) -> dict[str, bool]:
    snapshot_ids = {str(item.get("id") or "") for item in snapshot_items if item.get("id")}
    digest_ids = {str(item.get("id") or "") for item in digest_items if item.get("id")}
    forbidden_actions = set(_string_list(snapshot_payload.get("forbidden_actions")))
    required_completion_evidence = _string_list(
        snapshot_payload.get("required_completion_evidence")
    )
    return {
        "pointer_manual_fill_release_snapshot_present": bool(release_snapshot),
        "pointer_manual_fill_release_snapshot_ready": bool(release_snapshot.get("passed"))
        and str(release_snapshot.get("status") or "")
        == READY_POINTER_MANUAL_FILL_RELEASE_SNAPSHOT_STATUS,
        "pointer_manual_fill_release_snapshot_default_off": (
            not bool(release_snapshot.get("internal_gate_enablement_allowed"))
            and not bool(release_snapshot.get("release_claim_allowed"))
        ),
        "pointer_manual_fill_release_digest_present": bool(release_digest),
        "pointer_manual_fill_release_digest_ready": bool(release_digest.get("passed"))
        and str(release_digest.get("status") or "") == READY_POINTER_MANUAL_FILL_RELEASE_DIGEST_STATUS,
        "pointer_manual_fill_release_digest_default_off": (
            not bool(release_digest.get("internal_gate_enablement_allowed"))
            and not bool(release_digest.get("release_claim_allowed"))
        ),
        "artifact_snapshot_item_count_is_four": len(snapshot_items) == 4,
        "artifact_digest_item_count_is_four": len(digest_items) == 4,
        "artifact_roles_aligned": snapshot_ids == digest_ids == {
            "baseline_manifest_pointer",
            "probe_manifest_pointer",
            "before_after_probe_evidence",
            "manual_probe_review_notes",
        },
        "all_artifacts_ready_for_preview": all(
            bool(item.get("ready_for_manual_fill_preview")) for item in snapshot_items
        ),
        "operator_actions_visible": all(
            _string(item.get("human_operator_action")) == "fill_template_preview_manually"
            for item in snapshot_items
        ),
        "forbidden_actions_keep_execution_closed": {
            "write_pointer_files_automatically",
            "enable_internal_gate_now",
            "start_training_now",
            "approve_batch2_4_8_release_probe",
            "approve_release_claim",
        }.issubset(forbidden_actions),
        "required_completion_evidence_visible": len(required_completion_evidence) >= 4,
        "release_snapshot_forbidden_actions_match_release_digest": set(
            _string_list(digest_payload.get("forbidden_actions"))
        )
        == forbidden_actions,
    }


def _blockers(
    *,
    release_snapshot: Mapping[str, Any],
    release_digest: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not release_snapshot:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_release_snapshot_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_release_snapshot:{item}"
        for item in _string_list(release_snapshot.get("blockers"))
    )
    if not release_digest:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_release_digest_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_release_digest:{item}"
        for item in _string_list(release_digest.get("blockers"))
    )
    return _dedupe(blockers)


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


def _digest_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_manual_fill_release_digest"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "artifact_digest_item_count": len(_sequence_of_mappings(payload.get("artifact_digest_items"))),
        "required_completion_evidence_count": len(
            _string_list(payload.get("required_completion_evidence"))
        ),
    }


def _release_capsule(
    *,
    snapshot_items: Sequence[Mapping[str, Any]],
    digest_items: Sequence[Mapping[str, Any]],
    snapshot_payload: Mapping[str, Any],
    digest_payload: Mapping[str, Any],
) -> dict[str, Any]:
    digest_map = {str(item.get("id") or ""): item for item in digest_items}
    return {
        "release_capsule_kind": "manual_pointer_fill_release_capsule_v0",
        "release_capsule_scope": "batch1_non_release_pointer_manual_fill_release_capsule",
        "forbidden_actions": _string_list(snapshot_payload.get("forbidden_actions")),
        "required_completion_evidence": _string_list(
            snapshot_payload.get("required_completion_evidence")
        ),
        "artifact_capsule_items": [
            {
                "id": _string(item.get("id")),
                "target_path": _string(item.get("target_path")),
                "human_operator_action": _string(item.get("human_operator_action")),
                "ready_for_manual_fill_preview": bool(item.get("ready_for_manual_fill_preview")),
                "snapshot_alignment": True,
                "digest_alignment": bool(digest_map.get(_string(item.get("id")))),
                "cover_note_line": _string(item.get("cover_note_line")),
                "template_preview_key_count": int(item.get("template_preview_key_count") or 0),
            }
            for item in snapshot_items
        ],
        "release_capsule_note": (
            "manual_pointer_fill_release_capsule_is_summary_only_and_cannot_enable_runtime_execution"
        ),
        "release_digest_forbidden_actions": _string_list(digest_payload.get("forbidden_actions")),
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_pointer_fill_release_card_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_manual_fill_release_capsule_prerequisites"]
    if any("pointer_manual_fill_release_snapshot" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_release_snapshot")
    if any("pointer_manual_fill_release_digest" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_release_digest")
    if any("artifact_" in item or "operator_action" in item for item in blockers):
        actions.append("repair_pointer_manual_fill_release_capsule_inputs")
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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_CAPSULE",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_capsule",
]
