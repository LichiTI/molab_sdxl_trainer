"""Release digest for manual fill of pointer-only batch1 probe materials.

This digest remains report-only. It assembles the release brief and release
manifest into a lighter operator-facing summary without writing files,
enabling the internal gate, or starting any probe execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_DIGEST = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_digest_v0"
)
READY_POINTER_MANUAL_FILL_RELEASE_BRIEF_STATUS = (
    "ready_for_followup_pointer_manual_fill_release_brief"
)
READY_POINTER_MANUAL_FILL_RELEASE_MANIFEST_STATUS = (
    "ready_for_followup_pointer_manual_fill_release_manifest"
)


def build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_digest(
    *,
    internal_gate_limited_non_release_probe_pointer_manual_fill_release_brief: Mapping[str, Any]
    | None = None,
    internal_gate_limited_non_release_probe_pointer_manual_fill_release_manifest: Mapping[str, Any]
    | None = None,
) -> dict[str, Any]:
    """Build a fail-closed release digest for manual pointer fill handoff."""

    release_brief = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_release_brief)
    )
    release_manifest = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_release_manifest)
    )
    brief_payload = _mapping(release_brief.get("pointer_manual_fill_release_brief"))
    manifest_payload = _mapping(release_manifest.get("pointer_manual_fill_release_manifest"))
    brief_items = _sequence_of_mappings(brief_payload.get("artifact_brief_items"))
    manifest_items = _sequence_of_mappings(manifest_payload.get("artifact_manifest_items"))
    checks = _checks(
        release_brief=release_brief,
        release_manifest=release_manifest,
        brief_items=brief_items,
        manifest_items=manifest_items,
        brief_payload=brief_payload,
        manifest_payload=manifest_payload,
    )
    blockers = _blockers(
        release_brief=release_brief,
        release_manifest=release_manifest,
        checks=checks,
    )
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_DIGEST,
        "status": "ready_for_followup_pointer_manual_fill_release_digest" if ready else "blocked",
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
        "pointer_manual_fill_release_brief_summary": _brief_summary(release_brief),
        "pointer_manual_fill_release_manifest_summary": _manifest_summary(release_manifest),
        "pointer_manual_fill_release_digest": _release_digest(
            brief_items=brief_items,
            manifest_items=manifest_items,
            brief_payload=brief_payload,
            manifest_payload=manifest_payload,
        ),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    release_brief: Mapping[str, Any],
    release_manifest: Mapping[str, Any],
    brief_items: Sequence[Mapping[str, Any]],
    manifest_items: Sequence[Mapping[str, Any]],
    brief_payload: Mapping[str, Any],
    manifest_payload: Mapping[str, Any],
) -> dict[str, bool]:
    brief_ids = {str(item.get("id") or "") for item in brief_items if item.get("id")}
    manifest_ids = {str(item.get("id") or "") for item in manifest_items if item.get("id")}
    forbidden_actions = set(_string_list(brief_payload.get("forbidden_actions")))
    required_completion_evidence = _string_list(brief_payload.get("required_completion_evidence"))
    return {
        "pointer_manual_fill_release_brief_present": bool(release_brief),
        "pointer_manual_fill_release_brief_ready": bool(release_brief.get("passed"))
        and str(release_brief.get("status") or "") == READY_POINTER_MANUAL_FILL_RELEASE_BRIEF_STATUS,
        "pointer_manual_fill_release_brief_default_off": (
            not bool(release_brief.get("internal_gate_enablement_allowed"))
            and not bool(release_brief.get("release_claim_allowed"))
        ),
        "pointer_manual_fill_release_manifest_present": bool(release_manifest),
        "pointer_manual_fill_release_manifest_ready": bool(release_manifest.get("passed"))
        and str(release_manifest.get("status") or "")
        == READY_POINTER_MANUAL_FILL_RELEASE_MANIFEST_STATUS,
        "pointer_manual_fill_release_manifest_default_off": (
            not bool(release_manifest.get("internal_gate_enablement_allowed"))
            and not bool(release_manifest.get("release_claim_allowed"))
        ),
        "artifact_brief_item_count_is_four": len(brief_items) == 4,
        "artifact_manifest_item_count_is_four": len(manifest_items) == 4,
        "artifact_roles_aligned": brief_ids == manifest_ids == {
            "baseline_manifest_pointer",
            "probe_manifest_pointer",
            "before_after_probe_evidence",
            "manual_probe_review_notes",
        },
        "all_artifacts_ready_for_preview": all(
            bool(item.get("ready_for_manual_fill_preview")) for item in brief_items
        ),
        "operator_actions_visible": all(
            _string(item.get("human_operator_action")) == "fill_template_preview_manually"
            for item in brief_items
        ),
        "forbidden_actions_keep_execution_closed": {
            "write_pointer_files_automatically",
            "enable_internal_gate_now",
            "start_training_now",
            "approve_batch2_4_8_release_probe",
            "approve_release_claim",
        }.issubset(forbidden_actions),
        "required_completion_evidence_visible": len(required_completion_evidence) >= 4,
        "release_brief_forbidden_actions_match_release_manifest": set(
            _string_list(manifest_payload.get("forbidden_actions"))
        )
        == forbidden_actions,
    }


def _blockers(
    *,
    release_brief: Mapping[str, Any],
    release_manifest: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not release_brief:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_release_brief_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_release_brief:{item}"
        for item in _string_list(release_brief.get("blockers"))
    )
    if not release_manifest:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_release_manifest_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_release_manifest:{item}"
        for item in _string_list(release_manifest.get("blockers"))
    )
    return _dedupe(blockers)


def _brief_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_manual_fill_release_brief"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "artifact_brief_item_count": len(_sequence_of_mappings(payload.get("artifact_brief_items"))),
        "required_completion_evidence_count": len(
            _string_list(payload.get("required_completion_evidence"))
        ),
    }


def _manifest_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_manual_fill_release_manifest"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "artifact_manifest_item_count": len(
            _sequence_of_mappings(payload.get("artifact_manifest_items"))
        ),
        "required_completion_evidence_count": len(
            _string_list(payload.get("required_completion_evidence"))
        ),
    }


def _release_digest(
    *,
    brief_items: Sequence[Mapping[str, Any]],
    manifest_items: Sequence[Mapping[str, Any]],
    brief_payload: Mapping[str, Any],
    manifest_payload: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_map = {str(item.get("id") or ""): item for item in manifest_items}
    return {
        "release_digest_kind": "manual_pointer_fill_release_digest_v0",
        "release_digest_scope": "batch1_non_release_pointer_manual_fill_release_digest",
        "forbidden_actions": _string_list(brief_payload.get("forbidden_actions")),
        "required_completion_evidence": _string_list(
            brief_payload.get("required_completion_evidence")
        ),
        "artifact_digest_items": [
            {
                "id": _string(item.get("id")),
                "target_path": _string(item.get("target_path")),
                "human_operator_action": _string(item.get("human_operator_action")),
                "ready_for_manual_fill_preview": bool(item.get("ready_for_manual_fill_preview")),
                "brief_alignment": True,
                "manifest_alignment": bool(manifest_map.get(_string(item.get("id")))),
                "cover_note_line": _string(item.get("cover_note_line")),
                "template_preview_key_count": int(item.get("template_preview_key_count") or 0),
            }
            for item in brief_items
        ],
        "release_digest_note": (
            "manual_pointer_fill_release_digest_is_summary_only_and_cannot_enable_runtime_execution"
        ),
        "release_manifest_forbidden_actions": _string_list(manifest_payload.get("forbidden_actions")),
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_pointer_fill_release_snapshot_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_manual_fill_release_digest_prerequisites"]
    if any("pointer_manual_fill_release_brief" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_release_brief")
    if any("pointer_manual_fill_release_manifest" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_release_manifest")
    if any("artifact_" in item or "operator_action" in item for item in blockers):
        actions.append("repair_pointer_manual_fill_release_digest_inputs")
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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_DIGEST",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_digest",
]
