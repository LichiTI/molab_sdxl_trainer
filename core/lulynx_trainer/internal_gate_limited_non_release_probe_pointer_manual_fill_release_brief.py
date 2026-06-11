"""Release brief for manual fill of pointer-only batch1 probe materials.

This brief remains report-only. It assembles the release manifest and release
index into a short operator-facing handoff without writing files, enabling the
internal gate, or starting any probe execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_BRIEF = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_brief_v0"
)
READY_POINTER_MANUAL_FILL_RELEASE_MANIFEST_STATUS = (
    "ready_for_followup_pointer_manual_fill_release_manifest"
)
READY_POINTER_MANUAL_FILL_RELEASE_INDEX_STATUS = (
    "ready_for_followup_pointer_manual_fill_release_index"
)


def build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_brief(
    *,
    internal_gate_limited_non_release_probe_pointer_manual_fill_release_manifest: Mapping[str, Any]
    | None = None,
    internal_gate_limited_non_release_probe_pointer_manual_fill_release_index: Mapping[str, Any]
    | None = None,
) -> dict[str, Any]:
    """Build a fail-closed release brief for manual pointer fill handoff."""

    release_manifest = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_release_manifest)
    )
    release_index = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_release_index)
    )
    manifest_payload = _mapping(release_manifest.get("pointer_manual_fill_release_manifest"))
    index_payload = _mapping(release_index.get("pointer_manual_fill_release_index"))
    manifest_items = _sequence_of_mappings(manifest_payload.get("artifact_manifest_items"))
    index_items = _sequence_of_mappings(index_payload.get("artifact_index_items"))
    checks = _checks(
        release_manifest=release_manifest,
        release_index=release_index,
        manifest_items=manifest_items,
        index_items=index_items,
        manifest_payload=manifest_payload,
        index_payload=index_payload,
    )
    blockers = _blockers(
        release_manifest=release_manifest,
        release_index=release_index,
        checks=checks,
    )
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_BRIEF,
        "status": "ready_for_followup_pointer_manual_fill_release_brief" if ready else "blocked",
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
        "pointer_manual_fill_release_manifest_summary": _manifest_summary(release_manifest),
        "pointer_manual_fill_release_index_summary": _index_summary(release_index),
        "pointer_manual_fill_release_brief": _release_brief(
            manifest_items=manifest_items,
            index_items=index_items,
            manifest_payload=manifest_payload,
            index_payload=index_payload,
        ),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    release_manifest: Mapping[str, Any],
    release_index: Mapping[str, Any],
    manifest_items: Sequence[Mapping[str, Any]],
    index_items: Sequence[Mapping[str, Any]],
    manifest_payload: Mapping[str, Any],
    index_payload: Mapping[str, Any],
) -> dict[str, bool]:
    manifest_ids = {str(item.get("id") or "") for item in manifest_items if item.get("id")}
    index_ids = {str(item.get("id") or "") for item in index_items if item.get("id")}
    forbidden_actions = set(_string_list(manifest_payload.get("forbidden_actions")))
    required_completion_evidence = _string_list(
        manifest_payload.get("required_completion_evidence")
    )
    return {
        "pointer_manual_fill_release_manifest_present": bool(release_manifest),
        "pointer_manual_fill_release_manifest_ready": bool(release_manifest.get("passed"))
        and str(release_manifest.get("status") or "")
        == READY_POINTER_MANUAL_FILL_RELEASE_MANIFEST_STATUS,
        "pointer_manual_fill_release_manifest_default_off": (
            not bool(release_manifest.get("internal_gate_enablement_allowed"))
            and not bool(release_manifest.get("release_claim_allowed"))
        ),
        "pointer_manual_fill_release_index_present": bool(release_index),
        "pointer_manual_fill_release_index_ready": bool(release_index.get("passed"))
        and str(release_index.get("status") or "") == READY_POINTER_MANUAL_FILL_RELEASE_INDEX_STATUS,
        "pointer_manual_fill_release_index_default_off": (
            not bool(release_index.get("internal_gate_enablement_allowed"))
            and not bool(release_index.get("release_claim_allowed"))
        ),
        "artifact_manifest_item_count_is_four": len(manifest_items) == 4,
        "artifact_index_item_count_is_four": len(index_items) == 4,
        "artifact_roles_aligned": manifest_ids == index_ids == {
            "baseline_manifest_pointer",
            "probe_manifest_pointer",
            "before_after_probe_evidence",
            "manual_probe_review_notes",
        },
        "all_artifacts_ready_for_preview": all(
            bool(item.get("ready_for_manual_fill_preview")) for item in manifest_items
        ),
        "operator_actions_visible": all(
            _string(item.get("human_operator_action")) == "fill_template_preview_manually"
            for item in manifest_items
        ),
        "forbidden_actions_keep_execution_closed": {
            "write_pointer_files_automatically",
            "enable_internal_gate_now",
            "start_training_now",
            "approve_batch2_4_8_release_probe",
            "approve_release_claim",
        }.issubset(forbidden_actions),
        "required_completion_evidence_visible": len(required_completion_evidence) >= 4,
        "release_manifest_forbidden_actions_match_release_index": set(
            _string_list(index_payload.get("forbidden_actions"))
        )
        == forbidden_actions,
    }


def _blockers(
    *,
    release_manifest: Mapping[str, Any],
    release_index: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not release_manifest:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_release_manifest_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_release_manifest:{item}"
        for item in _string_list(release_manifest.get("blockers"))
    )
    if not release_index:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_release_index_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_release_index:{item}"
        for item in _string_list(release_index.get("blockers"))
    )
    return _dedupe(blockers)


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


def _release_brief(
    *,
    manifest_items: Sequence[Mapping[str, Any]],
    index_items: Sequence[Mapping[str, Any]],
    manifest_payload: Mapping[str, Any],
    index_payload: Mapping[str, Any],
) -> dict[str, Any]:
    index_map = {str(item.get("id") or ""): item for item in index_items}
    return {
        "release_brief_kind": "manual_pointer_fill_release_brief_v0",
        "release_brief_scope": "batch1_non_release_pointer_manual_fill_release_brief",
        "forbidden_actions": _string_list(manifest_payload.get("forbidden_actions")),
        "required_completion_evidence": _string_list(
            manifest_payload.get("required_completion_evidence")
        ),
        "artifact_brief_items": [
            {
                "id": _string(item.get("id")),
                "target_path": _string(item.get("target_path")),
                "human_operator_action": _string(item.get("human_operator_action")),
                "ready_for_manual_fill_preview": bool(item.get("ready_for_manual_fill_preview")),
                "manifest_alignment": True,
                "index_alignment": bool(index_map.get(_string(item.get("id")))),
                "cover_note_line": _string(item.get("cover_note_line")),
                "template_preview_key_count": len(_string_list(item.get("template_preview_keys"))),
            }
            for item in manifest_items
        ],
        "release_brief_note": (
            "manual_pointer_fill_release_brief_is_summary_only_and_cannot_enable_runtime_execution"
        ),
        "release_index_forbidden_actions": _string_list(index_payload.get("forbidden_actions")),
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_pointer_fill_release_digest_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_manual_fill_release_brief_prerequisites"]
    if any("pointer_manual_fill_release_manifest" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_release_manifest")
    if any("pointer_manual_fill_release_index" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_release_index")
    if any("artifact_" in item or "operator_action" in item for item in blockers):
        actions.append("repair_pointer_manual_fill_release_brief_inputs")
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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_BRIEF",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_brief",
]
