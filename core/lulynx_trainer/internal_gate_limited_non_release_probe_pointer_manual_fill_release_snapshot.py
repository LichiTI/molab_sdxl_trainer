"""Release snapshot for manual fill of pointer-only batch1 probe materials.

This snapshot remains report-only. It assembles the release digest and release
brief into a compact operator-facing snapshot without writing files, enabling
the internal gate, or starting any probe execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_SNAPSHOT = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_snapshot_v0"
)
READY_POINTER_MANUAL_FILL_RELEASE_DIGEST_STATUS = (
    "ready_for_followup_pointer_manual_fill_release_digest"
)
READY_POINTER_MANUAL_FILL_RELEASE_BRIEF_STATUS = (
    "ready_for_followup_pointer_manual_fill_release_brief"
)


def build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_snapshot(
    *,
    internal_gate_limited_non_release_probe_pointer_manual_fill_release_digest: Mapping[str, Any]
    | None = None,
    internal_gate_limited_non_release_probe_pointer_manual_fill_release_brief: Mapping[str, Any]
    | None = None,
) -> dict[str, Any]:
    """Build a fail-closed release snapshot for manual pointer fill handoff."""

    release_digest = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_release_digest)
    )
    release_brief = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_release_brief)
    )
    digest_payload = _mapping(release_digest.get("pointer_manual_fill_release_digest"))
    brief_payload = _mapping(release_brief.get("pointer_manual_fill_release_brief"))
    digest_items = _sequence_of_mappings(digest_payload.get("artifact_digest_items"))
    brief_items = _sequence_of_mappings(brief_payload.get("artifact_brief_items"))
    checks = _checks(
        release_digest=release_digest,
        release_brief=release_brief,
        digest_items=digest_items,
        brief_items=brief_items,
        digest_payload=digest_payload,
        brief_payload=brief_payload,
    )
    blockers = _blockers(
        release_digest=release_digest,
        release_brief=release_brief,
        checks=checks,
    )
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_SNAPSHOT,
        "status": "ready_for_followup_pointer_manual_fill_release_snapshot" if ready else "blocked",
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
        "pointer_manual_fill_release_digest_summary": _digest_summary(release_digest),
        "pointer_manual_fill_release_brief_summary": _brief_summary(release_brief),
        "pointer_manual_fill_release_snapshot": _release_snapshot(
            digest_items=digest_items,
            brief_items=brief_items,
            digest_payload=digest_payload,
            brief_payload=brief_payload,
        ),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    release_digest: Mapping[str, Any],
    release_brief: Mapping[str, Any],
    digest_items: Sequence[Mapping[str, Any]],
    brief_items: Sequence[Mapping[str, Any]],
    digest_payload: Mapping[str, Any],
    brief_payload: Mapping[str, Any],
) -> dict[str, bool]:
    digest_ids = {str(item.get("id") or "") for item in digest_items if item.get("id")}
    brief_ids = {str(item.get("id") or "") for item in brief_items if item.get("id")}
    forbidden_actions = set(_string_list(digest_payload.get("forbidden_actions")))
    required_completion_evidence = _string_list(digest_payload.get("required_completion_evidence"))
    return {
        "pointer_manual_fill_release_digest_present": bool(release_digest),
        "pointer_manual_fill_release_digest_ready": bool(release_digest.get("passed"))
        and str(release_digest.get("status") or "") == READY_POINTER_MANUAL_FILL_RELEASE_DIGEST_STATUS,
        "pointer_manual_fill_release_digest_default_off": (
            not bool(release_digest.get("internal_gate_enablement_allowed"))
            and not bool(release_digest.get("release_claim_allowed"))
        ),
        "pointer_manual_fill_release_brief_present": bool(release_brief),
        "pointer_manual_fill_release_brief_ready": bool(release_brief.get("passed"))
        and str(release_brief.get("status") or "") == READY_POINTER_MANUAL_FILL_RELEASE_BRIEF_STATUS,
        "pointer_manual_fill_release_brief_default_off": (
            not bool(release_brief.get("internal_gate_enablement_allowed"))
            and not bool(release_brief.get("release_claim_allowed"))
        ),
        "artifact_digest_item_count_is_four": len(digest_items) == 4,
        "artifact_brief_item_count_is_four": len(brief_items) == 4,
        "artifact_roles_aligned": digest_ids == brief_ids == {
            "baseline_manifest_pointer",
            "probe_manifest_pointer",
            "before_after_probe_evidence",
            "manual_probe_review_notes",
        },
        "all_artifacts_ready_for_preview": all(
            bool(item.get("ready_for_manual_fill_preview")) for item in digest_items
        ),
        "operator_actions_visible": all(
            _string(item.get("human_operator_action")) == "fill_template_preview_manually"
            for item in digest_items
        ),
        "forbidden_actions_keep_execution_closed": {
            "write_pointer_files_automatically",
            "enable_internal_gate_now",
            "start_training_now",
            "approve_batch2_4_8_release_probe",
            "approve_release_claim",
        }.issubset(forbidden_actions),
        "required_completion_evidence_visible": len(required_completion_evidence) >= 4,
        "release_digest_forbidden_actions_match_release_brief": set(
            _string_list(brief_payload.get("forbidden_actions"))
        )
        == forbidden_actions,
    }


def _blockers(
    *,
    release_digest: Mapping[str, Any],
    release_brief: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not release_digest:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_release_digest_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_release_digest:{item}"
        for item in _string_list(release_digest.get("blockers"))
    )
    if not release_brief:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_release_brief_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_release_brief:{item}"
        for item in _string_list(release_brief.get("blockers"))
    )
    return _dedupe(blockers)


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


def _release_snapshot(
    *,
    digest_items: Sequence[Mapping[str, Any]],
    brief_items: Sequence[Mapping[str, Any]],
    digest_payload: Mapping[str, Any],
    brief_payload: Mapping[str, Any],
) -> dict[str, Any]:
    brief_map = {str(item.get("id") or ""): item for item in brief_items}
    return {
        "release_snapshot_kind": "manual_pointer_fill_release_snapshot_v0",
        "release_snapshot_scope": "batch1_non_release_pointer_manual_fill_release_snapshot",
        "forbidden_actions": _string_list(digest_payload.get("forbidden_actions")),
        "required_completion_evidence": _string_list(
            digest_payload.get("required_completion_evidence")
        ),
        "artifact_snapshot_items": [
            {
                "id": _string(item.get("id")),
                "target_path": _string(item.get("target_path")),
                "human_operator_action": _string(item.get("human_operator_action")),
                "ready_for_manual_fill_preview": bool(item.get("ready_for_manual_fill_preview")),
                "digest_alignment": True,
                "brief_alignment": bool(brief_map.get(_string(item.get("id")))),
                "cover_note_line": _string(item.get("cover_note_line")),
                "template_preview_key_count": int(item.get("template_preview_key_count") or 0),
            }
            for item in digest_items
        ],
        "release_snapshot_note": (
            "manual_pointer_fill_release_snapshot_is_summary_only_and_cannot_enable_runtime_execution"
        ),
        "release_brief_forbidden_actions": _string_list(brief_payload.get("forbidden_actions")),
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_pointer_fill_release_capsule_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_manual_fill_release_snapshot_prerequisites"]
    if any("pointer_manual_fill_release_digest" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_release_digest")
    if any("pointer_manual_fill_release_brief" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_release_brief")
    if any("artifact_" in item or "operator_action" in item for item in blockers):
        actions.append("repair_pointer_manual_fill_release_snapshot_inputs")
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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_RELEASE_SNAPSHOT",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_release_snapshot",
]
