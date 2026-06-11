"""Checklist package for manual pointer-only batch1 probe preparation.

This package remains report-only. It organizes the manual preparation
checklists for pointer-only artifacts without writing files, enabling the
internal gate, or starting any probe execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_PREPARATION_CHECKLIST_PACKAGE = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_preparation_checklist_package_v0"
)
READY_POINTER_CONTENT_MANIFEST_STATUS = "ready_for_followup_pointer_content_manifest"
READY_POINTER_ARTIFACT_READINESS_STATUS = "ready_for_followup_pointer_artifact_readiness"


def build_lulynx_internal_gate_limited_non_release_probe_pointer_preparation_checklist_package(
    *,
    internal_gate_limited_non_release_probe_pointer_content_manifest: Mapping[str, Any] | None = None,
    internal_gate_limited_non_release_probe_pointer_artifact_readiness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed checklist package for manual pointer preparation."""

    content_manifest = dict(_mapping(internal_gate_limited_non_release_probe_pointer_content_manifest))
    readiness = dict(_mapping(internal_gate_limited_non_release_probe_pointer_artifact_readiness))
    manifest_payload = _mapping(content_manifest.get("pointer_content_manifest"))
    artifacts = _sequence_of_mappings(manifest_payload.get("artifacts"))
    readiness_payload = _mapping(readiness.get("pointer_artifact_readiness"))
    checks = _checks(
        content_manifest=content_manifest,
        readiness=readiness,
        artifacts=artifacts,
        readiness_payload=readiness_payload,
    )
    blockers = _blockers(content_manifest=content_manifest, readiness=readiness, checks=checks)
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_PREPARATION_CHECKLIST_PACKAGE,
        "status": "ready_for_followup_pointer_preparation_checklist_package" if ready else "blocked",
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
        "pointer_content_manifest_summary": _content_manifest_summary(content_manifest),
        "pointer_artifact_readiness_summary": _readiness_summary(readiness),
        "pointer_preparation_checklist_package": _pointer_preparation_checklist_package(
            artifacts=artifacts,
            readiness_payload=readiness_payload,
        ),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    content_manifest: Mapping[str, Any],
    readiness: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    readiness_payload: Mapping[str, Any],
) -> dict[str, bool]:
    artifact_ids = {str(item.get("id") or "") for item in artifacts if item.get("id")}
    return {
        "pointer_content_manifest_present": bool(content_manifest),
        "pointer_content_manifest_ready": bool(content_manifest.get("passed"))
        and str(content_manifest.get("status") or "") == READY_POINTER_CONTENT_MANIFEST_STATUS,
        "pointer_content_manifest_default_off": (
            not bool(content_manifest.get("internal_gate_enablement_allowed"))
            and not bool(content_manifest.get("release_claim_allowed"))
        ),
        "pointer_artifact_readiness_present": bool(readiness),
        "pointer_artifact_readiness_ready": bool(readiness.get("passed"))
        and str(readiness.get("status") or "") == READY_POINTER_ARTIFACT_READINESS_STATUS,
        "pointer_artifact_readiness_default_off": (
            not bool(readiness.get("internal_gate_enablement_allowed"))
            and not bool(readiness.get("release_claim_allowed"))
        ),
        "artifact_count_is_four": len(artifacts) == 4,
        "artifact_roles_complete": {
            "baseline_manifest_pointer",
            "probe_manifest_pointer",
            "before_after_probe_evidence",
            "manual_probe_review_notes",
        }.issubset(artifact_ids),
        "pointer_root_visible": bool(_string(readiness_payload.get("pointer_root"))),
        "comparison_count_visible": int(readiness_payload.get("required_comparison_count") or 0) >= 5,
        "acknowledgement_count_visible": int(
            readiness_payload.get("required_acknowledgement_count") or 0
        )
        >= 4,
        "all_artifacts_have_targets": all(bool(_string(item.get("target_path"))) for item in artifacts),
    }


def _blockers(
    *,
    content_manifest: Mapping[str, Any],
    readiness: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not content_manifest:
        blockers.append("internal_gate_limited_non_release_probe_pointer_content_manifest_missing")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_content_manifest:{item}"
        for item in _string_list(content_manifest.get("blockers"))
    )
    if not readiness:
        blockers.append("internal_gate_limited_non_release_probe_pointer_artifact_readiness_missing")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_artifact_readiness:{item}"
        for item in _string_list(readiness.get("blockers"))
    )
    return _dedupe(blockers)


def _content_manifest_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_content_manifest"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "artifact_count": len(_sequence_of_mappings(payload.get("artifacts"))),
        "execution_policy": str(payload.get("execution_policy") or ""),
    }


def _readiness_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_artifact_readiness"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "pointer_root": _string(payload.get("pointer_root")),
        "required_comparison_count": int(payload.get("required_comparison_count") or 0),
        "required_acknowledgement_count": int(payload.get("required_acknowledgement_count") or 0),
    }


def _pointer_preparation_checklist_package(
    *,
    artifacts: Sequence[Mapping[str, Any]],
    readiness_payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "package_kind": "manual_pointer_preparation_checklist_package_v0",
        "checklist_scope": "batch1_non_release_pointer_material_preparation",
        "pointer_root": _string(readiness_payload.get("pointer_root")),
        "forbidden_actions": [
            "write_pointer_files_automatically",
            "enable_internal_gate_now",
            "start_training_now",
            "approve_batch2_4_8_release_probe",
            "approve_release_claim",
        ],
        "required_completion_evidence": [
            "baseline_manifest_digest_recorded",
            "probe_manifest_capture_metadata_recorded",
            "before_after_comparison_placeholders_prepared",
            "review_acknowledgements_visible_before_manual_fill",
        ],
        "artifact_checklists": [_artifact_checklist(item) for item in artifacts],
    }


def _artifact_checklist(artifact: Mapping[str, Any]) -> dict[str, Any]:
    fill_fields = _string_list(artifact.get("manual_fill_fields"))
    fill_sections = _string_list(artifact.get("manual_fill_sections"))
    steps = [
        "confirm_target_path_under_pointer_root",
        "confirm_pointer_only_mode",
    ]
    if fill_fields:
        steps.append("prepare_manual_fill_fields")
    if fill_sections:
        steps.append("prepare_manual_fill_sections")
    return {
        "id": _string(artifact.get("id")),
        "target_path": _string(artifact.get("target_path")),
        "manual_fill_fields": fill_fields,
        "manual_fill_sections": fill_sections,
        "steps": steps,
        "completion_evidence": _artifact_completion_evidence(
            artifact_id=_string(artifact.get("id")),
            fill_fields=fill_fields,
            fill_sections=fill_sections,
        ),
    }


def _artifact_completion_evidence(
    *,
    artifact_id: str,
    fill_fields: Sequence[str],
    fill_sections: Sequence[str],
) -> list[str]:
    evidence = [f"{artifact_id}:target_path_confirmed", f"{artifact_id}:pointer_only_mode_confirmed"]
    if fill_fields:
        evidence.append(f"{artifact_id}:manual_fill_fields_listed")
    if fill_sections:
        evidence.append(f"{artifact_id}:manual_fill_sections_listed")
    return evidence


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_pointer_fill_templates_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_preparation_checklist_prerequisites"]
    if any("pointer_content_manifest" in item for item in blockers):
        actions.append("refresh_pointer_content_manifest")
    if any("pointer_artifact_readiness" in item for item in blockers):
        actions.append("refresh_pointer_artifact_readiness")
    if any("artifact_" in item or "pointer_root" in item for item in blockers):
        actions.append("repair_pointer_preparation_checklist_inputs")
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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_PREPARATION_CHECKLIST_PACKAGE",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_preparation_checklist_package",
]
