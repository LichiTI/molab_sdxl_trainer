"""Manual fill template bundle for pointer-only batch1 probe materials.

This bundle stays report-only. It expands the manual-fill preview templates for
the pointer-only artifacts without writing files, enabling the internal gate,
or starting any probe execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_TEMPLATE_BUNDLE = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_template_bundle_v0"
)
READY_POINTER_PREPARATION_CHECKLIST_PACKAGE_STATUS = (
    "ready_for_followup_pointer_preparation_checklist_package"
)
READY_POINTER_CONTENT_MANIFEST_STATUS = "ready_for_followup_pointer_content_manifest"


def build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_template_bundle(
    *,
    internal_gate_limited_non_release_probe_pointer_preparation_checklist_package: Mapping[str, Any]
    | None = None,
    internal_gate_limited_non_release_probe_pointer_content_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed preview bundle for manual pointer fill templates."""

    checklist_package = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_preparation_checklist_package)
    )
    content_manifest = dict(_mapping(internal_gate_limited_non_release_probe_pointer_content_manifest))
    checklist_payload = _mapping(checklist_package.get("pointer_preparation_checklist_package"))
    content_payload = _mapping(content_manifest.get("pointer_content_manifest"))
    artifact_checklists = _sequence_of_mappings(checklist_payload.get("artifact_checklists"))
    artifacts = _sequence_of_mappings(content_payload.get("artifacts"))
    checks = _checks(
        checklist_package=checklist_package,
        content_manifest=content_manifest,
        artifact_checklists=artifact_checklists,
        artifacts=artifacts,
    )
    blockers = _blockers(
        checklist_package=checklist_package,
        content_manifest=content_manifest,
        checks=checks,
    )
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_TEMPLATE_BUNDLE,
        "status": "ready_for_followup_pointer_manual_fill_template_bundle" if ready else "blocked",
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
        "pointer_preparation_checklist_package_summary": _checklist_summary(checklist_package),
        "pointer_content_manifest_summary": _content_summary(content_manifest),
        "pointer_manual_fill_template_bundle": _template_bundle(
            artifact_checklists=artifact_checklists,
            artifacts=artifacts,
        ),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    checklist_package: Mapping[str, Any],
    content_manifest: Mapping[str, Any],
    artifact_checklists: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    checklist_ids = {str(item.get("id") or "") for item in artifact_checklists if item.get("id")}
    artifact_ids = {str(item.get("id") or "") for item in artifacts if item.get("id")}
    forbidden_actions = set(
        _string_list(_mapping(checklist_package.get("pointer_preparation_checklist_package")).get("forbidden_actions"))
    )
    required_completion_evidence = _string_list(
        _mapping(checklist_package.get("pointer_preparation_checklist_package")).get(
            "required_completion_evidence"
        )
    )
    return {
        "pointer_preparation_checklist_package_present": bool(checklist_package),
        "pointer_preparation_checklist_package_ready": bool(checklist_package.get("passed"))
        and str(checklist_package.get("status") or "")
        == READY_POINTER_PREPARATION_CHECKLIST_PACKAGE_STATUS,
        "pointer_preparation_checklist_package_default_off": (
            not bool(checklist_package.get("internal_gate_enablement_allowed"))
            and not bool(checklist_package.get("release_claim_allowed"))
        ),
        "pointer_content_manifest_present": bool(content_manifest),
        "pointer_content_manifest_ready": bool(content_manifest.get("passed"))
        and str(content_manifest.get("status") or "") == READY_POINTER_CONTENT_MANIFEST_STATUS,
        "pointer_content_manifest_default_off": (
            not bool(content_manifest.get("internal_gate_enablement_allowed"))
            and not bool(content_manifest.get("release_claim_allowed"))
        ),
        "artifact_checklist_count_is_four": len(artifact_checklists) == 4,
        "artifact_template_count_is_four": len(artifacts) == 4,
        "artifact_roles_match_between_inputs": checklist_ids == artifact_ids == {
            "baseline_manifest_pointer",
            "probe_manifest_pointer",
            "before_after_probe_evidence",
            "manual_probe_review_notes",
        },
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
    checklist_package: Mapping[str, Any],
    content_manifest: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not checklist_package:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_preparation_checklist_package_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_preparation_checklist_package:{item}"
        for item in _string_list(checklist_package.get("blockers"))
    )
    if not content_manifest:
        blockers.append("internal_gate_limited_non_release_probe_pointer_content_manifest_missing")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_content_manifest:{item}"
        for item in _string_list(content_manifest.get("blockers"))
    )
    return _dedupe(blockers)


def _checklist_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_preparation_checklist_package"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "artifact_checklist_count": len(_sequence_of_mappings(payload.get("artifact_checklists"))),
        "required_completion_evidence_count": len(
            _string_list(payload.get("required_completion_evidence"))
        ),
    }


def _content_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_content_manifest"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "artifact_template_count": len(_sequence_of_mappings(payload.get("artifacts"))),
        "execution_policy": str(payload.get("execution_policy") or ""),
    }


def _template_bundle(
    *,
    artifact_checklists: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checklist_map = {str(item.get("id") or ""): item for item in artifact_checklists}
    artifact_map = {str(item.get("id") or ""): item for item in artifacts}
    return {
        "bundle_kind": "manual_pointer_fill_template_bundle_v0",
        "bundle_mode": "pointer_only_preview",
        "artifact_templates": [
            _baseline_template(
                artifact=artifact_map.get("baseline_manifest_pointer", {}),
                checklist=checklist_map.get("baseline_manifest_pointer", {}),
            ),
            _probe_template(
                artifact=artifact_map.get("probe_manifest_pointer", {}),
                checklist=checklist_map.get("probe_manifest_pointer", {}),
            ),
            _comparison_template(
                artifact=artifact_map.get("before_after_probe_evidence", {}),
                checklist=checklist_map.get("before_after_probe_evidence", {}),
            ),
            _review_template(
                artifact=artifact_map.get("manual_probe_review_notes", {}),
                checklist=checklist_map.get("manual_probe_review_notes", {}),
            ),
        ],
    }


def _baseline_template(*, artifact: Mapping[str, Any], checklist: Mapping[str, Any]) -> dict[str, Any]:
    template = _mapping(artifact.get("content_template"))
    return {
        "id": "baseline_manifest_pointer",
        "target_path": _string(artifact.get("target_path") or checklist.get("target_path")),
        "template_preview": {
            "artifact_kind": _string(template.get("artifact_kind")),
            "batch_contract": _string(template.get("batch_contract")),
            "mode": _string(template.get("mode")),
            "source_manifest_path": _string(template.get("source_manifest_path")),
            "baseline_manifest_digest": "<fill-manually>",
            "baseline_capture_timestamp": "<fill-manually>",
            "baseline_operator_note": "<fill-manually>",
        },
        "checklist_steps": _string_list(checklist.get("steps")),
    }


def _probe_template(*, artifact: Mapping[str, Any], checklist: Mapping[str, Any]) -> dict[str, Any]:
    template = _mapping(artifact.get("content_template"))
    return {
        "id": "probe_manifest_pointer",
        "target_path": _string(artifact.get("target_path") or checklist.get("target_path")),
        "template_preview": {
            "artifact_kind": _string(template.get("artifact_kind")),
            "expected_probe_scope": _string(template.get("expected_probe_scope")),
            "batch_contract": _string(template.get("batch_contract")),
            "mode": _string(template.get("mode")),
            "status": _string(template.get("status")),
            "probe_manifest_path": "<fill-manually>",
            "probe_capture_timestamp": "<fill-manually>",
            "probe_operator_note": "<fill-manually>",
            "probe_stop_condition_summary": "<fill-manually>",
        },
        "checklist_steps": _string_list(checklist.get("steps")),
    }


def _comparison_template(
    *, artifact: Mapping[str, Any], checklist: Mapping[str, Any]
) -> dict[str, Any]:
    template = _mapping(artifact.get("content_template"))
    required_comparisons = _string_list(template.get("required_comparisons"))
    return {
        "id": "before_after_probe_evidence",
        "target_path": _string(artifact.get("target_path") or checklist.get("target_path")),
        "template_preview": {
            "artifact_kind": _string(template.get("artifact_kind")),
            "before_manifest_pointer_path": _string(template.get("before_manifest_pointer_path")),
            "after_manifest_pointer_path": _string(template.get("after_manifest_pointer_path")),
            "mode": _string(template.get("mode")),
            "comparison_inputs": "<fill-manually>",
            "comparison_results": {item: "<fill-manually>" for item in required_comparisons},
            "regression_summary": "<fill-manually>",
            "hold_or_continue_recommendation": "<fill-manually>",
        },
        "checklist_steps": _string_list(checklist.get("steps")),
    }


def _review_template(*, artifact: Mapping[str, Any], checklist: Mapping[str, Any]) -> dict[str, Any]:
    template = _mapping(artifact.get("content_template"))
    acknowledgements = _string_list(template.get("required_acknowledgements"))
    return {
        "id": "manual_probe_review_notes",
        "target_path": _string(artifact.get("target_path") or checklist.get("target_path")),
        "template_preview": {
            "artifact_kind": _string(template.get("artifact_kind")),
            "mode": _string(template.get("mode")),
            "required_acknowledgements": {item: "pending_manual_ack" for item in acknowledgements},
            "reviewer_identity": "<fill-manually>",
            "review_decision": "<fill-manually>",
            "decision_rationale": "<fill-manually>",
            "followup_actions": ["<fill-manually>"],
        },
        "checklist_steps": _string_list(checklist.get("steps")),
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_pointer_fill_examples_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_manual_fill_template_bundle_prerequisites"]
    if any("pointer_preparation_checklist_package" in item for item in blockers):
        actions.append("refresh_pointer_preparation_checklist_package")
    if any("pointer_content_manifest" in item for item in blockers):
        actions.append("refresh_pointer_content_manifest")
    if any("artifact_" in item or "forbidden_actions" in item for item in blockers):
        actions.append("repair_pointer_manual_fill_template_inputs")
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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_TEMPLATE_BUNDLE",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_template_bundle",
]
