"""Content manifest for pointer-only batch1 non-release probe artifacts.

This manifest does not write any pointer files. It only organizes the content
that a future manual operator would need to prepare for the four pointer-only
artifacts after readiness has been confirmed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_CONTENT_MANIFEST = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_content_manifest_v0"
)
READY_POINTER_ARTIFACT_READINESS_STATUS = "ready_for_followup_pointer_artifact_readiness"
READY_POINTER_ARTIFACT_PACKAGE_STATUS = "ready_for_followup_pointer_artifact_package"


def build_lulynx_internal_gate_limited_non_release_probe_pointer_content_manifest(
    *,
    internal_gate_limited_non_release_probe_pointer_artifact_readiness: Mapping[str, Any] | None = None,
    internal_gate_limited_non_release_probe_pointer_artifact_package: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed content manifest for pointer-only artifacts."""

    readiness = dict(_mapping(internal_gate_limited_non_release_probe_pointer_artifact_readiness))
    pointer_package = dict(_mapping(internal_gate_limited_non_release_probe_pointer_artifact_package))
    pointer_artifacts = _mapping(pointer_package.get("pointer_artifacts"))
    baseline_pointer = _mapping(pointer_artifacts.get("baseline_manifest_pointer"))
    probe_pointer = _mapping(pointer_artifacts.get("probe_manifest_pointer"))
    comparison_pointer = _mapping(pointer_artifacts.get("before_after_probe_evidence"))
    review_pointer = _mapping(pointer_artifacts.get("manual_probe_review_notes"))
    checks = _checks(
        readiness=readiness,
        pointer_package=pointer_package,
        baseline_pointer=baseline_pointer,
        probe_pointer=probe_pointer,
        comparison_pointer=comparison_pointer,
        review_pointer=review_pointer,
    )
    blockers = _blockers(readiness=readiness, pointer_package=pointer_package, checks=checks)
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_CONTENT_MANIFEST,
        "status": "ready_for_followup_pointer_content_manifest" if ready else "blocked",
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
        "pointer_artifact_readiness_summary": _readiness_summary(readiness),
        "pointer_artifact_package_summary": _package_summary(pointer_package),
        "pointer_content_manifest": _pointer_content_manifest(
            baseline_pointer=baseline_pointer,
            probe_pointer=probe_pointer,
            comparison_pointer=comparison_pointer,
            review_pointer=review_pointer,
        ),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    readiness: Mapping[str, Any],
    pointer_package: Mapping[str, Any],
    baseline_pointer: Mapping[str, Any],
    probe_pointer: Mapping[str, Any],
    comparison_pointer: Mapping[str, Any],
    review_pointer: Mapping[str, Any],
) -> dict[str, bool]:
    readiness_payload = _mapping(readiness.get("pointer_artifact_readiness"))
    baseline_template = _mapping(baseline_pointer.get("template"))
    probe_template = _mapping(probe_pointer.get("template"))
    comparison_template = _mapping(comparison_pointer.get("template"))
    review_template = _mapping(review_pointer.get("template"))
    return {
        "pointer_artifact_readiness_present": bool(readiness),
        "pointer_artifact_readiness_ready": bool(readiness.get("passed"))
        and str(readiness.get("status") or "") == READY_POINTER_ARTIFACT_READINESS_STATUS,
        "pointer_artifact_readiness_default_off": (
            not bool(readiness.get("internal_gate_enablement_allowed"))
            and not bool(readiness.get("release_claim_allowed"))
        ),
        "pointer_artifact_package_present": bool(pointer_package),
        "pointer_artifact_package_ready": bool(pointer_package.get("passed"))
        and str(pointer_package.get("status") or "") == READY_POINTER_ARTIFACT_PACKAGE_STATUS,
        "pointer_artifact_package_default_off": (
            not bool(pointer_package.get("internal_gate_enablement_allowed"))
            and not bool(pointer_package.get("release_claim_allowed"))
        ),
        "pointer_root_visible": bool(_string(readiness_payload.get("pointer_root"))),
        "baseline_source_manifest_visible": bool(
            _string(readiness_payload.get("baseline_source_manifest_path"))
        ),
        "pointer_targets_present": all(
            bool(_string(item.get("path")))
            for item in (
                baseline_pointer,
                probe_pointer,
                comparison_pointer,
                review_pointer,
            )
        ),
        "comparison_references_match_pointer_targets": (
            _string(comparison_template.get("before_manifest_pointer_path"))
            == _string(baseline_pointer.get("path"))
            and _string(comparison_template.get("after_manifest_pointer_path"))
            == _string(probe_pointer.get("path"))
        ),
        "comparison_requirements_nonempty": len(
            _string_list(comparison_template.get("required_comparisons"))
        )
        >= 5,
        "review_acknowledgements_nonempty": len(
            _string_list(review_template.get("required_acknowledgements"))
        )
        >= 4,
        "manual_fill_sections_complete": (
            bool(_string(baseline_template.get("source_manifest_path")))
            and bool(_string(probe_template.get("expected_probe_scope")))
            and bool(_string(probe_template.get("status")))
            and bool(_string(comparison_template.get("artifact_kind")))
            and bool(_string(review_template.get("artifact_kind")))
        ),
    }


def _blockers(
    *,
    readiness: Mapping[str, Any],
    pointer_package: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not readiness:
        blockers.append("internal_gate_limited_non_release_probe_pointer_artifact_readiness_missing")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_artifact_readiness:{item}"
        for item in _string_list(readiness.get("blockers"))
    )
    if not pointer_package:
        blockers.append("internal_gate_limited_non_release_probe_pointer_artifact_package_missing")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_artifact_package:{item}"
        for item in _string_list(pointer_package.get("blockers"))
    )
    return _dedupe(blockers)


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


def _package_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "pointer_artifact_count": len(_mapping(report.get("pointer_artifacts"))),
    }


def _pointer_content_manifest(
    *,
    baseline_pointer: Mapping[str, Any],
    probe_pointer: Mapping[str, Any],
    comparison_pointer: Mapping[str, Any],
    review_pointer: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_template = _mapping(baseline_pointer.get("template"))
    probe_template = _mapping(probe_pointer.get("template"))
    comparison_template = _mapping(comparison_pointer.get("template"))
    review_template = _mapping(review_pointer.get("template"))
    return {
        "manifest_kind": "manual_pointer_content_manifest_v0",
        "content_mode": "pointer_only",
        "execution_policy": "manual_only_non_release_batch1",
        "artifacts": [
            {
                "id": "baseline_manifest_pointer",
                "target_path": _string(baseline_pointer.get("path")),
                "content_template": {
                    "artifact_kind": _string(baseline_template.get("artifact_kind")),
                    "batch_contract": _string(baseline_template.get("batch_contract")),
                    "mode": _string(baseline_template.get("mode")),
                    "source_manifest_path": _string(baseline_template.get("source_manifest_path")),
                },
                "manual_fill_fields": [
                    "baseline_manifest_digest",
                    "baseline_capture_timestamp",
                    "baseline_operator_note",
                ],
            },
            {
                "id": "probe_manifest_pointer",
                "target_path": _string(probe_pointer.get("path")),
                "content_template": {
                    "artifact_kind": _string(probe_template.get("artifact_kind")),
                    "expected_probe_scope": _string(probe_template.get("expected_probe_scope")),
                    "batch_contract": _string(probe_template.get("batch_contract")),
                    "mode": _string(probe_template.get("mode")),
                    "status": _string(probe_template.get("status")),
                },
                "manual_fill_fields": [
                    "probe_manifest_path",
                    "probe_capture_timestamp",
                    "probe_operator_note",
                    "probe_stop_condition_summary",
                ],
            },
            {
                "id": "before_after_probe_evidence",
                "target_path": _string(comparison_pointer.get("path")),
                "content_template": {
                    "artifact_kind": _string(comparison_template.get("artifact_kind")),
                    "before_manifest_pointer_path": _string(
                        comparison_template.get("before_manifest_pointer_path")
                    ),
                    "after_manifest_pointer_path": _string(
                        comparison_template.get("after_manifest_pointer_path")
                    ),
                    "required_comparisons": _string_list(
                        comparison_template.get("required_comparisons")
                    ),
                    "mode": _string(comparison_template.get("mode")),
                },
                "manual_fill_sections": [
                    "comparison_inputs",
                    "comparison_results",
                    "regression_summary",
                    "hold_or_continue_recommendation",
                ],
            },
            {
                "id": "manual_probe_review_notes",
                "target_path": _string(review_pointer.get("path")),
                "content_template": {
                    "artifact_kind": _string(review_template.get("artifact_kind")),
                    "required_acknowledgements": _string_list(
                        review_template.get("required_acknowledgements")
                    ),
                    "mode": _string(review_template.get("mode")),
                },
                "manual_fill_sections": [
                    "reviewer_identity",
                    "review_decision",
                    "decision_rationale",
                    "followup_actions",
                ],
            },
        ],
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_pointer_content_checklists_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_content_manifest_prerequisites"]
    if any("pointer_artifact_readiness" in item for item in blockers):
        actions.append("refresh_pointer_artifact_readiness")
    if any("pointer_artifact_package" in item for item in blockers):
        actions.append("refresh_pointer_artifact_package")
    if any("comparison_" in item or "review_" in item for item in blockers):
        actions.append("repair_pointer_content_cross_references")
    return _dedupe(actions)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_CONTENT_MANIFEST",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_content_manifest",
]
