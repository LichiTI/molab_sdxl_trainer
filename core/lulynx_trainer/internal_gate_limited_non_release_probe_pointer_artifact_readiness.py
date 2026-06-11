"""Readiness gate for pointer-only batch1 non-release probe artifacts.

This gate validates that the pointer artifact package is internally complete
and still fail-closed before any future manual material preparation happens.
It does not write files, start a probe, enable the internal gate, or relax
release boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_ARTIFACT_READINESS = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_artifact_readiness_v0"
)
READY_POINTER_ARTIFACT_PACKAGE_STATUS = "ready_for_followup_pointer_artifact_package"
EXPECTED_SCOPE = "behavior_equivalent_internal_gate_batch1_non_release_probe"
EXPECTED_BATCH_CONTRACT = "real_gpu_batch1_only"
EXPECTED_MODE = "pointer_only"


def build_lulynx_internal_gate_limited_non_release_probe_pointer_artifact_readiness(
    *,
    internal_gate_limited_non_release_probe_pointer_artifact_package: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed readiness report for pointer-only artifacts."""

    pointer_package = dict(_mapping(internal_gate_limited_non_release_probe_pointer_artifact_package))
    pointer_artifacts = _mapping(pointer_package.get("pointer_artifacts"))
    baseline_pointer = _mapping(pointer_artifacts.get("baseline_manifest_pointer"))
    probe_pointer = _mapping(pointer_artifacts.get("probe_manifest_pointer"))
    comparison_pointer = _mapping(pointer_artifacts.get("before_after_probe_evidence"))
    review_pointer = _mapping(pointer_artifacts.get("manual_probe_review_notes"))
    checks = _checks(
        pointer_package=pointer_package,
        baseline_pointer=baseline_pointer,
        probe_pointer=probe_pointer,
        comparison_pointer=comparison_pointer,
        review_pointer=review_pointer,
    )
    blockers = _blockers(pointer_package=pointer_package, checks=checks)
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_ARTIFACT_READINESS,
        "status": "ready_for_followup_pointer_artifact_readiness" if ready else "blocked",
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
        "pointer_artifact_package_summary": _pointer_artifact_package_summary(pointer_package),
        "pointer_artifact_readiness": _pointer_artifact_readiness(
            baseline_pointer=baseline_pointer,
            probe_pointer=probe_pointer,
            comparison_pointer=comparison_pointer,
            review_pointer=review_pointer,
        ),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    pointer_package: Mapping[str, Any],
    baseline_pointer: Mapping[str, Any],
    probe_pointer: Mapping[str, Any],
    comparison_pointer: Mapping[str, Any],
    review_pointer: Mapping[str, Any],
) -> dict[str, bool]:
    baseline_template = _mapping(baseline_pointer.get("template"))
    probe_template = _mapping(probe_pointer.get("template"))
    comparison_template = _mapping(comparison_pointer.get("template"))
    review_template = _mapping(review_pointer.get("template"))
    comparison_set = set(_string_list(comparison_template.get("required_comparisons")))
    review_ack_set = set(_string_list(review_template.get("required_acknowledgements")))
    pointer_root_candidates = {
        _extract_root(_string(baseline_pointer.get("path")), "before/baseline_manifest_pointer.json"),
        _extract_root(_string(probe_pointer.get("path")), "after/probe_manifest_pointer.json"),
        _extract_root(
            _string(comparison_pointer.get("path")),
            "comparison/before_after_probe_evidence.json",
        ),
        _extract_root(_string(review_pointer.get("path")), "review/manual_probe_review_notes.json"),
    }
    pointer_root_candidates.discard("")
    return {
        "pointer_artifact_package_present": bool(pointer_package),
        "pointer_artifact_package_ready": bool(pointer_package.get("passed"))
        and str(pointer_package.get("status") or "") == READY_POINTER_ARTIFACT_PACKAGE_STATUS,
        "pointer_artifact_package_default_off": (
            not bool(pointer_package.get("internal_gate_enablement_allowed"))
            and not bool(pointer_package.get("release_claim_allowed"))
        ),
        "pointer_artifact_count_is_four": len(pointer_package.get("pointer_artifacts") or {}) == 4,
        "pointer_root_layout_consistent": len(pointer_root_candidates) == 1,
        "baseline_manifest_pointer_template_complete": (
            _string(baseline_template.get("artifact_kind")) == "baseline_manifest_pointer_v0"
            and _string(baseline_template.get("batch_contract")) == EXPECTED_BATCH_CONTRACT
            and _string(baseline_template.get("mode")) == EXPECTED_MODE
            and bool(_string(baseline_template.get("source_manifest_path")))
        ),
        "probe_manifest_pointer_template_complete": (
            _string(probe_template.get("artifact_kind")) == "probe_manifest_pointer_v0"
            and _string(probe_template.get("batch_contract")) == EXPECTED_BATCH_CONTRACT
            and _string(probe_template.get("expected_probe_scope")) == EXPECTED_SCOPE
            and _string(probe_template.get("mode")) == EXPECTED_MODE
            and _string(probe_template.get("status")) == "awaiting_manual_probe_manifest"
        ),
        "before_after_probe_evidence_template_complete": (
            _string(comparison_template.get("artifact_kind")) == "before_after_probe_evidence_pointer_v0"
            and _string(comparison_template.get("mode")) == EXPECTED_MODE
            and _string(comparison_template.get("before_manifest_pointer_path"))
            == _string(baseline_pointer.get("path"))
            and _string(comparison_template.get("after_manifest_pointer_path"))
            == _string(probe_pointer.get("path"))
            and {
                "throughput_delta",
                "active_gpu_window_delta",
                "vram_delta",
                "loss_delta",
                "runtime_path_diff_summary",
            }.issubset(comparison_set)
        ),
        "manual_probe_review_notes_template_complete": (
            _string(review_template.get("artifact_kind")) == "manual_probe_review_notes_v0"
            and _string(review_template.get("mode")) == EXPECTED_MODE
            and {
                "internal_gate_stays_disabled",
                "batch1_only_non_release_probe",
                "batch2_4_8_release_probe_still_blocked",
                "release_claim_stays_closed",
            }.issubset(review_ack_set)
        ),
    }


def _blockers(*, pointer_package: Mapping[str, Any], checks: Mapping[str, bool]) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not pointer_package:
        blockers.append("internal_gate_limited_non_release_probe_pointer_artifact_package_missing")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_artifact_package:{item}"
        for item in _string_list(pointer_package.get("blockers"))
    )
    return _dedupe(blockers)


def _pointer_artifact_package_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    source_manifest_summary = _mapping(report.get("source_manifest_summary"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "pointer_artifact_count": len(_mapping(report.get("pointer_artifacts"))),
        "source_manifest_passed": bool(source_manifest_summary.get("passed")),
        "source_manifest_status": str(source_manifest_summary.get("status") or ""),
    }


def _pointer_artifact_readiness(
    *,
    baseline_pointer: Mapping[str, Any],
    probe_pointer: Mapping[str, Any],
    comparison_pointer: Mapping[str, Any],
    review_pointer: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_path = _string(baseline_pointer.get("path"))
    probe_path = _string(probe_pointer.get("path"))
    comparison_path = _string(comparison_pointer.get("path"))
    review_path = _string(review_pointer.get("path"))
    return {
        "pointer_root": _first_non_empty(
            [
                _extract_root(baseline_path, "before/baseline_manifest_pointer.json"),
                _extract_root(probe_path, "after/probe_manifest_pointer.json"),
                _extract_root(comparison_path, "comparison/before_after_probe_evidence.json"),
                _extract_root(review_path, "review/manual_probe_review_notes.json"),
            ]
        ),
        "baseline_manifest_pointer_path": baseline_path,
        "probe_manifest_pointer_path": probe_path,
        "before_after_probe_evidence_path": comparison_path,
        "manual_probe_review_notes_path": review_path,
        "baseline_source_manifest_path": _string(
            _mapping(baseline_pointer.get("template")).get("source_manifest_path")
        ),
        "expected_probe_scope": _string(
            _mapping(probe_pointer.get("template")).get("expected_probe_scope")
        ),
        "required_comparison_count": len(
            _string_list(_mapping(comparison_pointer.get("template")).get("required_comparisons"))
        ),
        "required_acknowledgement_count": len(
            _string_list(
                _mapping(review_pointer.get("template")).get("required_acknowledgements")
            )
        ),
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_pointer_contents_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_artifact_readiness_prerequisites"]
    if any("pointer_artifact_package" in item for item in blockers):
        actions.append("refresh_pointer_artifact_package")
    if any("pointer_root_layout_consistent" in item for item in blockers):
        actions.append("repair_pointer_artifact_layout_under_shared_root")
    if any("template_complete" in item for item in blockers):
        actions.append("repair_pointer_only_template_fields")
    return _dedupe(actions)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _string(value: Any) -> str:
    return str(value or "")


def _norm_path(path_text: str) -> str:
    return str(path_text or "").replace("\\", "/").strip()


def _extract_root(path_text: str, suffix: str) -> str:
    normalized_path = _norm_path(path_text)
    normalized_suffix = _norm_path(suffix)
    if not normalized_path or not normalized_path.endswith(normalized_suffix):
        return ""
    return normalized_path[: -len(normalized_suffix)].rstrip("/")


def _first_non_empty(values: Sequence[str]) -> str:
    for value in values:
        if value:
            return value
    return ""


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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_ARTIFACT_READINESS",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_artifact_readiness",
]
