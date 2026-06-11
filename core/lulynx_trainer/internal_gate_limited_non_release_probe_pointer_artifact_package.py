"""Report-only pointer artifact package for batch1 non-release follow-up materials.

This package does not write any pointer files. It only defines the content
shape for the pointer-only artifacts described by the source manifest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_ARTIFACT_PACKAGE = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_artifact_package_v0"
)
READY_SOURCE_MANIFEST_STATUS = "ready_for_followup_manual_probe_source_manifest"


def build_lulynx_internal_gate_limited_non_release_probe_pointer_artifact_package(
    *,
    internal_gate_limited_non_release_probe_followup_source_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed package describing pointer-only artifact contents."""

    source_manifest = dict(_mapping(internal_gate_limited_non_release_probe_followup_source_manifest))
    before_sources = _sequence_of_mappings(source_manifest.get("before_sources"))
    after_destination = _mapping(source_manifest.get("after_destination"))
    checks = _checks(
        source_manifest=source_manifest,
        before_sources=before_sources,
        after_destination=after_destination,
    )
    blockers = _blockers(source_manifest=source_manifest, checks=checks)
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_ARTIFACT_PACKAGE,
        "status": "ready_for_followup_pointer_artifact_package" if ready else "blocked",
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
        "source_manifest_summary": _source_manifest_summary(source_manifest),
        "pointer_artifacts": _pointer_artifacts(before_sources, after_destination),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    source_manifest: Mapping[str, Any],
    before_sources: Sequence[Mapping[str, Any]],
    after_destination: Mapping[str, Any],
) -> dict[str, bool]:
    expected_artifacts = _string_list(after_destination.get("expected_artifacts"))
    return {
        "source_manifest_present": bool(source_manifest),
        "source_manifest_ready": bool(source_manifest.get("passed"))
        and str(source_manifest.get("status") or "") == READY_SOURCE_MANIFEST_STATUS,
        "source_manifest_default_off": (
            not bool(source_manifest.get("internal_gate_enablement_allowed"))
            and not bool(source_manifest.get("release_claim_allowed"))
        ),
        "before_sources_complete": all(
            bool(item.get("exists")) and bool(item.get("is_file")) for item in before_sources
        ),
        "expected_artifact_count_is_four": len(expected_artifacts) == 4,
        "expected_pointer_paths_cover_before_after_comparison_review": _expected_paths_cover_roles(
            expected_artifacts
        ),
    }


def _expected_paths_cover_roles(paths: Sequence[str]) -> bool:
    text = " ".join(str(path) for path in paths)
    return all(token in text for token in ("before", "after", "comparison", "review"))


def _blockers(*, source_manifest: Mapping[str, Any], checks: Mapping[str, bool]) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not source_manifest:
        blockers.append("internal_gate_limited_non_release_probe_followup_source_manifest_missing")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_followup_source_manifest:{item}"
        for item in _string_list(source_manifest.get("blockers"))
    )
    return _dedupe(blockers)


def _source_manifest_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "before_source_count": len(_sequence_of_mappings(report.get("before_sources"))),
        "expected_artifact_count": len(
            _string_list(_mapping(report.get("after_destination")).get("expected_artifacts"))
        ),
    }


def _pointer_artifacts(
    before_sources: Sequence[Mapping[str, Any]],
    after_destination: Mapping[str, Any],
) -> dict[str, Any]:
    before_map = {str(item.get("id") or ""): item for item in before_sources}
    expected = _string_list(after_destination.get("expected_artifacts"))
    path_map = {
        "baseline_manifest_pointer": expected[0] if len(expected) > 0 else "",
        "probe_manifest_pointer": expected[1] if len(expected) > 1 else "",
        "before_after_probe_evidence": expected[2] if len(expected) > 2 else "",
        "manual_probe_review_notes": expected[3] if len(expected) > 3 else "",
    }
    return {
        "baseline_manifest_pointer": {
            "path": path_map["baseline_manifest_pointer"],
            "template": {
                "artifact_kind": "baseline_manifest_pointer_v0",
                "source_manifest_path": str(
                    _mapping(before_map.get("baseline_before_manifest")).get("path") or ""
                ),
                "batch_contract": "real_gpu_batch1_only",
                "mode": "pointer_only",
            },
        },
        "probe_manifest_pointer": {
            "path": path_map["probe_manifest_pointer"],
            "template": {
                "artifact_kind": "probe_manifest_pointer_v0",
                "expected_probe_scope": "behavior_equivalent_internal_gate_batch1_non_release_probe",
                "batch_contract": "real_gpu_batch1_only",
                "mode": "pointer_only",
                "status": "awaiting_manual_probe_manifest",
            },
        },
        "before_after_probe_evidence": {
            "path": path_map["before_after_probe_evidence"],
            "template": {
                "artifact_kind": "before_after_probe_evidence_pointer_v0",
                "before_manifest_pointer_path": path_map["baseline_manifest_pointer"],
                "after_manifest_pointer_path": path_map["probe_manifest_pointer"],
                "required_comparisons": [
                    "throughput_delta",
                    "active_gpu_window_delta",
                    "vram_delta",
                    "loss_delta",
                    "runtime_path_diff_summary",
                ],
                "mode": "pointer_only",
            },
        },
        "manual_probe_review_notes": {
            "path": path_map["manual_probe_review_notes"],
            "template": {
                "artifact_kind": "manual_probe_review_notes_v0",
                "required_acknowledgements": [
                    "internal_gate_stays_disabled",
                    "batch1_only_non_release_probe",
                    "batch2_4_8_release_probe_still_blocked",
                    "release_claim_stays_closed",
                ],
                "mode": "pointer_only",
            },
        },
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_pointer_only_artifact_contents_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_artifact_package_prerequisites"]
    if any("followup_source_manifest" in item for item in blockers):
        actions.append("refresh_followup_source_manifest")
    if any("before_sources_complete" in item for item in blockers):
        actions.append("refresh_before_sources_for_pointer_artifacts")
    if any("expected_artifact" in item for item in blockers):
        actions.append("repair_after_destination_expected_artifact_layout")
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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_ARTIFACT_PACKAGE",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_artifact_package",
]
