"""Report-only source manifest for batch1 non-release follow-up materials.

This manifest pins the real before-evidence sources and the intended
after-evidence destination shape for a future manual-only probe. It does not
create directories, start training, or enable the internal gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_FOLLOWUP_SOURCE_MANIFEST = (
    "lulynx_internal_gate_limited_non_release_probe_followup_source_manifest_v0"
)
READY_MATERIAL_READINESS_STATUS = "ready_for_followup_manual_probe_material_readiness"


def build_lulynx_internal_gate_limited_non_release_probe_followup_source_manifest(
    *,
    internal_gate_limited_non_release_probe_followup_material_readiness: Mapping[str, Any] | None = None,
    baseline_before_manifest_path: str = "",
    batch1_parity_smoke_path: str = "",
    real_gpu_batch1_golden_evidence_path: str = "",
    manual_review_packet_path: str = "",
    after_probe_artifact_root: str = "",
) -> dict[str, Any]:
    """Build a fail-closed source manifest for later manual probe materials."""

    readiness = dict(_mapping(internal_gate_limited_non_release_probe_followup_material_readiness))
    before_sources = _before_sources(
        baseline_before_manifest_path=baseline_before_manifest_path,
        batch1_parity_smoke_path=batch1_parity_smoke_path,
        real_gpu_batch1_golden_evidence_path=real_gpu_batch1_golden_evidence_path,
        manual_review_packet_path=manual_review_packet_path,
    )
    after_destination = _after_destination(after_probe_artifact_root)
    checks = _checks(readiness=readiness, before_sources=before_sources, after_destination=after_destination)
    blockers = _blockers(readiness=readiness, checks=checks)
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_FOLLOWUP_SOURCE_MANIFEST,
        "status": "ready_for_followup_manual_probe_source_manifest" if ready else "blocked",
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
        "followup_material_readiness_summary": _readiness_summary(readiness),
        "before_sources": before_sources,
        "after_destination": after_destination,
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _before_sources(
    *,
    baseline_before_manifest_path: str,
    batch1_parity_smoke_path: str,
    real_gpu_batch1_golden_evidence_path: str,
    manual_review_packet_path: str,
) -> list[dict[str, Any]]:
    rows = [
        ("baseline_before_manifest", baseline_before_manifest_path, "run_manifest_json"),
        ("batch1_parity_smoke", batch1_parity_smoke_path, "batch1_parity_smoke_json"),
        ("real_gpu_batch1_golden_evidence", real_gpu_batch1_golden_evidence_path, "golden_evidence_json"),
        ("manual_review_packet", manual_review_packet_path, "manual_review_packet_json"),
    ]
    result: list[dict[str, Any]] = []
    for source_id, raw_path, source_kind in rows:
        path = Path(str(raw_path or "").strip()) if str(raw_path or "").strip() else None
        result.append(
            {
                "id": source_id,
                "source_kind": source_kind,
                "path": str(path) if path is not None else "",
                "exists": bool(path and path.exists()),
                "is_file": bool(path and path.is_file()),
            }
        )
    return result


def _after_destination(root_text: str) -> dict[str, Any]:
    text = str(root_text or "").strip()
    root = Path(text) if text else None
    parent = root.parent if root is not None else None
    expected = []
    if root is not None:
        expected = [
            str(root / "before" / "baseline_manifest_pointer.json"),
            str(root / "after" / "probe_manifest_pointer.json"),
            str(root / "comparison" / "before_after_probe_evidence.json"),
            str(root / "review" / "manual_probe_review_notes.json"),
        ]
    return {
        "root": str(root) if root is not None else "",
        "root_parent_exists": bool(parent and parent.exists()),
        "root_is_file": bool(root and root.exists() and root.is_file()),
        "expected_artifacts": expected,
    }


def _checks(
    *,
    readiness: Mapping[str, Any],
    before_sources: Sequence[Mapping[str, Any]],
    after_destination: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "followup_material_readiness_present": bool(readiness),
        "followup_material_readiness_ready": bool(readiness.get("passed"))
        and str(readiness.get("status") or "") == READY_MATERIAL_READINESS_STATUS,
        "followup_material_readiness_default_off": (
            not bool(readiness.get("internal_gate_enablement_allowed"))
            and not bool(readiness.get("release_claim_allowed"))
        ),
        "before_sources_complete": all(
            bool(item.get("exists")) and bool(item.get("is_file")) for item in before_sources
        ),
        "after_destination_root_parent_exists": bool(after_destination.get("root_parent_exists")),
        "after_destination_root_not_file": not bool(after_destination.get("root_is_file")),
        "after_destination_expected_artifacts_listed": bool(
            _string_list(after_destination.get("expected_artifacts"))
        ),
    }


def _blockers(*, readiness: Mapping[str, Any], checks: Mapping[str, bool]) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not readiness:
        blockers.append("internal_gate_limited_non_release_probe_followup_material_readiness_missing")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_followup_material_readiness:{item}"
        for item in _string_list(readiness.get("blockers"))
    )
    return _dedupe(blockers)


def _readiness_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "release_claim_allowed": bool(report.get("release_claim_allowed")),
        "internal_gate_enablement_allowed": bool(report.get("internal_gate_enablement_allowed")),
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_report_only_pointer_files_for_before_and_after_evidence",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_followup_source_manifest_prerequisites"]
    if any("followup_material_readiness" in item for item in blockers):
        actions.append("refresh_followup_material_readiness")
    if any("before_sources_complete" in item for item in blockers):
        actions.append("refresh_before_source_paths")
    if any("after_destination" in item for item in blockers):
        actions.append("pick_valid_after_destination_root_under_temp")
    return _dedupe(actions)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_FOLLOWUP_SOURCE_MANIFEST",
    "build_lulynx_internal_gate_limited_non_release_probe_followup_source_manifest",
]
