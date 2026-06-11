"""Readiness packet for manual fill of pointer-only batch1 probe materials.

This packet remains report-only. It confirms that manual-fill previews and
preparation checklists are aligned and ready for human preparation without
writing files, enabling the internal gate, or starting probe execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_READINESS_PACKET = (
    "lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_readiness_packet_v0"
)
READY_POINTER_MANUAL_FILL_TEMPLATE_BUNDLE_STATUS = (
    "ready_for_followup_pointer_manual_fill_template_bundle"
)
READY_POINTER_PREPARATION_CHECKLIST_PACKAGE_STATUS = (
    "ready_for_followup_pointer_preparation_checklist_package"
)


def build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_readiness_packet(
    *,
    internal_gate_limited_non_release_probe_pointer_manual_fill_template_bundle: Mapping[str, Any]
    | None = None,
    internal_gate_limited_non_release_probe_pointer_preparation_checklist_package: Mapping[str, Any]
    | None = None,
) -> dict[str, Any]:
    """Build a fail-closed readiness packet for manual pointer fill work."""

    template_bundle = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_manual_fill_template_bundle)
    )
    checklist_package = dict(
        _mapping(internal_gate_limited_non_release_probe_pointer_preparation_checklist_package)
    )
    bundle_payload = _mapping(template_bundle.get("pointer_manual_fill_template_bundle"))
    checklist_payload = _mapping(checklist_package.get("pointer_preparation_checklist_package"))
    artifact_templates = _sequence_of_mappings(bundle_payload.get("artifact_templates"))
    artifact_checklists = _sequence_of_mappings(checklist_payload.get("artifact_checklists"))
    checks = _checks(
        template_bundle=template_bundle,
        checklist_package=checklist_package,
        artifact_templates=artifact_templates,
        artifact_checklists=artifact_checklists,
        checklist_payload=checklist_payload,
    )
    blockers = _blockers(
        template_bundle=template_bundle,
        checklist_package=checklist_package,
        checks=checks,
    )
    ready = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_READINESS_PACKET,
        "status": "ready_for_followup_pointer_manual_fill_readiness_packet" if ready else "blocked",
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
        "pointer_manual_fill_template_bundle_summary": _bundle_summary(template_bundle),
        "pointer_preparation_checklist_package_summary": _checklist_summary(checklist_package),
        "pointer_manual_fill_readiness_packet": _readiness_packet(
            artifact_templates=artifact_templates,
            artifact_checklists=artifact_checklists,
            checklist_payload=checklist_payload,
        ),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(
    *,
    template_bundle: Mapping[str, Any],
    checklist_package: Mapping[str, Any],
    artifact_templates: Sequence[Mapping[str, Any]],
    artifact_checklists: Sequence[Mapping[str, Any]],
    checklist_payload: Mapping[str, Any],
) -> dict[str, bool]:
    template_ids = {str(item.get("id") or "") for item in artifact_templates if item.get("id")}
    checklist_ids = {str(item.get("id") or "") for item in artifact_checklists if item.get("id")}
    forbidden_actions = set(_string_list(checklist_payload.get("forbidden_actions")))
    required_completion_evidence = _string_list(checklist_payload.get("required_completion_evidence"))
    return {
        "pointer_manual_fill_template_bundle_present": bool(template_bundle),
        "pointer_manual_fill_template_bundle_ready": bool(template_bundle.get("passed"))
        and str(template_bundle.get("status") or "") == READY_POINTER_MANUAL_FILL_TEMPLATE_BUNDLE_STATUS,
        "pointer_manual_fill_template_bundle_default_off": (
            not bool(template_bundle.get("internal_gate_enablement_allowed"))
            and not bool(template_bundle.get("release_claim_allowed"))
        ),
        "pointer_preparation_checklist_package_present": bool(checklist_package),
        "pointer_preparation_checklist_package_ready": bool(checklist_package.get("passed"))
        and str(checklist_package.get("status") or "")
        == READY_POINTER_PREPARATION_CHECKLIST_PACKAGE_STATUS,
        "pointer_preparation_checklist_package_default_off": (
            not bool(checklist_package.get("internal_gate_enablement_allowed"))
            and not bool(checklist_package.get("release_claim_allowed"))
        ),
        "artifact_template_count_is_four": len(artifact_templates) == 4,
        "artifact_checklist_count_is_four": len(artifact_checklists) == 4,
        "artifact_roles_aligned": template_ids == checklist_ids == {
            "baseline_manifest_pointer",
            "probe_manifest_pointer",
            "before_after_probe_evidence",
            "manual_probe_review_notes",
        },
        "template_previews_present": all(bool(_mapping(item.get("template_preview"))) for item in artifact_templates),
        "checklist_steps_present": all(bool(_string_list(item.get("checklist_steps") or item.get("steps"))) for item in artifact_templates + artifact_checklists),  # type: ignore[operator]
        "required_completion_evidence_visible": len(required_completion_evidence) >= 4,
        "forbidden_actions_keep_execution_closed": {
            "write_pointer_files_automatically",
            "enable_internal_gate_now",
            "start_training_now",
            "approve_batch2_4_8_release_probe",
            "approve_release_claim",
        }.issubset(forbidden_actions),
    }


def _blockers(
    *,
    template_bundle: Mapping[str, Any],
    checklist_package: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not template_bundle:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_manual_fill_template_bundle_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_manual_fill_template_bundle:{item}"
        for item in _string_list(template_bundle.get("blockers"))
    )
    if not checklist_package:
        blockers.append(
            "internal_gate_limited_non_release_probe_pointer_preparation_checklist_package_missing"
        )
    blockers.extend(
        f"internal_gate_limited_non_release_probe_pointer_preparation_checklist_package:{item}"
        for item in _string_list(checklist_package.get("blockers"))
    )
    return _dedupe(blockers)


def _bundle_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(report.get("pointer_manual_fill_template_bundle"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "artifact_template_count": len(_sequence_of_mappings(payload.get("artifact_templates"))),
        "bundle_mode": str(payload.get("bundle_mode") or ""),
    }


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


def _readiness_packet(
    *,
    artifact_templates: Sequence[Mapping[str, Any]],
    artifact_checklists: Sequence[Mapping[str, Any]],
    checklist_payload: Mapping[str, Any],
) -> dict[str, Any]:
    checklist_map = {str(item.get("id") or ""): item for item in artifact_checklists}
    return {
        "packet_kind": "manual_pointer_fill_readiness_packet_v0",
        "readiness_scope": "batch1_non_release_pointer_manual_fill",
        "forbidden_actions": _string_list(checklist_payload.get("forbidden_actions")),
        "required_completion_evidence": _string_list(
            checklist_payload.get("required_completion_evidence")
        ),
        "artifact_readiness": [
            {
                "id": _string(item.get("id")),
                "target_path": _string(item.get("target_path")),
                "template_preview_keys": sorted(_mapping(item.get("template_preview")).keys()),
                "checklist_steps": _string_list(
                    _mapping(checklist_map.get(_string(item.get("id")))).get("steps")
                    or item.get("checklist_steps")
                ),
                "ready_for_manual_fill_preview": bool(_mapping(item.get("template_preview"))),
            }
            for item in artifact_templates
        ],
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "prepare_manual_pointer_fill_examples_without_writing_runtime_state",
            "keep_internal_gate_disabled_until_any_later_explicit_probe_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_pointer_manual_fill_readiness_packet_prerequisites"]
    if any("pointer_manual_fill_template_bundle" in item for item in blockers):
        actions.append("refresh_pointer_manual_fill_template_bundle")
    if any("pointer_preparation_checklist_package" in item for item in blockers):
        actions.append("refresh_pointer_preparation_checklist_package")
    if any("artifact_" in item or "checklist_steps" in item for item in blockers):
        actions.append("repair_pointer_manual_fill_readiness_inputs")
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
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_POINTER_MANUAL_FILL_READINESS_PACKET",
    "build_lulynx_internal_gate_limited_non_release_probe_pointer_manual_fill_readiness_packet",
]
