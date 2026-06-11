# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Contract-only boundary map for P7 lossless replacement phase guards.

The module is intentionally report-only. It translates the P7 guarded variant
mitigation blueprint into runtime/request boundary decisions without importing
trainer, opening feature gates, or mutating the lossless DataLoader path.
"""

from __future__ import annotations

from typing import Any, Mapping


REQUEST_CONFIG_BOUNDARY = "backend/core/configs.py"
TRAINING_REQUEST_BOUNDARY = "backend/core/contracts/training.py"
REQUEST_ADAPTER_BOUNDARY = "backend/core/services/training_request_adapter.py"
TRAINER_LOADER_BOUNDARY = "backend/core/lulynx_trainer/trainer.py"
LXFS_LOADER_BOUNDARY = "backend/core/lulynx_trainer/lossless_anima_cache_replacement_dataloader.py"
LYNX_MANIFEST_BOUNDARY = "backend/core/lulynx_trainer/lossless_anima_lynx_manifest_dataloader.py"
PHASE_PROFILE_BOUNDARY = "backend/core/lulynx_trainer/step_phase_profile.py"


ABSOLUTE_NO_TOUCH_PATTERNS = (
    "backend/lulynx_launcher_web_v2/**",
    "plugin/**",
    "backend/native/**",
    "backend/core/lulynx_trainer/bubble_*",
    "backend/core/**/turbocore*",
    "devtools/audit_lossless_tensor_transfer_roadmap.py",
    "devtools/refresh_lossless_tensor_transfer_evidence.py",
)


def _summary(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    value = payload.get("summary")
    return value if isinstance(value, Mapping) else {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _gate_closed(payload: Mapping[str, Any], key: str) -> bool:
    return not bool(payload.get(key) or _summary(payload).get(key))


def _unit_decision(unit: Mapping[str, Any]) -> dict[str, Any]:
    group_id = str(unit.get("group_id") or "")
    unit_kind = str(unit.get("unit_kind") or "")
    blockers = [str(item) for item in unit.get("blockers") or [] if str(item)]
    if unit_kind == "replacement_phase_guard_contract":
        decision = "contract_candidate_report_only"
        editable_runtime_boundary = True
        next_step = "add report-only phase guard metadata behind explicit validation"
    elif unit_kind == "mixed_regression_split_contract":
        decision = "split_required_no_runtime_change"
        editable_runtime_boundary = False
        next_step = "separate replacement regression from raw/control jitter first"
    else:
        decision = "raw_control_jitter_no_replacement_change"
        editable_runtime_boundary = False
        next_step = "keep as baseline jitter evidence and do not patch replacement runtime"
    return {
        "group_id": group_id,
        "unit_kind": unit_kind,
        "decision": decision,
        "blockers": blockers,
        "editable_runtime_boundary": editable_runtime_boundary,
        "opens_training_path": False,
        "opens_product_gate": False,
        "safe_to_auto_execute": False,
        "next_step": next_step,
    }


def build_lossless_replacement_phase_guard_contract(
    mitigation_blueprint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build a report-only contract from a P7 guarded mitigation blueprint."""

    blueprint = dict(mitigation_blueprint or {})
    summary = _summary(blueprint)
    units = _list_of_dicts(blueprint.get("implementation_units"))
    gates = _list_of_dicts(blueprint.get("validation_gates"))
    decisions = [_unit_decision(unit) for unit in units]
    replacement_candidates = [
        item for item in decisions if item["decision"] == "contract_candidate_report_only"
    ]
    validation_issues = [str(item) for item in blueprint.get("validation_issues") or []]
    gate_issues = [
        key
        for key in (
            "training_path_enabled",
            "resource_center_allowed",
            "resource_center_candidate",
            "default_enabled",
            "product_ready",
            "safe_to_auto_execute",
        )
        if not _gate_closed(blueprint, key)
    ]
    if gate_issues:
        validation_issues.append("p7_product_or_runtime_gate_open")
    ready = bool(
        blueprint.get("ok")
        and summary.get("guarded_variant_mitigation_blueprint_ready")
        and units
        and gates
        and not validation_issues
    )
    return {
        "contract": "lossless_replacement_phase_guard_contract_v1",
        "ok": ready,
        "report_only": True,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "does_not_run_nvcomp": True,
        "does_not_run_cache_scan": True,
        "does_not_mutate_runtime": True,
        "training_path_enabled": False,
        "resource_center_allowed": False,
        "resource_center_candidate": False,
        "default_enabled": False,
        "product_ready": False,
        "safe_to_auto_execute": False,
        "boundary_map": [
            {
                "boundary": "request_config",
                "files": [REQUEST_CONFIG_BOUNDARY],
                "role": "existing explicit dev-only lossless replacement fields",
                "edit_now": False,
            },
            {
                "boundary": "request_contract_adapter",
                "files": [TRAINING_REQUEST_BOUNDARY, REQUEST_ADAPTER_BOUNDARY],
                "role": "request-native normalization and config resolution surface",
                "edit_now": False,
            },
            {
                "boundary": "runtime_loader_selection",
                "files": [TRAINER_LOADER_BOUNDARY],
                "role": "existing Anima lossless replacement DataLoader switch",
                "edit_now": False,
            },
            {
                "boundary": "replacement_loader_facade",
                "files": [LXFS_LOADER_BOUNDARY, LYNX_MANIFEST_BOUNDARY],
                "role": "future report-only phase guard metadata, explicit validation only",
                "edit_now": bool(replacement_candidates),
            },
            {
                "boundary": "phase_profiler",
                "files": [PHASE_PROFILE_BOUNDARY],
                "role": "existing backward/optimizer/data-wait attribution evidence",
                "edit_now": False,
            },
        ],
        "unit_decisions": decisions,
        "minimal_write_set": {
            "safe_now": [
                {
                    "file": "backend/core/lulynx_trainer/lossless_replacement_phase_guard_contract.py",
                    "action": "new contract-only/report-only facade",
                },
                {
                    "file": "backend/core/tests/test_lossless_replacement_phase_guard_contract.py",
                    "action": "contract-only assertions; no trainer execution",
                },
            ],
            "future_after_design_review": [
                {
                    "file": LXFS_LOADER_BOUNDARY,
                    "action": "optional report-only guard metadata; no default behavior change",
                },
                {
                    "file": LYNX_MANIFEST_BOUNDARY,
                    "action": "optional report-only guard metadata; no default behavior change",
                },
            ],
        },
        "absolute_no_touch_patterns": list(ABSOLUTE_NO_TOUCH_PATTERNS),
        "validation_issues": validation_issues,
        "summary": {
            "replacement_phase_guard_contract_ready": ready,
            "source_blueprint_ready": bool(
                summary.get("guarded_variant_mitigation_blueprint_ready")
            ),
            "implementation_unit_count": len(units),
            "validation_gate_count": len(gates),
            "replacement_report_only_candidate_count": len(replacement_candidates),
            "runtime_default_change_allowed": False,
            "requires_manual_heavy_validation": True,
            "training_path_enabled": False,
            "resource_center_allowed": False,
            "product_ready": False,
            "safe_to_auto_execute": False,
        },
    }


__all__ = ["build_lossless_replacement_phase_guard_contract"]
