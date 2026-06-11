"""Batch scorecard for selected plugin factored-memory family evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_plugin_factored_memory_state_layout_scorecard import (
    build_plugin_factored_memory_state_layout_scorecard,
)
from core.turbocore_plugin_factored_memory_execution_matrix import (
    build_factored_memory_execution_matrix,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_plugin_factored_memory_family_batch_scorecard(*, write_artifact: bool = False) -> dict[str, Any]:
    """Aggregate selected factored-memory plugin evidence without dispatch."""

    layout = build_plugin_factored_memory_state_layout_scorecard()
    layout_rows = [row for row in layout.get("rows", []) if isinstance(row, Mapping)]
    execution_matrix = build_factored_memory_execution_matrix(layout_rows)
    summary = _summary(layout)
    unsafe = _unsafe_claims(layout)
    ready = (
        layout.get("state_layout_reference_ready") is True
        and layout.get("selected_native_layout_abi_ready") is True
        and layout.get("layout_quality_matrix_ready") is True
        and execution_matrix.get("ok") is True
        and not unsafe
    )
    dispatch_review = _dispatch_integration_review(layout_rows, execution_matrix, ready)
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_factored_memory_family_batch_scorecard_v0",
        "gate": "plugin_factored_memory_selected_family_batch",
        "ok": ready,
        "promotion_ready": False,
        "selected_factored_memory_family_batch_ready": ready,
        "selected_native_layout_abi_ready": layout.get("selected_native_layout_abi_ready") is True,
        "layout_quality_matrix_ready": layout.get("layout_quality_matrix_ready") is True,
        "native_kernel_entry_conditions_ready": layout.get("native_kernel_entry_conditions_ready") is True,
        "selected_optimizer_abi_ready": False,
        "report_only": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "plugin_selected_native_ready_count": 0,
        "selected_optimizer_family": "factored_memory_layout",
        "default_off_contract": {
            "training_path_enabled": False,
            "default_behavior_changed": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "native_kernel_ready": False,
            "native_ready_count": 0,
            "plugin_selected_native_ready_count": 0,
        },
        "state_layout_scorecard": _compact_layout(layout),
        "execution_matrix": _compact_execution_matrix(execution_matrix),
        "dispatch_integration_review": dispatch_review,
        "rows": [_row(row, execution_matrix, dispatch_review) for row in layout_rows],
        "summary": {
            "selected_optimizer_count": int(summary.get("case_count", 0) or 0),
            "selector_factored_memory_count": int(summary.get("selector_factored_memory_count", 0) or 0),
            "observed_resume_layout_count": int(summary.get("observed_resume_layout_count", 0) or 0),
            "manual_contract_pending_count": int(summary.get("manual_contract_pending_count", 0) or 0),
            "native_layout_abi_ready_count": int(summary.get("native_layout_abi_ready_count", 0) or 0),
            "quality_matrix_ready_count": int(summary.get("layout_quality_matrix_ready_count", 0) or 0),
            "native_kernel_entry_condition_ready_count": int(
                summary.get("native_kernel_entry_condition_ready_count", 0) or 0
            ),
            "formula_tensor_binding_matrix_artifact_ready_count": sum(
                1 for row in layout.get("rows", []) if isinstance(row, Mapping)
            ),
            "formula_tensor_binding_matrix_implementation_ready_count": int(
                _summary(execution_matrix).get("matrix_implementation_ready_count", 0) or 0
            ),
            "formula_step_execution_ready_count": int(
                _summary(execution_matrix).get("formula_step_execution_ready_count", 0) or 0
            ),
            "resume_next_step_replay_ready_count": int(
                _summary(execution_matrix).get("resume_next_step_replay_ready_count", 0) or 0
            ),
            "tensor_binding_ready_count": int(_summary(execution_matrix).get("tensor_binding_ready_count", 0) or 0),
            "dispatch_review_gate_ready": dispatch_review["dispatch_review_gate_ready"],
            "dispatch_review_ready_count": len(layout_rows)
            if dispatch_review["dispatch_review_gate_ready"]
            else 0,
            "formula_parity_case_planned_count": sum(
                len(_formula_tensor_binding_matrix(str(row.get("optimizer_name", "")), {})["planned_cases"])
                for row in layout.get("rows", [])
                if isinstance(row, Mapping)
            ),
            "tensor_binding_case_planned_count": sum(
                1
                for row in layout.get("rows", [])
                if isinstance(row, Mapping)
                for case in _formula_tensor_binding_matrix(str(row.get("optimizer_name", "")), {})["planned_cases"]
                if case["case_group"] == "tensor_binding"
            ),
            "native_ready_count": 0,
            "product_native_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "plugin_selected_native_ready_count": 0,
            "unsafe_claim_count": len(unsafe),
        },
        "promotion_blockers": _promotion_blockers(
            unsafe,
            selected_native_layout_abi_ready=layout.get("selected_native_layout_abi_ready") is True,
            layout_quality_matrix_ready=layout.get("layout_quality_matrix_ready") is True,
            matrix_implementation_ready=execution_matrix.get("ok") is True,
            dispatch_review_ready=dispatch_review["dispatch_review_gate_ready"],
        ),
        "blocked_reasons": unsafe,
        "recommended_next_step": (
            "prepare selected factored-memory owner/release hold with product dispatch still default-off"
            if ready
            else "fix selected factored-memory state-layout blockers"
        ),
        "notes": [
            "This batch is report-only and never enables native dispatch.",
            "Factored-memory layout ABI and quality matrix are selected-family entry evidence, not native readiness.",
            "Formula parity and tensor binding matrices execute against the existing plugin reference only.",
            "Dispatch review readiness is not owner/release approval and does not enable native dispatch.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _row(
    row: Mapping[str, Any],
    execution_matrix: Mapping[str, Any],
    dispatch_review: Mapping[str, Any],
) -> dict[str, Any]:
    native_layout_abi = _as_dict(row.get("native_layout_abi"))
    layout_quality = _as_dict(row.get("layout_quality_matrix"))
    entry_ready = row.get("native_kernel_entry_condition_ready") is True
    name = str(row.get("optimizer_name", "")).strip().lower()
    execution = _execution_row(execution_matrix, name)
    return {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "native_route_family": "factored_memory_layout",
        "state_layout_status": str(row.get("state_layout_status", "")),
        "covers_trainer_plugin_request_path": row.get("covers_trainer_plugin_request_path") is True,
        "covers_small_tensor_step": row.get("covers_small_tensor_step") is True,
        "covers_resume": row.get("covers_resume") is True,
        "batch_status": (
            "dispatch_review_ready_report_only"
            if dispatch_review.get("dispatch_review_gate_ready") is True
            else "layout_abi_quality_ready_report_only"
            if entry_ready
            else "layout_entry_blocked"
        ),
        "native_layout_abi_ready": row.get("native_layout_abi_ready") is True,
        "quality_matrix_ready": row.get("layout_quality_matrix_ready") is True,
        "native_kernel_entry_condition_ready": entry_ready,
        "state_key_count": int(native_layout_abi.get("state_key_count", 0) or 0),
        "tensor_state_count": int(native_layout_abi.get("tensor_state_count", 0) or 0),
        "non_tensor_state_count": int(native_layout_abi.get("non_tensor_state_count", 0) or 0),
        "layout_quality_ready_criteria": dict(_as_dict(layout_quality.get("criteria"))),
        "formula_tensor_binding_matrix": _formula_tensor_binding_matrix(name, execution),
        "dispatch_integration_review_ready": dispatch_review.get("dispatch_review_gate_ready") is True,
        "pending_native_gates": list(row.get("pending_native_gates", []) or []),
        "plugin_selected_native_ready": False,
        "product_native_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "next_gate": (
            "selected_plugin_factored_memory_owner_release_hold"
            if dispatch_review.get("dispatch_review_gate_ready") is True
            else "selected_plugin_factored_memory_formula_parity_and_tensor_binding"
        ),
    }


def _dispatch_integration_review(
    layout_rows: list[Mapping[str, Any]],
    execution_matrix: Mapping[str, Any],
    family_ready: bool,
) -> dict[str, Any]:
    ready_names = [
        str(row.get("optimizer_name", "")).strip().lower()
        for row in layout_rows
        if isinstance(row, Mapping)
    ]
    ready = family_ready and execution_matrix.get("ok") is True and bool(ready_names)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_factored_memory_dispatch_integration_review_v0",
        "gate": "plugin_factored_memory_dispatch_integration_review",
        "ok": ready,
        "dispatch_review_gate_ready": ready,
        "manual_review_required": True,
        "owner_approval_recorded": False,
        "release_approval_recorded": False,
        "selected_optimizer_names": ready_names,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "backend_router_registered": False,
        "plugin_selected_native_ready_count": 0,
        "product_native_ready_count": 0,
        "blocked_reasons": [] if ready else ["plugin_factored_memory_dispatch_review_preconditions_missing"],
    }


def _compact_layout(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "ok": report.get("ok") is True,
        "state_layout_reference_ready": report.get("state_layout_reference_ready") is True,
        "selected_native_layout_abi_ready": report.get("selected_native_layout_abi_ready") is True,
        "layout_quality_matrix_ready": report.get("layout_quality_matrix_ready") is True,
        "native_kernel_entry_conditions_ready": report.get("native_kernel_entry_conditions_ready") is True,
        "selected_optimizer_abi_ready": report.get("selected_optimizer_abi_ready") is True,
        "observed_resume_layout_count": int(summary.get("observed_resume_layout_count", 0) or 0),
        "manual_contract_pending_count": int(summary.get("manual_contract_pending_count", 0) or 0),
        "native_layout_abi_ready_count": int(summary.get("native_layout_abi_ready_count", 0) or 0),
        "layout_quality_matrix_ready_count": int(summary.get("layout_quality_matrix_ready_count", 0) or 0),
        "native_kernel_entry_condition_ready_count": int(
            summary.get("native_kernel_entry_condition_ready_count", 0) or 0
        ),
        "native_ready_count": int(summary.get("native_ready_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
    }


def _formula_tensor_binding_matrix(name: str, execution: Mapping[str, Any]) -> dict[str, Any]:
    cases = [
        ("formula_parity", "single_tensor_reference_step"),
        ("formula_parity", "state_resume_next_step"),
        ("tensor_binding", "flat_param_buffer_binding"),
        ("tensor_binding", "factored_state_buffer_binding"),
        ("tensor_binding", "non_tensor_hparam_snapshot"),
    ]
    implementation_ready = execution.get("matrix_implementation_ready") is True
    return {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "matrix_artifact_ready": True,
        "matrix_implementation_ready": implementation_ready,
        "implementation_reference": str(execution.get("execution_reference", "")),
        "formula_step_execution_ready": execution.get("formula_step_execution_ready") is True,
        "resume_next_step_replay_ready": execution.get("resume_next_step_replay_ready") is True,
        "tensor_binding_ready": execution.get("tensor_binding_ready") is True,
        "planned_cases": [
            {
                "case_id": f"{name}:{case_id}",
                "case_group": group,
                "status": "implementation_ready" if implementation_ready else "planned",
                "requires_layout_abi": True,
                "requires_quality_guard": True,
                "native_dispatch_allowed": False,
            }
            for group, case_id in cases
        ],
    }


def _execution_row(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    rows = report.get("rows", []) if isinstance(report, Mapping) else []
    for row in rows:
        if isinstance(row, Mapping) and str(row.get("selected_optimizer_name", "")).strip().lower() == name:
            return dict(row)
    return {}


def _compact_execution_matrix(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "execution_matrix_ready": report.get("execution_matrix_ready") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "native_kernel_ready": report.get("native_kernel_ready") is True,
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "matrix_implementation_ready_count": int(summary.get("matrix_implementation_ready_count", 0) or 0),
        "formula_step_execution_ready_count": int(summary.get("formula_step_execution_ready_count", 0) or 0),
        "resume_next_step_replay_ready_count": int(summary.get("resume_next_step_replay_ready_count", 0) or 0),
        "tensor_binding_ready_count": int(summary.get("tensor_binding_ready_count", 0) or 0),
        "execution_failed_count": int(summary.get("execution_failed_count", 0) or 0),
        "blocked_reasons": [str(reason) for reason in report.get("blocked_reasons", []) or [] if str(reason)],
    }


def _unsafe_claims(report: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for field in ("training_path_enabled", "default_behavior_changed", "runtime_dispatch_ready", "native_dispatch_allowed"):
        if report.get(field) is True:
            out.append(f"plugin_factored_memory_state_layout:{field}")
    if report.get("native_kernel_ready") is True or report.get("selected_optimizer_abi_ready") is True:
        out.append("plugin_factored_memory_state_layout:native_ready_claim")
    return out


def _promotion_blockers(
    unsafe: list[str],
    *,
    selected_native_layout_abi_ready: bool,
    layout_quality_matrix_ready: bool,
    matrix_implementation_ready: bool,
    dispatch_review_ready: bool,
) -> list[str]:
    pending = list(unsafe)
    if not selected_native_layout_abi_ready:
        pending.append("selected_plugin_factored_memory_native_layout_abi_missing")
    if not layout_quality_matrix_ready:
        pending.append("selected_plugin_factored_memory_quality_matrix_missing")
    if not matrix_implementation_ready:
        pending.extend(
            [
                "selected_plugin_factored_memory_formula_parity_missing",
                "selected_plugin_factored_memory_training_tensor_binding_missing",
                "selected_plugin_factored_memory_formula_tensor_binding_matrix_implementation_missing",
            ]
        )
    if not dispatch_review_ready:
        pending.append("selected_plugin_factored_memory_dispatch_review_missing")
    pending.append("selected_plugin_factored_memory_owner_release_approval_missing")
    return _dedupe(pending)


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_factored_memory_family_batch_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report.get("summary")
    return dict(summary) if isinstance(summary, Mapping) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


__all__ = ["build_plugin_factored_memory_family_batch_scorecard"]
