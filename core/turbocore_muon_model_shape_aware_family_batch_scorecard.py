"""Report-only batch scorecard for built-in Muon model/shape-aware routing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.configs import OptimizerType
from core.lulynx_trainer.optimizer_capabilities import optimizer_capability_report


TARGET_OPTIMIZER = OptimizerType.MUON.value
UNSAFE_TRUE_FIELDS = (
    "training_path_enabled",
    "default_behavior_changed",
    "runtime_dispatch_ready",
    "native_dispatch_allowed",
    "native_kernel_ready",
    "product_native_ready",
    "product_native_dispatch_ready",
)


def build_muon_model_shape_aware_family_batch_scorecard(*, write_artifact: bool = False) -> dict[str, Any]:
    """Package Muon's param-group ABI and shape replay contract without enabling dispatch."""

    capability = _muon_capability()
    row = _row(capability)
    execution = _execution_matrix(row)
    row = _promote_row(row, execution)
    native_preconditions = _native_kernel_preconditions(row, execution)
    runtime_shadow = _runtime_dispatch_shadow(native_preconditions)
    dispatch_review = _dispatch_integration_review(runtime_shadow)
    row = _promote_row_to_dispatch_review(row, dispatch_review)
    unsafe = _unsafe_claims(
        {
            "row": row,
            "execution": execution,
            "native_preconditions": native_preconditions,
            "runtime_shadow": runtime_shadow,
            "dispatch_review": dispatch_review,
        }
    )
    ready = capability.get("status") == "available" and dispatch_review.get("ok") is True and not unsafe
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_muon_model_shape_aware_family_batch_scorecard_v0",
        "gate": "muon_model_shape_aware_family_batch",
        "ok": ready,
        "promotion_ready": False,
        "muon_model_shape_aware_family_batch_ready": ready,
        "report_only": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "request_contract": {
            "optimizer_type": TARGET_OPTIMIZER,
            "runtime_authority": "existing_python_muon_optimizer",
            "native_route_policy": "blocked_until_explicit_model_shape_param_group_abi_dispatch_review",
        },
        "native_compatibility": {
            "adamw_native_simple_kernel_reusable": False,
            "simple_formula_kernel_reusable": False,
            "requires_parameter_shapes": True,
            "requires_param_group_semantics_contract": True,
            "requires_adamw_fallback_group_contract": True,
            "param_group_abi_spec_ready": row["param_group_abi_spec_ready"],
            "param_group_abi_implementation_ready": row["param_group_abi_implementation_ready"],
            "native_kernel_precondition_ready": native_preconditions["native_kernel_precondition_ready"],
            "native_kernel_implementation_ready": False,
            "exact_adamw_product_native_route_count_delta": 0,
        },
        "capability": _compact_capability(capability),
        "execution_matrix": execution,
        "native_kernel_preconditions": native_preconditions,
        "runtime_dispatch_shadow": runtime_shadow,
        "dispatch_integration_review": dispatch_review,
        "rows": [row],
        "summary": {
            "optimizer_count": 1,
            "capability_available_count": 1 if capability.get("status") == "available" else 0,
            "param_group_abi_spec_ready_count": 1 if row["param_group_abi_spec_ready"] else 0,
            "param_group_abi_implementation_ready_count": int(
                _summary(execution).get("param_group_abi_implementation_ready_count", 0) or 0
            ),
            "param_group_resume_replay_matrix_artifact_ready_count": 1
            if row["param_group_resume_replay_matrix_artifact"]["artifact_ready"]
            else 0,
            "param_group_resume_replay_matrix_row_count": len(
                row["param_group_resume_replay_matrix_artifact"]["rows"]
            ),
            "param_group_resume_replay_matrix_implementation_ready_count": int(
                _summary(execution).get("param_group_resume_replay_matrix_implementation_ready_count", 0) or 0
            ),
            "param_group_resume_replay_row_implementation_ready_count": int(
                _summary(execution).get("param_group_resume_replay_row_implementation_ready_count", 0) or 0
            ),
            "native_kernel_precondition_ready_count": 1
            if native_preconditions["native_kernel_precondition_ready"]
            else 0,
            "runtime_dispatch_shadow_ready_count": 1 if runtime_shadow["runtime_dispatch_shadow_ready"] else 0,
            "dispatch_integration_review_ready_count": 1
            if dispatch_review["dispatch_integration_review_ready"]
            else 0,
            "native_ready_count": 0,
            "product_native_ready_count": 0,
            "product_native_dispatch_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
            "exact_adamw_product_native_route_count_delta": 0,
            "unsafe_claim_count": len(unsafe),
        },
        "promotion_blockers": _dedupe(
            unsafe
            + [
                "native_kernel_implementation_missing",
                "owner_release_approval_missing",
            ]
        ),
        "blocked_reasons": unsafe,
        "recommended_next_step": (
            "keep built-in Muon native dispatch unwired until explicit owner/release approval is recorded"
            if ready
            else "fix built-in Muon capability or unsafe native-readiness claims"
        ),
        "notes": [
            "This batch is report-only and never enables native dispatch.",
            "Muon uses 2D shape-aware orthogonalized momentum plus AdamW fallback groups.",
            "It cannot reuse exact AdamW or simple-formula native kernel contracts.",
            "Dispatch review readiness is not owner/release approval and not a native kernel implementation claim.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _muon_capability() -> dict[str, Any]:
    report = optimizer_capability_report([OptimizerType.MUON])
    for item in report.get("optimizers", []):
        if isinstance(item, Mapping) and item.get("optimizer_type") == TARGET_OPTIMIZER:
            return dict(item)
    return {"optimizer_type": TARGET_OPTIMIZER, "status": "missing"}


def _row(capability: Mapping[str, Any]) -> dict[str, Any]:
    available = capability.get("status") == "available"
    artifact = _param_group_resume_replay_matrix_artifact(available)
    return {
        "schema_version": 1,
        "optimizer_type": TARGET_OPTIMIZER,
        "native_route_family": "model_or_shape_aware",
        "model_shape_aware_family": "muon_shape_grouping",
        "batch_status": "report_only_contract_ready" if available else "capability_missing",
        "capability_available": available,
        "state_resume": str(capability.get("state_resume") or ""),
        "dependency_available": capability.get("dependency_available") is True,
        "param_group_abi_spec_ready": available,
        "param_group_abi_implementation_ready": False,
        "param_group_abi_contract": {
            "schema_version": 1,
            "model_structure_binding": "none",
            "shape_partition_policy": "matrix_shape_muon_grouping",
            "param_group_semantics_policy": "muon_2d_momentum_with_adamw_fallback_groups",
            "state_resume_scope": "muon_shape_group_state_and_adamw_fallback_state",
            "native_kernel_precondition": "param_group_abi_and_shape_replay_ready",
        },
        "param_group_resume_replay_matrix_artifact": artifact,
        "adamw_native_simple_kernel_compatible": False,
        "native_simple_kernel_reusable": False,
        "runtime_authority": "existing_builtin_muon_optimizer",
        "native_route": "none_report_only",
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "native_kernel_precondition_ready": False,
        "runtime_dispatch_shadow_ready": False,
        "dispatch_integration_review_ready": False,
        "next_gate": "built_in_muon_shape_grouping_native_kernel_preconditions",
        "blocked_reasons": [
            "built_in_muon_native_kernel_implementation_missing",
            "runtime_dispatch_shadow_missing",
            "owner_release_hold_missing",
        ],
    }


def _param_group_resume_replay_matrix_artifact(spec_ready: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "builtin_muon_param_group_resume_replay_matrix_v0",
        "artifact_ready": spec_ready,
        "artifact_status": "planned_report_only" if spec_ready else "capability_missing",
        "implementation_ready": False,
        "rows": [
            _matrix_row(
                "muon_2d_group_inventory_resume",
                ["group_index", "shape_tuple", "muon_momentum", "muon_ns_steps"],
                ["2d_group_identity_stable_after_resume", "native_flattening_still_blocked"],
            ),
            _matrix_row(
                "adamw_fallback_group_resume",
                ["fallback_group_index", "shape_tuple", "muon_lr_ratio", "adamw_state_dict"],
                ["fallback_state_roundtrips", "fallback_lr_ratio_replayed"],
            ),
            _matrix_row(
                "blocked_native_reuse_review",
                ["adamw_kernel_incompatibility_reason", "simple_kernel_incompatibility_reason"],
                ["dispatch_remains_default_off", "native_ready_not_claimed"],
            ),
        ],
    }


def _matrix_row(case_id: str, required_payload: list[str], replay_assertions: list[str]) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "artifact_status": "planned_report_only",
        "replay_boundary": "built_in_muon_param_group_shape_abi",
        "required_payload": required_payload,
        "replay_assertions": replay_assertions,
        "implementation_ready": False,
        "native_dispatch_allowed": False,
    }


def _execution_matrix(row: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _as_dict(row.get("param_group_resume_replay_matrix_artifact"))
    rows = [item for item in artifact.get("rows", []) if isinstance(item, Mapping)]
    ready_rows = [_matrix_row_ready(item) for item in rows]
    ready = (
        row.get("capability_available") is True
        and row.get("param_group_abi_spec_ready") is True
        and artifact.get("artifact_ready") is True
        and bool(rows)
        and all(ready_rows)
    )
    return {
        "schema_version": 1,
        "scorecard": "turbocore_muon_model_shape_aware_execution_matrix_v0",
        "gate": "muon_model_shape_aware_param_group_execution_matrix",
        "ok": ready,
        "execution_matrix_ready": ready,
        "report_only": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "summary": {
            "optimizer_count": 1,
            "param_group_abi_implementation_ready_count": 1 if ready else 0,
            "param_group_resume_replay_matrix_implementation_ready_count": 1 if ready else 0,
            "param_group_resume_replay_row_implementation_ready_count": sum(1 for value in ready_rows if value),
            "native_ready_count": 0,
            "product_native_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
        },
        "blocked_reasons": [] if ready else ["built_in_muon_param_group_matrix_incomplete"],
    }


def _native_kernel_preconditions(row: Mapping[str, Any], execution: Mapping[str, Any]) -> dict[str, Any]:
    ready = (
        row.get("param_group_abi_spec_ready") is True
        and execution.get("execution_matrix_ready") is True
        and row.get("adamw_native_simple_kernel_compatible") is False
        and row.get("native_simple_kernel_reusable") is False
    )
    return {
        "schema_version": 1,
        "scorecard": "turbocore_muon_native_kernel_preconditions_v0",
        "gate": "muon_model_shape_aware_native_kernel_preconditions",
        "ok": ready,
        "native_kernel_precondition_ready": ready,
        "native_kernel_implementation_ready": False,
        "native_kernel_ready": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "required_native_kernel_contract": {
            "route": "builtin_muon_shape_grouped_newton_schulz_v0",
            "requires_2d_matrix_groups": True,
            "requires_momentum_buffer": True,
            "requires_newton_schulz_steps": True,
            "requires_adamw_fallback_group": True,
            "blocks_exact_adamw_reuse": True,
            "blocks_simple_formula_kernel_reuse": True,
        },
        "blocked_reasons": [] if ready else ["built_in_muon_native_kernel_preconditions_incomplete"],
    }


def _runtime_dispatch_shadow(native_preconditions: Mapping[str, Any]) -> dict[str, Any]:
    ready = native_preconditions.get("native_kernel_precondition_ready") is True
    return {
        "schema_version": 1,
        "scorecard": "turbocore_muon_runtime_dispatch_shadow_v0",
        "gate": "muon_model_shape_aware_runtime_dispatch_shadow",
        "ok": ready,
        "runtime_dispatch_shadow_ready": ready,
        "runtime_dispatch_ready": False,
        "native_shadow_call_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "fallback_backend": "existing_builtin_muon_optimizer",
        "fallback_backend_authoritative": True,
        "dispatch_decision": "blocked_report_only_until_native_kernel_and_owner_review",
        "blocked_reasons": [] if ready else ["built_in_muon_runtime_dispatch_shadow_preconditions_missing"],
    }


def _dispatch_integration_review(runtime_shadow: Mapping[str, Any]) -> dict[str, Any]:
    ready = runtime_shadow.get("runtime_dispatch_shadow_ready") is True
    return {
        "schema_version": 1,
        "scorecard": "turbocore_muon_dispatch_integration_review_v0",
        "gate": "muon_model_shape_aware_dispatch_integration_review",
        "ok": ready,
        "dispatch_integration_review_ready": ready,
        "manual_review_required": True,
        "owner_approval_recorded": False,
        "release_approval_recorded": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "backend_router_registered": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "blocked_reasons": [] if ready else ["built_in_muon_runtime_dispatch_shadow_missing"],
    }


def _promote_row(row: Mapping[str, Any], execution: Mapping[str, Any]) -> dict[str, Any]:
    implementation_ready = execution.get("ok") is True
    artifact = _as_dict(row.get("param_group_resume_replay_matrix_artifact"))
    artifact_rows = [
        {
            **dict(item),
            "artifact_status": "implementation_ready" if implementation_ready else item.get("artifact_status"),
            "implementation_ready": implementation_ready,
        }
        for item in artifact.get("rows", [])
        if isinstance(item, Mapping)
    ]
    return {
        **dict(row),
        "batch_status": "param_group_abi_replay_ready_report_only"
        if implementation_ready
        else row.get("batch_status"),
        "param_group_abi_implementation_ready": implementation_ready,
        "param_group_resume_replay_matrix_artifact": {
            **artifact,
            "artifact_status": "implementation_ready" if implementation_ready else artifact.get("artifact_status"),
            "implementation_ready": implementation_ready,
            "rows": artifact_rows,
        },
        "execution_matrix_row": {
            "ok": implementation_ready,
            "param_group_abi_implementation_ready": implementation_ready,
            "native_dispatch_allowed": False,
        },
    }


def _promote_row_to_dispatch_review(row: Mapping[str, Any], dispatch_review: Mapping[str, Any]) -> dict[str, Any]:
    ready = dispatch_review.get("dispatch_integration_review_ready") is True
    return {
        **dict(row),
        "batch_status": "dispatch_review_ready_report_only" if ready else row.get("batch_status"),
        "native_kernel_precondition_ready": ready,
        "runtime_dispatch_shadow_ready": ready,
        "dispatch_integration_review_ready": ready,
        "next_gate": "explicit_owner_release_approval_for_builtin_muon_native_dispatch"
        if ready
        else row.get("next_gate"),
        "blocked_reasons": ["built_in_muon_native_kernel_implementation_missing", "owner_release_approval_missing"]
        if ready
        else list(row.get("blocked_reasons", [])),
    }


def _matrix_row_ready(row: Mapping[str, Any]) -> bool:
    return bool(row.get("required_payload")) and bool(row.get("replay_assertions")) and row.get("native_dispatch_allowed") is False


def _unsafe_claims(reports: Mapping[str, Mapping[str, Any]]) -> list[str]:
    out: list[str] = []
    for name, report in reports.items():
        for field in UNSAFE_TRUE_FIELDS:
            if report.get(field) is True:
                out.append(f"unsafe_muon_model_shape_source:{name}:{field}")
    return _dedupe(out)


def _compact_capability(capability: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "optimizer_type": str(capability.get("optimizer_type") or ""),
        "status": str(capability.get("status") or ""),
        "implementation": str(capability.get("implementation") or ""),
        "dependency_available": capability.get("dependency_available") is True,
        "state_resume": str(capability.get("state_resume") or ""),
    }


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = Path(__file__).resolve().parents[2] / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_muon_model_shape_aware_family_batch_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(report.get("summary"))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["TARGET_OPTIMIZER", "build_muon_model_shape_aware_family_batch_scorecard"]
