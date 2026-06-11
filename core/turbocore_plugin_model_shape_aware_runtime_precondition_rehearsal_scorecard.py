"""Batch runtime precondition rehearsal for selected plugin model/shape-aware optimizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_plugin_model_shape_aware_family_batch_scorecard import (
    TARGET_PLUGIN_OPTIMIZERS,
    build_plugin_model_shape_aware_family_batch_scorecard,
)
from core.turbocore_muon_training_loop_canary_scorecard import build_muon_training_loop_canary_scorecard
from core.turbocore_plugin_runtime_adapter_coverage import build_family_runtime_launch_adapter_coverage


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"
ARTIFACT_NAME = "turbocore_plugin_model_shape_aware_runtime_precondition_rehearsal_scorecard.json"


def build_plugin_model_shape_aware_runtime_precondition_rehearsal_scorecard(
    *,
    family_batch_report: Mapping[str, Any] | None = None,
    include_representative_runtime_canary: bool = False,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Normalize model/shape-aware ABI and replay matrix readiness as one precondition gate."""

    family = _as_dict(
        family_batch_report
        if family_batch_report is not None
        else build_plugin_model_shape_aware_family_batch_scorecard(write_artifact=True)
    )
    representative_runtime = (
        _compact_muon_runtime_canary(build_muon_training_loop_canary_scorecard(write_artifact=True))
        if include_representative_runtime_canary
        else {}
    )
    rows = [_case_row(row) for row in family.get("rows", []) if isinstance(row, Mapping)]
    blockers = _dedupe(reason for row in rows for reason in row.get("blocked_reasons", []) or [])
    runtime_ready = representative_runtime.get("runtime_dispatch_rehearsal_ready") is True
    adapter_coverage = build_family_runtime_launch_adapter_coverage(
        family="model_or_shape_aware",
        cases=rows,
        representative_runtime=representative_runtime,
        adapter_kind="model_shape_aware_runtime_adapter",
        representative_optimizer_name="muon",
    )
    ready = (
        family.get("selected_model_shape_aware_family_batch_ready") is True
        and len(rows) == len(TARGET_PLUGIN_OPTIMIZERS)
        and all(row.get("runtime_precondition_rehearsal_ready") is True for row in rows)
        and not blockers
    )
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_model_shape_aware_runtime_precondition_rehearsal_scorecard_v0",
        "gate": "plugin_model_shape_aware_runtime_precondition_rehearsal",
        "roadmap": ROADMAP,
        "ok": ready,
        "promotion_ready": False,
        "runtime_precondition_rehearsal_ready": ready,
        "runtime_dispatch_rehearsal_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_dispatch_ready": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "internal_rehearsal_executed": False,
        "selected_optimizer_family": "model_or_shape_aware",
        "family_specific_runtime_launch_adapter_ready": adapter_coverage[
            "family_specific_runtime_launch_adapter_ready"
        ],
        "family_batch_scorecard": _compact_family_batch(family),
        "representative_runtime_canary": representative_runtime,
        "family_specific_runtime_launch_adapter_coverage": adapter_coverage,
        "cases": rows,
        "summary": {
            "selected_optimizer_count": len(rows),
            "case_count": len(rows),
            "runtime_precondition_rehearsal_ready_count": sum(
                1 for row in rows if row.get("runtime_precondition_rehearsal_ready") is True
            ),
            "param_group_abi_ready_count": sum(1 for row in rows if row.get("param_group_abi_ready") is True),
            "resume_replay_matrix_ready_count": sum(
                1 for row in rows if row.get("resume_replay_matrix_ready") is True
            ),
            "resume_replay_row_ready_count": sum(
                int(row.get("resume_replay_row_ready_count", 0) or 0) for row in rows
            ),
            "model_structure_contract_count": sum(
                1 for row in rows if row.get("requires_model_structure") is True
            ),
            "shape_partition_contract_count": sum(
                1 for row in rows if row.get("requires_parameter_shapes") is True
            ),
            "distributed_collective_contract_count": sum(
                1 for row in rows if row.get("requires_distributed_collective_semantics") is True
            ),
            "representative_runtime_dispatch_rehearsal_ready_count": 1 if runtime_ready else 0,
            "representative_runtime_dispatch_case_count": 1 if representative_runtime else 0,
            "family_specific_runtime_launch_adapter_ready_count": int(
                adapter_coverage["summary"]["family_specific_runtime_launch_adapter_ready_count"]
            ),
            "family_specific_runtime_launch_adapter_case_count": int(adapter_coverage["summary"]["case_count"]),
            "native_step_count": int(representative_runtime.get("native_step_count", 0) or 0),
            "native_kernel_launch_count": int(representative_runtime.get("native_kernel_launch_count", 0) or 0),
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "plugin_model_shape_aware_native_kernel_implementation_missing",
                "plugin_model_shape_aware_runtime_dispatch_rehearsal_missing",
                "plugin_model_shape_aware_owner_release_review_missing",
                "plugin_model_shape_aware_product_training_route_not_bound",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add selected plugin model/shape-aware native executor representatives"
            if ready
            else "fix selected plugin model/shape-aware runtime precondition blockers"
        ),
        "notes": [
            "This is a precondition rehearsal, not a CUDA runtime dispatch rehearsal.",
            "It batches param-group ABI, model/shape dependency, and resume replay matrix readiness.",
            "When requested, it records a Muon representative runtime canary without promoting the whole family.",
            "Native steps, kernel launches, request/schema/UI, and product training dispatch remain closed.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _case_row(row: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _as_dict(row.get("param_group_resume_replay_matrix_artifact"))
    matrix_rows = [item for item in artifact.get("rows", []) if isinstance(item, Mapping)]
    dependency = _as_dict(row.get("dependency_contract"))
    ready = (
        row.get("param_group_abi_spec_ready") is True
        and row.get("param_group_abi_implementation_ready") is True
        and artifact.get("artifact_ready") is True
        and artifact.get("implementation_ready") is True
        and bool(matrix_rows)
        and all(item.get("implementation_ready") is True for item in matrix_rows)
        and all(item.get("native_dispatch_allowed") is False for item in matrix_rows)
    )
    name = str(row.get("selected_optimizer_name") or "")
    return {
        "schema_version": 1,
        "ok": ready,
        "selected_optimizer_name": name,
        "selected_optimizer_family": "model_or_shape_aware",
        "model_shape_aware_family": str(row.get("model_shape_aware_family") or ""),
        "runtime_precondition_rehearsal_ready": ready,
        "runtime_dispatch_rehearsal_ready": False,
        "param_group_abi_ready": row.get("param_group_abi_implementation_ready") is True,
        "resume_replay_matrix_ready": artifact.get("implementation_ready") is True,
        "resume_replay_row_ready_count": sum(1 for item in matrix_rows if item.get("implementation_ready") is True),
        "requires_model_structure": dependency.get("requires_model_structure") is True,
        "requires_parameter_shapes": dependency.get("requires_parameter_shapes") is True,
        "requires_param_group_semantics": dependency.get("requires_param_group_semantics") is True,
        "requires_distributed_collective_semantics": (
            dependency.get("requires_distributed_collective_semantics") is True
        ),
        "native_step_executed": False,
        "native_kernel_launched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [] if ready else [f"plugin_{name}_model_shape_runtime_precondition_missing"],
    }


def _compact_family_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "selected_model_shape_aware_family_batch_ready": report.get(
            "selected_model_shape_aware_family_batch_ready"
        )
        is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "param_group_abi_implementation_ready_count": int(
            summary.get("param_group_abi_implementation_ready_count", 0) or 0
        ),
        "param_group_resume_replay_matrix_implementation_ready_count": int(
            summary.get("param_group_resume_replay_matrix_implementation_ready_count", 0) or 0
        ),
        "param_group_resume_replay_row_implementation_ready_count": int(
            summary.get("param_group_resume_replay_row_implementation_ready_count", 0) or 0
        ),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_muon_runtime_canary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    ready = (
        report.get("ok") is True
        and int(summary.get("native_step_count", 0) or 0) > 0
        and int(summary.get("native_kernel_launch_count", 0) or 0) > 0
    )
    return {
        "schema_version": 1,
        "selected_optimizer_name": "muon",
        "source_scorecard": str(report.get("scorecard", "") or ""),
        "runtime_dispatch_rehearsal_ready": ready,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "blocked_reasons": [] if ready else [str(reason) for reason in report.get("blocked_reasons", []) or []],
    }


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / ARTIFACT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["ARTIFACT_NAME", "ROADMAP", "build_plugin_model_shape_aware_runtime_precondition_rehearsal_scorecard"]
