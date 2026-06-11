"""Batch runtime precondition rehearsal for selected plugin closure/second-order optimizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_plugin_closure_second_order_family_batch_scorecard import (
    TARGET_PLUGIN_OPTIMIZERS,
    build_plugin_closure_second_order_family_batch_scorecard,
)
from core.turbocore_alig_closure_canary_scorecard import build_alig_closure_canary_scorecard
from core.turbocore_plugin_runtime_adapter_coverage import build_family_runtime_launch_adapter_coverage


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"
ARTIFACT_NAME = "turbocore_plugin_closure_second_order_runtime_precondition_rehearsal_scorecard.json"


def build_plugin_closure_second_order_runtime_precondition_rehearsal_scorecard(
    *,
    family_batch_report: Mapping[str, Any] | None = None,
    include_representative_runtime_canary: bool = False,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Normalize closure/create-graph/replay ABI readiness into one precondition gate."""

    family = _as_dict(
        family_batch_report
        if family_batch_report is not None
        else build_plugin_closure_second_order_family_batch_scorecard(write_artifact=True)
    )
    representative_runtime = (
        _compact_alig_runtime_canary(build_alig_closure_canary_scorecard())
        if include_representative_runtime_canary
        else {}
    )
    rows = [_case_row(row) for row in family.get("rows", []) if isinstance(row, Mapping)]
    blockers = _dedupe(reason for row in rows for reason in row.get("blocked_reasons", []) or [])
    adapter_coverage = build_family_runtime_launch_adapter_coverage(
        family="closure_or_second_order",
        cases=rows,
        representative_runtime=representative_runtime,
        adapter_kind="closure_second_order_runtime_adapter",
        representative_optimizer_name="alig",
    )
    ready = (
        family.get("selected_closure_second_order_family_batch_ready") is True
        and len(rows) == len(TARGET_PLUGIN_OPTIMIZERS)
        and all(row.get("runtime_precondition_rehearsal_ready") is True for row in rows)
        and not blockers
    )
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_closure_second_order_runtime_precondition_rehearsal_scorecard_v0",
        "gate": "plugin_closure_second_order_runtime_precondition_rehearsal",
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
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "internal_rehearsal_executed": False,
        "selected_optimizer_family": "closure_or_second_order",
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
            "training_loop_abi_ready_count": sum(
                1 for row in rows if row.get("training_loop_abi_ready") is True
            ),
            "resume_parity_matrix_ready_count": sum(
                1 for row in rows if row.get("resume_parity_matrix_ready") is True
            ),
            "closure_resume_replay_artifact_ready_count": sum(
                1 for row in rows if row.get("closure_resume_replay_artifact_ready") is True
            ),
            "closure_resume_replay_row_ready_count": sum(
                int(row.get("closure_resume_replay_row_ready_count", 0) or 0) for row in rows
            ),
            "state_resume_adapter_ready_count": sum(
                1 for row in rows if row.get("state_resume_adapter_ready") is True
            ),
            "native_kernel_precondition_ready_count": sum(
                1 for row in rows if row.get("native_kernel_precondition_ready") is True
            ),
            "representative_runtime_dispatch_rehearsal_ready_count": 1
            if representative_runtime.get("runtime_dispatch_rehearsal_ready") is True
            else 0,
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
                "plugin_closure_second_order_native_kernel_implementation_missing",
                "plugin_closure_second_order_runtime_dispatch_rehearsal_missing",
                "plugin_closure_second_order_owner_release_review_missing",
                "plugin_closure_second_order_training_loop_route_not_bound",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add selected plugin closure/second-order native executor representatives"
            if ready
            else "fix selected plugin closure/second-order runtime precondition blockers"
        ),
        "notes": [
            "This is a precondition rehearsal, not a CUDA runtime dispatch rehearsal.",
            "It batches closure replay, create-graph/HVP lifetime, state resume, and native precondition readiness.",
            "When requested, it records an AliG representative closure canary without promoting the whole family.",
            "Native steps, kernel launches, request/schema/UI, and product training dispatch remain closed.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _case_row(row: Mapping[str, Any]) -> dict[str, Any]:
    matrix = _as_dict(row.get("resume_parity_matrix_plan"))
    artifact = _as_dict(matrix.get("closure_resume_replay_artifact"))
    replay_rows = [item for item in artifact.get("rows", []) if isinstance(item, Mapping)]
    execution = _as_dict(row.get("execution_matrix_row"))
    safe_closed = all(item.get("native_dispatch_allowed") is False for item in replay_rows) and all(
        execution.get(field) is False
        for field in (
            "training_path_enabled",
            "runtime_dispatch_ready",
            "native_dispatch_allowed",
            "native_kernel_ready",
            "product_native_ready",
        )
    )
    training_loop_ready = row.get("training_loop_abi_implementation_ready") is True
    resume_ready = matrix.get("implementation_ready") is True
    artifact_ready = artifact.get("implementation_ready") is True
    replay_rows_ready = bool(replay_rows) and all(item.get("implementation_ready") is True for item in replay_rows)
    state_resume_ready = execution.get("state_resume_adapter_implementation_ready") is True
    native_precondition_ready = execution.get("native_kernel_preconditions_implementation_ready") is True
    ready = (
        training_loop_ready
        and resume_ready
        and artifact_ready
        and replay_rows_ready
        and state_resume_ready
        and native_precondition_ready
        and safe_closed
    )
    name = str(row.get("selected_optimizer_name") or "")
    return {
        "schema_version": 1,
        "ok": ready,
        "selected_optimizer_name": name,
        "selected_optimizer_family": "closure_or_second_order",
        "runtime_precondition_rehearsal_ready": ready,
        "runtime_dispatch_rehearsal_ready": False,
        "training_loop_abi_ready": training_loop_ready,
        "resume_parity_matrix_ready": resume_ready,
        "closure_resume_replay_artifact_ready": artifact_ready,
        "closure_resume_replay_row_ready_count": sum(
            1 for item in replay_rows if item.get("implementation_ready") is True
        ),
        "state_resume_adapter_ready": state_resume_ready,
        "native_kernel_precondition_ready": native_precondition_ready,
        "native_step_executed": False,
        "native_kernel_launched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [] if ready else [f"plugin_{name}_closure_second_order_runtime_precondition_missing"],
    }


def _compact_family_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "selected_closure_second_order_family_batch_ready": report.get(
            "selected_closure_second_order_family_batch_ready"
        )
        is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "training_loop_abi_implementation_ready_count": int(
            summary.get("training_loop_abi_implementation_ready_count", 0) or 0
        ),
        "resume_parity_matrix_implementation_ready_count": int(
            summary.get("resume_parity_matrix_implementation_ready_count", 0) or 0
        ),
        "closure_resume_replay_artifact_implementation_ready_count": int(
            summary.get("closure_resume_replay_artifact_implementation_ready_count", 0) or 0
        ),
        "closure_resume_replay_row_implementation_ready_count": int(
            summary.get("closure_resume_replay_row_implementation_ready_count", 0) or 0
        ),
        "native_kernel_preconditions_implementation_ready_count": int(
            summary.get("native_kernel_preconditions_implementation_ready_count", 0) or 0
        ),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_alig_runtime_canary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    ready = (
        report.get("ok") is True
        and int(summary.get("native_step_count", 0) or 0) > 0
        and int(summary.get("native_kernel_launch_count", 0) or 0) > 0
        and int(summary.get("closure_call_count", 0) or 0) == 1
        and int(summary.get("optimizer_step_called_count", 0) or 0) == 0
    )
    return {
        "schema_version": 1,
        "selected_optimizer_name": "alig",
        "source_scorecard": str(report.get("scorecard", "") or ""),
        "runtime_dispatch_rehearsal_ready": ready,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "closure_replay": _as_dict(report.get("case")).get("closure_replay") is True,
        "closure_call_count": int(summary.get("closure_call_count", 0) or 0),
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "optimizer_step_called_count": int(summary.get("optimizer_step_called_count", 0) or 0),
        "blocked_reasons": _dedupe(report.get("blocked_reasons", []) or []),
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


__all__ = [
    "ARTIFACT_NAME",
    "ROADMAP",
    "build_plugin_closure_second_order_runtime_precondition_rehearsal_scorecard",
]
