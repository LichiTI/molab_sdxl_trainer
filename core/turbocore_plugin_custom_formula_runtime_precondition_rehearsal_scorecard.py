"""Batch runtime precondition rehearsal for selected plugin custom-formula optimizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_plugin_custom_formula_family_batch_scorecard import (
    build_plugin_custom_formula_family_batch_scorecard,
)
from core.turbocore_pnm_training_loop_canary_scorecard import build_pnm_training_loop_canary_scorecard
from core.turbocore_plugin_runtime_adapter_coverage import build_family_runtime_launch_adapter_coverage


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"
ARTIFACT_NAME = "turbocore_plugin_custom_formula_runtime_precondition_rehearsal_scorecard.json"


def build_plugin_custom_formula_runtime_precondition_rehearsal_scorecard(
    *,
    family_batch_report: Mapping[str, Any] | None = None,
    include_representative_runtime_canary: bool = False,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Normalize custom-formula formula/state/quality/replay readiness into one gate."""

    family = _as_dict(
        family_batch_report
        if family_batch_report is not None
        else build_plugin_custom_formula_family_batch_scorecard(write_artifact=True)
    )
    representative_runtime = (
        _compact_pnm_runtime_canary(build_pnm_training_loop_canary_scorecard())
        if include_representative_runtime_canary
        else {}
    )
    rows = [_case_row(row) for row in family.get("rows", []) if isinstance(row, Mapping)]
    summary = _as_dict(family.get("summary"))
    expected = int(summary.get("selected_optimizer_count", 0) or 0)
    blockers = _dedupe(reason for row in rows for reason in row.get("blocked_reasons", []) or [])
    adapter_coverage = build_family_runtime_launch_adapter_coverage(
        family="custom_formula",
        cases=rows,
        representative_runtime=representative_runtime,
        adapter_kind="custom_formula_runtime_adapter",
        representative_optimizer_name="pnm",
    )
    ready = (
        family.get("selected_custom_formula_family_batch_ready") is True
        and expected > 0
        and len(rows) == expected
        and all(row.get("runtime_precondition_rehearsal_ready") is True for row in rows)
        and not blockers
    )
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_custom_formula_runtime_precondition_rehearsal_scorecard_v0",
        "gate": "plugin_custom_formula_runtime_precondition_rehearsal",
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
        "selected_optimizer_family": "custom_formula",
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
            "formula_spec_ready_count": sum(1 for row in rows if row.get("formula_spec_ready") is True),
            "state_inventory_ready_count": sum(1 for row in rows if row.get("state_inventory_ready") is True),
            "quality_guard_ready_count": sum(1 for row in rows if row.get("quality_guard_ready") is True),
            "formula_parity_ready_count": sum(1 for row in rows if row.get("formula_parity_ready") is True),
            "resume_parity_ready_count": sum(1 for row in rows if row.get("resume_parity_ready") is True),
            "formula_step_execution_ready_count": sum(
                1 for row in rows if row.get("formula_step_execution_ready") is True
            ),
            "resume_next_step_replay_ready_count": sum(
                1 for row in rows if row.get("resume_next_step_replay_ready") is True
            ),
            "quality_guard_case_ready_count": sum(int(row.get("quality_guard_case_count", 0) or 0) for row in rows),
            "formula_parity_case_ready_count": sum(
                int(row.get("formula_parity_case_count", 0) or 0) for row in rows
            ),
            "resume_parity_case_ready_count": sum(
                int(row.get("resume_parity_case_count", 0) or 0) for row in rows
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
                "plugin_custom_formula_native_kernel_implementation_missing",
                "plugin_custom_formula_runtime_dispatch_rehearsal_missing",
                "plugin_custom_formula_owner_release_review_missing",
                "plugin_custom_formula_training_route_not_bound",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add selected plugin custom-formula native executor representatives"
            if ready
            else "fix selected plugin custom-formula runtime precondition blockers"
        ),
        "notes": [
            "This is a precondition rehearsal, not a CUDA runtime dispatch rehearsal.",
            "It batches formula spec, state inventory, quality guard, formula parity, and resume replay readiness.",
            "When requested, it records a PNM representative runtime canary without promoting the whole family.",
            "Native steps, kernel launches, request/schema/UI, and product training dispatch remain closed.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _case_row(row: Mapping[str, Any]) -> dict[str, Any]:
    evidence = _as_dict(row.get("evidence_status"))
    formula = _as_dict(row.get("formula_parity_matrix_artifact"))
    resume = _as_dict(row.get("resume_parity_matrix_artifact"))
    quality = _as_dict(row.get("quality_guard_artifact"))
    formula_evidence = _as_dict(formula.get("implementation_evidence"))
    resume_evidence = _as_dict(resume.get("implementation_evidence"))
    formula_spec_ready = evidence.get("formula_spec") == "ready"
    state_inventory_ready = evidence.get("state_inventory") == "ready"
    quality_guard_ready = evidence.get("quality_guard_matrix") == "ready"
    formula_parity_ready = formula.get("implementation_ready") is True
    resume_parity_ready = resume.get("implementation_ready") is True
    formula_step_ready = formula_evidence.get("formula_step_execution_ready") is True
    resume_replay_ready = (
        formula_evidence.get("resume_next_step_replay_ready") is True
        or resume_evidence.get("resume_next_step_replay_ready") is True
    )
    safe_closed = all(
        item.get("native_dispatch_allowed") is False
        and item.get("runtime_dispatch_ready") is False
        and item.get("product_native_ready") is False
        for item in (formula, resume)
    )
    ready = (
        formula_spec_ready
        and state_inventory_ready
        and quality_guard_ready
        and formula_parity_ready
        and resume_parity_ready
        and formula_step_ready
        and resume_replay_ready
        and safe_closed
    )
    name = str(row.get("selected_optimizer_name") or "")
    return {
        "schema_version": 1,
        "ok": ready,
        "selected_optimizer_name": name,
        "selected_optimizer_family": "custom_formula",
        "runtime_precondition_rehearsal_ready": ready,
        "runtime_dispatch_rehearsal_ready": False,
        "formula_spec_ready": formula_spec_ready,
        "state_inventory_ready": state_inventory_ready,
        "quality_guard_ready": quality_guard_ready,
        "formula_parity_ready": formula_parity_ready,
        "resume_parity_ready": resume_parity_ready,
        "formula_step_execution_ready": formula_step_ready,
        "resume_next_step_replay_ready": resume_replay_ready,
        "quality_guard_case_count": int(quality.get("guard_case_count", 0) or 0),
        "formula_parity_case_count": int(formula.get("case_count", 0) or 0),
        "resume_parity_case_count": int(resume.get("case_count", 0) or 0),
        "native_step_executed": False,
        "native_kernel_launched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [] if ready else [f"plugin_{name}_custom_formula_runtime_precondition_missing"],
    }


def _compact_family_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "selected_custom_formula_family_batch_ready": report.get("selected_custom_formula_family_batch_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "formula_spec_artifact_ready_count": int(summary.get("formula_spec_artifact_ready_count", 0) or 0),
        "state_inventory_artifact_ready_count": int(summary.get("state_inventory_artifact_ready_count", 0) or 0),
        "quality_guard_matrix_artifact_ready_count": int(
            summary.get("quality_guard_matrix_artifact_ready_count", 0) or 0
        ),
        "formula_parity_matrix_implementation_ready_count": int(
            summary.get("formula_parity_matrix_implementation_ready_count", 0) or 0
        ),
        "resume_parity_matrix_implementation_ready_count": int(
            summary.get("resume_parity_matrix_implementation_ready_count", 0) or 0
        ),
        "formula_step_execution_ready_count": int(summary.get("formula_step_execution_ready_count", 0) or 0),
        "resume_next_step_replay_ready_count": int(summary.get("resume_next_step_replay_ready_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
    }


def _compact_pnm_runtime_canary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    ready = (
        report.get("ok") is True
        and int(summary.get("native_step_count", 0) or 0) > 0
        and int(summary.get("native_kernel_launch_count", 0) or 0) > 0
    )
    return {
        "schema_version": 1,
        "selected_optimizer_name": "pnm",
        "source_scorecard": str(report.get("scorecard", "") or ""),
        "runtime_dispatch_rehearsal_ready": ready,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "training_executor_called_count": int(summary.get("training_executor_called_count", 0) or 0),
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


__all__ = ["ARTIFACT_NAME", "ROADMAP", "build_plugin_custom_formula_runtime_precondition_rehearsal_scorecard"]
