"""Report-only batch scorecard for selected plugin custom-formula routes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from core.configs import OptimizerType
from core.turbocore_plugin_custom_formula_artifacts import (
    BACKLOG_TIERS,
    EVIDENCE_STAGES,
    build_custom_formula_backlog,
    build_custom_formula_evidence_artifacts,
    build_custom_formula_evidence_plan,
    build_custom_formula_evidence_status,
    build_custom_formula_parallel_next_actions,
    build_custom_formula_spec_artifact,
    custom_formula_backlog_tier,
    custom_formula_blocked_reasons,
    custom_formula_hint,
)
from core.turbocore_plugin_custom_formula_source_inventory import build_custom_formula_state_inventory_artifact
from core.turbocore_plugin_custom_formula_quality_guard import build_custom_formula_quality_guard_artifact
from core.turbocore_plugin_custom_formula_parity_plan import build_custom_formula_parity_plan_artifacts
from core.turbocore_plugin_custom_formula_parity_plan import promote_custom_formula_parity_artifacts
from core.turbocore_plugin_custom_formula_execution_matrix import build_custom_formula_execution_matrix
from core.turbocore_plugin_optimizer_selector_scorecard import build_plugin_optimizer_selector_scorecard


CUSTOM_FORMULA_ROUTE_FAMILY = "custom_formula"
UNSAFE_TRUE_FIELDS = (
    "training_path_enabled",
    "default_behavior_changed",
    "runtime_dispatch_ready",
    "native_dispatch_allowed",
    "product_native_ready",
    "product_native_dispatch_ready",
    "native_kernel_ready",
)


def build_plugin_custom_formula_family_batch_scorecard(
    *,
    selector_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Aggregate selected plugin custom-formula status without dispatch."""

    selector = _as_dict(selector_report) if selector_report is not None else _call_selector()
    selector_rows = _selected_custom_formula_rows(selector)
    rows = [_row(row) for row in selector_rows]
    execution_matrix = build_custom_formula_execution_matrix(rows)
    rows = _apply_execution_matrix(rows, execution_matrix)
    tier_counts = Counter(str(row["custom_formula_backlog"]["backlog_tier"]) for row in rows)
    formula_spec_ready_rows = [row for row in rows if row["evidence_status"]["formula_spec"] == "ready"]
    state_inventory_ready_rows = [row for row in rows if row["evidence_status"]["state_inventory"] == "ready"]
    quality_guard_ready_rows = [row for row in rows if row["evidence_status"]["quality_guard_matrix"] == "ready"]
    stage_pending_counts = {
        stage: sum(1 for row in rows if row["evidence_status"][stage] == "pending") for stage in EVIDENCE_STAGES
    }
    stage_ready_counts = {
        stage: sum(1 for row in rows if row["evidence_status"][stage] == "ready") for stage in EVIDENCE_STAGES
    }
    selector_count = int(
        _as_dict(_summary(selector).get("route_family_counts")).get(CUSTOM_FORMULA_ROUTE_FAMILY, 0) or 0
    )
    missing = max(selector_count - len(rows), 0)
    unsafe = _unsafe_claims({"selector": selector}, rows)
    ready = selector.get("ok") is True and bool(rows) and missing == 0 and not unsafe

    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_custom_formula_family_batch_scorecard_v0",
        "gate": "plugin_custom_formula_selected_family_batch",
        "ok": ready,
        "promotion_ready": False,
        "selected_custom_formula_family_batch_ready": ready,
        "report_only": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "plugin_selected_native_ready_count": 0,
        "selected_optimizer_family": CUSTOM_FORMULA_ROUTE_FAMILY,
        "selector_scorecard": _compact_selector(selector),
        "request_contract": {
            "optimizer_type": OptimizerType.PYTORCH_OPTIMIZER.value,
            "selected_optimizer_source": "optimizer_args.name",
            "selected_optimizer_names": [row["selected_optimizer_name"] for row in rows],
            "runtime_authority": "existing_pytorch_optimizer_plugin",
            "native_route_policy": "blocked_until_each_optimizer_formula_state_quality_parity_contract",
        },
        "native_compatibility": {
            "requires_per_optimizer_formula_spec": True,
            "requires_per_optimizer_state_inventory": True,
            "requires_quality_guard_matrix": True,
            "requires_formula_parity_matrix": True,
            "requires_resume_parity_matrix": True,
            "adamw_state_schema_compatible": False,
            "adamw_step_kernel_compatible": False,
            "simple_formula_kernel_compatible": False,
            "can_reuse_exact_adamw_native_dispatch": False,
            "exact_adamw_product_native_route_count_delta": 0,
        },
        "execution_matrix": _compact_execution_matrix(execution_matrix),
        "backlog_plan": {
            "schema_version": 1,
            "plan": "selected_plugin_custom_formula_backlog_v0",
            "tier_order": sorted(BACKLOG_TIERS, key=lambda tier: int(BACKLOG_TIERS[tier]["priority"])),
            "evidence_stages": list(EVIDENCE_STAGES),
            "status": "ready_for_parallel_per_optimizer_evidence",
            "native_work_policy": "blocked_until_optimizer_evidence_stages_pass",
        },
        "rows": rows,
        "summary": {
            "selected_optimizer_count": len(rows),
            "selector_custom_formula_count": selector_count,
            "selector_classified_count": sum(1 for row in rows if row["selector_classified"] is True),
            "resume_proven_count": sum(1 for row in rows if row["resume_proven"] is True),
            "formula_spec_required_count": sum(
                1 for row in rows if row["custom_formula_contract"]["requires_per_optimizer_formula_spec"] is True
            ),
            "state_inventory_required_count": sum(
                1 for row in rows if row["custom_formula_contract"]["requires_per_optimizer_state_inventory"] is True
            ),
            "quality_guard_required_count": sum(
                1 for row in rows if row["custom_formula_contract"]["requires_quality_guard_matrix"] is True
            ),
            "formula_parity_required_count": sum(
                1 for row in rows if row["custom_formula_contract"]["requires_formula_parity_matrix"] is True
            ),
            "resume_parity_required_count": sum(
                1 for row in rows if row["custom_formula_contract"]["requires_resume_parity_matrix"] is True
            ),
            "backlog_tier_counts": dict(sorted(tier_counts.items())),
            "backlog_ready_count": sum(
                1 for row in rows if row["custom_formula_backlog"]["backlog_ready_for_owner"] is True
            ),
            "evidence_stage_pending_counts": stage_pending_counts,
            "evidence_stage_ready_counts": stage_ready_counts,
            "evidence_artifact_planned_count": sum(len(row["evidence_artifacts"]) for row in rows),
            "evidence_status_pending_total": sum(stage_pending_counts.values()),
            "formula_spec_artifact_ready_count": len(formula_spec_ready_rows),
            "formula_spec_artifact_pending_count": stage_pending_counts["formula_spec"],
            "formula_spec_ready_tier_counts": dict(
                sorted(Counter(str(row["custom_formula_backlog"]["backlog_tier"]) for row in formula_spec_ready_rows).items())
            ),
            "formula_spec_ready_optimizer_names": [row["selected_optimizer_name"] for row in formula_spec_ready_rows],
            "formula_state_inventory_skeleton_count": sum(
                1 for row in rows if isinstance(row.get("formula_spec_artifact"), Mapping)
            ),
            "state_inventory_artifact_ready_count": len(state_inventory_ready_rows),
            "state_inventory_artifact_pending_count": stage_pending_counts["state_inventory"],
            "state_inventory_ready_optimizer_names": [row["selected_optimizer_name"] for row in state_inventory_ready_rows],
            "quality_guard_matrix_artifact_ready_count": len(quality_guard_ready_rows),
            "quality_guard_matrix_artifact_pending_count": stage_pending_counts["quality_guard_matrix"],
            "quality_guard_matrix_case_planned_count": sum(
                int(_as_dict(row.get("quality_guard_artifact")).get("guard_case_count", 0) or 0)
                for row in quality_guard_ready_rows
            ),
            "formula_parity_matrix_artifact_planned_count": sum(
                1 for row in rows if isinstance(row.get("formula_parity_matrix_artifact"), Mapping)
            ),
            "formula_parity_matrix_implementation_ready_count": sum(
                1
                for row in rows
                if _as_dict(row.get("formula_parity_matrix_artifact")).get("implementation_ready") is True
            ),
            "formula_parity_case_planned_count": sum(
                int(_as_dict(row.get("formula_parity_matrix_artifact")).get("case_count", 0) or 0)
                for row in rows
            ),
            "resume_parity_matrix_artifact_planned_count": sum(
                1 for row in rows if isinstance(row.get("resume_parity_matrix_artifact"), Mapping)
            ),
            "resume_parity_matrix_implementation_ready_count": sum(
                1
                for row in rows
                if _as_dict(row.get("resume_parity_matrix_artifact")).get("implementation_ready") is True
            ),
            "resume_parity_case_planned_count": sum(
                int(_as_dict(row.get("resume_parity_matrix_artifact")).get("case_count", 0) or 0)
                for row in rows
            ),
            "execution_matrix_ready": execution_matrix.get("execution_matrix_ready") is True,
            "formula_step_execution_ready_count": int(
                _summary(execution_matrix).get("formula_step_execution_ready_count", 0) or 0
            ),
            "resume_next_step_replay_ready_count": int(
                _summary(execution_matrix).get("resume_next_step_replay_ready_count", 0) or 0
            ),
            "execution_failed_count": int(_summary(execution_matrix).get("execution_failed_count", 0) or 0),
            "per_optimizer_next_action_count": sum(len(row["parallel_next_actions"]) for row in rows),
            "adamw_kernel_compatible_count": 0,
            "simple_kernel_compatible_count": 0,
            "native_ready_count": 0,
            "product_native_ready_count": 0,
            "product_native_dispatch_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
            "plugin_selected_native_ready_count": 0,
            "exact_adamw_product_native_route_count_delta": 0,
            "missing_selector_classification_count": missing,
            "unsafe_claim_count": len(unsafe),
        },
        "promotion_blockers": _dedupe(
            unsafe
            + ([f"selector_custom_formula_count_mismatch:{selector_count}:{len(rows)}"] if missing else [])
            + _missing_evidence_blockers(rows)
            + [
                "adamw_native_simple_kernel_not_reusable",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": _dedupe(
            unsafe + ([f"selector_custom_formula_count_mismatch:{selector_count}:{len(rows)}"] if missing else [])
        ),
        "recommended_next_step": (
            "owner/release hold for implementation-ready custom-formula parity/resume matrices with dispatch default-off"
            if ready
            else "fix selector custom-formula blockers before per-optimizer evidence planning"
        ),
        "notes": [
            "This batch is report-only and never enables native dispatch.",
            "Custom-formula plugins are not AdamW-compatible by default and cannot reuse simple-formula evidence.",
            "Each selected optimizer needs separate formula, state, quality, and independent parity evidence.",
            "The execution matrix proves plugin-owned tiny step/resume replay only; native/kernel readiness remains closed.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _call_selector() -> dict[str, Any]:
    try:
        return dict(build_plugin_optimizer_selector_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_optimizer_selector_scorecard", exc)


def _selected_custom_formula_rows(selector: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = selector.get("rows", [])
    if not isinstance(rows, list):
        return []
    selected = [
        row
        for row in rows
        if isinstance(row, Mapping) and str(row.get("native_route_family", "")) == CUSTOM_FORMULA_ROUTE_FAMILY
    ]
    return sorted(selected, key=lambda row: str(row.get("optimizer_name", "")).strip().lower())


def _row(selector_row: Mapping[str, Any]) -> dict[str, Any]:
    name = str(selector_row.get("optimizer_name", "")).strip().lower()
    resume_proven = selector_row.get("resume_proven") is True
    contract = _custom_formula_contract(name)
    backlog = build_custom_formula_backlog(name)
    evidence_artifacts = build_custom_formula_evidence_artifacts(name)
    evidence_status = build_custom_formula_evidence_status(name)
    formula_spec_artifact = build_custom_formula_spec_artifact(name, backlog)
    state_inventory_artifact = build_custom_formula_state_inventory_artifact(name, evidence_artifacts)
    quality_guard_artifact = build_custom_formula_quality_guard_artifact(
        name,
        evidence_artifacts,
        formula_spec_artifact,
        state_inventory_artifact,
    )
    formula_parity_artifact, resume_parity_artifact = build_custom_formula_parity_plan_artifacts(
        name,
        evidence_artifacts,
        formula_spec_artifact,
        state_inventory_artifact,
        quality_guard_artifact,
    )
    return {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "selector": str(selector_row.get("selector", OptimizerType.PYTORCH_OPTIMIZER.value)),
        "native_route_family": CUSTOM_FORMULA_ROUTE_FAMILY,
        "custom_formula_family_hint": custom_formula_hint(name),
        "selector_classified": True,
        "resume_proven": resume_proven,
        "special_handling": str(selector_row.get("special_handling", "")),
        "batch_status": "formula_state_quality_parity_required_report_only",
        "custom_formula_contract": contract,
        "custom_formula_backlog": backlog,
        "evidence_status": evidence_status,
        "evidence_artifacts": evidence_artifacts,
        "evidence_plan": build_custom_formula_evidence_plan(backlog, evidence_artifacts, evidence_status),
        "formula_spec_artifact": formula_spec_artifact,
        "state_inventory_artifact": state_inventory_artifact,
        "quality_guard_artifact": quality_guard_artifact,
        "formula_parity_matrix_artifact": formula_parity_artifact,
        "resume_parity_matrix_artifact": resume_parity_artifact,
        "parallel_next_actions": build_custom_formula_parallel_next_actions(name, backlog, evidence_status),
        "runtime_authority": "existing_pytorch_optimizer_plugin",
        "native_route": "none_report_only",
        "adamw_state_schema_compatible": False,
        "adamw_kernel_compatible": False,
        "simple_formula_kernel_compatible": False,
        "can_reuse_exact_adamw_native_dispatch": False,
        "plugin_selected_native_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "next_gate": f"selected_plugin_{name}_formula_state_quality_parity_contract",
        "formula_state_quality_parity_work_items": [
            evidence_artifacts["formula_spec"],
            evidence_artifacts["state_inventory"],
            evidence_artifacts["quality_guard_matrix"],
            evidence_artifacts["formula_parity_matrix"],
            evidence_artifacts["resume_parity_matrix"],
        ],
        "blocked_reasons": custom_formula_blocked_reasons(evidence_status),
    }


def _apply_execution_matrix(
    rows: list[dict[str, Any]],
    execution_matrix: Mapping[str, Any],
) -> list[dict[str, Any]]:
    execution_rows = {
        str(row.get("selected_optimizer_name", "")): row
        for row in execution_matrix.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        selected = str(row.get("selected_optimizer_name", ""))
        execution = execution_rows.get(selected)
        formula, resume = promote_custom_formula_parity_artifacts(
            _as_dict(row.get("formula_parity_matrix_artifact")),
            _as_dict(row.get("resume_parity_matrix_artifact")),
            execution,
        )
        evidence_status = dict(_as_dict(row.get("evidence_status")))
        if formula and formula.get("implementation_ready") is True:
            evidence_status["formula_parity_matrix"] = "ready"
        if resume and resume.get("implementation_ready") is True:
            evidence_status["resume_parity_matrix"] = "ready"
        out.append(
            {
                **row,
                "evidence_status": evidence_status,
                "evidence_plan": build_custom_formula_evidence_plan(
                    _as_dict(row.get("custom_formula_backlog")),
                    _as_dict(row.get("evidence_artifacts")),
                    evidence_status,
                ),
                "formula_parity_matrix_artifact": formula,
                "resume_parity_matrix_artifact": resume,
                "parallel_next_actions": build_custom_formula_parallel_next_actions(
                    selected,
                    _as_dict(row.get("custom_formula_backlog")),
                    evidence_status,
                ),
                "blocked_reasons": custom_formula_blocked_reasons(evidence_status),
            }
        )
    return out


def _missing_evidence_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    stage_blockers = {
        "formula_spec": "selected_plugin_custom_formula_spec_missing",
        "state_inventory": "selected_plugin_custom_state_inventory_missing",
        "quality_guard_matrix": "selected_plugin_custom_quality_guard_matrix_missing",
        "formula_parity_matrix": "selected_plugin_custom_formula_parity_matrix_missing",
        "resume_parity_matrix": "selected_plugin_custom_resume_parity_matrix_missing",
    }
    blockers = []
    for stage, blocker in stage_blockers.items():
        if any(_as_dict(row.get("evidence_status")).get(stage) != "ready" for row in rows):
            blockers.append(blocker)
    return blockers


def _custom_formula_contract(name: str) -> dict[str, Any]:
    return {
        "contract_family": "per_optimizer_custom_formula",
        "formula_hint": custom_formula_hint(name),
        "backlog_tier": custom_formula_backlog_tier(name),
        "requires_backlog_tier_owner": True,
        "requires_evidence_artifact_plan": True,
        "requires_per_optimizer_formula_spec": True,
        "requires_per_optimizer_state_inventory": True,
        "requires_quality_guard_matrix": True,
        "requires_formula_parity_matrix": True,
        "requires_resume_parity_matrix": True,
        "requires_state_dict_key_inventory": True,
        "requires_hparam_surface_inventory": True,
        "requires_step_order_contract": True,
        "requires_dtype_device_quality_guard": True,
        "adamw_state_schema_compatible": False,
        "adamw_step_kernel_compatible": False,
        "simple_formula_kernel_compatible": False,
        "can_reuse_exact_adamw_native_dispatch": False,
    }


def _compact_selector(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    counts = _as_dict(summary.get("route_family_counts"))
    return {
        "ok": report.get("ok") is True,
        "plugin_selector_classification_ready": report.get("plugin_selector_classification_ready") is True,
        "selector_boundary_ready": report.get("selector_boundary_ready") is True,
        "all_discovered_plugins_resume_proven": report.get("all_discovered_plugins_resume_proven") is True,
        "plugin_optimizer_count": int(summary.get("plugin_optimizer_count", 0) or 0),
        "custom_formula_count": int(counts.get(CUSTOM_FORMULA_ROUTE_FAMILY, 0) or 0),
        "missing_resume_count": int(summary.get("missing_resume_count", 0) or 0),
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
    }


def _unsafe_claims(
    reports: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> list[str]:
    out: list[str] = []
    for name, report in reports.items():
        scorecard = str(report.get("scorecard", name))
        for field in UNSAFE_TRUE_FIELDS:
            if report.get(field) is True:
                out.append(f"unsafe_plugin_custom_formula_source:{scorecard}:{field}")
    for row in rows:
        selected = str(row.get("selected_optimizer_name", "unknown"))
        for field in UNSAFE_TRUE_FIELDS:
            if row.get(field) is True:
                out.append(f"unsafe_plugin_custom_formula_row:{selected}:{field}")
        if row.get("can_reuse_exact_adamw_native_dispatch") is True:
            out.append(f"unsafe_plugin_custom_formula_row:{selected}:adamw_dispatch_reuse")
        if row.get("adamw_kernel_compatible") is True or row.get("simple_formula_kernel_compatible") is True:
            out.append(f"unsafe_plugin_custom_formula_row:{selected}:kernel_compatible")
    return _dedupe(out)


def _failed_report(builder_name: str, exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "scorecard": builder_name,
        "error": f"{type(exc).__name__}: {exc}",
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "blocked_reasons": [f"builder_failed:{builder_name}"],
    }


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = Path(__file__).resolve().parents[2] / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_custom_formula_family_batch_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        "formula_step_execution_ready_count": int(summary.get("formula_step_execution_ready_count", 0) or 0),
        "resume_next_step_replay_ready_count": int(summary.get("resume_next_step_replay_ready_count", 0) or 0),
        "execution_failed_count": int(summary.get("execution_failed_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(report.get("summary"))


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "CUSTOM_FORMULA_ROUTE_FAMILY",
    "build_plugin_custom_formula_family_batch_scorecard",
]
