"""Readiness gate for promoting Lulynx staged pipeline execution.

This module is diagnostic only. It summarizes already-collected runtime
evidence and decides whether the next behavior-equivalent orchestrator slice is
ready to wire. It does not execute training, inspect tensors, or start
DataLoader iteration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .training_pipeline_contract import lulynx_training_stage_ids


LULYNX_TRAINING_PIPELINE_EXECUTION_READINESS = "lulynx_training_pipeline_execution_readiness_v0"

_DATA_PLAN_BY_STAGE = {
    "dataset_scan": "dataset_scan_stage_plan",
    "bucket_plan": "bucket_plan_stage_plan",
    "batch_collate": "batch_collate_stage_plan",
}

_TRACE_PLAN_BY_STAGE = {
    "batch_contract": "batch_stage_plan",
    "host_to_device": "transfer_stage_plan",
    "noise_timestep": "noise_timestep_stage_plan",
    "conditioning": "conditioning_stage_plan",
    "forward": "forward_stage_plan",
    "loss": "loss_stage_plan",
    "backward": "backward_stage_plan",
    "optimizer_step": "optimizer_stage_plan",
    "telemetry": "telemetry_stage_plan",
}


def build_lulynx_training_pipeline_execution_readiness(
    *,
    runtime_features: Mapping[str, Any] | None = None,
    training_data_pipeline: Mapping[str, Any] | None = None,
    training_pipeline_trace: Mapping[str, Any] | None = None,
    multi_batch_promotion_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return staged-pipeline execution readiness from existing evidence."""

    features = _mapping(runtime_features)
    data_report = dict(_mapping(training_data_pipeline) or _mapping(features.get("training_data_pipeline")))
    trace = dict(_mapping(training_pipeline_trace) or _trace_from_features(features))
    gate = dict(_mapping(multi_batch_promotion_gate) or _mapping(features.get("multi_batch_promotion_gate")))
    stage_reports = [
        _stage_readiness(stage, data_report=data_report, trace=trace)
        for stage in lulynx_training_stage_ids()
    ]
    blockers = _readiness_blockers(stage_reports=stage_reports, data_report=data_report, trace=trace)
    multi_batch_gate_blockers = _string_list(gate.get("blockers")) if gate else []
    ready_for_multi_batch_long_window_probe = bool(
        gate and gate.get("ready_for_long_window_probe")
    )
    return {
        "schema_version": 1,
        "gate": LULYNX_TRAINING_PIPELINE_EXECUTION_READINESS,
        "status": "ready_for_behavior_equivalent_orchestrator_slice" if not blockers else "blocked",
        "ready_for_behavior_equivalent_orchestrator_slice": not blockers,
        "ready_for_multi_batch_long_window_probe": ready_for_multi_batch_long_window_probe,
        "release_claim_allowed": False,
        "does_not_start_gpu_work": True,
        "does_not_start_dataloader_iteration": True,
        "stage_ids": lulynx_training_stage_ids(),
        "stage_readiness": stage_reports,
        "ready_stage_ids": [item["stage_id"] for item in stage_reports if item["ready_for_orchestrator_slice"]],
        "blocked_stage_ids": [item["stage_id"] for item in stage_reports if not item["ready_for_orchestrator_slice"]],
        "blockers": blockers,
        "multi_batch_promotion_gate_status": str(gate.get("status") or "") if gate else "",
        "multi_batch_promotion_gate_blockers": multi_batch_gate_blockers,
        "recommended_next_actions": _recommended_next_actions(
            blockers,
            multi_batch_gate_blockers=multi_batch_gate_blockers,
            ready_for_multi_batch_long_window_probe=ready_for_multi_batch_long_window_probe,
        ),
    }


def _stage_readiness(
    stage_id: str,
    *,
    data_report: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    plan_key = _DATA_PLAN_BY_STAGE.get(stage_id)
    evidence_source = "missing"
    observed = False
    plan: Mapping[str, Any] = {}
    reasons: list[str] = []
    if plan_key:
        plan = _mapping(data_report.get(plan_key))
        observed = bool(plan)
        evidence_source = "data_pipeline_report" if observed else "missing"
        if not data_report:
            reasons.append("missing_training_data_pipeline_report")
        if stage_id == "batch_collate" and "batch_collate_not_observed_without_dataloader_iteration" in _string_list(
            data_report.get("missing_runtime_evidence")
        ):
            reasons.append("batch_collate_not_observed_without_dataloader_iteration")
    else:
        stage_ids = set(_string_list(trace.get("stage_ids")))
        stage_plans = _mapping(trace.get("stage_plans"))
        plan_key = _TRACE_PLAN_BY_STAGE.get(stage_id, "")
        plan = _mapping(stage_plans.get(plan_key))
        observed = stage_id in stage_ids
        evidence_source = "training_pipeline_trace" if observed else "missing"
        if not trace:
            reasons.append("missing_training_pipeline_trace")
        elif str(trace.get("status") or "") != "completed":
            reasons.append("training_pipeline_trace_not_completed")
        if not plan:
            reasons.append("missing_stage_plan")

    if observed and not plan:
        reasons.append("stage_observed_without_plan")
    if plan and bool(plan.get("compile_static_graph_risk")):
        reasons.append("compile_static_graph_risk")
    if plan and not bool(plan.get("ok", True)):
        reasons.append("stage_plan_not_ok")
    blocking = [
        reason
        for reason in reasons
        if reason
        in {
            "missing_training_data_pipeline_report",
            "missing_training_pipeline_trace",
            "training_pipeline_trace_not_completed",
            "missing_stage_plan",
            "stage_observed_without_plan",
            "stage_plan_not_ok",
            "batch_collate_not_observed_without_dataloader_iteration",
        }
    ]
    return {
        "stage_id": stage_id,
        "evidence_source": evidence_source,
        "observed": observed,
        "stage_plan_key": plan_key or "",
        "stage_plan_present": bool(plan),
        "ready_for_orchestrator_slice": observed and bool(plan) and not blocking,
        "reasons": _dedupe(reasons),
    }


def _readiness_blockers(
    *,
    stage_reports: Sequence[Mapping[str, Any]],
    data_report: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not data_report:
        blockers.append("missing_training_data_pipeline_report")
    if not trace:
        blockers.append("missing_training_pipeline_trace")
    elif str(trace.get("status") or "") != "completed":
        blockers.append("training_pipeline_trace_not_completed")
    blocked_stages = [str(item.get("stage_id")) for item in stage_reports if not item.get("ready_for_orchestrator_slice")]
    if blocked_stages:
        blockers.append("pipeline_stage_evidence_incomplete")
    if "batch_collate" in blocked_stages:
        blockers.append("batch_collate_runtime_evidence_missing")
    return _dedupe(blockers)


def _trace_from_features(features: Mapping[str, Any]) -> Mapping[str, Any]:
    loop_runtime = _mapping(features.get("training_loop_runtime"))
    return _mapping(loop_runtime.get("training_pipeline_trace"))


def _recommended_next_actions(
    blockers: Sequence[str],
    *,
    multi_batch_gate_blockers: Sequence[str] = (),
    ready_for_multi_batch_long_window_probe: bool = False,
) -> list[str]:
    if not blockers:
        actions = [
            "wire_behavior_equivalent_stage_orchestrator_slice_behind_internal_gate",
            "verify_batch1_loss_and_trace_parity",
            "keep_multi_batch_release_claim_blocked_until_long_window_matrix_passes",
        ]
        if multi_batch_gate_blockers or not ready_for_multi_batch_long_window_probe:
            actions.append("keep_batch2_4_8_probe_separate_from_batch1_orchestrator_readiness")
        return actions
    actions = ["complete_pipeline_stage_evidence_before_execution_boundary"]
    if "batch_collate_runtime_evidence_missing" in blockers:
        actions.append("record_first_real_batch_collate_contract_without_extra_dataloader_iteration")
    if "missing_training_pipeline_trace" in blockers or "pipeline_stage_evidence_incomplete" in blockers:
        actions.append("complete_report_only_trace_for_all_contract_stages")
    if multi_batch_gate_blockers or not ready_for_multi_batch_long_window_probe:
        actions.append("fix_multi_batch_gate_blockers_before_batch2_4_8_probe")
    return actions


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
    "LULYNX_TRAINING_PIPELINE_EXECUTION_READINESS",
    "build_lulynx_training_pipeline_execution_readiness",
]
