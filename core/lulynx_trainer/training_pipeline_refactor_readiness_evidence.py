"""Aggregated evidence for the staged Lulynx training pipeline refactor.

This module is report-only. It combines already-collected runtime features,
the batch1 handler parity smoke, and the refactor plan into one small evidence
document for devtools and release review.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .training_pipeline_execution_readiness import build_lulynx_training_pipeline_execution_readiness
from .training_pipeline_refactor_plan import build_lulynx_training_pipeline_refactor_plan
from .training_step_orchestrator import build_lulynx_training_step_orchestrator_slice


LULYNX_TRAINING_PIPELINE_REFACTOR_READINESS_EVIDENCE = (
    "lulynx_training_pipeline_refactor_readiness_evidence_v0"
)


def build_lulynx_training_pipeline_refactor_readiness_evidence(
    *,
    runtime_features: Mapping[str, Any] | None = None,
    batch1_parity_smoke: Mapping[str, Any] | None = None,
    real_gpu_batch1_golden_evidence: Mapping[str, Any] | None = None,
    contracts_locked: bool = True,
    batch2_long_window_failed: bool | None = None,
    compile_release_gate_passed: bool | None = None,
    known_blockers: Sequence[str] = (),
    target_physical_batch_sizes: Sequence[int] = (2, 4, 8),
) -> dict[str, Any]:
    """Build a single report-only readiness document for the refactor path."""

    features = dict(_mapping(runtime_features))
    parity = dict(_mapping(batch1_parity_smoke) or _mapping(features.get("batch1_parity_smoke")))
    golden = dict(
        _mapping(real_gpu_batch1_golden_evidence)
        or _mapping(features.get("real_gpu_batch1_golden_evidence"))
    )
    if parity:
        features["batch1_parity_smoke"] = parity
    if golden:
        features["real_gpu_batch1_golden_evidence"] = golden
    features["contracts_locked"] = bool(contracts_locked)
    if batch2_long_window_failed is not None:
        features["batch2_long_window_failed"] = bool(batch2_long_window_failed)
    if compile_release_gate_passed is not None:
        features["compile_release_gate_passed"] = bool(compile_release_gate_passed)
    blockers = _string_list(known_blockers)
    if blockers:
        features["known_blockers"] = blockers

    execution_readiness = build_lulynx_training_pipeline_execution_readiness(runtime_features=features)
    orchestrator_slice = build_lulynx_training_step_orchestrator_slice(
        runtime_features=features,
        execution_readiness=execution_readiness,
        internal_gate_enabled=False,
    )
    plan = build_lulynx_training_pipeline_refactor_plan(
        current_state=features,
        target_physical_batch_sizes=target_physical_batch_sizes,
    )
    normalized_parity = dict(_mapping(plan.get("batch1_parity_smoke")))
    promotion_blockers = _string_list(plan.get("multi_batch_promotion_blockers"))
    evidence_blockers = _evidence_blockers(
        execution_readiness=execution_readiness,
        orchestrator_slice=orchestrator_slice,
        batch1_parity_smoke=normalized_parity,
        refactor_plan=plan,
    )
    ready_for_internal_gate = (
        bool(execution_readiness.get("ready_for_behavior_equivalent_orchestrator_slice"))
        and bool(orchestrator_slice.get("ready_for_internal_gate"))
        and bool(normalized_parity.get("passed"))
    )
    action_statuses = {
        str(action.get("id")): str(action.get("status"))
        for action in _sequence_of_mappings(plan.get("actions"))
        if action.get("id")
    }
    return {
        "schema_version": 1,
        "report": LULYNX_TRAINING_PIPELINE_REFACTOR_READINESS_EVIDENCE,
        "status": "ready_for_internal_orchestrator_gate_non_gpu" if ready_for_internal_gate else "blocked",
        "brand": "Lulynx",
        "release_claim_allowed": False,
        "safe_to_auto_start": False,
        "does_not_start_gpu_work": True,
        "does_not_start_dataloader_iteration": True,
        "does_not_add_training_entrypoint": True,
        "agpl_risk_policy": "behavior_contract_only_reimplemented_in_house",
        "ready_for_internal_orchestrator_gate": ready_for_internal_gate,
        "internal_orchestrator_gate_enabled": False,
        "ready_for_batch2_4_8_release_probe": False,
        "pipeline_execution_readiness": execution_readiness,
        "training_step_orchestrator_slice": orchestrator_slice,
        "batch1_parity_smoke": normalized_parity,
        "real_gpu_batch1_golden_evidence": golden,
        "refactor_plan": _compact_plan(plan),
        "action_statuses": action_statuses,
        "blockers": _dedupe([*evidence_blockers, *promotion_blockers]),
        "recommended_next_actions": _recommended_next_actions(
            ready_for_internal_gate=ready_for_internal_gate,
            blockers=evidence_blockers,
        ),
    }


def _evidence_blockers(
    *,
    execution_readiness: Mapping[str, Any],
    orchestrator_slice: Mapping[str, Any],
    batch1_parity_smoke: Mapping[str, Any],
    refactor_plan: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    for blocker in _string_list(execution_readiness.get("blockers")):
        blockers.append(f"pipeline_execution_readiness:{blocker}")
    if not bool(orchestrator_slice.get("ready_for_internal_gate")):
        slice_blockers = _string_list(orchestrator_slice.get("blockers")) or ["not_ready"]
        blockers.extend(f"training_step_orchestrator_slice:{item}" for item in slice_blockers)
    if not bool(batch1_parity_smoke.get("passed")):
        blockers.append("batch1_parity_smoke_missing")
    for action in _sequence_of_mappings(refactor_plan.get("actions")):
        status = str(action.get("status") or "")
        if status.startswith("blocked_"):
            blockers.append(f"refactor_action:{action.get('id')}:{status}")
    return _dedupe(blockers)


def _compact_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "report": str(plan.get("report") or ""),
        "status": str(plan.get("status") or ""),
        "release_claim_allowed": bool(plan.get("release_claim_allowed")),
        "does_not_start_gpu_work": bool(plan.get("does_not_start_gpu_work")),
        "target_physical_batch_sizes": list(plan.get("target_physical_batch_sizes") or []),
        "promotion_order": list(plan.get("promotion_order") or []),
        "multi_batch_promotion_blockers": _string_list(plan.get("multi_batch_promotion_blockers")),
    }


def _recommended_next_actions(*, ready_for_internal_gate: bool, blockers: Sequence[str]) -> list[str]:
    if ready_for_internal_gate:
        return [
            "wire_behavior_equivalent_orchestrator_slice_behind_disabled_internal_gate",
            "run_real_gpu_batch1_golden_before_enabling_internal_gate",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["complete_refactor_readiness_evidence_before_training_loop_promotion"]
    if any("batch1_parity_smoke" in blocker for blocker in blockers):
        actions.append("run_lulynx_batch1_handler_parity_smoke")
    if any("training_data_pipeline" in blocker or "batch_collate" in blocker for blocker in blockers):
        actions.append("record_real_training_data_pipeline_and_batch_collate_runtime_evidence")
    if any("training_pipeline_trace" in blocker or "pipeline_stage" in blocker for blocker in blockers):
        actions.append("complete_report_only_training_pipeline_trace_for_all_contract_stages")
    return _dedupe(actions)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _sequence_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [item for item in value if isinstance(item, Mapping)]


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
    "LULYNX_TRAINING_PIPELINE_REFACTOR_READINESS_EVIDENCE",
    "build_lulynx_training_pipeline_refactor_readiness_evidence",
]
