"""Golden evidence for real GPU batch1 staged-pipeline readiness.

This module is report-only. It reads an already-produced run manifest plus
batch1 parity evidence and does not start training, DataLoader iteration, or
GPU work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .training_pipeline_execution_readiness import (
    build_lulynx_training_pipeline_execution_readiness,
)
from .training_step_orchestrator import build_lulynx_training_step_orchestrator_slice


LULYNX_REAL_GPU_BATCH1_GOLDEN_EVIDENCE = "lulynx_real_gpu_batch1_golden_evidence_v0"


def build_lulynx_real_gpu_batch1_golden_evidence(
    *,
    run_manifest: Mapping[str, Any] | None = None,
    batch1_parity_smoke: Mapping[str, Any] | None = None,
    min_steps: int = 1,
) -> dict[str, Any]:
    """Return fail-closed evidence for real GPU batch1 internal-gate review."""

    manifest = dict(_mapping(run_manifest))
    features = _runtime_features_from_manifest(manifest)
    parity = dict(_mapping(batch1_parity_smoke))
    data_pipeline = _mapping(features.get("training_data_pipeline"))
    loop_runtime = _mapping(features.get("training_loop_runtime"))
    trace = _mapping(loop_runtime.get("training_pipeline_trace"))
    batch_contract = _mapping(trace.get("batch_contract"))
    step_phase = _mapping(loop_runtime.get("step_phase_profile"))
    step_phase_last = _mapping(step_phase.get("last"))
    manifest_orchestrator_slice = _mapping(features.get("training_step_orchestrator_slice"))
    runtime_orchestrator = _mapping(features.get("training_step_orchestrator_runtime"))
    execution_readiness = build_lulynx_training_pipeline_execution_readiness(
        runtime_features=features
    )
    orchestrator_slice = build_lulynx_training_step_orchestrator_slice(
        runtime_features=features,
        execution_readiness=execution_readiness,
        internal_gate_enabled=False,
    )

    global_step = _safe_int(manifest.get("global_step"), 0)
    required_steps = max(_safe_int(min_steps, 1), 1)
    real_gpu_runtime_evidence_present = bool(
        step_phase_last.get("cuda_event_profile_available")
        or _mapping(step_phase_last.get("cuda_event_ms"))
    )
    parity_passed = bool(parity.get("passed"))
    release_claim_leaks = _release_claim_leaks(
        manifest=manifest,
        features=features,
        parity=parity,
        execution_readiness=execution_readiness,
        orchestrator_slice=orchestrator_slice,
    )

    checks = {
        "manifest_completed": str(manifest.get("status") or "") == "completed",
        "min_steps_completed": global_step >= required_steps,
        "training_data_pipeline_ok": bool(data_pipeline.get("ok")),
        "dataset_scan_stage_plan_present": bool(data_pipeline.get("dataset_scan_stage_plan")),
        "bucket_plan_stage_plan_present": bool(data_pipeline.get("bucket_plan_stage_plan")),
        "batch_collate_stage_plan_present": bool(data_pipeline.get("batch_collate_stage_plan")),
        "batch_collate_runtime_observed": bool(data_pipeline.get("batch_collate_runtime_observed")),
        "training_pipeline_trace_completed": str(trace.get("status") or "") == "completed",
        "batch_contract_is_batch1": bool(
            batch_contract.get("ok")
            and _safe_int(batch_contract.get("expected_physical_batch_size"), 0) == 1
            and _safe_int(batch_contract.get("inferred_physical_batch_size"), 0) == 1
            and not bool(batch_contract.get("real_multi_batch"))
        ),
        "real_gpu_runtime_evidence_present": real_gpu_runtime_evidence_present,
        "batch1_parity_smoke_passed": parity_passed,
        "pipeline_execution_readiness_ready": bool(
            execution_readiness.get("ready_for_behavior_equivalent_orchestrator_slice")
        ),
        "orchestrator_slice_ready_behind_disabled_gate": bool(
            orchestrator_slice.get("ready_for_internal_gate")
            and not orchestrator_slice.get("internal_gate_enabled")
        ),
        "manifest_orchestrator_gate_disabled": not bool(
            manifest_orchestrator_slice.get("internal_gate_enabled")
        ),
        "runtime_orchestrator_gate_disabled": not bool(
            runtime_orchestrator.get("internal_gate_enabled")
        ),
        "release_claim_closed": not release_claim_leaks,
    }
    blockers = _blockers_from_checks(checks)
    for item in _string_list(execution_readiness.get("blockers")):
        blockers.append(f"pipeline_execution_readiness:{item}")
    for item in release_claim_leaks:
        blockers.append(f"release_claim_leak:{item}")
    blockers = _dedupe(blockers)
    ok = not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_REAL_GPU_BATCH1_GOLDEN_EVIDENCE,
        "status": "ready_for_internal_gate_enablement_review" if ok else "blocked",
        "passed": ok,
        "release_claim_allowed": False,
        "safe_to_auto_start": False,
        "does_not_start_gpu_work": True,
        "does_not_start_dataloader_iteration": True,
        "does_not_add_training_entrypoint": True,
        "min_steps_required": required_steps,
        "global_step": global_step,
        "checks": checks,
        "blockers": blockers,
        "pipeline_execution_readiness": execution_readiness,
        "training_step_orchestrator_slice": orchestrator_slice,
        "multi_batch_promotion_gate_blockers": _string_list(
            execution_readiness.get("multi_batch_promotion_gate_blockers")
        ),
        "recommended_next_actions": _recommended_next_actions(ok=ok, blockers=blockers),
    }


def _runtime_features_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    extra = _mapping(manifest.get("extra"))
    if extra:
        return dict(extra)
    runtime_features = _mapping(manifest.get("runtime_features"))
    if runtime_features:
        return dict(runtime_features)
    return dict(manifest)


def _release_claim_leaks(
    *,
    manifest: Mapping[str, Any],
    features: Mapping[str, Any],
    parity: Mapping[str, Any],
    execution_readiness: Mapping[str, Any],
    orchestrator_slice: Mapping[str, Any],
) -> list[str]:
    leaks: list[str] = []
    for name, payload in (
        ("manifest", manifest),
        ("features", features),
        ("batch1_parity_smoke", parity),
        ("pipeline_execution_readiness", execution_readiness),
        ("training_step_orchestrator_slice", orchestrator_slice),
    ):
        if bool(_mapping(payload).get("release_claim_allowed")):
            leaks.append(name)
    return leaks


def _blockers_from_checks(checks: Mapping[str, bool]) -> list[str]:
    return [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]


def _recommended_next_actions(*, ok: bool, blockers: Sequence[str]) -> list[str]:
    if ok:
        return [
            "review_internal_gate_enablement_with_real_gpu_batch1_golden",
            "keep_internal_gate_disabled_until_explicit_enablement_review",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["rerun_real_gpu_batch1_golden_probe_after_fixing_blockers"]
    if any("batch1_parity" in item for item in blockers):
        actions.append("refresh_lulynx_batch1_handler_parity_smoke")
    if any("training_data_pipeline" in item or "batch_collate" in item for item in blockers):
        actions.append("refresh_real_run_manifest_with_observed_training_data_pipeline")
    if any("pipeline_execution_readiness" in item for item in blockers):
        actions.append("refresh_pipeline_refactor_readiness_from_current_manifest")
    return _dedupe(actions)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


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
    "LULYNX_REAL_GPU_BATCH1_GOLDEN_EVIDENCE",
    "build_lulynx_real_gpu_batch1_golden_evidence",
]
