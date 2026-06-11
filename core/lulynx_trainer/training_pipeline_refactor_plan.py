"""Executable plan for the Lulynx staged training refactor.

This is a planning/reporting module only. It does not change the active
training path and does not start GPU work. The goal is to keep the pre-release
pipeline refactor ordered: stage contracts first, native multi-batch after the
batch contract is visible, then long-window stability evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from backend.core.lulynx_trainer.multi_batch_contract import normalize_multi_batch_request
from backend.core.lulynx_trainer.training_pipeline_contract import (
    build_lulynx_pipeline_refactor_roadmap_item,
    build_lulynx_training_pipeline_contract,
    lulynx_training_stage_ids,
)
from backend.core.lulynx_trainer.training_pipeline_execution_readiness import (
    build_lulynx_training_pipeline_execution_readiness,
)
from backend.core.lulynx_trainer.training_step_orchestrator import build_lulynx_training_step_orchestrator_slice


LULYNX_TRAINING_PIPELINE_REFACTOR_PLAN_REPORT = "lulynx_training_pipeline_refactor_plan_v0"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _safe_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _multi_batch_execution_strategy(state: Mapping[str, Any]) -> dict[str, Any]:
    training_loop_runtime = _mapping(state.get("training_loop_runtime"))
    trace = _mapping(training_loop_runtime.get("training_pipeline_trace") or state.get("training_pipeline_trace"))
    strategy = _mapping(trace.get("multi_batch_execution_strategy"))
    if strategy:
        return dict(strategy)
    return {}


def _multi_batch_execution_strategy_gate(state: Mapping[str, Any]) -> dict[str, Any]:
    training_loop_runtime = _mapping(state.get("training_loop_runtime"))
    trace = _mapping(training_loop_runtime.get("training_pipeline_trace") or state.get("training_pipeline_trace"))
    gate = _mapping(trace.get("multi_batch_execution_strategy_gate"))
    if gate:
        return dict(gate)
    return {}


def _multi_batch_promotion_gate(state: Mapping[str, Any]) -> dict[str, Any]:
    gate = _mapping(state.get("multi_batch_promotion_gate"))
    if gate:
        return dict(gate)
    return {}


def _multi_batch_promotion_gate_blockers(state: Mapping[str, Any]) -> list[str]:
    gate = _multi_batch_promotion_gate(state)
    if not gate or _safe_bool(gate.get("ready_for_long_window_probe")):
        return []
    blockers = _string_list(gate.get("blockers")) or ["multi_batch_promotion_gate_blocked"]
    return [f"multi_batch_promotion_gate:{item}" for item in blockers]


def _batch1_parity_smoke(state: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(state.get("batch1_parity_smoke") or state.get("batch1_handler_parity_smoke"))
    passed = _safe_bool(raw.get("passed")) or _safe_bool(state.get("batch1_handler_parity_smoke_passed"))
    status = str(raw.get("status") or ("passed" if passed else "missing"))
    report = {
        "schema_version": 1,
        "smoke": "lulynx_batch1_handler_parity_smoke_v0",
        "status": status,
        "passed": passed,
        "release_claim_allowed": False,
        "does_not_start_gpu_work": True,
        "required_before_internal_orchestrator_gate": True,
        "evidence_source": str(raw.get("evidence_source") or ("unit_smoke" if passed else "missing")),
    }
    for key in ("checks", "blockers", "direct", "orchestrated"):
        value = raw.get(key)
        if value is not None:
            report[key] = value
    return report


def _real_gpu_batch1_golden_evidence(state: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(state.get("real_gpu_batch1_golden_evidence"))
    if raw:
        return dict(raw)
    return {
        "schema_version": 1,
        "report": "lulynx_real_gpu_batch1_golden_evidence_v0",
        "status": "missing",
        "passed": False,
        "release_claim_allowed": False,
        "blockers": ["real_gpu_batch1_golden_evidence_missing"],
    }


def _action(
    *,
    action_id: str,
    phase: str,
    priority: int,
    status: str,
    required_contracts: Sequence[str],
    unlocks: Sequence[str],
    rationale: str,
    gates: Sequence[str] = (),
    files: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "id": action_id,
        "phase": phase,
        "priority": int(priority),
        "status": status,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "requires_gpu_if_executed": False,
        "required_contracts": list(required_contracts),
        "unlocks": list(unlocks),
        "gates": list(gates),
        "files": list(files),
        "rationale": rationale,
    }


def _planned_actions(
    *,
    include_training_loop_extraction: bool,
    execution_readiness: Mapping[str, Any],
    orchestrator_slice: Mapping[str, Any],
    multi_batch_execution_strategy: Mapping[str, Any],
    multi_batch_execution_strategy_gate: Mapping[str, Any],
    batch1_parity_smoke: Mapping[str, Any],
    real_gpu_batch1_golden_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    readiness = _mapping(execution_readiness)
    parity_ready = _safe_bool(batch1_parity_smoke.get("passed"))
    golden_ready = _safe_bool(real_gpu_batch1_golden_evidence.get("passed"))
    extraction_status = "planned_non_gpu" if include_training_loop_extraction else "blocked_until_contracts_locked"
    if include_training_loop_extraction and readiness and not _safe_bool(
        readiness.get("ready_for_behavior_equivalent_orchestrator_slice")
    ):
        extraction_status = "blocked_until_pipeline_execution_readiness"
    elif include_training_loop_extraction and not parity_ready:
        extraction_status = "blocked_until_batch1_parity_smoke"
    orchestrator_status = (
        "ready_non_gpu"
        if _safe_bool(orchestrator_slice.get("ready_for_internal_gate")) and parity_ready
        else "blocked_until_pipeline_execution_readiness"
        if not _safe_bool(orchestrator_slice.get("ready_for_internal_gate"))
        else "blocked_until_batch1_parity_smoke"
    )
    golden_review_status = (
        "ready_for_internal_gate_enablement_review"
        if golden_ready
        else "blocked_until_real_gpu_batch1_golden_evidence"
        if _safe_bool(orchestrator_slice.get("ready_for_internal_gate")) and parity_ready
        else "blocked_until_behavior_equivalent_orchestrator_slice"
    )
    strategy_evidence_present = (
        str(multi_batch_execution_strategy.get("contract") or "")
        == "lulynx_multi_batch_execution_strategy_v0"
        and bool(multi_batch_execution_strategy.get("strategy"))
    )
    strategy_gate_present = (
        str(multi_batch_execution_strategy_gate.get("gate") or "")
        == "lulynx_multi_batch_execution_strategy_gate_v0"
        and bool(multi_batch_execution_strategy_gate.get("status"))
    )
    execution_strategy_status = (
        "ready_report_only_strategy_evidence"
        if (
            strategy_evidence_present
            and strategy_gate_present
            and _safe_bool(readiness.get("ready_for_behavior_equivalent_orchestrator_slice"))
            and parity_ready
        )
        else "blocked_until_batch1_parity_smoke"
        if strategy_evidence_present and strategy_gate_present and not parity_ready
        else "blocked_until_stage_extraction_evidence"
        if strategy_evidence_present or strategy_gate_present
        else "planned_after_stage_extraction"
    )
    actions = [
        _action(
            action_id="lock_lulynx_stage_contract_and_hook_tiers",
            phase="contract_lock",
            priority=10,
            status="ready_non_gpu",
            required_contracts=["lulynx_training_pipeline_contract_v0"],
            unlocks=["training_step_stage_extraction", "plugin_hook_registry_skeleton"],
            gates=[
                "no_new_training_entrypoint",
                "runtime_request_boundary_preserved",
                "launcher_webui_backend_boundaries_preserved",
            ],
            files=["backend/core/lulynx_trainer/training_pipeline_contract.py"],
            rationale="Freeze the staged runtime vocabulary before touching the training loop.",
        ),
        _action(
            action_id="attach_lulynx_multi_batch_contract_to_dataloader",
            phase="batch_contract",
            priority=20,
            status="planned_non_gpu",
            required_contracts=[
                "lulynx_multi_batch_dataloader_contract_v0",
                "lulynx_multi_batch_training_batch_contract_v0",
            ],
            unlocks=["native_batch_forward_strategy", "batch2_4_8_stability_matrix"],
            gates=[
                "bucket_uniform_batches_for_physical_batch_gt1",
                "physical_batch_size_separate_from_effective_batch_size",
                "tail_batch_policy_visible",
            ],
            files=[
                "backend/core/lulynx_trainer/dataset_loader.py",
                "backend/core/lulynx_trainer/multi_batch_contract.py",
            ],
            rationale="Expose real physical batch semantics at the dataloader boundary before promoting batch N.",
        ),
        _action(
            action_id="extract_lulynx_train_step_stages_without_behavior_change",
            phase="stage_extraction",
            priority=30,
            status=extraction_status,
            required_contracts=["lulynx_training_pipeline_contract_v0"],
            unlocks=["per_stage_telemetry", "plugin_hook_gate", "execution_strategy_switch"],
            gates=[
                "golden_batch1_loss_smoke_unchanged",
                "batch_dim_preserved_across_noise_conditioning_forward_loss",
                "no_reference_project_code_or_names",
                "training_pipeline_execution_readiness_clear",
            ],
            files=["backend/core/lulynx_trainer/training_loop.py"],
            rationale="Split the current train step into named internal stages while keeping the active behavior stable.",
        ),
        _action(
            action_id="wire_lulynx_behavior_equivalent_stage_orchestrator_slice",
            phase="stage_orchestrator",
            priority=35,
            status=orchestrator_status,
            required_contracts=[
                "lulynx_training_pipeline_contract_v0",
                "lulynx_training_pipeline_execution_readiness_v0",
            ],
            unlocks=["training_loop_internal_stage_gate", "per_stage_plugin_hook_mount_points"],
            gates=[
                "internal_gate_disabled_by_default",
                "stage_handlers_supplied_by_existing_training_loop_only",
                "no_new_training_entrypoint",
                "no_reference_project_code_or_names",
            ],
            files=[
                "backend/core/lulynx_trainer/training_step_orchestrator.py",
                "backend/core/lulynx_trainer/training_loop.py",
            ],
            rationale=(
                "Add the Lulynx-owned stage envelope that can call existing train-step handlers in contract order "
                "once runtime evidence proves the slice is ready."
            ),
        ),
        _action(
            action_id="review_lulynx_real_gpu_batch1_golden_before_internal_gate_enablement",
            phase="stage_orchestrator_enablement_review",
            priority=36,
            status=golden_review_status,
            required_contracts=[
                "lulynx_real_gpu_batch1_golden_evidence_v0",
                "lulynx_batch1_handler_parity_smoke_v0",
            ],
            unlocks=["training_step_orchestrator_internal_gate_enablement_review"],
            gates=[
                "real_gpu_batch1_manifest_completed",
                "training_data_pipeline_observed",
                "batch1_parity_smoke_passed",
                "internal_gate_still_disabled_before_review",
                "release_claim_closed",
            ],
            files=[
                "backend/core/lulynx_trainer/real_gpu_batch1_golden_evidence.py",
                "devtools/build_lulynx_real_gpu_batch1_golden_evidence.py",
            ],
            rationale=(
                "Require a real GPU batch1 golden package before any review of the internal orchestrator gate."
            ),
        ),
        _action(
            action_id="add_lulynx_plugin_hook_gate_for_training_stages",
            phase="plugin_hook_gate",
            priority=40,
            status="planned_non_gpu",
            required_contracts=["lulynx_training_pipeline_contract_v0"],
            unlocks=["readonly_advisory_transform_control_training_hooks"],
            gates=[
                "plugins_declare_stage_id_and_access_tier",
                "mutations_rejected_when_field_not_declared",
                "diagnostic_hooks_do_not_create_release_claims",
            ],
            files=["backend/core/lulynx_trainer/training_pipeline_contract.py"],
            rationale="Give plugins a narrow, auditable way to observe or influence training stages.",
        ),
        _action(
            action_id="wire_lulynx_native_microbatch_single_item_strategy",
            phase="execution_strategy",
            priority=50,
            status=execution_strategy_status,
            required_contracts=["lulynx_multi_batch_execution_strategy_v0"],
            unlocks=["batch2_4_8_runtime_rollout"],
            gates=[
                "native_batch_forward_is_default_only_when_contract_passes",
                "microbatch_is_diagnostic_not_release_claim",
                "single_item_debug_isolation_available",
            ],
            files=[
                "backend/core/lulynx_trainer/training_loop.py",
                "backend/core/lulynx_trainer/multi_batch_contract.py",
            ],
            rationale="Select native batch, diagnostic microbatch, or single-item isolation from Lulynx contract evidence.",
        ),
        _action(
            action_id="run_lulynx_batch2_4_8_long_window_matrix_after_refactor",
            phase="release_evidence",
            priority=60,
            status="manual_gpu_later",
            required_contracts=[
                "lulynx_training_pipeline_contract_v0",
                "lulynx_multi_batch_training_batch_contract_v0",
            ],
            unlocks=["multi_batch_release_claim_review"],
            gates=[
                "no_cuda_backward_failure",
                "throughput_gain_met",
                "loss_stability_required",
                "vram_headroom_required",
                "phase_telemetry_shows_bubble_reduction",
            ],
            files=["devtools/docs/bubble_aware_runtime_controller_roadmap.md"],
            rationale="Only long-window batch2/4/8 stability can justify native multi-batch release claims.",
        ),
    ]
    return sorted(actions, key=lambda item: (int(item["priority"]), str(item["id"])))


def build_lulynx_training_pipeline_refactor_plan(
    *,
    current_state: Mapping[str, Any] | None = None,
    target_physical_batch_sizes: Sequence[int] = (2, 4, 8),
) -> dict[str, Any]:
    """Return the pre-release staged refactor plan as structured evidence."""

    state = _mapping(current_state)
    pipeline_contract = build_lulynx_training_pipeline_contract()
    roadmap_item = build_lulynx_pipeline_refactor_roadmap_item()
    include_extraction = _safe_bool(state.get("contracts_locked"), default=True)
    physical_targets = [
        parsed
        for parsed in (_safe_positive_int(size) for size in target_physical_batch_sizes)
        if parsed is not None
    ]
    if not physical_targets:
        physical_targets = [2, 4, 8]
    batch_targets = [
        {
            "physical_batch_size": request.physical_batch_size,
            "gradient_accumulation_steps": request.gradient_accumulation_steps,
            "data_parallel_world_size": request.data_parallel_world_size,
            "effective_batch_size": request.effective_batch_size,
        }
        for request in (
            normalize_multi_batch_request(train_batch_size=size, gradient_accumulation_steps=1)
            for size in physical_targets
        )
    ]
    execution_readiness = build_lulynx_training_pipeline_execution_readiness(runtime_features=state)
    multi_batch_execution_strategy = _multi_batch_execution_strategy(state)
    multi_batch_execution_strategy_gate = _multi_batch_execution_strategy_gate(state)
    batch1_parity_smoke = _batch1_parity_smoke(state)
    real_gpu_batch1_golden_evidence = _real_gpu_batch1_golden_evidence(state)
    orchestrator_slice = build_lulynx_training_step_orchestrator_slice(
        runtime_features=state,
        execution_readiness=execution_readiness,
        internal_gate_enabled=False,
        internal_gate_requested=_safe_bool(state.get("training_step_orchestrator_internal_gate_enabled")),
    )
    actions = _planned_actions(
        include_training_loop_extraction=include_extraction,
        execution_readiness=execution_readiness,
        orchestrator_slice=orchestrator_slice,
        multi_batch_execution_strategy=multi_batch_execution_strategy,
        multi_batch_execution_strategy_gate=multi_batch_execution_strategy_gate,
        batch1_parity_smoke=batch1_parity_smoke,
        real_gpu_batch1_golden_evidence=real_gpu_batch1_golden_evidence,
    )
    for index, action in enumerate(actions, start=1):
        action["run_order"] = index

    blockers = _string_list(state.get("known_blockers"))
    if state.get("batch2_long_window_failed") is True:
        blockers.append("batch2_long_window_cuda_backward_failure")
    if state.get("compile_release_gate_passed") is False:
        blockers.append("compile_gain_below_release_gate")
    for blocker in _string_list(execution_readiness.get("blockers")):
        blockers.append(f"pipeline_execution_readiness:{blocker}")
    blockers.extend(_multi_batch_promotion_gate_blockers(state))
    if not _safe_bool(batch1_parity_smoke.get("passed")):
        blockers.append("batch1_parity_smoke_missing")

    return {
        "schema_version": 1,
        "report": LULYNX_TRAINING_PIPELINE_REFACTOR_PLAN_REPORT,
        "status": "ready_non_gpu_planning",
        "brand": "Lulynx",
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "does_not_start_gpu_work": True,
        "does_not_add_training_entrypoint": True,
        "agpl_risk_policy": "behavior_contract_only_reimplemented_in_house",
        "stage_ids": lulynx_training_stage_ids(),
        "source_contract": pipeline_contract["contract"],
        "source_roadmap_item": roadmap_item["id"],
        "pipeline_execution_readiness": execution_readiness,
        "training_step_orchestrator_slice": orchestrator_slice,
        "batch1_parity_smoke": batch1_parity_smoke,
        "real_gpu_batch1_golden_evidence": real_gpu_batch1_golden_evidence,
        "multi_batch_execution_strategy": multi_batch_execution_strategy,
        "multi_batch_execution_strategy_gate": multi_batch_execution_strategy_gate,
        "target_physical_batch_sizes": physical_targets,
        "batch_targets": batch_targets,
        "promotion_order": [
            "pipeline_contract",
            "dataloader_batch_contract",
            "train_step_stage_extraction",
            "plugin_hook_gate",
            "execution_strategy_switch",
            "batch2_4_8_long_window_evidence",
        ],
        "multi_batch_promotion_blockers": sorted(set(blockers)),
        "actions": actions,
        "notes": [
            "Reference projects are used only as behavior observations; this plan keeps Lulynx-owned names and contracts.",
            "Native batch promotion remains blocked until the long-window CUDA/backward failures are explained or fixed.",
            "Gradient accumulation can raise effective batch size but is not treated as a physical GPU-utilization fix.",
        ],
    }


__all__ = [
    "LULYNX_TRAINING_PIPELINE_REFACTOR_PLAN_REPORT",
    "build_lulynx_training_pipeline_refactor_plan",
]
