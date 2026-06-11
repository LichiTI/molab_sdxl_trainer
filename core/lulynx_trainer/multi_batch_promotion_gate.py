"""Promotion gate for Lulynx native multi-batch long-window probes.

The gate is diagnostic only. It decides whether current runtime evidence is
strong enough to start a batch2/4/8 long-window stability probe; it never marks
performance claims as releasable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .multi_batch_contract import recommend_execution_strategy


LULYNX_MULTI_BATCH_PROMOTION_GATE = "lulynx_multi_batch_promotion_gate_v0"

_REQUIRED_STAGE_IDS = ("batch_contract", "host_to_device", "conditioning", "noise_timestep", "forward", "loss", "backward")
_REQUIRED_STAGE_PLANS = (
    "batch_stage_plan",
    "conditioning_stage_plan",
    "noise_timestep_stage_plan",
    "forward_stage_plan",
    "loss_stage_plan",
)


def build_lulynx_multi_batch_promotion_gate(
    *,
    training_pipeline_trace: Mapping[str, Any] | None,
    multi_batch_dataloader: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    trace = dict(training_pipeline_trace or {}) if isinstance(training_pipeline_trace, Mapping) else {}
    dataloader = dict(multi_batch_dataloader or {}) if isinstance(multi_batch_dataloader, Mapping) else {}
    batch_contract = dict(trace.get("batch_contract") or {}) if isinstance(trace.get("batch_contract"), Mapping) else {}
    stage_plans = dict(trace.get("stage_plans") or {}) if isinstance(trace.get("stage_plans"), Mapping) else {}
    stage_ids = _string_list(trace.get("stage_ids"))
    blockers: list[str] = []
    cautions: list[str] = []

    status = str(trace.get("status") or "")
    if not trace:
        blockers.append("missing_training_pipeline_trace")
    elif status != "completed":
        blockers.append("training_pipeline_trace_not_completed")
    if str(trace.get("error") or ""):
        blockers.append("training_pipeline_trace_has_error")

    if not batch_contract:
        blockers.append("missing_batch_contract")
    elif not bool(batch_contract.get("ok")):
        blockers.append("batch_contract_not_ok")
    if not bool(batch_contract.get("real_multi_batch")):
        blockers.append("not_real_physical_multi_batch")

    missing_stages = [stage for stage in _REQUIRED_STAGE_IDS if stage not in stage_ids]
    if missing_stages:
        blockers.append("missing_required_training_stages")

    missing_plans = [name for name in _REQUIRED_STAGE_PLANS if name not in stage_plans]
    if missing_plans:
        blockers.append("missing_required_stage_plans")

    dataloader_status = "missing"
    if dataloader:
        dataloader_status = "ok" if bool(dataloader.get("ok")) else "blocked"
        if not bool(dataloader.get("ok")):
            blockers.append("multi_batch_dataloader_contract_not_ok")
        if _as_int(dataloader.get("physical_batch_size")) > 1 and not bool(dataloader.get("drop_last")):
            blockers.append("multi_batch_dataloader_tail_batch_not_dropped")
        inferred = _as_int(batch_contract.get("inferred_physical_batch_size"), 0)
        requested = _as_int(dataloader.get("physical_batch_size"), 0)
        if inferred > 0 and requested > 0 and inferred != requested:
            blockers.append("dataloader_physical_batch_differs_from_observed_batch")
    else:
        cautions.append("multi_batch_dataloader_contract_missing_from_runtime_profile")

    for key, plan in stage_plans.items():
        if not isinstance(plan, Mapping):
            continue
        if bool(plan.get("compile_static_graph_risk")):
            cautions.append(f"{key}_reports_compile_static_graph_risk")
        for reason in _string_list(plan.get("compile_caution_reasons")):
            cautions.append(f"{key}:{reason}")

    observed_strategy = (
        trace.get("multi_batch_execution_strategy")
        if isinstance(trace.get("multi_batch_execution_strategy"), Mapping)
        else None
    )
    strategy_gate = (
        dict(trace.get("multi_batch_execution_strategy_gate"))
        if isinstance(trace.get("multi_batch_execution_strategy_gate"), Mapping)
        else {}
    )
    strategy = (
        dict(observed_strategy)
        if isinstance(observed_strategy, Mapping) and observed_strategy
        else recommend_execution_strategy(batch_contract=batch_contract)
        if batch_contract
        else {}
    )
    ready = not blockers
    return {
        "schema_version": 1,
        "gate": LULYNX_MULTI_BATCH_PROMOTION_GATE,
        "status": "ready_for_long_window_probe" if ready else "blocked",
        "ready_for_long_window_probe": ready,
        "release_claim_allowed": False,
        "candidate_physical_batch_size": _as_int(batch_contract.get("inferred_physical_batch_size"), 0),
        "dataloader_contract_status": dataloader_status,
        "execution_strategy": dict(strategy) if isinstance(strategy, Mapping) else {},
        "execution_strategy_gate": strategy_gate,
        "required_stage_ids": list(_REQUIRED_STAGE_IDS),
        "observed_stage_ids": stage_ids,
        "missing_required_stage_ids": missing_stages,
        "required_stage_plans": list(_REQUIRED_STAGE_PLANS),
        "observed_stage_plans": sorted(str(key) for key in stage_plans.keys()),
        "missing_required_stage_plans": missing_plans,
        "blockers": _dedupe(blockers),
        "cautions": _dedupe(cautions),
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers, cautions=cautions),
    }


def _recommended_next_actions(*, ready: bool, blockers: list[str], cautions: list[str]) -> list[str]:
    if ready:
        actions = [
            "run_batch2_4_8_long_window_stability_matrix",
            "record_throughput_loss_vram_active_gpu_and_failure_stage",
            "keep_release_claim_blocked_until_long_window_matrix_passes",
        ]
        if cautions:
            actions.append("run_compile_randomization_or_static_shape_probe_separately")
        return actions
    actions = ["fix_promotion_gate_blockers_before_long_window_probe"]
    if "missing_required_stage_plans" in blockers or "missing_required_training_stages" in blockers:
        actions.append("complete_training_pipeline_stage_trace_extraction")
    if "multi_batch_dataloader_contract_not_ok" in blockers or "multi_batch_dataloader_tail_batch_not_dropped" in blockers:
        actions.append("fix_bucket_batch_sampler_drop_last_contract")
    if "batch_contract_not_ok" in blockers or "not_real_physical_multi_batch" in blockers:
        actions.append("inspect_first_real_batch_contract")
    return actions


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if item is not None]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


__all__ = [
    "LULYNX_MULTI_BATCH_PROMOTION_GATE",
    "build_lulynx_multi_batch_promotion_gate",
]
