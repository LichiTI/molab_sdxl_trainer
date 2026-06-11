"""Report-only batch gate for AdamW variant TurboCore native evidence.

This module aggregates existing AdamW variant scorecards without running large
training jobs or changing request/runtime behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from core.configs import OptimizerType


TARGET_OPTIMIZERS: tuple[OptimizerType, ...] = (
    OptimizerType.ADAMW_8BIT,
    OptimizerType.PAGED_ADAMW,
    OptimizerType.PAGED_ADAMW_32BIT,
    OptimizerType.PAGED_ADAMW_8BIT,
    OptimizerType.KAHAN_ADAMW_8BIT,
    OptimizerType.ADAMW_SCHEDULE_FREE,
)

VARIANT_STATE_STAGE = "variant_state"
SCHEDULE_FREE_STAGE = "schedule_free_state_machine"
SCHEDULE_FREE_ABI_STAGE = "schedule_free_native_abi"
SCHEDULE_FREE_SCRATCH_STAGE = "schedule_free_scratch_formula_canary"
SCHEDULE_FREE_NATIVE_SCRATCH_STAGE = "schedule_free_native_scratch_kernel"
SCHEDULE_FREE_RUNTIME_STAGE = "schedule_free_runtime_canary"
SCHEDULE_FREE_LOOP_STAGE = "schedule_free_training_loop_canary"
ADAMW_8BIT_RUNTIME_STAGE = "adamw8bit_runtime_canary"
ADAMW_8BIT_LOOP_STAGE = "adamw8bit_training_loop_canary"
PAGED_32_RUNTIME_STAGE = "paged_adamw32_runtime_canary"
PAGED_32_LOOP_STAGE = "paged_adamw32_training_loop_canary"
PAGED_8BIT_RUNTIME_STAGE = "paged_adamw8bit_runtime_canary"
PAGED_8BIT_LOOP_STAGE = "paged_adamw8bit_training_loop_canary"
KAHAN_8BIT_RUNTIME_STAGE = "kahan_adamw8bit_runtime_canary"
KAHAN_8BIT_LOOP_STAGE = "kahan_adamw8bit_training_loop_canary"
PLUGIN_SCHEDULE_FREE_STAGE = "plugin_schedulefree_runtime_dispatch_shadow"

UNSAFE_TRUE_FIELDS = (
    "default_behavior_changed",
    "training_path_enabled",
    "native_dispatch_allowed",
    "runtime_dispatch_ready",
    "product_exposure_allowed",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "ui_exposure_allowed",
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_adamw_variant_family_batch_scorecard(
    *,
    stage_reports: Mapping[str, Any] | None = None,
    include_live_training_loop_canaries: bool = False,
) -> dict[str, Any]:
    """Aggregate AdamW variant evidence while keeping native dispatch closed."""

    stages = _stage_reports(
        stage_reports,
        include_live_training_loop_canaries=include_live_training_loop_canaries,
    )
    rows = [_row(optimizer, stages) for optimizer in TARGET_OPTIMIZERS]
    e2e = _artifact_report("turbocore_adamw_variant_e2e_shadow_matrix_scorecard.json")
    rollout = _artifact_report("turbocore_adamw_variant_canary_rollout_policy_scorecard.json")
    review = _artifact_report("turbocore_adamw_variant_dispatch_integration_review_scorecard.json")
    unsafe = _unsafe_claims(stages)
    native_canary_stage_ready_count = sum(1 for row in rows if row["native_ready"] is True)
    canary_manifest_count = sum(1 for row in rows if row["native_canary_manifest_present"] is True)
    pending = [row["optimizer_type"] for row in rows if row["batch_status"] == "pending"]
    blockers = _dedupe(
        unsafe
        + [reason for row in rows for reason in row["blocked_reasons"]]
        + ["product_rollout_review_missing"]
    )
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adamw_variant_family_batch_scorecard_v0",
        "gate": "adamw_variant_family_batch_native_canary",
        "ok": not unsafe and len(rows) == len(TARGET_OPTIMIZERS),
        "promotion_ready": False,
        "report_only": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "product_native_dispatch_ready": False,
        "native_ready": False,
        "target_optimizer_types": [optimizer.value for optimizer in TARGET_OPTIMIZERS],
        "rows": rows,
        "stage_summaries": {name: _compact_report(report) for name, report in stages.items()},
        "summary": {
            "target_count": len(TARGET_OPTIMIZERS),
            "native_ready_count": native_canary_stage_ready_count,
            "native_canary_stage_evidence_ready_count": native_canary_stage_ready_count,
            "product_native_ready_count": 0,
            "state_reference_ready_count": sum(1 for row in rows if row["state_reference_ready"] is True),
            "native_canary_manifest_count": canary_manifest_count,
            "native_canary_manifest_ready_count": sum(
                1 for row in rows if row["native_canary_manifest_ready"] is True
            ),
            "training_loop_canary_ready_count": sum(
                1 for row in rows if row["training_loop_canary_ready"] is True
            ),
            "schedule_free_native_abi_ready_count": _stage_ready_count(rows, SCHEDULE_FREE_ABI_STAGE),
            "schedule_free_scratch_formula_canary_ready_count": _stage_ready_count(
                rows,
                SCHEDULE_FREE_SCRATCH_STAGE,
            ),
            "schedule_free_native_scratch_kernel_ready_count": _stage_ready_count(
                rows,
                SCHEDULE_FREE_NATIVE_SCRATCH_STAGE,
            ),
            "schedule_free_runtime_canary_manifest_ready_count": _stage_ready_count(
                rows,
                SCHEDULE_FREE_RUNTIME_STAGE,
            ),
            "schedule_free_training_loop_canary_manifest_ready_count": _stage_ready_count(
                rows,
                SCHEDULE_FREE_LOOP_STAGE,
            ),
            "pending_count": len(pending),
            "unsafe_claim_count": len(unsafe),
            "training_path_enabled_count": sum(1 for row in rows if row["training_path_enabled"] is True),
            "native_dispatch_allowed_count": sum(1 for row in rows if row["native_dispatch_allowed"] is True),
            "exact_adamw_included": False,
            "e2e_shadow_matrix_ready": e2e.get("e2e_shadow_matrix_ready") is True,
            "canary_rollout_policy_ready": rollout.get("canary_rollout_policy_ready") is True,
            "dispatch_integration_review_ready": review.get("review_gate_ready") is True,
        },
        "e2e_shadow_matrix": _compact_gate(e2e, ready_field="e2e_shadow_matrix_ready"),
        "canary_rollout_policy": _compact_gate(rollout, ready_field="canary_rollout_policy_ready"),
        "dispatch_integration_review": _compact_gate(review, ready_field="review_gate_ready"),
        "native_ready_count_policy": {
            "counts_exact_adamw": False,
            "counts_layout_only": False,
            "counts_manifest_only": False,
            "counts_product_default_dispatch": False,
            "requires_training_loop_native_canary": True,
            "note": "native_ready_count is family-local canary evidence; product native readiness remains product_native_ready_count=0.",
        },
        "promotion_blockers": blockers,
        "blocked_reasons": _dedupe(unsafe),
        "recommended_next_step": _recommended_next_step(canary_manifest_count, e2e, rollout, review),
        "notes": [
            "This batch gate covers AdamW variants only and deliberately excludes exact AdamW.",
            "State/layout-only rows are pending and do not increment family-local native canary evidence.",
            "No request, UI, schema, training_path, or native dispatch behavior is changed.",
        ],
    }


def _stage_reports(
    stage_reports: Mapping[str, Any] | None,
    *,
    include_live_training_loop_canaries: bool,
) -> dict[str, dict[str, Any]]:
    if stage_reports is not None:
        return {str(name): _as_dict(report) for name, report in stage_reports.items()}

    stages: dict[str, dict[str, Any]] = {}
    _try_stage(
        stages,
        VARIANT_STATE_STAGE,
        lambda: _build_variant_state_report(),
    )
    _try_stage(
        stages,
        SCHEDULE_FREE_STAGE,
        lambda: _call("core.turbocore_adamw_schedule_free_state_machine_scorecard", "build_adamw_schedule_free_state_machine_scorecard"),
    )
    _try_stage(
        stages,
        SCHEDULE_FREE_ABI_STAGE,
        lambda: _call("core.turbocore_adamw_schedule_free_native_abi_scorecard", "build_adamw_schedule_free_native_abi_scorecard"),
    )
    _try_stage(
        stages,
        SCHEDULE_FREE_SCRATCH_STAGE,
        lambda: _call("core.turbocore_adamw_schedule_free_scratch_canary_scorecard", "build_adamw_schedule_free_scratch_canary_scorecard"),
    )
    _try_stage(
        stages,
        SCHEDULE_FREE_NATIVE_SCRATCH_STAGE,
        lambda: _call("core.turbocore_adamw_schedule_free_native_scratch_kernel_scorecard", "build_adamw_schedule_free_native_scratch_kernel_scorecard"),
    )
    _try_stage(
        stages,
        SCHEDULE_FREE_RUNTIME_STAGE,
        lambda: _call("core.turbocore_adamw_schedule_free_runtime_canary_scorecard", "build_adamw_schedule_free_runtime_canary_scorecard"),
    )
    _try_stage(
        stages,
        ADAMW_8BIT_RUNTIME_STAGE,
        lambda: _call("core.turbocore_adamw8bit_runtime_canary_scorecard", "build_adamw8bit_runtime_canary_scorecard"),
    )
    _try_stage(
        stages,
        PAGED_32_RUNTIME_STAGE,
        lambda: _call("core.turbocore_paged_adamw32_runtime_canary_scorecard", "build_paged_adamw32_runtime_canary_scorecard"),
    )
    _try_stage(
        stages,
        PAGED_8BIT_RUNTIME_STAGE,
        lambda: _call("core.turbocore_paged_adamw8bit_runtime_canary_scorecard", "build_paged_adamw8bit_runtime_canary_scorecard"),
    )
    _try_stage(
        stages,
        KAHAN_8BIT_RUNTIME_STAGE,
        lambda: _call("core.turbocore_kahan_adamw8bit_runtime_canary_scorecard", "build_kahan_adamw8bit_runtime_canary_scorecard"),
    )
    _try_stage(
        stages,
        PLUGIN_SCHEDULE_FREE_STAGE,
        lambda: _call("core.turbocore_plugin_schedulefree_runtime_dispatch_shadow_scorecard", "build_plugin_schedulefree_runtime_dispatch_shadow_scorecard"),
    )
    if include_live_training_loop_canaries:
        _try_stage(
            stages,
            SCHEDULE_FREE_LOOP_STAGE,
            lambda: _call("core.turbocore_adamw_schedule_free_training_loop_canary_scorecard", "build_adamw_schedule_free_training_loop_canary_scorecard"),
        )
        _try_stage(
            stages,
            ADAMW_8BIT_LOOP_STAGE,
            lambda: _call("core.turbocore_adamw8bit_training_loop_canary_scorecard", "build_adamw8bit_training_loop_canary_scorecard"),
        )
        _try_stage(
            stages,
            PAGED_32_LOOP_STAGE,
            lambda: _call("core.turbocore_paged_adamw32_training_loop_canary_scorecard", "build_paged_adamw32_training_loop_canary_scorecard"),
        )
        _try_stage(
            stages,
            PAGED_8BIT_LOOP_STAGE,
            lambda: _call("core.turbocore_paged_adamw8bit_training_loop_canary_scorecard", "build_paged_adamw8bit_training_loop_canary_scorecard"),
        )
        _try_stage(
            stages,
            KAHAN_8BIT_LOOP_STAGE,
            lambda: _call("core.turbocore_kahan_adamw8bit_training_loop_canary_scorecard", "build_kahan_adamw8bit_training_loop_canary_scorecard"),
        )
    return stages


def _stage_ready_count(rows: list[Mapping[str, Any]], stage: str) -> int:
    return sum(1 for row in rows if _as_dict(row.get("stage_ready")).get(stage) is True)


def _build_variant_state_report() -> dict[str, Any]:
    from core.turbocore_adamw_variant_state_scorecard import build_adamw_variant_state_scorecard

    return build_adamw_variant_state_scorecard(run_cuda_optional=False)


def _call(module_name: str, builder_name: str) -> dict[str, Any]:
    module = __import__(module_name, fromlist=[builder_name])
    builder: Callable[[], dict[str, Any]] = getattr(module, builder_name)
    return builder()


def _try_stage(stages: dict[str, dict[str, Any]], name: str, builder: Callable[[], dict[str, Any]]) -> None:
    try:
        stages[name] = _as_dict(builder())
    except Exception as exc:
        stages[name] = {
            "schema_version": 1,
            "scorecard": "",
            "gate": name,
            "ok": False,
            "pending": True,
            "blocked_reasons": [f"{name}_evidence_unavailable:{type(exc).__name__}"],
            "error": f"{type(exc).__name__}: {exc}",
            "training_path_enabled": False,
            "default_behavior_changed": False,
            "native_dispatch_allowed": False,
        }


def _row(optimizer: OptimizerType, stages: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if optimizer == OptimizerType.ADAMW_8BIT:
        return _native_variant_row(
            optimizer,
            "adamw_quantized",
            stages,
            runtime_stage=ADAMW_8BIT_RUNTIME_STAGE,
            loop_stage=ADAMW_8BIT_LOOP_STAGE,
            next_gate="adamw8bit_cuda_training_loop_canary",
        )
    if optimizer == OptimizerType.PAGED_ADAMW:
        return _paged_adamw32_row(optimizer, "paged_adamw", stages)
    if optimizer == OptimizerType.PAGED_ADAMW_32BIT:
        return _paged_adamw32_row(optimizer, "paged_adamw32bit", stages)
    if optimizer == OptimizerType.PAGED_ADAMW_8BIT:
        return _native_variant_row(
            optimizer,
            "adamw_quantized_paged",
            stages,
            runtime_stage=PAGED_8BIT_RUNTIME_STAGE,
            loop_stage=PAGED_8BIT_LOOP_STAGE,
            next_gate="paged_adamw8bit_cuda_training_loop_canary",
        )
    if optimizer == OptimizerType.KAHAN_ADAMW_8BIT:
        return _native_variant_row(
            optimizer,
            "adamw_quantized_kahan",
            stages,
            runtime_stage=KAHAN_8BIT_RUNTIME_STAGE,
            loop_stage=KAHAN_8BIT_LOOP_STAGE,
            next_gate="kahan_adamw8bit_cuda_training_loop_canary",
        )
    if optimizer == OptimizerType.ADAMW_SCHEDULE_FREE:
        return _schedule_free_row(optimizer, stages)
    return _layout_pending_row(optimizer, stages)


def _paged_adamw32_row(
    optimizer: OptimizerType,
    optimizer_kind: str,
    stages: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    runtime = stages.get(PAGED_32_RUNTIME_STAGE, {})
    loop = stages.get(PAGED_32_LOOP_STAGE, {})
    manifest_present = bool(runtime)
    manifest_ready = bool(runtime.get("runtime_canary_manifest_ready", False))
    loop_ready = _loop_case_ready(loop, optimizer_kind)
    native_ready = manifest_ready and loop_ready
    reasons = []
    if not manifest_present:
        reasons.append("paged_adamw32_runtime_canary_missing")
    elif not manifest_ready:
        reasons.extend(_strings(runtime.get("blocked_reasons")) or ["paged_adamw32_runtime_canary_not_ready"])
    if not loop:
        reasons.append("paged_adamw32_training_loop_canary_not_run")
    elif not loop_ready:
        reasons.extend(_case_blockers(loop, optimizer_kind) or [f"{optimizer_kind}_training_loop_canary_not_ready"])
    return _base_row(
        optimizer,
        "adamw_paged",
        "native_canary_ready" if native_ready else "pending",
        stages,
        evidence_stages=[PAGED_32_RUNTIME_STAGE, PAGED_32_LOOP_STAGE],
        state_reference_ready=_variant_state_ready(optimizer, stages),
        native_canary_manifest_present=manifest_present,
        native_canary_manifest_ready=manifest_ready,
        training_loop_canary_ready=loop_ready,
        native_ready=native_ready,
        next_gate=f"{optimizer_kind}_product_canary_review",
        blocked_reasons=reasons,
    )


def _native_variant_row(
    optimizer: OptimizerType,
    family: str,
    stages: Mapping[str, Mapping[str, Any]],
    *,
    runtime_stage: str,
    loop_stage: str,
    next_gate: str,
) -> dict[str, Any]:
    runtime = stages.get(runtime_stage, {})
    loop = stages.get(loop_stage, {})
    manifest_present = bool(runtime)
    manifest_ready = bool(runtime.get("runtime_canary_manifest_ready", False))
    loop_present = bool(loop)
    loop_ready = bool(loop.get("ok", False)) and _native_loop_executed(loop)
    native_ready = manifest_ready and loop_ready
    status = "native_canary_ready" if native_ready else "pending"
    reasons = []
    if not manifest_present:
        reasons.append(f"{runtime_stage}_missing")
    elif not manifest_ready:
        reasons.extend(_strings(runtime.get("blocked_reasons")) or [f"{runtime_stage}_not_ready"])
    if not loop_present:
        reasons.append(f"{loop_stage}_not_run")
    elif not loop_ready:
        reasons.extend(_strings(loop.get("blocked_reasons")) or [f"{loop_stage}_not_ready"])
    return _base_row(
        optimizer,
        family,
        status,
        stages,
        evidence_stages=[runtime_stage, loop_stage],
        state_reference_ready=_variant_state_ready(optimizer, stages),
        native_canary_manifest_present=manifest_present,
        native_canary_manifest_ready=manifest_ready,
        training_loop_canary_ready=loop_ready,
        native_ready=native_ready,
        next_gate=next_gate,
        blocked_reasons=reasons,
    )


def _schedule_free_row(optimizer: OptimizerType, stages: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    state = stages.get(SCHEDULE_FREE_STAGE, {})
    abi = stages.get(SCHEDULE_FREE_ABI_STAGE, {})
    scratch = stages.get(SCHEDULE_FREE_SCRATCH_STAGE, {})
    native_scratch = stages.get(SCHEDULE_FREE_NATIVE_SCRATCH_STAGE, {})
    runtime = stages.get(SCHEDULE_FREE_RUNTIME_STAGE, {})
    loop = stages.get(SCHEDULE_FREE_LOOP_STAGE, {})
    plugin_shadow = stages.get(PLUGIN_SCHEDULE_FREE_STAGE, {})
    state_ready = bool(state.get("state_machine_reference_ready", False))
    abi_ready = bool(abi.get("abi_contract_ready", False))
    scratch_ready = bool(scratch.get("scratch_formula_canary_ready", False))
    native_scratch_ready = bool(native_scratch.get("native_scratch_kernel_parity_ready", False))
    runtime_manifest_ready = bool(runtime.get("runtime_canary_manifest_ready", False))
    loop_manifest_ready = bool(loop.get("training_loop_canary_manifest_ready", False))
    loop_ready = bool(loop.get("training_loop_canary_ready", False)) and _native_loop_executed(loop)
    shadow_ready = bool(plugin_shadow.get("runtime_dispatch_shadow_ready", False))
    reasons = []
    if not state:
        reasons.append("adamw_schedule_free_state_machine_missing")
    elif not state_ready:
        reasons.extend(_strings(state.get("blocked_reasons")) or ["adamw_schedule_free_state_machine_not_ready"])
    if not plugin_shadow:
        reasons.append("plugin_schedulefree_runtime_dispatch_shadow_missing")
    elif not shadow_ready:
        reasons.extend(_strings(plugin_shadow.get("blocked_reasons")) or ["plugin_schedulefree_shadow_not_ready"])
    if not abi:
        reasons.append("adamw_schedule_free_native_abi_missing")
    elif not abi_ready:
        reasons.extend(_strings(abi.get("blocked_reasons")) or ["adamw_schedule_free_native_abi_not_ready"])
    if not scratch:
        reasons.append("adamw_schedule_free_scratch_canary_missing")
    elif not scratch_ready:
        reasons.extend(_strings(scratch.get("blocked_reasons")) or ["adamw_schedule_free_scratch_canary_not_ready"])
    if not native_scratch:
        reasons.append("adamw_schedule_free_native_scratch_kernel_missing")
    elif not native_scratch_ready:
        reasons.extend(
            _strings(native_scratch.get("blocked_reasons"))
            or ["adamw_schedule_free_native_scratch_kernel_not_ready"]
        )
    if not runtime:
        reasons.append("adamw_schedule_free_runtime_canary_manifest_missing")
    elif not runtime_manifest_ready:
        reasons.extend(
            _strings(runtime.get("blocked_reasons"))
            or ["adamw_schedule_free_runtime_canary_manifest_not_ready"]
        )
    if not loop:
        reasons.append("adamw_schedule_free_training_loop_canary_manifest_missing")
    elif not loop_manifest_ready:
        reasons.extend(
            _strings(loop.get("blocked_reasons"))
            or ["adamw_schedule_free_training_loop_canary_manifest_not_ready"]
        )
    if not loop_ready:
        reasons.append("adamw_schedule_free_training_loop_native_dispatch_missing")
    status = (
        "native_canary_ready"
        if (
            state_ready
            and abi_ready
            and shadow_ready
            and scratch_ready
            and native_scratch_ready
            and runtime_manifest_ready
            and loop_ready
        )
        else
        "training_loop_canary_manifest_ready_dispatch_pending"
        if (
            state_ready
            and abi_ready
            and shadow_ready
            and scratch_ready
            and native_scratch_ready
            and runtime_manifest_ready
            and loop_manifest_ready
        )
        else
        "runtime_canary_manifest_ready_training_loop_pending"
        if state_ready and abi_ready and shadow_ready and scratch_ready and native_scratch_ready and runtime_manifest_ready
        else
        "native_scratch_kernel_ready_runtime_pending"
        if state_ready and abi_ready and shadow_ready and scratch_ready and native_scratch_ready
        else "scratch_formula_canary_ready_kernel_pending"
        if state_ready and abi_ready and shadow_ready and scratch_ready
        else ("native_abi_ready_kernel_pending" if state_ready and abi_ready and shadow_ready else "pending")
    )
    return _base_row(
        optimizer,
        "adamw_schedule_free",
        status,
        stages,
        evidence_stages=[
            SCHEDULE_FREE_STAGE,
            SCHEDULE_FREE_ABI_STAGE,
            SCHEDULE_FREE_SCRATCH_STAGE,
            SCHEDULE_FREE_NATIVE_SCRATCH_STAGE,
            SCHEDULE_FREE_RUNTIME_STAGE,
            SCHEDULE_FREE_LOOP_STAGE,
            PLUGIN_SCHEDULE_FREE_STAGE,
        ],
        state_reference_ready=state_ready,
        native_canary_manifest_present=runtime_manifest_ready,
        native_canary_manifest_ready=runtime_manifest_ready,
        training_loop_canary_ready=loop_ready,
        native_ready=loop_ready,
        next_gate="adamw_schedule_free_e2e_shadow_matrix_and_rollout_policy",
        blocked_reasons=reasons,
    )


def _layout_pending_row(optimizer: OptimizerType, stages: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    state_ready = _variant_state_ready(optimizer, stages)
    reasons = [] if state_ready else ["adamw_variant_state_layout_missing"]
    reasons.append("variant_native_canary_missing")
    return _base_row(
        optimizer,
        _family(optimizer),
        "pending",
        stages,
        evidence_stages=[VARIANT_STATE_STAGE],
        state_reference_ready=state_ready,
        native_canary_manifest_present=False,
        native_canary_manifest_ready=False,
        training_loop_canary_ready=False,
        native_ready=False,
        next_gate="variant_specific_native_canary_manifest",
        blocked_reasons=reasons,
    )


def _base_row(
    optimizer: OptimizerType,
    family: str,
    status: str,
    stages: Mapping[str, Mapping[str, Any]],
    *,
    evidence_stages: list[str],
    state_reference_ready: bool,
    native_canary_manifest_present: bool,
    native_canary_manifest_ready: bool,
    training_loop_canary_ready: bool,
    native_ready: bool,
    next_gate: str,
    blocked_reasons: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "optimizer_type": optimizer.value,
        "optimizer_family": family,
        "batch_status": status,
        "state_reference_ready": state_reference_ready,
        "native_canary_manifest_present": native_canary_manifest_present,
        "native_canary_manifest_ready": native_canary_manifest_ready,
        "training_loop_canary_ready": training_loop_canary_ready,
        "native_ready": native_ready,
        "product_native_dispatch_ready": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "runtime_authority": "existing_python_or_third_party_optimizer",
        "evidence_stages": evidence_stages,
        "stage_ready": {stage: _stage_ready(stage, stages.get(stage, {})) for stage in evidence_stages},
        "blocked_reasons": _dedupe(blocked_reasons),
        "next_gate": next_gate,
    }


def _variant_state_ready(optimizer: OptimizerType, stages: Mapping[str, Mapping[str, Any]]) -> bool:
    report = stages.get(VARIANT_STATE_STAGE, {})
    for row in report.get("rows", []) if isinstance(report.get("rows"), list) else []:
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer.value:
            return row.get("state_layout_status") == "layout_reference_ready"
    return False


def _native_loop_executed(report: Mapping[str, Any]) -> bool:
    summary = report.get("summary", {}) if isinstance(report.get("summary"), Mapping) else {}
    return bool(
        report.get("ok", False)
        and int(summary.get("native_step_count") or 0) > 0
        and int(summary.get("native_kernel_launch_count") or 0) > 0
    )


def _stage_ready(stage: str, report: Mapping[str, Any]) -> bool:
    if not report or not _default_off(report):
        return False
    ready_fields = {
        VARIANT_STATE_STAGE: "state_layout_stage_ready",
        SCHEDULE_FREE_STAGE: "state_machine_reference_ready",
        SCHEDULE_FREE_ABI_STAGE: "abi_contract_ready",
        SCHEDULE_FREE_SCRATCH_STAGE: "scratch_formula_canary_ready",
        SCHEDULE_FREE_NATIVE_SCRATCH_STAGE: "native_scratch_kernel_parity_ready",
        SCHEDULE_FREE_RUNTIME_STAGE: "runtime_canary_manifest_ready",
        SCHEDULE_FREE_LOOP_STAGE: "training_loop_canary_manifest_ready",
        ADAMW_8BIT_RUNTIME_STAGE: "runtime_canary_manifest_ready",
        ADAMW_8BIT_LOOP_STAGE: "ok",
        PAGED_32_RUNTIME_STAGE: "runtime_canary_manifest_ready",
        PAGED_32_LOOP_STAGE: "ok",
        PAGED_8BIT_RUNTIME_STAGE: "runtime_canary_manifest_ready",
        PAGED_8BIT_LOOP_STAGE: "ok",
        KAHAN_8BIT_RUNTIME_STAGE: "runtime_canary_manifest_ready",
        KAHAN_8BIT_LOOP_STAGE: "ok",
        PLUGIN_SCHEDULE_FREE_STAGE: "runtime_dispatch_shadow_ready",
    }
    field = ready_fields.get(stage)
    return bool(field and report.get(field) is True)


def _loop_case_ready(report: Mapping[str, Any], optimizer_kind: str) -> bool:
    for case in report.get("cases", []) if isinstance(report.get("cases"), list) else []:
        if isinstance(case, Mapping) and case.get("optimizer_kind") == optimizer_kind:
            return bool(
                case.get("ok", False)
                and case.get("native_step_executed") is True
                and case.get("native_kernel_launched") is True
            )
    return False


def _case_blockers(report: Mapping[str, Any], optimizer_kind: str) -> list[str]:
    for case in report.get("cases", []) if isinstance(report.get("cases"), list) else []:
        if isinstance(case, Mapping) and case.get("optimizer_kind") == optimizer_kind:
            return _strings(case.get("blocked_reasons"))
    return []


def _family(optimizer: OptimizerType) -> str:
    if optimizer == OptimizerType.ADAMW_8BIT:
        return "adamw_quantized"
    if optimizer in {OptimizerType.PAGED_ADAMW, OptimizerType.PAGED_ADAMW_32BIT}:
        return "adamw_paged"
    return "adamw_variant"


def _compact_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(report),
        "scorecard": str(report.get("scorecard") or ""),
        "gate": str(report.get("gate") or ""),
        "ok": report.get("ok") is True,
        "default_off": _default_off(report),
        "summary": _as_dict(report.get("summary")),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_gate(report: Mapping[str, Any], *, ready_field: str) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        ready_field: report.get(ready_field) is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "case_count": int(summary.get("case_count", 0) or 0),
        "manual_review_required": report.get("manual_review_required") is True,
    }


def _artifact_report(filename: str) -> dict[str, Any]:
    path = REPO_ROOT / "temp" / "turbocore_optimizer" / filename
    if not path.exists():
        return {}
    try:
        return _as_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _recommended_next_step(
    canary_manifest_count: int,
    e2e: Mapping[str, Any],
    rollout: Mapping[str, Any],
    review: Mapping[str, Any],
) -> str:
    if not canary_manifest_count:
        return "complete variant-specific native canary manifests before batch promotion"
    if e2e.get("e2e_shadow_matrix_ready") is not True:
        return "add AdamW variant e2e shadow matrix for canary-ready rows"
    if rollout.get("canary_rollout_policy_ready") is not True:
        return "add AdamW variant default-off canary rollout policy"
    if review.get("review_gate_ready") is not True:
        return "prepare AdamW variant owner/release hold package with product dispatch still default-off"
    return "keep AdamW variant canary dispatch unwired until explicit owner approval is recorded"


def _default_off(report: Mapping[str, Any]) -> bool:
    return bool(report) and all(report.get(field) is not True for field in UNSAFE_TRUE_FIELDS)


def _unsafe_claims(stages: Mapping[str, Mapping[str, Any]]) -> list[str]:
    claims: list[str] = []
    for stage, report in stages.items():
        for field in UNSAFE_TRUE_FIELDS:
            if report.get(field) is True:
                claims.append(f"unsafe_stage_claim:{stage}:{field}")
    return claims


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list | tuple | set):
        return []
    return [str(item) for item in value if str(item or "")]


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["TARGET_OPTIMIZERS", "build_adamw_variant_family_batch_scorecard"]
