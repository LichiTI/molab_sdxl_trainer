"""Batch scorecard for simple-formula TurboCore optimizer adaptation.

This module aggregates the existing Lion and SGD Nesterov native evidence into
one family-level status.  It deliberately keeps product dispatch off; the goal
is to make optimizer adaptation progress visible without opening request/UI or
training defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from core.configs import OptimizerType
from core.turbocore_simple_optimizer_abi_scorecard import build_simple_optimizer_abi_scorecard
from core.turbocore_simple_optimizer_dispatch_runtime_scorecard import (
    build_simple_optimizer_dispatch_runtime_scorecard,
)
from core.turbocore_simple_optimizer_e2e_no_regression_scorecard import (
    build_simple_optimizer_e2e_no_regression_scorecard,
)
from core.turbocore_simple_optimizer_kernel_parity_scorecard import (
    build_simple_optimizer_kernel_parity_scorecard,
)
from core.turbocore_simple_optimizer_reference_scorecard import (
    build_simple_optimizer_reference_scorecard,
)
from core.turbocore_simple_optimizer_registry_scorecard import (
    build_simple_optimizer_registry_scorecard,
)
from core.turbocore_simple_optimizer_runtime_canary_scorecard import (
    build_simple_optimizer_runtime_canary_scorecard,
)
from core.turbocore_simple_optimizer_runtime_launch_scorecard import (
    build_simple_optimizer_runtime_launch_scorecard,
)
from core.turbocore_simple_optimizer_training_executor_scorecard import (
    build_simple_optimizer_training_executor_scorecard,
)
from core.turbocore_simple_optimizer_training_loop_canary_scorecard import (
    build_simple_optimizer_training_loop_canary_scorecard,
)
from core.turbocore_simple_optimizer_variant_state_scorecard import (
    build_simple_optimizer_variant_state_scorecard,
)
from core.turbocore_simple_optimizer_variant_native_abi_scorecard import (
    build_simple_optimizer_variant_native_abi_scorecard,
)
from core.turbocore_simple_optimizer_variant_native_canary_scorecard import (
    build_simple_optimizer_variant_native_canary_scorecard,
)
from core.turbocore_simple_optimizer_variant_resume_parity_scorecard import (
    build_simple_optimizer_variant_resume_parity_scorecard,
)


EXACT_BATCH_TARGETS = (OptimizerType.LION, OptimizerType.SGD_NESTEROV)
PENDING_VARIANTS = (
    OptimizerType.LION_8BIT,
    OptimizerType.PAGED_LION_8BIT,
    OptimizerType.SGD_NESTEROV_8BIT,
    OptimizerType.SGD_SCHEDULE_FREE,
    OptimizerType.RADAM_SCHEDULE_FREE,
)
TARGET_KINDS = ("lion", "sgd_nesterov")

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


def build_simple_optimizer_family_batch_scorecard(
    *,
    stage_reports: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate simple optimizer stages into one batch adaptation report."""

    stages = _stage_reports(stage_reports, workspace_root)
    stage_rows = [_stage_row(name, report) for name, report in stages.items()]
    missing = [row["stage"] for row in stage_rows if not row["present"]]
    not_ready = [row["stage"] for row in stage_rows if row["present"] and not row["ready"]]
    unsafe = _unsafe_claims(stages)
    batch_ready = not missing and not not_ready and not unsafe
    variant_state = build_simple_optimizer_variant_state_scorecard()
    variant_abi = build_simple_optimizer_variant_native_abi_scorecard(variant_state_report=variant_state)
    variant_canary = build_simple_optimizer_variant_native_canary_scorecard(native_abi_report=variant_abi)
    variant_resume = build_simple_optimizer_variant_resume_parity_scorecard(
        variant_canary_report=variant_canary,
        variant_state_report=variant_state,
        workspace_root=workspace_root,
    )
    exact_rows = [_exact_target_row(optimizer, stage_rows, batch_ready) for optimizer in EXACT_BATCH_TARGETS]
    pending_rows = [
        _pending_variant_row(optimizer, variant_state, variant_abi, variant_canary, variant_resume)
        for optimizer in PENDING_VARIANTS
    ]
    variant_summary = _as_dict(variant_state.get("summary"))
    variant_abi_summary = _as_dict(variant_abi.get("summary"))
    variant_canary_summary = _as_dict(variant_canary.get("summary"))
    variant_resume_summary = _as_dict(variant_resume.get("summary"))
    blockers = _dedupe(
        [f"simple_optimizer_stage_missing:{stage}" for stage in missing]
        + [f"simple_optimizer_stage_not_ready:{stage}" for stage in not_ready]
        + unsafe
        + [reason for row in stage_rows for reason in row["blocked_reasons"]]
    )
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_family_batch_scorecard_v0",
        "gate": "simple_formula_family_batch_native_canary",
        "ok": not missing and not unsafe,
        "promotion_ready": False,
        "simple_formula_native_batch_canary_ready": batch_ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "product_native_dispatch_ready": False,
        "runtime_dispatch_ready": False,
        "target_optimizer_types": [optimizer.value for optimizer in EXACT_BATCH_TARGETS],
        "target_optimizer_kinds": list(TARGET_KINDS),
        "pending_optimizer_types": [optimizer.value for optimizer in PENDING_VARIANTS],
        "rows": exact_rows + pending_rows,
        "stage_rows": stage_rows,
        "stage_summaries": {name: _compact_stage_report(report) for name, report in stages.items()},
        "variant_state_layout_scorecard": _compact_variant_state_report(variant_state),
        "variant_native_abi_scorecard": _compact_variant_native_abi_report(variant_abi),
        "variant_native_canary_scorecard": _compact_variant_native_canary_report(variant_canary),
        "variant_resume_parity_scorecard": _compact_variant_resume_report(variant_resume),
        "summary": {
            "exact_target_count": len(EXACT_BATCH_TARGETS),
            "batch_canary_ready_count": len(EXACT_BATCH_TARGETS) if batch_ready else 0,
            "product_native_ready_count": 0,
            "pending_variant_count": len(PENDING_VARIANTS),
            "variant_layout_spec_ready_count": int(variant_summary.get("layout_spec_ready_count", 0) or 0),
            "variant_state_machine_reference_ready_count": int(
                variant_summary.get("state_machine_reference_ready_count", 0) or 0
            ),
            "variant_native_abi_spec_ready_count": int(variant_abi_summary.get("native_abi_spec_ready_count", 0) or 0),
            "variant_formula_parity_matrix_artifact_ready_count": int(
                variant_abi_summary.get("formula_parity_matrix_artifact_ready_count", 0) or 0
            ),
            "variant_formula_parity_matrix_implementation_ready_count": int(
                variant_canary_summary.get("quantized_formula_parity_ready_count", 0) or 0
            ),
            "variant_resume_parity_matrix_artifact_ready_count": int(
                variant_abi_summary.get("resume_parity_matrix_artifact_ready_count", 0) or 0
            ),
            "variant_resume_parity_matrix_implementation_ready_count": int(
                variant_resume_summary.get("resume_parity_matrix_implementation_ready_count", 0) or 0
            ),
            "variant_quantized_resume_parity_ready_count": int(
                variant_resume_summary.get("quantized_resume_parity_ready_count", 0) or 0
            ),
            "variant_schedule_free_resume_parity_ready_count": int(
                variant_resume_summary.get("schedule_free_resume_parity_ready_count", 0) or 0
            ),
            "variant_schedule_free_native_canary_ready_count": int(
                variant_canary_summary.get("schedule_free_native_canary_ready_count", 0) or 0
            ),
            "variant_quantized_formula_parity_ready_count": int(
                variant_canary_summary.get("quantized_formula_parity_ready_count", 0) or 0
            ),
            "variant_quantized_native_scratch_kernel_ready_count": int(
                variant_canary_summary.get("quantized_native_scratch_kernel_ready_count", 0) or 0
            ),
            "variant_quantized_runtime_canary_manifest_ready_count": int(
                variant_canary_summary.get("quantized_runtime_canary_manifest_ready_count", 0) or 0
            ),
            "variant_quantized_training_loop_canary_manifest_ready_count": int(
                variant_canary_summary.get("quantized_training_loop_canary_manifest_ready_count", 0) or 0
            ),
            "variant_quantized_training_loop_canary_ready_count": int(
                variant_canary_summary.get("quantized_training_loop_canary_ready_count", 0) or 0
            ),
            "variant_quantized_e2e_no_regression_ready_count": int(
                variant_canary_summary.get("quantized_e2e_no_regression_ready_count", 0) or 0
            ),
            "variant_quantized_product_state_sync_review_ready_count": int(
                variant_canary_summary.get("quantized_product_state_sync_review_ready_count", 0) or 0
            ),
            "variant_quantized_product_optimizer_state_sync_ready_count": int(
                variant_canary_summary.get("quantized_product_optimizer_state_sync_ready_count", 0) or 0
            ),
            "variant_quantized_optimizer_state_sync_state_tensor_count": int(
                variant_canary_summary.get("quantized_optimizer_state_sync_state_tensor_count", 0) or 0
            ),
            "variant_quantized_optimizer_state_sync_parameter_tensor_count": int(
                variant_canary_summary.get("quantized_optimizer_state_sync_parameter_tensor_count", 0) or 0
            ),
            "variant_quantized_rollout_policy_ready_count": int(
                variant_canary_summary.get("quantized_rollout_policy_ready_count", 0) or 0
            ),
            "variant_quantized_dispatch_integration_review_ready_count": int(
                variant_canary_summary.get("quantized_dispatch_integration_review_ready_count", 0) or 0
            ),
            "variant_quantized_owner_approval_hold_ready_count": int(
                variant_canary_summary.get("quantized_owner_approval_hold_ready_count", 0) or 0
            ),
            "variant_quantized_native_canary_pending_count": int(
                variant_canary_summary.get("quantized_native_canary_pending_count", 0) or 0
            ),
            "variant_native_kernel_ready_count": int(
                variant_canary_summary.get("native_kernel_ready_count", 0) or 0
            ),
            "stage_count": len(stage_rows),
            "ready_stage_count": sum(1 for row in stage_rows if row["ready"]),
            "missing_stage_count": len(missing),
            "unsafe_claim_count": len(unsafe),
        },
        "promotion_blockers": _dedupe(blockers + ["product_rollout_review_missing"]),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "record simple-formula owner/release hold for Lion and SGDNesterov with product dispatch default-off"
            if batch_ready
            else "complete simple-formula family batch blockers"
        ),
        "notes": [
            "This batch gate covers fp32 Lion and SGDNesterov only.",
            "Schedule-free simple variants reuse existing default-off native canaries; 8-bit variants remain ABI-spec-only.",
            "The family batch is canary-ready evidence, not product native dispatch.",
        ],
    }


def _stage_reports(stage_reports: Mapping[str, Any] | None, workspace_root: str | Path | None) -> dict[str, dict[str, Any]]:
    if stage_reports is not None:
        return {name: _as_dict(report) for name, report in stage_reports.items()}

    root = Path(workspace_root).resolve() if workspace_root is not None else None
    reference = build_simple_optimizer_reference_scorecard(dtype_cases=("float32",))
    abi = build_simple_optimizer_abi_scorecard(reference_report=reference)
    kernel = _call_with_root(build_simple_optimizer_kernel_parity_scorecard, root)
    runtime_canary = build_simple_optimizer_runtime_canary_scorecard(kernel_parity_report=kernel)
    return {
        "reference": reference,
        "abi": abi,
        "registry": build_simple_optimizer_registry_scorecard(),
        "kernel_parity": kernel,
        "runtime_launch": _call_with_root(build_simple_optimizer_runtime_launch_scorecard, root),
        "training_executor": _call_with_root(build_simple_optimizer_training_executor_scorecard, root),
        "dispatch_runtime": _call_with_root(build_simple_optimizer_dispatch_runtime_scorecard, root),
        "runtime_canary": runtime_canary,
        "training_loop_canary": build_simple_optimizer_training_loop_canary_scorecard(),
        "e2e_no_regression": build_simple_optimizer_e2e_no_regression_scorecard(
            runtime_canary_report=runtime_canary
        ),
    }


def _call_with_root(builder: Callable[..., dict[str, Any]], root: Path | None) -> dict[str, Any]:
    if root is None:
        return builder()
    return builder(workspace_root=root)


def _stage_row(stage: str, report: Mapping[str, Any]) -> dict[str, Any]:
    ready = _stage_ready(stage, report)
    return {
        "stage": stage,
        "present": bool(report),
        "scorecard": str(report.get("scorecard") or ""),
        "gate": str(report.get("gate") or ""),
        "ok": report.get("ok") is True,
        "ready": ready,
        "default_off": _default_off(report),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _stage_ready(stage: str, report: Mapping[str, Any]) -> bool:
    if not report or not _default_off(report):
        return False
    ready_fields = {
        "reference": "first_stage_ready",
        "abi": "first_abi_stage_ready",
        "registry": "registry_stage_ready",
        "kernel_parity": "kernel_parity_stage_ready",
        "runtime_launch": "runtime_launch_stage_ready",
        "training_executor": "training_executor_stage_ready",
        "dispatch_runtime": "dispatch_runtime_stage_ready",
        "runtime_canary": "runtime_canary_ready",
        "training_loop_canary": "ok",
        "e2e_no_regression": "e2e_no_regression_ready",
    }
    field = ready_fields.get(stage)
    return bool(field and report.get(field) is True)


def _exact_target_row(
    optimizer: OptimizerType,
    stage_rows: list[dict[str, Any]],
    batch_ready: bool,
) -> dict[str, Any]:
    status = "simple_formula_native_batch_canary_ready" if batch_ready else "simple_formula_batch_blocked"
    return {
        "optimizer_type": optimizer.value,
        "optimizer_kind": "lion" if optimizer == OptimizerType.LION else "sgd_nesterov",
        "optimizer_family": "simple_formula",
        "batch_status": status,
        "native_route": "rust_cuda_simple_formula_runtime_v0",
        "native_kernel_ready": batch_ready,
        "runtime_canary_ready": batch_ready,
        "training_loop_canary_ready": batch_ready,
        "product_native_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "stage_ready": {row["stage"]: row["ready"] for row in stage_rows},
        "next_gate": "simple_formula_owner_release_hold_with_product_dispatch_default_off",
    }


def _pending_variant_row(
    optimizer: OptimizerType,
    variant_report: Mapping[str, Any],
    variant_abi_report: Mapping[str, Any],
    variant_canary_report: Mapping[str, Any],
    variant_resume_report: Mapping[str, Any],
) -> dict[str, Any]:
    variant_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in variant_report.get("rows", [])
        if isinstance(row, Mapping)
    }
    abi_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in variant_abi_report.get("rows", [])
        if isinstance(row, Mapping)
    }
    canary_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in variant_canary_report.get("rows", [])
        if isinstance(row, Mapping)
    }
    resume_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in variant_resume_report.get("rows", [])
        if isinstance(row, Mapping)
    }
    variant = _as_dict(variant_rows.get(optimizer.value))
    abi = _as_dict(abi_rows.get(optimizer.value))
    canary = _as_dict(canary_rows.get(optimizer.value))
    resume = _as_dict(resume_rows.get(optimizer.value))
    variant_status = str(variant.get("variant_status") or "")
    if canary.get("variant_status") == "schedule_free_native_canary_ready":
        batch_status = "simple_formula_variant_schedule_free_native_canary_ready"
        next_gate = str(canary.get("next_gate") or "default_off_rollout_review_and_e2e_shadow_matrix")
    elif canary.get("variant_status") == "quantized_owner_approval_hold_ready":
        batch_status = "simple_formula_variant_quantized_owner_approval_hold_ready"
        next_gate = str(canary.get("next_gate") or "quantized_variant_explicit_owner_approval_record")
    elif canary.get("variant_status") == "quantized_dispatch_integration_review_ready":
        batch_status = "simple_formula_variant_quantized_dispatch_integration_review_ready"
        next_gate = str(canary.get("next_gate") or "quantized_variant_owner_approval_hold")
    elif canary.get("variant_status") == "quantized_rollout_policy_ready":
        batch_status = "simple_formula_variant_quantized_rollout_policy_ready"
        next_gate = str(canary.get("next_gate") or "quantized_variant_dispatch_integration_review")
    elif canary.get("variant_status") == "quantized_product_state_sync_ready":
        batch_status = "simple_formula_variant_quantized_product_state_sync_ready"
        next_gate = str(canary.get("next_gate") or "quantized_variant_default_off_rollout_review")
    elif canary.get("variant_status") == "quantized_e2e_no_regression_ready":
        batch_status = "simple_formula_variant_quantized_e2e_no_regression_ready"
        next_gate = str(canary.get("next_gate") or "quantized_variant_product_state_sync_review")
    elif canary.get("variant_status") == "quantized_training_loop_canary_ready":
        batch_status = "simple_formula_variant_quantized_training_loop_canary_ready"
        next_gate = str(canary.get("next_gate") or "quantized_variant_e2e_no_regression")
    elif canary.get("variant_status") == "quantized_training_loop_canary_manifest_ready":
        batch_status = "simple_formula_variant_quantized_training_loop_canary_manifest_ready"
        next_gate = str(canary.get("next_gate") or "quantized_variant_training_loop_executor")
    elif canary.get("variant_status") == "quantized_runtime_canary_manifest_ready":
        batch_status = "simple_formula_variant_quantized_runtime_canary_manifest_ready"
        next_gate = str(canary.get("next_gate") or "quantized_variant_training_loop_canary")
    elif canary.get("variant_status") == "quantized_native_scratch_kernel_ready":
        batch_status = "simple_formula_variant_quantized_native_scratch_kernel_ready"
        next_gate = str(canary.get("next_gate") or "quantized_variant_runtime_canary")
    elif canary.get("variant_status") == "quantized_formula_parity_ready":
        batch_status = "simple_formula_variant_quantized_formula_parity_ready"
        next_gate = str(canary.get("next_gate") or "quantized_variant_cuda_scratch_kernel")
    elif abi.get("variant_status") == "native_abi_spec_ready":
        batch_status = "simple_formula_variant_native_abi_spec_ready"
        next_gate = str(abi.get("next_gate") or "variant_parity_scratch_kernel_canary")
    elif variant_status == "layout_spec_ready":
        batch_status = "simple_formula_variant_layout_spec_ready"
        next_gate = str(variant.get("next_gate") or "quantized_state_formula_parity_and_tensor_binding_matrix")
    elif variant_status == "state_machine_reference_ready":
        batch_status = "simple_formula_variant_state_machine_reference_ready"
        next_gate = str(variant.get("next_gate") or "schedule_free_variant_native_abi_and_resume_matrix")
    elif optimizer in {OptimizerType.LION_8BIT, OptimizerType.PAGED_LION_8BIT, OptimizerType.SGD_NESTEROV_8BIT}:
        batch_status = "simple_formula_variant_layout_pending"
        next_gate = "quantized_or_paged_state_layout_parity"
    else:
        batch_status = "simple_formula_variant_state_machine_pending"
        next_gate = "schedule_free_state_machine_parity"
    return {
        "optimizer_type": optimizer.value,
        "optimizer_family": "simple_formula",
        "batch_status": batch_status,
        "native_route": str(variant.get("native_route") or "dedicated_variant_kernel_required"),
        "state_layout_ready": variant.get("state_layout_ready") is True,
        "state_machine_reference_ready": variant.get("state_machine_reference_ready") is True,
        "native_abi_spec_ready": abi.get("native_abi_spec_ready") is True,
        "native_canary_ready": canary.get("native_canary_ready") is True,
        "quantized_formula_parity_ready": canary.get("formula_parity_ready") is True,
        "native_scratch_kernel_ready": canary.get("native_scratch_kernel_ready") is True,
        "runtime_canary_manifest_ready": canary.get("runtime_canary_manifest_ready") is True,
        "training_loop_canary_manifest_ready": canary.get("training_loop_canary_manifest_ready") is True,
        "e2e_no_regression_ready": canary.get("e2e_no_regression_ready") is True,
        "product_state_sync_review_ready": canary.get("product_state_sync_review_ready") is True,
        "product_optimizer_state_sync_ready": canary.get("product_optimizer_state_sync_ready") is True,
        "canary_rollout_policy_ready": canary.get("canary_rollout_policy_ready") is True,
        "dispatch_integration_review_ready": canary.get("dispatch_integration_review_ready") is True,
        "owner_approval_hold_ready": canary.get("owner_approval_hold_ready") is True,
        "formula_parity_matrix_artifact_ready": abi.get("formula_parity_matrix_artifact_ready") is True,
        "formula_parity_matrix_implementation_ready": False,
        "resume_parity_matrix_artifact_ready": abi.get("resume_parity_matrix_artifact_ready") is True,
        "resume_parity_matrix_implementation_ready": resume.get("resume_parity_matrix_implementation_ready") is True,
        "native_kernel_ready": canary.get("native_kernel_ready") is True,
        "runtime_canary_ready": canary.get("native_canary_ready") is True,
        "training_loop_canary_ready": canary.get("training_loop_canary_ready") is True,
        "product_native_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "next_gate": next_gate,
    }


def _compact_stage_report(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "scorecard": str(report.get("scorecard") or ""),
        "gate": str(report.get("gate") or ""),
        "ok": report.get("ok") is True,
        "default_off": _default_off(report),
        "summary": summary,
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_variant_state_report(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "variant_state_layout_stage_ready": report.get("variant_state_layout_stage_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "layout_spec_ready_count": int(summary.get("layout_spec_ready_count", 0) or 0),
        "state_machine_reference_ready_count": int(summary.get("state_machine_reference_ready_count", 0) or 0),
        "native_kernel_ready_count": int(summary.get("native_kernel_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_variant_native_abi_report(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "variant_native_abi_spec_ready": report.get("variant_native_abi_spec_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "native_abi_spec_ready_count": int(summary.get("native_abi_spec_ready_count", 0) or 0),
        "formula_parity_matrix_artifact_ready_count": int(
            summary.get("formula_parity_matrix_artifact_ready_count", 0) or 0
        ),
        "formula_parity_matrix_implementation_ready_count": int(
            summary.get("formula_parity_matrix_implementation_ready_count", 0) or 0
        ),
        "resume_parity_matrix_artifact_ready_count": int(
            summary.get("resume_parity_matrix_artifact_ready_count", 0) or 0
        ),
        "resume_parity_matrix_implementation_ready_count": int(
            summary.get("resume_parity_matrix_implementation_ready_count", 0) or 0
        ),
        "native_kernel_ready_count": int(summary.get("native_kernel_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_variant_native_canary_report(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "variant_schedule_free_native_canary_ready": report.get("variant_schedule_free_native_canary_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "schedule_free_native_canary_ready_count": int(
            summary.get("schedule_free_native_canary_ready_count", 0) or 0
        ),
        "quantized_formula_parity_ready_count": int(
            summary.get("quantized_formula_parity_ready_count", 0) or 0
        ),
        "quantized_native_scratch_kernel_ready_count": int(
            summary.get("quantized_native_scratch_kernel_ready_count", 0) or 0
        ),
        "quantized_runtime_canary_manifest_ready_count": int(
            summary.get("quantized_runtime_canary_manifest_ready_count", 0) or 0
        ),
        "quantized_training_loop_canary_manifest_ready_count": int(
            summary.get("quantized_training_loop_canary_manifest_ready_count", 0) or 0
        ),
        "quantized_training_loop_canary_ready_count": int(
            summary.get("quantized_training_loop_canary_ready_count", 0) or 0
        ),
        "quantized_e2e_no_regression_ready_count": int(
            summary.get("quantized_e2e_no_regression_ready_count", 0) or 0
        ),
        "quantized_product_state_sync_review_ready_count": int(
            summary.get("quantized_product_state_sync_review_ready_count", 0) or 0
        ),
        "quantized_product_optimizer_state_sync_ready_count": int(
            summary.get("quantized_product_optimizer_state_sync_ready_count", 0) or 0
        ),
        "quantized_optimizer_state_sync_state_tensor_count": int(
            summary.get("quantized_optimizer_state_sync_state_tensor_count", 0) or 0
        ),
        "quantized_optimizer_state_sync_parameter_tensor_count": int(
            summary.get("quantized_optimizer_state_sync_parameter_tensor_count", 0) or 0
        ),
        "quantized_rollout_policy_ready_count": int(
            summary.get("quantized_rollout_policy_ready_count", 0) or 0
        ),
        "quantized_dispatch_integration_review_ready_count": int(
            summary.get("quantized_dispatch_integration_review_ready_count", 0) or 0
        ),
        "quantized_owner_approval_hold_ready_count": int(
            summary.get("quantized_owner_approval_hold_ready_count", 0) or 0
        ),
        "quantized_native_canary_pending_count": int(summary.get("quantized_native_canary_pending_count", 0) or 0),
        "native_kernel_ready_count": int(summary.get("native_kernel_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_variant_resume_report(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "resume_parity_matrix_implementation_ready_count": int(
            summary.get("resume_parity_matrix_implementation_ready_count", 0) or 0
        ),
        "quantized_resume_parity_ready_count": int(summary.get("quantized_resume_parity_ready_count", 0) or 0),
        "schedule_free_resume_parity_ready_count": int(
            summary.get("schedule_free_resume_parity_ready_count", 0) or 0
        ),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _default_off(report: Mapping[str, Any]) -> bool:
    return bool(report) and all(report.get(field) is not True for field in UNSAFE_TRUE_FIELDS)


def _unsafe_claims(stages: Mapping[str, Mapping[str, Any]]) -> list[str]:
    claims: list[str] = []
    for stage, report in stages.items():
        for field in UNSAFE_TRUE_FIELDS:
            if report.get(field) is True:
                claims.append(f"{stage}:unsafe_true:{field}")
    return _dedupe(claims)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


__all__ = [
    "EXACT_BATCH_TARGETS",
    "PENDING_VARIANTS",
    "build_simple_optimizer_family_batch_scorecard",
]
