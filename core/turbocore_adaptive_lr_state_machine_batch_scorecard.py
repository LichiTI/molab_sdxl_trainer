"""Report-only batch scorecard for adaptive-LR optimizer state machines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.configs import OptimizerType


TARGET_OPTIMIZERS: tuple[OptimizerType, ...] = (
    OptimizerType.AUTO_PRODIGY,
    OptimizerType.PRODIGY,
    OptimizerType.DADAPTATION,
    OptimizerType.DADAPT_ADAM_PREPRINT,
    OptimizerType.DADAPT_ADAGRAD,
    OptimizerType.DADAPT_ADAM,
    OptimizerType.DADAPT_ADAN,
    OptimizerType.DADAPT_ADAN_IP,
    OptimizerType.DADAPT_LION,
    OptimizerType.DADAPT_SGD,
    OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE,
)

_PRODIGY_FAMILY = {
    OptimizerType.AUTO_PRODIGY,
    OptimizerType.PRODIGY,
    OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE,
}


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_adaptive_lr_state_machine_batch_scorecard(*, write_artifact: bool = False) -> dict[str, Any]:
    """Build a lightweight readiness report without native dispatch claims."""

    rows = [_row(optimizer) for optimizer in TARGET_OPTIMIZERS]
    ready_count = sum(1 for row in rows if row["state_machine_reference_ready"] is True)
    abi_spec_ready_count = sum(1 for row in rows if row["state_machine_abi_spec_ready"] is True)
    native_ready_count = sum(1 for row in rows if row["native_ready"] is True)

    report = {
        "schema_version": 1,
        "scorecard": "turbocore_adaptive_lr_state_machine_batch_scorecard_v0",
        "gate": "adaptive_lr_state_machine_batch_reference",
        "ok": ready_count == len(TARGET_OPTIMIZERS) and abi_spec_ready_count == len(TARGET_OPTIMIZERS) and native_ready_count == 0,
        "report_only": True,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "native_ready": False,
        "state_machine_abi_spec_ready": abi_spec_ready_count == len(TARGET_OPTIMIZERS),
        "rows": rows,
        "summary": {
            "target_count": len(TARGET_OPTIMIZERS),
            "state_machine_reference_ready_count": ready_count,
            "state_machine_abi_spec_ready_count": abi_spec_ready_count,
            "dynamic_lr_scalar_state_spec_ready_count": _count_spec(rows, "dynamic_lr_scalar_state"),
            "d_estimator_global_state_spec_ready_count": _count_spec(rows, "d_estimator_global_state"),
            "per_step_quality_guard_spec_ready_count": _count_spec(rows, "per_step_quality_guard"),
            "resume_scope_spec_ready_count": _count_spec(rows, "resume_scope"),
            "native_kernel_preconditions_spec_ready_count": _count_spec(rows, "native_kernel_preconditions"),
            "native_kernel_preconditions_implementation_ready_count": 0,
            "native_ready_count": native_ready_count,
            "training_path_enabled_count": sum(1 for row in rows if row["training_path_enabled"] is True),
            "runtime_dispatch_ready_count": sum(1 for row in rows if row["runtime_dispatch_ready"] is True),
            "native_dispatch_allowed_count": sum(1 for row in rows if row["native_dispatch_allowed"] is True),
        },
        "blocked_reasons": [],
        "recommended_next_step": "review adaptive-LR native ABI and batch resume parity before any dispatch route",
        "notes": [
            "This scorecard only classifies state-machine reference readiness.",
            "It intentionally does not open training_path, native_dispatch, native_ready, or default behavior.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _row(optimizer: OptimizerType) -> dict[str, Any]:
    family = _family(optimizer)
    return {
        "schema_version": 1,
        "optimizer_type": optimizer.value,
        "family": family,
        "state_machine_status": "reference_ready_report_only",
        "state_machine_reference_ready": True,
        "batch_reference_ready": True,
        "next_gate": _next_gate(family),
        "blocked_reasons": [
            "native_state_machine_abi_missing",
            "native_state_machine_abi_implementation_missing",
            "batch_resume_parity_not_validated",
            "native_dispatch_gate_not_requested",
        ],
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_ready": False,
        "default_behavior_changed": False,
        "runtime_authority": "existing_python_or_third_party_optimizer",
        "state_model": _state_model(family),
        "state_machine_abi_status": "report_only_spec_ready",
        "state_machine_abi_spec_ready": True,
        "state_machine_abi_implementation_ready": False,
        "native_kernel_preconditions_spec_ready": True,
        "native_kernel_preconditions_implementation_ready": False,
        "state_machine_abi_spec": _state_machine_abi_spec(optimizer, family),
    }


def _family(optimizer: OptimizerType) -> str:
    if optimizer in _PRODIGY_FAMILY:
        return "adaptive_lr_prodigy"
    return "adaptive_lr_dadapt"


def _next_gate(family: str) -> str:
    if family == "adaptive_lr_prodigy":
        return "prodigy_native_state_machine_abi_review"
    return "dadapt_native_state_machine_abi_review"


def _state_model(family: str) -> dict[str, Any]:
    if family == "adaptive_lr_prodigy":
        return {
            "requires_dynamic_lr_estimate": True,
            "requires_global_distance_state": True,
            "requires_schedule_free_mode_review": True,
            "adamw_kernel_compatible": False,
        }
    return {
        "requires_dynamic_lr_estimate": True,
        "requires_global_d_adaptation_state": True,
        "requires_variant_specific_accumulators": True,
        "adamw_kernel_compatible": False,
    }


def _state_machine_abi_spec(optimizer: OptimizerType, family: str) -> dict[str, Any]:
    if family == "adaptive_lr_prodigy":
        estimator = {
            "kind": "prodigy_distance_estimator",
            "global_state_fields": ("d", "d0", "d_max", "dlr", "growth_rate"),
            "requires_bias_correction_inputs": True,
            "requires_decoupled_weight_decay_policy": True,
        }
        resume_scope = ("param_groups", "per_param_moments", "global_d_state", "step", "rng_unowned")
        preconditions = (
            "global d scalar must be updated before per-parameter kernel launch",
            "per-group lr/dlr values must be materialized as scalar launch inputs",
            "schedule-free variant requires an explicit z/state buffer contract",
        )
    else:
        estimator = {
            "kind": "dadapt_global_d_estimator",
            "global_state_fields": ("d", "d0", "gsq_weighted", "growth_rate", "sk_l1"),
            "requires_variant_specific_accumulators": True,
            "requires_layer_scale_policy": optimizer in {OptimizerType.DADAPT_ADAGRAD, OptimizerType.DADAPT_SGD},
        }
        resume_scope = ("param_groups", "per_param_moments", "global_d_state", "step", "variant_accumulators")
        preconditions = (
            "global d estimator reduction must complete before param update",
            "variant accumulator buffers must match the selected DAdapt optimizer",
            "lr scalar is derived from d/lr policy and cannot reuse AdamW fixed-lr ABI",
        )

    return {
        "schema_version": 1,
        "report_only": True,
        "dynamic_lr_scalar_state": {
            "spec_ready": True,
            "implementation_ready": False,
            "fields": ("lr", "d", "d0", "growth_rate", "step"),
            "source_of_truth": "optimizer_state_and_param_group",
            "kernel_input_policy": "materialize_scalars_before_launch",
        },
        "d_estimator_global_state": {
            "spec_ready": True,
            "implementation_ready": False,
            **estimator,
        },
        "per_step_quality_guard": {
            "spec_ready": True,
            "implementation_ready": False,
            "checks": (
                "finite_dynamic_lr",
                "finite_global_d",
                "nonnegative_denominator",
                "monotonic_step",
                "fallback_on_invalid_scalar",
            ),
        },
        "resume_scope": {
            "spec_ready": True,
            "implementation_ready": False,
            "required_state": resume_scope,
            "resume_parity_gate": "step_state_dict_load_state_dict_next_step",
        },
        "native_kernel_preconditions": {
            "spec_ready": True,
            "implementation_ready": False,
            "preconditions": preconditions,
            "blocked_until": (
                "state_machine_abi_implementation",
                "batch_resume_parity_matrix",
                "owner_release_hold",
            ),
        },
    }


def _count_spec(rows: list[dict[str, Any]], key: str) -> int:
    return sum(
        1
        for row in rows
        if row.get("state_machine_abi_spec_ready") is True
        and _as_dict(row.get("state_machine_abi_spec")).get(key, {}).get("spec_ready") is True
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _write_artifact(report: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_adaptive_lr_state_machine_batch_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["TARGET_OPTIMIZERS", "build_adaptive_lr_state_machine_batch_scorecard"]
