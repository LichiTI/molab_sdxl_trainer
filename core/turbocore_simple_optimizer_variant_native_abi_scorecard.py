"""Report-only native ABI plans for simple optimizer variants.

The variant state/layout scorecard proves what state exists.  This module turns
that into native launch contracts and parity/resume matrix plans.  It does not
register kernels, call native code, or alter training/request/UI behavior.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.configs import OptimizerType
from core.turbocore_simple_optimizer_variant_state_scorecard import (
    ALL_VARIANT_TARGETS,
    QUANTIZED_LAYOUT_TARGETS,
    SCHEDULE_FREE_TARGETS,
    build_simple_optimizer_variant_state_scorecard,
)


UNSAFE_TRUE_FIELDS = (
    "training_path_enabled",
    "default_behavior_changed",
    "native_dispatch_allowed",
    "runtime_dispatch_ready",
    "product_native_dispatch_ready",
    "native_kernel_ready",
)


def build_simple_optimizer_variant_native_abi_scorecard(
    *,
    variant_state_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build launch ABI and parity plans without enabling native dispatch."""

    state_report = dict(variant_state_report or build_simple_optimizer_variant_state_scorecard())
    state_rows = {
        str(row.get("optimizer_type") or ""): dict(row)
        for row in state_report.get("rows", [])
        if isinstance(row, Mapping)
    }
    contracts = [_contract_for_optimizer(optimizer, state_rows.get(optimizer.value, {})) for optimizer in ALL_VARIANT_TARGETS]
    validations = [_validate_contract(contract, state_rows.get(str(contract.get("optimizer_type") or ""), {})) for contract in contracts]
    parity_plans = [_parity_plan(contract) for contract in contracts]
    resume_plans = [_resume_plan(contract) for contract in contracts]
    failed = [item for item in validations if item.get("ok") is not True]
    unsafe = _unsafe_claims(state_report)
    abi_ready = state_report.get("variant_state_layout_stage_ready") is True and not failed and not unsafe
    rows = [
        _row_for_optimizer(contract, validation, parity_plan, resume_plan, abi_ready)
        for contract, validation, parity_plan, resume_plan in zip(contracts, validations, parity_plans, resume_plans)
    ]
    blockers = _dedupe(
        [reason for item in failed for reason in _strings(item.get("blocked_reasons"))]
        + unsafe
        + ([] if state_report.get("variant_state_layout_stage_ready") is True else ["variant_state_layout_stage_not_ready"])
    )
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_variant_native_abi_scorecard_v0",
        "gate": "simple_formula_variant_native_abi_and_parity_plan",
        "ok": abi_ready,
        "promotion_ready": False,
        "variant_native_abi_spec_ready": abi_ready,
        "variant_state_layout_stage_ready": state_report.get("variant_state_layout_stage_ready") is True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "native_kernel_ready": False,
        "product_native_dispatch_ready": False,
        "target_optimizer_types": [optimizer.value for optimizer in ALL_VARIANT_TARGETS],
        "rows": rows,
        "contracts": contracts,
        "validations": validations,
        "formula_parity_plans": parity_plans,
        "resume_parity_plans": resume_plans,
        "variant_state_summary": dict(state_report.get("summary") or {}),
        "summary": {
            "target_optimizer_count": len(ALL_VARIANT_TARGETS),
            "native_abi_spec_ready_count": sum(1 for row in rows if row["native_abi_spec_ready"]),
            "quantized_native_abi_spec_ready_count": sum(
                1 for row in rows if row["variant_kind"] == "quantized_state" and row["native_abi_spec_ready"]
            ),
            "schedule_free_native_abi_spec_ready_count": sum(
                1 for row in rows if row["variant_kind"] == "schedule_free_state_machine" and row["native_abi_spec_ready"]
            ),
            "formula_parity_matrix_artifact_ready_count": sum(1 for item in parity_plans if item["artifact_ready"]),
            "formula_parity_matrix_implementation_ready_count": 0,
            "formula_parity_case_planned_count": sum(int(item["case_count"]) for item in parity_plans),
            "resume_parity_matrix_artifact_ready_count": sum(1 for item in resume_plans if item["artifact_ready"]),
            "resume_parity_matrix_implementation_ready_count": 0,
            "resume_parity_case_planned_count": sum(int(item["case_count"]) for item in resume_plans),
            "native_kernel_ready_count": 0,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": blockers
        + [
            "simple_variant_native_kernel_missing",
            "simple_variant_runtime_canary_missing",
            "simple_variant_parity_matrix_implementation_missing",
            "simple_variant_product_rollout_review_missing",
        ],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "implement parity-only scratch kernels for simple variants behind default-off canaries"
            if abi_ready
            else "fix simple variant native ABI blockers before kernel work"
        ),
        "notes": [
            "This scorecard names native ABI contracts and matrix plans only.",
            "Quantized/paged variants need dequantize-update-requant parity before runtime canary.",
            "Schedule-free variants need explicit train/eval and param-group state ABI before runtime canary.",
        ],
    }


def _contract_for_optimizer(optimizer: OptimizerType, state_row: Mapping[str, Any]) -> dict[str, Any]:
    if optimizer in QUANTIZED_LAYOUT_TARGETS:
        return _quantized_contract(optimizer, state_row)
    if optimizer in SCHEDULE_FREE_TARGETS:
        return _schedule_free_contract(optimizer, state_row)
    raise ValueError(f"unsupported simple variant native ABI target: {optimizer}")


def _quantized_contract(optimizer: OptimizerType, state_row: Mapping[str, Any]) -> dict[str, Any]:
    role = "momentum_buffer" if optimizer == OptimizerType.SGD_NESTEROV_8BIT else "exp_avg"
    return {
        "schema_version": 1,
        "contract": f"{_kind(optimizer)}_native_quantized_launch_plan_v0",
        "optimizer_type": optimizer.value,
        "optimizer_kind": _kind(optimizer),
        "optimizer_family": "simple_formula",
        "variant_kind": "quantized_state",
        "source_state_status": str(state_row.get("variant_status") or ""),
        "launch_plan": f"{_kind(optimizer)}_flat_quantized_launch_plan_v0",
        "kernel_name": f"{_kind(optimizer)}_flat_quantized_cuda_v0",
        "status": "abi_spec_only",
        "native_kernel_present": False,
        "training_path_enabled": False,
        "input_buffers": [
            _buffer("param_flat", "parameter values", "float32|float16|bfloat16", True),
            _buffer("grad_flat", "gradient values", "float32|float16|bfloat16", False),
            _buffer(f"{role}_uint8", f"quantized {role} state", "uint8", True),
            _buffer(f"{role}_scale_fp32", f"{role} block scale or absmax", "float32", True),
            _buffer(f"{role}_qmap_fp32", f"{role} quantization map", "float32", False),
            _scalar("step", "optimizer step", "int64"),
            *_hyperparameter_scalars(optimizer),
        ],
        "numeric_policy": {
            "state_role": role,
            "state_precision": "uint8_blockwise",
            "reference_replay_dtype": "float32",
            "requires_dequantize_before_formula": True,
            "requires_requantize_after_formula": True,
            "paged_state": optimizer == OptimizerType.PAGED_LION_8BIT,
        },
        "route_policy": "report_only_until_parity_kernel_and_runtime_canary_exist",
    }


def _schedule_free_contract(optimizer: OptimizerType, state_row: Mapping[str, Any]) -> dict[str, Any]:
    required_state = ["z", "exp_avg_sq"] if optimizer == OptimizerType.RADAM_SCHEDULE_FREE else ["z"]
    return {
        "schema_version": 1,
        "contract": f"{_kind(optimizer)}_native_stateful_launch_plan_v0",
        "optimizer_type": optimizer.value,
        "optimizer_kind": _kind(optimizer),
        "optimizer_family": "simple_formula",
        "variant_kind": "schedule_free_state_machine",
        "source_state_status": str(state_row.get("variant_status") or ""),
        "launch_plan": f"{_kind(optimizer)}_stateful_launch_plan_v0",
        "kernel_name": f"{_kind(optimizer)}_stateful_cuda_v0",
        "status": "abi_spec_only",
        "native_kernel_present": False,
        "training_path_enabled": False,
        "input_buffers": [
            _buffer("param_flat", "parameter values", "float32|float16|bfloat16", True),
            _buffer("grad_flat", "gradient values", "float32|float16|bfloat16", False),
            *[_buffer(role, f"schedule-free param state {role}", "float32|float16|bfloat16", True) for role in required_state],
            _scalar("train_mode", "optimizer train/eval mode flag", "bool"),
            _scalar("k", "schedule-free update counter", "int64"),
            _scalar("lr", "requested learning rate", "float32"),
            _scalar("scheduled_lr", "schedule-free effective learning rate", "float32"),
            _scalar("lr_max", "maximum scheduled learning rate", "float32"),
            _scalar("weight_sum", "schedule-free averaging accumulator", "float32"),
            _scalar("weight_lr_power", "schedule-free LR weighting power", "float32"),
            _scalar("r", "schedule-free weight exponent", "float32"),
            *_hyperparameter_scalars(optimizer),
        ],
        "numeric_policy": {
            "required_param_state_keys": required_state,
            "scheduler_coupled": True,
            "requires_explicit_train_eval_mode": True,
            "external_scheduler_policy": "constant_required",
        },
        "route_policy": "report_only_until_stateful_kernel_and_runtime_canary_exist",
    }


def _hyperparameter_scalars(optimizer: OptimizerType) -> list[dict[str, Any]]:
    if optimizer in {OptimizerType.LION_8BIT, OptimizerType.PAGED_LION_8BIT}:
        return [
            _scalar("lr", "learning rate", "float32"),
            _scalar("beta1", "Lion beta1", "float32"),
            _scalar("beta2", "Lion beta2", "float32"),
            _scalar("weight_decay", "decoupled weight decay", "float32"),
        ]
    if optimizer == OptimizerType.SGD_NESTEROV_8BIT:
        return [
            _scalar("lr", "learning rate", "float32"),
            _scalar("momentum", "SGD momentum", "float32"),
            _scalar("weight_decay", "coupled weight decay", "float32"),
            _scalar("nesterov", "Nesterov flag", "bool"),
        ]
    if optimizer == OptimizerType.RADAM_SCHEDULE_FREE:
        return [
            _scalar("beta1", "RAdam beta1", "float32"),
            _scalar("beta2", "RAdam beta2", "float32"),
            _scalar("eps", "RAdam epsilon", "float32"),
            _scalar("weight_decay", "weight decay", "float32"),
        ]
    return [
        _scalar("momentum", "SGD momentum", "float32"),
        _scalar("weight_decay", "weight decay", "float32"),
    ]


def _buffer(role: str, meaning: str, dtype: str, mutable: bool) -> dict[str, Any]:
    return {
        "role": role,
        "kind": "tensor",
        "meaning": meaning,
        "dtype": dtype,
        "mutable": mutable,
        "required": True,
    }


def _scalar(role: str, meaning: str, dtype: str) -> dict[str, Any]:
    return {
        "role": role,
        "kind": "scalar",
        "meaning": meaning,
        "dtype": dtype,
        "mutable": role in {"step", "k", "scheduled_lr", "lr_max", "weight_sum"},
        "required": True,
    }


def _validate_contract(contract: Mapping[str, Any], state_row: Mapping[str, Any]) -> dict[str, Any]:
    roles = {str(item.get("role") or "") for item in contract.get("input_buffers", []) if isinstance(item, Mapping)}
    variant_kind = str(contract.get("variant_kind") or "")
    required_roles = {"param_flat", "grad_flat", "lr", "weight_decay"}
    if variant_kind == "quantized_state":
        state_role = str(dict(contract.get("numeric_policy") or {}).get("state_role") or "")
        required_roles.update({f"{state_role}_uint8", f"{state_role}_scale_fp32", f"{state_role}_qmap_fp32", "step"})
    else:
        required_roles.update({"train_mode", "k", "scheduled_lr", "lr_max", "weight_sum", "weight_lr_power", "r", "z"})
    missing_roles = sorted(required_roles - roles)
    source_ready = str(state_row.get("variant_status") or "") in {
        "layout_spec_ready",
        "state_machine_reference_ready",
    }
    ok = (
        source_ready
        and not missing_roles
        and bool(contract.get("native_kernel_present", True)) is False
        and bool(contract.get("training_path_enabled", True)) is False
    )
    return {
        "schema_version": 1,
        "validator": "simple_variant_native_abi_validator_v0",
        "ok": ok,
        "optimizer_type": str(contract.get("optimizer_type") or ""),
        "optimizer_kind": str(contract.get("optimizer_kind") or ""),
        "variant_kind": variant_kind,
        "source_state_ready": source_ready,
        "missing_roles": missing_roles,
        "native_kernel_present": bool(contract.get("native_kernel_present", False)),
        "training_path_enabled": bool(contract.get("training_path_enabled", False)),
        "blocked_reasons": [] if ok else _validation_blockers(source_ready, missing_roles),
    }


def _validation_blockers(source_ready: bool, missing_roles: Sequence[str]) -> list[str]:
    reasons: list[str] = []
    if not source_ready:
        reasons.append("simple_variant_source_state_not_ready")
    if missing_roles:
        reasons.append("simple_variant_native_abi_roles_missing")
    return reasons


def _row_for_optimizer(
    contract: Mapping[str, Any],
    validation: Mapping[str, Any],
    parity_plan: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    abi_ready: bool,
) -> dict[str, Any]:
    row_ready = abi_ready and validation.get("ok") is True
    return {
        "optimizer_type": str(contract.get("optimizer_type") or ""),
        "optimizer_kind": str(contract.get("optimizer_kind") or ""),
        "optimizer_family": "simple_formula",
        "variant_kind": str(contract.get("variant_kind") or ""),
        "variant_status": "native_abi_spec_ready" if row_ready else "native_abi_spec_blocked",
        "native_abi_spec_ready": row_ready,
        "formula_parity_matrix_artifact_ready": parity_plan.get("artifact_ready") is True,
        "formula_parity_matrix_implementation_ready": False,
        "resume_parity_matrix_artifact_ready": resume_plan.get("artifact_ready") is True,
        "resume_parity_matrix_implementation_ready": False,
        "native_kernel_ready": False,
        "runtime_canary_ready": False,
        "product_native_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "next_gate": "variant_parity_scratch_kernel_canary" if row_ready else "fix_variant_native_abi_spec",
    }


def _parity_plan(contract: Mapping[str, Any]) -> dict[str, Any]:
    optimizer = str(contract.get("optimizer_type") or "")
    variant_kind = str(contract.get("variant_kind") or "")
    cases = _quantized_parity_cases(optimizer) if variant_kind == "quantized_state" else _schedule_free_parity_cases(optimizer)
    return {
        "schema_version": 1,
        "artifact": f"simple_variant/{optimizer}/formula_parity_matrix_v0",
        "status": "planned",
        "report_only": True,
        "optimizer_type": optimizer,
        "optimizer_kind": str(contract.get("optimizer_kind") or ""),
        "variant_kind": variant_kind,
        "artifact_ready": True,
        "implementation_ready": False,
        "native_kernel_ready": False,
        "case_count": len(cases),
        "cases": cases,
    }


def _resume_plan(contract: Mapping[str, Any]) -> dict[str, Any]:
    optimizer = str(contract.get("optimizer_type") or "")
    variant_kind = str(contract.get("variant_kind") or "")
    cases = _quantized_resume_cases(optimizer) if variant_kind == "quantized_state" else _schedule_free_resume_cases(optimizer)
    return {
        "schema_version": 1,
        "artifact": f"simple_variant/{optimizer}/resume_parity_matrix_v0",
        "status": "planned",
        "report_only": True,
        "optimizer_type": optimizer,
        "optimizer_kind": str(contract.get("optimizer_kind") or ""),
        "variant_kind": variant_kind,
        "artifact_ready": True,
        "implementation_ready": False,
        "native_kernel_ready": False,
        "case_count": len(cases),
        "cases": cases,
    }


def _quantized_parity_cases(optimizer: str) -> list[dict[str, str]]:
    return [
        _case(optimizer, "dense_fp32_reference_step", "compare dequantized update against fp32 simple reference"),
        _case(optimizer, "quant_dequant_roundtrip", "state quantization must round-trip within planned tolerance"),
        _case(optimizer, "weight_decay_branch", "cover weight decay placement for the variant formula"),
        _case(optimizer, "none_grad_skip", "grad=None must not mutate param or quantized state"),
        _case(optimizer, "dtype_boundary", "cover fp32, fp16, and bf16 parameter tensors"),
        _case(optimizer, "paged_residency_boundary", "cover paged residency or explicitly no-page policy"),
    ]


def _schedule_free_parity_cases(optimizer: str) -> list[dict[str, str]]:
    return [
        _case(optimizer, "train_mode_step", "native path must update only while optimizer is in train mode"),
        _case(optimizer, "eval_mode_no_update", "eval mode must block optimizer update semantics"),
        _case(optimizer, "stateful_reference_step", "compare z and schedule counters against schedulefree reference"),
        _case(optimizer, "weight_sum_boundary", "cover schedule-free averaging accumulator transitions"),
        _case(optimizer, "constant_scheduler_boundary", "verify external scheduler policy stays constant"),
        _case(optimizer, "dtype_boundary", "cover fp32, fp16, and bf16 parameter tensors"),
    ]


def _quantized_resume_cases(optimizer: str) -> list[dict[str, str]]:
    return [
        _case(optimizer, "state_dict_quant_buffers_roundtrip", "round-trip quantized state buffers and scales"),
        _case(optimizer, "next_step_after_restore", "next step after load matches uninterrupted reference"),
        _case(optimizer, "device_dtype_restore", "restore state tensors to expected device and dtype"),
        _case(optimizer, "paged_checkpoint_restore", "paged variants must preserve residency metadata"),
    ]


def _schedule_free_resume_cases(optimizer: str) -> list[dict[str, str]]:
    return [
        _case(optimizer, "state_dict_roundtrip", "round-trip z and param-group schedule state"),
        _case(optimizer, "train_eval_restore", "restore train/eval mode before next step"),
        _case(optimizer, "next_step_after_restore", "next step after load matches uninterrupted reference"),
        _case(optimizer, "scheduler_counter_restore", "restore k, scheduled_lr, lr_max, and weight_sum"),
    ]


def _case(optimizer: str, suffix: str, expectation: str) -> dict[str, str]:
    return {
        "case_id": f"{optimizer}:{suffix}",
        "expectation": expectation,
    }


def _unsafe_claims(report: Mapping[str, Any]) -> list[str]:
    claims: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if report.get(field) is True:
            claims.append(f"variant_state_report_unsafe_true:{field}")
    return claims


def _kind(optimizer: OptimizerType) -> str:
    mapping = {
        OptimizerType.LION_8BIT: "lion_8bit",
        OptimizerType.PAGED_LION_8BIT: "paged_lion_8bit",
        OptimizerType.SGD_NESTEROV_8BIT: "sgd_nesterov_8bit",
        OptimizerType.RADAM_SCHEDULE_FREE: "radam_schedule_free",
        OptimizerType.SGD_SCHEDULE_FREE: "sgd_schedule_free",
    }
    return mapping[optimizer]


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_simple_optimizer_variant_native_abi_scorecard"]
