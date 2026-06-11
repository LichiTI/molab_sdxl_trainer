"""Report-only ABI scorecard for V2-P7 simple optimizer kernels."""

from __future__ import annotations

from typing import Any, Mapping

from core.configs import OptimizerType


EXACT_SIMPLE_ABI_TARGETS = (OptimizerType.LION, OptimizerType.SGD_NESTEROV)


def build_simple_optimizer_abi_scorecard(
    *,
    reference_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe optimizer_kind contracts before any native kernel exists."""

    reference = dict(reference_report or {})
    reference_ready = bool(reference.get("first_stage_ready", False))
    contracts = [_contract_for_optimizer(optimizer) for optimizer in EXACT_SIMPLE_ABI_TARGETS]
    validations = [_validate_contract(contract) for contract in contracts]
    rows = [_row_for_contract(contract, validation) for contract, validation in zip(contracts, validations)]
    layout_pending = _layout_pending_rows()
    state_machine_pending = _state_machine_pending_rows()
    all_contracts_ok = all(bool(validation.get("ok", False)) for validation in validations)
    first_abi_stage_ready = reference_ready and all_contracts_ok
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_abi_scorecard_v0",
        "gate": "simple_formula_optimizer_kind_abi",
        "ok": all_contracts_ok,
        "promotion_ready": False,
        "first_abi_stage_ready": first_abi_stage_ready,
        "reference_stage_ready": reference_ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "native_contract_stage_ready": first_abi_stage_ready,
        "exact_kernel_targets": [optimizer.value for optimizer in EXACT_SIMPLE_ABI_TARGETS],
        "rows": rows + layout_pending + state_machine_pending,
        "contracts": contracts,
        "validations": validations,
        "summary": {
            "exact_kernel_target_count": len(EXACT_SIMPLE_ABI_TARGETS),
            "abi_ready_optimizer_count": sum(1 for row in rows if row["abi_status"] == "optimizer_kind_contract_ready"),
            "native_contract_ready_count": sum(1 for row in rows if row["abi_status"] == "optimizer_kind_contract_ready"),
            "layout_pending_count": len(layout_pending),
            "state_machine_pending_count": len(state_machine_pending),
            "native_kernel_ready_count": 0,
        },
        "promotion_blockers": [
            "lion_native_kernel_parity_missing",
            "sgd_nesterov_native_kernel_parity_missing",
            "runtime_canary_hit_missing",
            "e2e_no_regression_missing",
        ],
        "blocked_reasons": [] if first_abi_stage_ready else _blocked_reasons(reference_ready, validations),
        "recommended_next_step": "add kernel registry dry-run launch and CPU reference guard for Lion and SGDNesterov",
        "notes": [
            "This ABI scorecard names optimizer_kind values but does not register kernels.",
            "Lion8bit, PagedLion8bit, and SGDNesterov8bit wait for quantized/paged state layout.",
            "Schedule-free optimizers wait for state-machine reference before ABI promotion.",
        ],
    }


def _contract_for_optimizer(optimizer: OptimizerType) -> dict[str, Any]:
    if optimizer == OptimizerType.LION:
        return {
            "schema_version": 1,
            "contract": "lion_flat_fp32_cuda_kernel_v0",
            "optimizer": optimizer.value,
            "optimizer_kind": "lion",
            "optimizer_family": "simple_formula",
            "status": "contract_only",
            "available": False,
            "native_kernel_present": False,
            "training_path_enabled": False,
            "launch_plan": "lion_flat_fp32_launch_plan_v0",
            "kernel_name": "lion_flat_fp32_cuda_v0",
            "input_buffers": [
                {"role": "param_flat", "dtype": "float32", "mutable": True},
                {"role": "grad_flat", "dtype": "float32", "mutable": False},
                {"role": "exp_avg", "dtype": "float32", "mutable": True},
            ],
            "required_hyperparameters": ["lr", "beta1", "beta2", "weight_decay"],
            "numeric_policy": {
                "state_dtype": "float32",
                "parameter_dtype": "float32",
                "gradient_dtype": "float32",
                "decoupled_weight_decay": True,
                "sign_update": True,
            },
        }
    if optimizer == OptimizerType.SGD_NESTEROV:
        return {
            "schema_version": 1,
            "contract": "sgd_nesterov_flat_fp32_cuda_kernel_v0",
            "optimizer": optimizer.value,
            "optimizer_kind": "sgd_nesterov",
            "optimizer_family": "simple_formula",
            "status": "contract_only",
            "available": False,
            "native_kernel_present": False,
            "training_path_enabled": False,
            "launch_plan": "sgd_nesterov_flat_fp32_launch_plan_v0",
            "kernel_name": "sgd_nesterov_flat_fp32_cuda_v0",
            "input_buffers": [
                {"role": "param_flat", "dtype": "float32", "mutable": True},
                {"role": "grad_flat", "dtype": "float32", "mutable": False},
                {"role": "momentum_buffer", "dtype": "float32", "mutable": True},
            ],
            "required_hyperparameters": ["lr", "momentum", "weight_decay"],
            "numeric_policy": {
                "state_dtype": "float32",
                "parameter_dtype": "float32",
                "gradient_dtype": "float32",
                "coupled_weight_decay": True,
                "nesterov": True,
            },
        }
    raise ValueError(f"unsupported simple optimizer ABI target: {optimizer}")


def _validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    roles = {str(item.get("role", "")) for item in contract.get("input_buffers", []) if isinstance(item, Mapping)}
    required_roles = {"param_flat", "grad_flat"}
    optimizer_kind = str(contract.get("optimizer_kind", ""))
    if optimizer_kind == "lion":
        required_roles.add("exp_avg")
    if optimizer_kind == "sgd_nesterov":
        required_roles.add("momentum_buffer")
    missing_roles = sorted(required_roles - roles)
    missing_hparams = [
        name for name in contract.get("required_hyperparameters", []) if not str(name or "").strip()
    ]
    ok = (
        not missing_roles
        and not missing_hparams
        and str(contract.get("optimizer_family")) == "simple_formula"
        and bool(contract.get("native_kernel_present", True)) is False
        and bool(contract.get("training_path_enabled", True)) is False
    )
    return {
        "schema_version": 1,
        "validator": "simple_optimizer_kind_contract_validator_v0",
        "ok": ok,
        "optimizer": str(contract.get("optimizer", "")),
        "optimizer_kind": optimizer_kind,
        "contract": str(contract.get("contract", "")),
        "launch_plan": str(contract.get("launch_plan", "")),
        "missing_roles": missing_roles,
        "missing_hyperparameters": missing_hparams,
        "native_kernel_present": bool(contract.get("native_kernel_present", False)),
        "training_path_enabled": bool(contract.get("training_path_enabled", False)),
    }


def _row_for_contract(contract: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "optimizer_type": str(contract.get("optimizer", "")),
        "optimizer_kind": str(contract.get("optimizer_kind", "")),
        "family": "simple_formula",
        "abi_status": "optimizer_kind_contract_ready" if validation.get("ok", False) else "optimizer_kind_contract_invalid",
        "launch_plan": str(contract.get("launch_plan", "")),
        "kernel_name": str(contract.get("kernel_name", "")),
        "native_kernel_status": "scratch_parity_tracked_by_kernel_scorecard",
        "training_path_enabled": False,
        "default_behavior_changed": False,
    }


def _layout_pending_rows() -> list[dict[str, Any]]:
    return [
        _pending_row(OptimizerType.LION_8BIT, "lion_8bit", "quantized_state_layout_pending"),
        _pending_row(OptimizerType.PAGED_LION_8BIT, "paged_lion_8bit", "paged_quantized_state_layout_pending"),
        _pending_row(OptimizerType.SGD_NESTEROV_8BIT, "sgd_nesterov_8bit", "quantized_momentum_layout_pending"),
    ]


def _state_machine_pending_rows() -> list[dict[str, Any]]:
    return [
        _pending_row(OptimizerType.RADAM_SCHEDULE_FREE, "radam_schedule_free", "state_machine_reference_pending"),
        _pending_row(OptimizerType.SGD_SCHEDULE_FREE, "sgd_schedule_free", "state_machine_reference_pending"),
    ]


def _pending_row(optimizer: OptimizerType, optimizer_kind: str, status: str) -> dict[str, Any]:
    return {
        "optimizer_type": optimizer.value,
        "optimizer_kind": optimizer_kind,
        "family": "simple_formula",
        "abi_status": status,
        "launch_plan": "",
        "kernel_name": "",
        "native_kernel_status": "not_started",
        "training_path_enabled": False,
        "default_behavior_changed": False,
    }


def _blocked_reasons(reference_ready: bool, validations: list[Mapping[str, Any]]) -> list[str]:
    reasons = []
    if not reference_ready:
        reasons.append("simple_formula_reference_stage_not_ready")
    for validation in validations:
        if not bool(validation.get("ok", False)):
            reasons.append(f"invalid_optimizer_kind_contract:{validation.get('optimizer_kind')}")
    return reasons


__all__ = ["build_simple_optimizer_abi_scorecard"]
