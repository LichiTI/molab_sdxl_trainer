"""Report-only scorecard for simple optimizer variants.

The fp32 Lion/SGD Nesterov path has its own native-canary batch.  The variants
below still need dedicated state layouts or schedule-free state machines before
any kernel can be promoted.  This module records those contracts without
touching trainer defaults or native dispatch.
"""

from __future__ import annotations

import copy
import importlib.util
from typing import Any, Callable, Mapping

import torch

from core.configs import OptimizerType


QUANTIZED_LAYOUT_TARGETS = (
    OptimizerType.LION_8BIT,
    OptimizerType.PAGED_LION_8BIT,
    OptimizerType.SGD_NESTEROV_8BIT,
)
SCHEDULE_FREE_TARGETS = (
    OptimizerType.RADAM_SCHEDULE_FREE,
    OptimizerType.SGD_SCHEDULE_FREE,
)
ALL_VARIANT_TARGETS = QUANTIZED_LAYOUT_TARGETS + SCHEDULE_FREE_TARGETS


def build_simple_optimizer_variant_state_scorecard() -> dict[str, Any]:
    """Build variant state/layout evidence while keeping product routes off."""

    layout_contracts = [_layout_contract(optimizer) for optimizer in QUANTIZED_LAYOUT_TARGETS]
    layout_validations = [_validate_layout_contract(contract) for contract in layout_contracts]
    schedule_cases = _schedule_free_cases()
    rows = _layout_rows(layout_contracts, layout_validations) + _schedule_free_rows(schedule_cases)
    failed_layout = [item for item in layout_validations if item.get("ok") is not True]
    failed_schedule = [case for case in schedule_cases if case.get("ok") is not True]
    ready_layout_count = sum(1 for row in rows if row.get("variant_status") == "layout_spec_ready")
    ready_state_count = sum(1 for row in rows if row.get("variant_status") == "state_machine_reference_ready")
    blockers = _dedupe(
        [f"variant_layout_invalid:{item.get('optimizer_type')}" for item in failed_layout]
        + [
            f"variant_state_machine_not_ready:{case.get('optimizer_type')}:{reason}"
            for case in failed_schedule
            for reason in _strings(case.get("blocked_reasons"))
        ]
    )
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_variant_state_scorecard_v0",
        "gate": "simple_formula_variant_state_layout_reference",
        "ok": not blockers,
        "promotion_ready": False,
        "variant_state_layout_stage_ready": not blockers,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "native_kernel_ready": False,
        "product_native_dispatch_ready": False,
        "target_optimizer_types": [optimizer.value for optimizer in ALL_VARIANT_TARGETS],
        "rows": rows,
        "layout_contracts": layout_contracts,
        "layout_validations": layout_validations,
        "schedule_free_cases": schedule_cases,
        "summary": {
            "target_optimizer_count": len(ALL_VARIANT_TARGETS),
            "layout_target_count": len(QUANTIZED_LAYOUT_TARGETS),
            "layout_spec_ready_count": ready_layout_count,
            "state_machine_target_count": len(SCHEDULE_FREE_TARGETS),
            "state_machine_reference_ready_count": ready_state_count,
            "state_machine_case_count": len(schedule_cases),
            "state_machine_passed_case_count": len(schedule_cases) - len(failed_schedule),
            "native_kernel_ready_count": 0,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": blockers
        + [
            "variant_native_kernel_missing",
            "variant_runtime_canary_missing",
            "variant_product_rollout_review_missing",
        ],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add report-only variant kernel ABI plans for quantized/paged Lion/SGD and schedule-free RAdam/SGD"
            if not blockers
            else "fix simple optimizer variant state/layout blockers"
        ),
        "notes": [
            "This scorecard records state/layout contracts only; it never enables training dispatch.",
            "8-bit and paged variants require dedicated quantized state tensor bindings.",
            "Schedule-free variants require optimizer-owned train/eval and param-group state handling.",
        ],
    }


def _layout_contract(optimizer: OptimizerType) -> dict[str, Any]:
    if optimizer == OptimizerType.LION_8BIT:
        return _quantized_contract(
            optimizer=optimizer,
            optimizer_kind="lion_8bit",
            formula_family="lion",
            state_role="exp_avg",
            implementation="bitsandbytes.optim.Lion8bit",
            paged=False,
            launch_plan="lion_8bit_quantized_state_layout_v0",
        )
    if optimizer == OptimizerType.PAGED_LION_8BIT:
        return _quantized_contract(
            optimizer=optimizer,
            optimizer_kind="paged_lion_8bit",
            formula_family="lion",
            state_role="exp_avg",
            implementation="bitsandbytes.optim.PagedLion8bit",
            paged=True,
            launch_plan="paged_lion_8bit_quantized_state_layout_v0",
        )
    if optimizer == OptimizerType.SGD_NESTEROV_8BIT:
        return _quantized_contract(
            optimizer=optimizer,
            optimizer_kind="sgd_nesterov_8bit",
            formula_family="sgd_nesterov",
            state_role="momentum_buffer",
            implementation="bitsandbytes.optim.SGD8bit(nesterov=True)",
            paged=False,
            launch_plan="sgd_nesterov_8bit_quantized_momentum_layout_v0",
        )
    raise ValueError(f"unsupported simple variant layout target: {optimizer}")


def _quantized_contract(
    *,
    optimizer: OptimizerType,
    optimizer_kind: str,
    formula_family: str,
    state_role: str,
    implementation: str,
    paged: bool,
    launch_plan: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract": launch_plan,
        "optimizer_type": optimizer.value,
        "optimizer_kind": optimizer_kind,
        "optimizer_family": "simple_formula",
        "formula_family": formula_family,
        "implementation_reference": implementation,
        "status": "layout_spec_only",
        "training_path_enabled": False,
        "native_kernel_present": False,
        "launch_plan": launch_plan,
        "state_roles": [
            {"role": "param_flat", "dtype": "param_dtype", "mutable": True},
            {"role": "grad_flat", "dtype": "param_dtype|float32", "mutable": False},
            {
                "role": state_role,
                "dtype": "uint8_blockwise_with_scale",
                "mutable": True,
                "requires_dequantize_before_formula": True,
            },
        ],
        "layout_policy": {
            "quantized_state": True,
            "paged_state": paged,
            "state_precision": "8bit_blockwise",
            "master_state_dtype": "float32_for_reference_replay",
            "requires_bitsandbytes_layout_parity": True,
        },
        "required_hyperparameters": _layout_hparams(formula_family),
    }


def _layout_hparams(formula_family: str) -> list[str]:
    if formula_family == "lion":
        return ["lr", "beta1", "beta2", "weight_decay"]
    return ["lr", "momentum", "weight_decay", "nesterov"]


def _validate_layout_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    roles = {str(item.get("role") or "") for item in contract.get("state_roles", []) if isinstance(item, Mapping)}
    formula_family = str(contract.get("formula_family") or "")
    required_roles = {"param_flat", "grad_flat"}
    required_roles.add("exp_avg" if formula_family == "lion" else "momentum_buffer")
    missing_roles = sorted(required_roles - roles)
    policy = contract.get("layout_policy", {})
    policy_map = policy if isinstance(policy, Mapping) else {}
    missing_hparams = [name for name in contract.get("required_hyperparameters", []) if not str(name or "").strip()]
    ok = (
        not missing_roles
        and not missing_hparams
        and bool(policy_map.get("quantized_state", False))
        and bool(contract.get("native_kernel_present", True)) is False
        and bool(contract.get("training_path_enabled", True)) is False
    )
    return {
        "schema_version": 1,
        "validator": "simple_variant_quantized_layout_validator_v0",
        "ok": ok,
        "optimizer_type": str(contract.get("optimizer_type") or ""),
        "optimizer_kind": str(contract.get("optimizer_kind") or ""),
        "missing_roles": missing_roles,
        "missing_hyperparameters": missing_hparams,
        "native_kernel_present": bool(contract.get("native_kernel_present", False)),
        "training_path_enabled": bool(contract.get("training_path_enabled", False)),
    }


def _layout_rows(
    contracts: list[Mapping[str, Any]],
    validations: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contract, validation in zip(contracts, validations):
        rows.append(
            {
                "optimizer_type": str(contract.get("optimizer_type") or ""),
                "optimizer_kind": str(contract.get("optimizer_kind") or ""),
                "optimizer_family": "simple_formula",
                "variant_status": "layout_spec_ready" if validation.get("ok") is True else "layout_spec_invalid",
                "native_route": "dedicated_quantized_variant_kernel_required",
                "state_layout_ready": validation.get("ok") is True,
                "state_machine_reference_ready": False,
                "native_kernel_ready": False,
                "runtime_canary_ready": False,
                "product_native_dispatch_ready": False,
                "training_path_enabled": False,
                "default_behavior_changed": False,
                "native_dispatch_allowed": False,
                "next_gate": "quantized_state_formula_parity_and_tensor_binding_matrix",
            }
        )
    return rows


def _schedule_free_cases() -> list[dict[str, Any]]:
    if importlib.util.find_spec("schedulefree") is None:
        return [_blocked_schedule_case(optimizer, "schedulefree_unavailable") for optimizer in SCHEDULE_FREE_TARGETS]
    import schedulefree

    return [
        case
        for optimizer, optimizer_class in (
            (OptimizerType.RADAM_SCHEDULE_FREE, schedulefree.RAdamScheduleFree),
            (OptimizerType.SGD_SCHEDULE_FREE, schedulefree.SGDScheduleFree),
        )
        for case in (
            _schedule_free_requires_train_mode_case(optimizer, optimizer_class),
            _schedule_free_roundtrip_case(optimizer, optimizer_class),
        )
    ]


def _schedule_free_requires_train_mode_case(
    optimizer: OptimizerType,
    optimizer_class: Callable[..., torch.optim.Optimizer],
) -> dict[str, Any]:
    param = torch.nn.Parameter(torch.linspace(-0.2, 0.3, 4))
    opt = _make_schedule_free_optimizer(optimizer, optimizer_class, [param])
    param.grad = torch.linspace(0.01, 0.04, 4)
    try:
        opt.step()
    except Exception as exc:
        message = str(exc)
        ok = "train mode" in message
        return {
            "schema_version": 1,
            "case": "step_requires_train_mode",
            "optimizer_type": optimizer.value,
            "ok": ok,
            "covers_mode_toggle": True,
            "error_message": message,
            "blocked_reasons": [] if ok else ["schedule_free_missing_train_mode_guard"],
        }
    return {
        "schema_version": 1,
        "case": "step_requires_train_mode",
        "optimizer_type": optimizer.value,
        "ok": False,
        "covers_mode_toggle": True,
        "blocked_reasons": ["schedule_free_step_allowed_without_train_mode"],
    }


def _schedule_free_roundtrip_case(
    optimizer: OptimizerType,
    optimizer_class: Callable[..., torch.optim.Optimizer],
) -> dict[str, Any]:
    value = torch.linspace(-0.25, 0.35, steps=16).reshape(4, 4)
    grad1 = torch.linspace(0.01, 0.05, steps=value.numel()).reshape_as(value)
    grad2 = torch.linspace(-0.03, 0.02, steps=value.numel()).reshape_as(value)
    param = torch.nn.Parameter(value.detach().clone())
    opt = _make_schedule_free_optimizer(optimizer, optimizer_class, [param])
    opt.train()
    _step(param, opt, grad1)
    after_step = _schedule_free_state_contract(optimizer, opt.state_dict())
    opt.eval()
    after_eval = _schedule_free_state_contract(optimizer, opt.state_dict())
    opt.train()
    after_train = _schedule_free_state_contract(optimizer, opt.state_dict())
    saved_state = copy.deepcopy(opt.state_dict())
    saved_param = param.detach().clone()

    restored_param = torch.nn.Parameter(saved_param.detach().clone())
    restored = _make_schedule_free_optimizer(optimizer, optimizer_class, [restored_param])
    restored.load_state_dict(saved_state)
    _step(param, opt, grad2)
    _step(restored_param, restored, grad2)
    diff = _max_abs(param.detach(), restored_param.detach())
    ok = (
        after_step["has_required_param_state"]
        and after_eval["train_mode"] is False
        and after_train["train_mode"] is True
        and diff <= 1e-5
    )
    return {
        "schema_version": 1,
        "case": "roundtrip_state_machine",
        "optimizer_type": optimizer.value,
        "ok": ok,
        "covers_resume": True,
        "covers_mode_toggle": True,
        "after_step": after_step,
        "after_eval": after_eval,
        "after_train": after_train,
        "max_resume_diff": diff,
        "tolerance": 1e-5,
        "blocked_reasons": [] if ok else ["schedule_free_state_machine_roundtrip_failed"],
    }


def _make_schedule_free_optimizer(
    optimizer: OptimizerType,
    optimizer_class: Callable[..., torch.optim.Optimizer],
    params: list[torch.nn.Parameter],
) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {"lr": 1e-3, "weight_decay": 0.01}
    if optimizer == OptimizerType.SGD_SCHEDULE_FREE:
        kwargs["momentum"] = 0.9
    return optimizer_class(params, **kwargs)


def _step(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer, grad: torch.Tensor) -> None:
    param.grad = grad.detach().clone().to(device=param.device, dtype=param.dtype)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _schedule_free_state_contract(optimizer: OptimizerType, state_dict: Mapping[str, Any]) -> dict[str, Any]:
    state = state_dict.get("state", {}) if isinstance(state_dict, Mapping) else {}
    groups = state_dict.get("param_groups", []) if isinstance(state_dict, Mapping) else []
    first_state = next(iter(state.values()), {}) if isinstance(state, Mapping) and state else {}
    group = groups[0] if groups else {}
    state_keys = sorted(str(key) for key in first_state.keys()) if isinstance(first_state, Mapping) else []
    required = {"z", "exp_avg_sq"} if optimizer == OptimizerType.RADAM_SCHEDULE_FREE else {"z"}
    return {
        "param_state_keys": state_keys,
        "required_param_state_keys": sorted(required),
        "has_required_param_state": required.issubset(set(state_keys)),
        "train_mode": bool(group.get("train_mode", False)),
        "param_group_state_keys": sorted(str(key) for key in group.keys()),
        "k": int(group.get("k", 0) or 0),
        "weight_sum": float(group.get("weight_sum", 0.0) or 0.0),
        "scheduled_lr": float(group.get("scheduled_lr", 0.0) or 0.0),
        "lr_max": float(group.get("lr_max", 0.0) or 0.0),
    }


def _schedule_free_rows(cases: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_optimizer = {
        optimizer: [case for case in cases if case.get("optimizer_type") == optimizer.value]
        for optimizer in SCHEDULE_FREE_TARGETS
    }
    rows: list[dict[str, Any]] = []
    for optimizer, optimizer_cases in by_optimizer.items():
        ready = bool(optimizer_cases) and all(case.get("ok") is True for case in optimizer_cases)
        rows.append(
            {
                "optimizer_type": optimizer.value,
                "optimizer_kind": _schedule_free_kind(optimizer),
                "optimizer_family": "simple_formula",
                "variant_status": "state_machine_reference_ready" if ready else "state_machine_reference_pending",
                "native_route": "dedicated_schedule_free_variant_kernel_required",
                "state_layout_ready": ready,
                "state_machine_reference_ready": ready,
                "native_kernel_ready": False,
                "runtime_canary_ready": False,
                "product_native_dispatch_ready": False,
                "training_path_enabled": False,
                "default_behavior_changed": False,
                "native_dispatch_allowed": False,
                "next_gate": "schedule_free_variant_native_abi_and_resume_matrix",
                "case_count": len(optimizer_cases),
                "passed_case_count": sum(1 for case in optimizer_cases if case.get("ok") is True),
            }
        )
    return rows


def _schedule_free_kind(optimizer: OptimizerType) -> str:
    if optimizer == OptimizerType.RADAM_SCHEDULE_FREE:
        return "radam_schedule_free"
    return "sgd_schedule_free"


def _blocked_schedule_case(optimizer: OptimizerType, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "case": "schedule_free_unavailable",
        "optimizer_type": optimizer.value,
        "ok": False,
        "blocked_reasons": [reason],
    }


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


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
    "ALL_VARIANT_TARGETS",
    "QUANTIZED_LAYOUT_TARGETS",
    "SCHEDULE_FREE_TARGETS",
    "build_simple_optimizer_variant_state_scorecard",
]
