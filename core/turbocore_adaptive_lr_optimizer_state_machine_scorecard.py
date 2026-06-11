"""Report-only state-machine scorecard for adaptive LR optimizers.

Prodigy, DAdapt, and local AutoProdigy own dynamic learning-rate estimates.
They cannot safely reuse the exact AdamW native update path until their global
state, scheduler coupling, checkpoint shape, and runtime authority are modeled.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch

from core.configs import OptimizerType
from core.lulynx_trainer.auto_prodigy_optimizer import AutoProdigy
from core.lulynx_trainer.optimizer_capabilities import optimizer_capability_report


ADAPTIVE_LR_OPTIMIZERS = (
    OptimizerType.PRODIGY,
    OptimizerType.AUTO_PRODIGY,
    OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE,
    OptimizerType.DADAPTATION,
    OptimizerType.DADAPT_ADAM_PREPRINT,
    OptimizerType.DADAPT_ADAGRAD,
    OptimizerType.DADAPT_ADAM,
    OptimizerType.DADAPT_ADAN,
    OptimizerType.DADAPT_ADAN_IP,
    OptimizerType.DADAPT_LION,
    OptimizerType.DADAPT_SGD,
)


def build_adaptive_lr_optimizer_state_machine_scorecard() -> dict[str, Any]:
    """Validate adaptive LR optimizer contracts without native dispatch."""

    rows = _contract_rows()
    cases = [_auto_prodigy_roundtrip_case(), _adamw_reuse_guard_case(rows)]
    failed = [case for case in cases if not bool(case.get("ok", False))]
    blockers = _dedupe(str(reason) for case in failed for reason in case.get("blocked_reasons", []) or [])
    expected = {optimizer.value for optimizer in ADAPTIVE_LR_OPTIMIZERS}
    present = {str(row.get("optimizer_type", "")) for row in rows}
    missing = sorted(expected - present)
    classified = not missing and all(bool(row.get("state_machine_required", False)) for row in rows)
    scheduler_constrained = all(
        str(row.get("native_external_scheduler_policy", "")) == "constant_required_before_native_dispatch"
        for row in rows
    )
    adamw_reuse_blocked = all(not bool(row.get("adamw_kernel_compatible", True)) for row in rows)
    ready = not failed and classified and scheduler_constrained and adamw_reuse_blocked
    if missing:
        blockers.extend(f"missing_adaptive_lr_contract:{name}" for name in missing)

    return {
        "schema_version": 1,
        "scorecard": "turbocore_adaptive_lr_optimizer_state_machine_scorecard_v0",
        "gate": "adaptive_lr_optimizer_state_machine_reference",
        "ok": ready,
        "promotion_ready": False,
        "state_machine_reference_ready": ready,
        "adaptive_family_classified": classified,
        "scheduler_coupling_constrained": scheduler_constrained,
        "adamw_kernel_reuse_blocked": adamw_reuse_blocked,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "optimizer_group": "adaptive_lr_prodigy_dadapt",
        "request_contract": {
            "optimizer_types": sorted(expected),
            "native_route_policy": "no_dispatch_until_state_machine_native_abi_review",
            "external_scheduler_policy": "constant_required_before_native_dispatch",
            "runtime_authority": "existing_python_or_third_party_optimizer",
            "checkpoint_resume_required": True,
        },
        "state_contract": {
            "requires_optimizer_owned_dynamic_lr": True,
            "requires_global_or_param_group_state": True,
            "requires_resume_shape_validation": True,
            "requires_non_finite_and_growth_guard": True,
            "adamw_state_schema_compatible": False,
        },
        "rows": rows,
        "cases": cases,
        "summary": {
            "optimizer_count": len(rows),
            "expected_optimizer_count": len(expected),
            "classified_count": sum(1 for row in rows if bool(row.get("state_machine_required", False))),
            "dependency_available_count": sum(1 for row in rows if bool(row.get("dependency_available", False))),
            "local_live_reference_count": sum(1 for case in cases if case.get("uses_local_optimizer") is True),
            "passed_case_count": len(cases) - len(failed),
            "required_case_count": len(cases),
        },
        "promotion_blockers": blockers
        + ["native_state_machine_abi_missing", "runtime_canary_missing", "owner_release_hold_missing"],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "design adaptive LR native ABI with explicit global state and constant scheduler boundary"
            if ready
            else "fix adaptive LR state-machine reference blockers"
        ),
        "notes": [
            "This scorecard is report-only and does not instantiate third-party Prodigy or DAdapt optimizers.",
            "AutoProdigy is used as the local live reference because its state_dict is owned by this project.",
            "Adaptive LR optimizers must not share the exact AdamW native kernel.",
        ],
    }


def _contract_rows() -> list[dict[str, Any]]:
    capabilities = optimizer_capability_report(ADAPTIVE_LR_OPTIMIZERS).get("optimizers", [])
    by_name = {str(item.get("optimizer_type", "")): item for item in capabilities if isinstance(item, Mapping)}
    return [_contract_row(optimizer, by_name.get(optimizer.value, {})) for optimizer in ADAPTIVE_LR_OPTIMIZERS]


def _contract_row(optimizer: OptimizerType, capability: Mapping[str, Any]) -> dict[str, Any]:
    schema = _state_schema(optimizer)
    current_scheduler = str(capability.get("scheduler_policy", "standard") or "standard")
    return {
        "optimizer_type": optimizer.value,
        "implementation": str(capability.get("implementation", "") or ""),
        "dependency": str(capability.get("dependency", "") or ""),
        "dependency_available": bool(capability.get("dependency_available", False)),
        "fallback_optimizer": str(capability.get("fallback_optimizer", "") or ""),
        "current_scheduler_policy": current_scheduler,
        "native_external_scheduler_policy": "constant_required_before_native_dispatch",
        "current_scheduler_guard": "already_constant" if current_scheduler == "constant" else "native_gate_requires_constant_review",
        "state_machine_kind": schema["kind"],
        "state_schema_level": schema["level"],
        "dynamic_global_fields": schema["dynamic_global_fields"],
        "param_state_model": schema["param_state_model"],
        "state_machine_required": True,
        "requires_dynamic_global_state": True,
        "checkpoint_resume_required": True,
        "adamw_kernel_compatible": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "notes": schema["notes"],
    }


def _state_schema(optimizer: OptimizerType) -> dict[str, Any]:
    if optimizer == OptimizerType.AUTO_PRODIGY:
        return {
            "kind": "global_distance_estimate_with_average_weights",
            "level": "observed_local_state_dict",
            "dynamic_global_fields": ["auto_prodigy_global.distance", "auto_prodigy_global.steps", "auto_prodigy_eval_mode"],
            "param_state_model": ["tick", "m1", "m2", "origin", "average", "train_weight_when_eval"],
            "notes": ["Local optimizer; live roundtrip case validates these keys."],
        }
    if optimizer == OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE:
        return {
            "kind": "prodigy_distance_plus_schedule_free",
            "level": "conceptual_external_contract",
            "dynamic_global_fields": ["distance_estimate", "growth_limiter", "schedule_free_mode", "averaging_state"],
            "param_state_model": ["moment_estimates", "origin_or_reference_weight", "schedule_free_average"],
            "notes": ["Third-party implementation; native ABI must be reviewed before dispatch."],
        }
    if optimizer in {OptimizerType.PRODIGY}:
        return {
            "kind": "prodigy_global_distance_estimate",
            "level": "conceptual_external_contract",
            "dynamic_global_fields": ["distance_estimate", "growth_limiter", "safeguard_warmup"],
            "param_state_model": ["moment_estimates", "initial_or_reference_weight", "distance_accumulators"],
            "notes": ["Third-party implementation; exact state names are not copied into this scorecard."],
        }
    return {
        "kind": "dadaptation_global_distance_estimate",
        "level": "conceptual_external_contract",
        "dynamic_global_fields": ["global_d_estimate", "growth_rate", "adaptation_accumulators"],
        "param_state_model": ["optimizer_specific_moments", "adaptation_accumulators"],
        "notes": ["DAdapt variants need per-variant ABI review; plugin fallback is not a native route."],
    }


def _auto_prodigy_roundtrip_case() -> dict[str, Any]:
    value = torch.linspace(-0.25, 0.35, steps=16, dtype=torch.float32).reshape(4, 4)
    grad1 = torch.linspace(0.01, 0.08, steps=value.numel(), dtype=torch.float32).reshape_as(value)
    grad2 = torch.linspace(-0.04, 0.03, steps=value.numel(), dtype=torch.float32).reshape_as(value)
    param = torch.nn.Parameter(value.clone())
    optimizer = AutoProdigy([param], lr=1.0, d0=1e-5, growth_rate=1.01, max_update_rms_ratio=None)

    _step(param, optimizer, grad1)
    after_step = _auto_prodigy_state_contract(optimizer.state_dict())
    optimizer.eval()
    after_eval = _auto_prodigy_state_contract(optimizer.state_dict())
    optimizer.train()
    after_train = _auto_prodigy_state_contract(optimizer.state_dict())
    saved_state = copy.deepcopy(optimizer.state_dict())
    saved_param = param.detach().clone()

    restored_param = torch.nn.Parameter(saved_param.clone())
    restored = AutoProdigy([restored_param], lr=1.0, d0=1e-5, growth_rate=1.01, max_update_rms_ratio=None)
    restored.load_state_dict(saved_state)
    _step(param, optimizer, grad2)
    _step(restored_param, restored, grad2)
    diff = _max_abs(param.detach(), restored_param.detach())
    ok = (
        after_step["has_required_param_state"]
        and after_step["has_required_global_state"]
        and after_eval["eval_mode"] is True
        and after_eval["has_train_weight_stash"] is True
        and after_train["eval_mode"] is False
        and diff <= 1e-6
    )
    return {
        "schema_version": 1,
        "case": "auto_prodigy_state_roundtrip",
        "ok": ok,
        "uses_local_optimizer": True,
        "covers_resume": True,
        "covers_eval_train_toggle": True,
        "after_step": after_step,
        "after_eval": after_eval,
        "after_train": after_train,
        "max_resume_diff": diff,
        "tolerance": 1e-6,
        "blocked_reasons": [] if ok else ["auto_prodigy_state_roundtrip_failed"],
    }


def _adamw_reuse_guard_case(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    incompatible = [str(row.get("optimizer_type", "")) for row in rows if bool(row.get("adamw_kernel_compatible", True))]
    missing_dynamic_state = [
        str(row.get("optimizer_type", "")) for row in rows if not bool(row.get("requires_dynamic_global_state", False))
    ]
    ok = not incompatible and not missing_dynamic_state
    return {
        "schema_version": 1,
        "case": "adamw_kernel_reuse_guard",
        "ok": ok,
        "incompatible_with_exact_adamw_kernel_count": len(rows) - len(incompatible),
        "requires_dynamic_global_state_count": len(rows) - len(missing_dynamic_state),
        "blocked_reasons": [] if ok else ["adaptive_lr_adamw_reuse_guard_failed"],
    }


def _auto_prodigy_state_contract(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    state = state_dict.get("state", {}) if isinstance(state_dict, Mapping) else {}
    groups = state_dict.get("param_groups", []) if isinstance(state_dict, Mapping) else []
    first_state = next(iter(state.values()), {}) if isinstance(state, Mapping) and state else {}
    state_keys = sorted(str(key) for key in first_state.keys()) if isinstance(first_state, Mapping) else []
    global_state = dict(state_dict.get("auto_prodigy_global", {}) or {})
    return {
        "top_level_keys": sorted(str(key) for key in state_dict.keys()),
        "param_group_keys": sorted(str(key) for key in (groups[0].keys() if groups else []) if key != "params"),
        "param_state_keys": state_keys,
        "has_required_param_state": {"tick", "m1", "m2", "origin", "average"}.issubset(set(state_keys)),
        "has_train_weight_stash": "train_weight" in set(state_keys),
        "has_required_global_state": {"distance", "steps"}.issubset(set(global_state.keys())),
        "distance": float(global_state.get("distance", 0.0) or 0.0),
        "steps": int(global_state.get("steps", 0) or 0),
        "eval_mode": bool(state_dict.get("auto_prodigy_eval_mode", False)),
    }


def _step(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer, grad: torch.Tensor) -> None:
    param.grad = grad.detach().clone().to(dtype=param.dtype)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["ADAPTIVE_LR_OPTIMIZERS", "build_adaptive_lr_optimizer_state_machine_scorecard"]
