"""Report-only native ABI gate for AdamWScheduleFree.

This scorecard reuses the AdamWScheduleFree state-machine proof and names the
native ABI a future kernel would need.  It deliberately does not implement a
kernel, open native dispatch, or change the existing optimizer path.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.configs import OptimizerType
from core.turbocore_adamw_schedule_free_state_machine_scorecard import (
    build_adamw_schedule_free_state_machine_scorecard,
)


TARGET_OPTIMIZER = OptimizerType.ADAMW_SCHEDULE_FREE
OPTIMIZER_KIND = "adamw_schedule_free"
OPTIMIZER_FAMILY = "adamw_schedule_free"
LAUNCH_PLAN = "adamw_schedule_free_stateful_launch_plan_v0"

_REQUIRED_STATE_ROLES = (
    "z",
    "exp_avg_sq",
)

_REQUIRED_GROUP_ROLES = (
    "train_mode",
    "k",
    "step",
    "lr",
    "scheduled_lr",
    "lr_max",
    "warmup_steps",
    "r",
    "weight_sum",
    "weight_lr_power",
    "weight_decay",
)


def build_adamw_schedule_free_native_abi_scorecard(
    *,
    state_machine_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a report-only ABI contract from the state-machine scorecard."""

    state_machine = dict(
        state_machine_report
        or build_adamw_schedule_free_state_machine_scorecard()
    )
    buffer_contract = _buffer_contract()
    mode_contract = _mode_contract()
    validations = _validations(state_machine, buffer_contract, mode_contract)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    source_ready = bool(state_machine.get("state_machine_reference_ready", False))
    abi_contract_ready = source_ready and not failed
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )

    return {
        "schema_version": 1,
        "scorecard": "turbocore_adamw_schedule_free_native_abi_scorecard_v0",
        "gate": "adamw_schedule_free_native_abi_report_only",
        "ok": abi_contract_ready,
        "promotion_ready": False,
        "abi_contract_ready": abi_contract_ready,
        "state_machine_reference_ready": source_ready,
        "native_ready": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "native_kernel_ready": False,
        "optimizer_type": TARGET_OPTIMIZER.value,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "launch_plan": LAUNCH_PLAN,
        "request_contract": {
            "optimizer_type": TARGET_OPTIMIZER.value,
            "optimizer_kind": OPTIMIZER_KIND,
            "optimizer_family": OPTIMIZER_FAMILY,
            "launch_plan": LAUNCH_PLAN,
            "external_scheduler_policy": "constant_required",
            "route_policy": "report_only_until_native_kernel_and_canary_exist",
        },
        "mode_contract": mode_contract,
        "buffer_contract": buffer_contract,
        "route_decision": _route_decision(abi_contract_ready),
        "validations": validations,
        "state_machine_summary": dict(state_machine.get("summary") or {}),
        "state_machine_blocked_reasons": [
            str(reason) for reason in state_machine.get("blocked_reasons", []) or []
        ],
        "summary": {
            "mode_role_count": len(mode_contract),
            "buffer_role_count": len(buffer_contract),
            "required_param_state_role_count": len(_REQUIRED_STATE_ROLES),
            "required_param_group_role_count": len(_REQUIRED_GROUP_ROLES),
            "validation_count": len(validations),
            "passed_validation_count": len(validations) - len(failed),
            "source_state_machine_ready": source_ready,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "adamw_schedule_free_native_kernel_missing",
                "adamw_schedule_free_runtime_canary_missing",
                "native_dispatch_not_allowed",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "design a parity-only native scratch kernel around the named schedule-free ABI"
            if abi_contract_ready
            else "fix AdamWScheduleFree state-machine blockers before native ABI work"
        ),
        "notes": [
            "This gate is report-only and does not enable native dispatch.",
            "AdamWScheduleFree owns train/eval mode and param_group schedule state.",
            "A future native path must preserve z and exp_avg_sq roles plus param_group counters.",
        ],
    }


def _buffer_contract() -> list[dict[str, Any]]:
    return [
        _tensor("param", "parameter values", "float32|float16|bfloat16", True, "trainer_param"),
        _tensor("grad", "gradient values", "float32|float16|bfloat16", False, "trainer_grad"),
        _tensor("z", "schedule-free interpolation state", "float32|float16|bfloat16", True, "optimizer_state.z"),
        _tensor(
            "exp_avg_sq",
            "AdamW second moment state",
            "float32|float16|bfloat16",
            True,
            "optimizer_state.exp_avg_sq",
        ),
        _scalar("k", "schedule-free raw update counter", "int64", "param_group.k"),
        _scalar("step", "native launch step alias derived from k", "int64", "param_group.k"),
        _scalar("lr", "requested learning rate", "float32", "param_group.lr"),
        _scalar("scheduled_lr", "warmup-adjusted learning rate", "float32", "param_group.scheduled_lr"),
        _scalar("lr_max", "maximum scheduled learning rate", "float32", "param_group.lr_max"),
        _scalar("warmup_steps", "schedule-free warmup length", "int64", "param_group.warmup_steps"),
        _scalar("r", "schedule-free weight exponent", "float32", "param_group.r"),
        _scalar("weight_sum", "schedule-free averaging accumulator", "float32", "param_group.weight_sum"),
        _scalar("weight_lr_power", "schedule-free LR weighting power", "float32", "param_group.weight_lr_power"),
        _scalar("weight_decay", "decoupled AdamW weight decay", "float32", "param_group.weight_decay"),
    ]


def _mode_contract() -> list[dict[str, Any]]:
    return [
        {
            "role": "train_mode",
            "kind": "mode_flag",
            "source": "param_group.train_mode",
            "required": True,
            "native_train_behavior": "updates z, exp_avg_sq, k, scheduled_lr, lr_max, weight_sum",
            "native_eval_behavior": "must not perform optimizer update while eval state is active",
        },
        {
            "role": "train_eval_transition",
            "kind": "state_machine",
            "source": "optimizer.train()/optimizer.eval()",
            "required": True,
            "native_requirement": "future native route must receive the current mode explicitly",
        },
    ]


def _tensor(role: str, meaning: str, dtype: str, mutable: bool, source: str) -> dict[str, Any]:
    return {
        "role": role,
        "kind": "tensor",
        "meaning": meaning,
        "dtype": dtype,
        "mutable": mutable,
        "source": source,
        "required": True,
    }


def _scalar(role: str, meaning: str, dtype: str, source: str) -> dict[str, Any]:
    return {
        "role": role,
        "kind": "scalar",
        "meaning": meaning,
        "dtype": dtype,
        "mutable": role in {"step", "scheduled_lr", "lr_max", "weight_sum"},
        "source": source,
        "required": True,
    }


def _validations(
    state_machine: Mapping[str, Any],
    buffer_contract: Sequence[Mapping[str, Any]],
    mode_contract: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    state_contract = dict(state_machine.get("state_contract") or {})
    observed_param_state = [str(item) for item in state_contract.get("param_state_keys", []) or []]
    observed_group_state = [
        "step" if str(item) == "k" else str(item)
        for item in state_contract.get("param_group_state_keys", []) or []
    ]
    buffer_roles = [str(item.get("role", "")) for item in buffer_contract]
    mode_roles = [str(item.get("role", "")) for item in mode_contract]
    return [
        _validation(
            "state_machine_reference_ready",
            bool(state_machine.get("state_machine_reference_ready", False)),
            "adamw_schedule_free_state_machine_not_ready",
            {"state_machine_blocked_reasons": list(state_machine.get("blocked_reasons", []) or [])},
        ),
        _validation(
            "mode_contract_names_train_eval",
            {"train_mode", "train_eval_transition"}.issubset(set(mode_roles)),
            "adamw_schedule_free_train_eval_mode_contract_missing",
            {"mode_roles": mode_roles},
        ),
        _validation(
            "param_state_roles_match_state_machine",
            set(_REQUIRED_STATE_ROLES).issubset(set(buffer_roles))
            and set(_REQUIRED_STATE_ROLES).issubset(set(observed_param_state)),
            "adamw_schedule_free_param_state_roles_missing",
            {
                "required_state_roles": list(_REQUIRED_STATE_ROLES),
                "observed_param_state_keys": observed_param_state,
            },
        ),
        _validation(
            "param_group_roles_named",
            set(_REQUIRED_GROUP_ROLES).issubset(set(buffer_roles) | {"train_mode"})
            and {"train_mode", "step", "warmup_steps", "r", "weight_sum"}.issubset(
                set(observed_group_state)
            ),
            "adamw_schedule_free_param_group_roles_missing",
            {
                "required_group_roles": list(_REQUIRED_GROUP_ROLES),
                "observed_group_state_keys": observed_group_state,
            },
        ),
        _validation(
            "report_only_native_route_locked",
            True,
            "adamw_schedule_free_native_route_unexpectedly_enabled",
            {
                "native_ready": False,
                "training_path_enabled": False,
                "native_dispatch_allowed": False,
                "default_behavior_changed": False,
            },
        ),
    ]


def _validation(name: str, ok: bool, blocker: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }
    if extra:
        payload.update(dict(extra))
    return payload


def _route_decision(abi_contract_ready: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "decision": "blocked_before_dispatch" if abi_contract_ready else "blocked",
        "reason": (
            "abi_contract_named_but_kernel_and_canary_missing"
            if abi_contract_ready
            else "state_machine_or_abi_contract_not_ready"
        ),
        "native_ready": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
    }


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_adamw_schedule_free_native_abi_scorecard"]
