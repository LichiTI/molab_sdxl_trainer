"""Report-only training tensor binding canary for schedule-free plugin optimizers."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch

from core.configs import SchedulerType
from core.lulynx_trainer.config import LulynxConfig, OptimizerType
from core.lulynx_trainer.optimizer_plugin_support import PLUGIN_SCHEDULE_FREE_OPTIMIZERS
from core.lulynx_trainer.trainer import LulynxTrainer
from core.turbocore_plugin_schedulefree_checkpoint_adapter_scorecard import (
    build_plugin_schedulefree_checkpoint_adapter_scorecard,
)
from core.turbocore_tensor_handle_registry import TurboCoreTensorHandleRegistry


TARGET_PLUGIN_OPTIMIZERS = tuple(sorted(PLUGIN_SCHEDULE_FREE_OPTIMIZERS))
BINDING_SCHEMA = "plugin_schedulefree_training_tensor_binding_request_v0"
PROBE_KIND = "plugin_schedulefree_training_tensor_binding_canary_v0"
TOLERANCE = 1e-5


class _Injector:
    def __init__(self, value: torch.Tensor) -> None:
        self.param = torch.nn.Parameter(value.detach().clone())
        self.injected_layers = {}

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [self.param]


def build_plugin_schedulefree_training_tensor_binding_canary_scorecard(
    *,
    checkpoint_adapter_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate schedule-free tensor binding shape without native dispatch."""

    checkpoint_adapter = dict(
        checkpoint_adapter_report or build_plugin_schedulefree_checkpoint_adapter_scorecard()
    )
    cases = [_training_tensor_binding_case(name) for name in TARGET_PLUGIN_OPTIMIZERS]
    failed = [case for case in cases if not bool(case.get("ok", False))]
    blockers = _dedupe(str(reason) for case in failed for reason in case.get("blocked_reasons", []) or [])
    checkpoint_ready = bool(checkpoint_adapter.get("checkpoint_adapter_proof_ready", False))
    ready = checkpoint_ready and not failed and bool(cases)
    if not checkpoint_ready:
        blockers.append("selected_schedulefree_checkpoint_adapter_proof_missing")
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_schedulefree_training_tensor_binding_canary_scorecard_v0",
        "gate": "plugin_schedulefree_training_tensor_binding_canary",
        "ok": ready,
        "promotion_ready": False,
        "training_tensor_binding_canary_ready": ready,
        "checkpoint_adapter_proof_ready": checkpoint_ready,
        "binding_request_shape_ready": ready,
        "non_mutating_binding_probe_ready": all(
            bool(case.get("non_mutating_binding_probe", False)) for case in cases
        ),
        "e2e_no_regression_ready": all(bool(case.get("e2e_no_regression", False)) for case in cases),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "probe_kind": PROBE_KIND,
        "binding_schema": BINDING_SCHEMA,
        "selected_optimizer_family": "schedule_free_state_machine",
        "checkpoint_adapter_summary": dict(checkpoint_adapter.get("summary") or {}),
        "binding_contract": {
            "schema": BINDING_SCHEMA,
            "handle_kind": "current_process_torch_tensor_handles",
            "selected_optimizer_names": list(TARGET_PLUGIN_OPTIMIZERS),
            "required_roles": ["param", "grad"],
            "state_roles": "optimizer_state_tensors_by_key",
            "pointer_exported": False,
            "training_dispatch": False,
            "native_update_authority": "none_until_review",
            "fallback_authority": "selected_pytorch_optimizer_plugin",
        },
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "passed_case_count": len(cases) - len(failed),
            "binding_request_shape_case_count": sum(
                1 for case in cases if case.get("binding_request_shape_ready") is True
            ),
            "non_mutating_binding_case_count": sum(
                1 for case in cases if case.get("non_mutating_binding_probe") is True
            ),
            "e2e_no_regression_case_count": sum(
                1 for case in cases if case.get("e2e_no_regression") is True
            ),
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "selected_schedulefree_native_kernel_missing",
                "selected_schedulefree_runtime_dispatch_disabled_pending_review",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "build report-only runtime dispatch shadow for selected schedule-free plugin optimizers"
            if ready
            else "fix selected schedule-free training tensor binding canary blockers"
        ),
        "notes": [
            "This canary registers current-process tensor handles only.",
            "It proves the binding request is non-mutating and leaves the selected plugin optimizer authoritative.",
            "No native kernel is called and no training dispatch path is enabled.",
        ],
    }


def _training_tensor_binding_case(name: str) -> dict[str, Any]:
    try:
        return _run_training_tensor_binding_case(name)
    except Exception as exc:
        return {
            "schema_version": 1,
            "case": f"training_tensor_binding_{name}",
            "optimizer_name": name,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": [f"plugin_schedulefree_training_tensor_binding_failed:{name}:{type(exc).__name__}"],
        }


def _run_training_tensor_binding_case(name: str) -> dict[str, Any]:
    value = torch.linspace(-0.25, 0.35, steps=16, dtype=torch.float32).reshape(4, 4)
    grad1 = torch.linspace(0.01, 0.05, steps=value.numel(), dtype=torch.float32).reshape_as(value)
    grad2 = torch.linspace(-0.03, 0.02, steps=value.numel(), dtype=torch.float32).reshape_as(value)

    prime = _make_trainer(name, value)
    prime_optimizer = prime._create_optimizer()
    _step(prime, prime_optimizer, grad1)
    saved_param = prime.lora_injector.param.detach().clone()
    saved_state = copy.deepcopy(prime_optimizer.state_dict())

    reference = _make_trainer(name, saved_param)
    reference_optimizer = reference._create_optimizer()
    reference_optimizer.load_state_dict(copy.deepcopy(saved_state))

    candidate = _make_trainer(name, saved_param)
    candidate_optimizer = candidate._create_optimizer()
    candidate_optimizer.load_state_dict(copy.deepcopy(saved_state))
    candidate_param = candidate.lora_injector.param
    candidate_param.grad = grad2.detach().clone().to(dtype=candidate_param.dtype)

    before_binding = candidate_param.detach().clone()
    binding_request = _build_binding_request(name, candidate_optimizer, candidate_param)
    after_binding_diff = _max_abs(before_binding, candidate_param.detach())

    _step(reference, reference_optimizer, grad2)
    if hasattr(candidate_optimizer, "train"):
        candidate_optimizer.train()
    candidate_optimizer.step()
    candidate_optimizer.zero_grad(set_to_none=True)

    param_diff = _max_abs(reference.lora_injector.param.detach(), candidate_param.detach())
    state_compare = _compare_state_dicts(reference_optimizer.state_dict(), candidate_optimizer.state_dict())
    e2e_ok = param_diff <= TOLERANCE and bool(state_compare["ok"])
    binding_ready = bool(binding_request["readiness"]["request_shape_ready"])
    non_mutating = after_binding_diff == 0.0
    ok = binding_ready and non_mutating and e2e_ok
    return {
        "schema_version": 1,
        "case": f"training_tensor_binding_{name}",
        "optimizer_name": name,
        "optimizer_class": type(candidate_optimizer).__name__,
        "ok": ok,
        "binding_request_shape_ready": binding_ready,
        "non_mutating_binding_probe": non_mutating,
        "e2e_no_regression": e2e_ok,
        "binding_request": binding_request,
        "max_binding_param_diff": after_binding_diff,
        "max_e2e_param_diff": param_diff,
        "state_compare": state_compare,
        "tolerance": TOLERANCE,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [] if ok else _case_blockers(binding_ready, non_mutating, e2e_ok, state_compare),
    }


def _build_binding_request(
    name: str,
    optimizer: torch.optim.Optimizer,
    param: torch.nn.Parameter,
) -> dict[str, Any]:
    if param.grad is None:
        raise ValueError("schedule-free tensor binding canary requires a live gradient")
    registry = TurboCoreTensorHandleRegistry(namespace=f"schedulefree_{name}")
    roles: dict[str, str] = {
        "param": registry.register(param, role="param").handle_id,
        "grad": registry.register(param.grad.detach().contiguous(), role="grad").handle_id,
    }
    state = _first_live_state(optimizer)
    for key, value in state.items():
        if torch.is_tensor(value) and value.dtype.is_floating_point:
            role = f"state_{key}"
            roles[role] = registry.register(value.detach().contiguous(), role=role).handle_id
    snapshot = registry.snapshot()
    state_dict_contract = _state_contract(optimizer.state_dict())
    readiness = _binding_readiness(snapshot, roles, state_dict_contract)
    return {
        "schema_version": 1,
        "schema": BINDING_SCHEMA,
        "probe_kind": PROBE_KIND,
        "selected_optimizer_name": name,
        "optimizer_class": type(optimizer).__name__,
        "selected_optimizer_family": "schedule_free_state_machine",
        "handle_kind": "current_process_torch_tensor_handles",
        "required_roles": ["param", "grad"],
        "reported_roles": sorted(roles),
        "role_handles": roles,
        "registry_snapshot": snapshot,
        "state_dict_contract": state_dict_contract,
        "pointer_exported": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "readiness": readiness,
    }


def _binding_readiness(
    snapshot: Mapping[str, Any],
    roles: Mapping[str, str],
    state_contract: Mapping[str, Any],
) -> dict[str, Any]:
    reported = set(roles)
    required_roles_present = {"param", "grad"}.issubset(reported)
    state_present = bool(state_contract.get("state_present", False))
    pointer_exported = bool(snapshot.get("pointer_exported", False))
    training_path_enabled = bool(snapshot.get("training_path_enabled", False))
    request_shape_ready = bool(
        required_roles_present
        and state_present
        and int(snapshot.get("handle_count", 0) or 0) >= 2
        and not pointer_exported
        and not training_path_enabled
    )
    return {
        "schema_version": 1,
        "request_shape_ready": request_shape_ready,
        "native_binding_ready": False,
        "performance_test_ready": False,
        "training_path_enabled": False,
        "required_roles_present": required_roles_present,
        "state_present": state_present,
        "pointer_exported": pointer_exported,
        "blocked_reasons": [
            "selected_schedulefree_native_kernel_missing",
            "runtime_dispatch_review_missing",
        ],
    }


def _make_trainer(name: str, value: torch.Tensor) -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = LulynxConfig(
        optimizer_type=OptimizerType.PYTORCH_OPTIMIZER,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer_args=f"name={name}",
        lr_scheduler=SchedulerType.COSINE,
        warmup_ratio=0.0,
    )
    trainer.config.semantic_tuner_enabled = False
    trainer.lora_injector = _Injector(value)
    trainer.model = None
    trainer.trainable_params = []
    trainer._block_weight_manager = None
    trainer._easy_control = None
    trainer._ip_adapter = None
    trainer._repa_projector = None
    trainer._advanced_optimizer_strategy_profile = {}
    trainer._optimizer_backend_profile = {}
    trainer._log_messages = []
    trainer._log = lambda msg: trainer._log_messages.append(str(msg))
    return trainer


def _step(trainer: LulynxTrainer, optimizer: torch.optim.Optimizer, grad: torch.Tensor) -> None:
    if hasattr(optimizer, "train"):
        optimizer.train()
    trainer.lora_injector.param.grad = grad.detach().clone().to(dtype=trainer.lora_injector.param.dtype)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _first_live_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    if not optimizer.state:
        return {}
    first = next(iter(optimizer.state.values()))
    return dict(first) if isinstance(first, Mapping) else {}


def _state_contract(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    state = state_dict.get("state", {}) if isinstance(state_dict, Mapping) else {}
    groups = state_dict.get("param_groups", []) if isinstance(state_dict, Mapping) else []
    first_state = next(iter(state.values()), {}) if isinstance(state, Mapping) and state else {}
    group = groups[0] if groups else {}
    state_keys = sorted(str(key) for key in first_state.keys()) if isinstance(first_state, Mapping) else []
    group_keys = sorted(str(key) for key in group.keys() if key != "params") if isinstance(group, Mapping) else []
    return {
        "state_present": bool(state_keys),
        "param_state_keys": state_keys,
        "param_group_keys": group_keys,
        "train_mode": bool(group.get("train_mode", False)) if isinstance(group, Mapping) else False,
    }


def _compare_state_dicts(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_contract = _state_contract(left)
    right_contract = _state_contract(right)
    first_left = next(iter((left.get("state", {}) or {}).values()), {})
    first_right = next(iter((right.get("state", {}) or {}).values()), {})
    max_diff = 0.0
    mismatches: list[str] = []
    for key in sorted(set(first_left) | set(first_right)):
        lval = first_left.get(key) if isinstance(first_left, Mapping) else None
        rval = first_right.get(key) if isinstance(first_right, Mapping) else None
        if torch.is_tensor(lval) and torch.is_tensor(rval):
            diff = _max_abs(lval.detach(), rval.detach())
            max_diff = max(max_diff, diff)
            if diff > TOLERANCE:
                mismatches.append(str(key))
        elif _scalar_value(lval) != _scalar_value(rval):
            mismatches.append(str(key))
    ok = left_contract == right_contract and not mismatches
    return {
        "schema_version": 1,
        "ok": ok,
        "max_state_tensor_diff": max_diff,
        "mismatched_keys": mismatches,
        "left_contract": left_contract,
        "right_contract": right_contract,
    }


def _scalar_value(value: Any) -> Any:
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    return repr(value)


def _case_blockers(
    binding_ready: bool,
    non_mutating: bool,
    e2e_ok: bool,
    state_compare: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not binding_ready:
        blockers.append("selected_schedulefree_training_tensor_binding_shape_failed")
    if not non_mutating:
        blockers.append("selected_schedulefree_training_tensor_binding_mutated_parameter")
    if not e2e_ok:
        blockers.append("selected_schedulefree_training_tensor_binding_e2e_regression")
    if not bool(state_compare.get("ok", False)):
        blockers.append("selected_schedulefree_training_tensor_binding_state_mismatch")
    return blockers


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "BINDING_SCHEMA",
    "PROBE_KIND",
    "TARGET_PLUGIN_OPTIMIZERS",
    "build_plugin_schedulefree_training_tensor_binding_canary_scorecard",
]
