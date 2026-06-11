"""Report-only e2e shadow matrix for selected schedule-free plugin optimizers."""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from core.configs import SchedulerType
from core.lulynx_trainer.config import LulynxConfig, OptimizerType
from core.lulynx_trainer.optimizer_plugin_support import PLUGIN_SCHEDULE_FREE_OPTIMIZERS
from core.lulynx_trainer.trainer import LulynxTrainer
from core.turbocore_plugin_schedulefree_runtime_dispatch_shadow_scorecard import (
    build_plugin_schedulefree_runtime_dispatch_shadow_scorecard,
)
from core.turbocore_tensor_handle_registry import TurboCoreTensorHandleRegistry


TARGET_PLUGIN_OPTIMIZERS = tuple(sorted(PLUGIN_SCHEDULE_FREE_OPTIMIZERS))
MATRIX_KIND = "plugin_schedulefree_e2e_shadow_training_matrix_v0"
MATRIX_CASES = tuple(
    {"case": f"shadow_{name}_numel_{numel}", "optimizer_name": name, "numel": numel, "shadow_step_count": 3}
    for name in TARGET_PLUGIN_OPTIMIZERS
    for numel in (16, 32)
)
TOLERANCE = 1e-5


class _Injector:
    def __init__(self, value: torch.Tensor) -> None:
        self.param = torch.nn.Parameter(value.detach().clone())
        self.injected_layers = {}

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [self.param]


def build_plugin_schedulefree_e2e_shadow_training_matrix_scorecard(
    *,
    adapter_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "canary",
) -> dict[str, Any]:
    """Run fallback-authoritative shadow checks on cloned tensors."""

    adapter = dict(
        adapter_report
        or build_plugin_schedulefree_runtime_dispatch_shadow_scorecard(
            native_training_mode=native_training_mode
        )
    )
    cases = [_safe_case(case, lambda item=case: _run_case(item)) for case in MATRIX_CASES]
    matrix_ready = all(bool(case.get("shadow_e2e_ready", False)) for case in cases)
    matrix_passed = all(str(case.get("status", "unknown")) == "passed" for case in cases)
    validations = _validations(adapter, cases, matrix_ready)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(str(reason) for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_schedulefree_e2e_shadow_training_matrix_scorecard_v0",
        "gate": "plugin_schedulefree_e2e_shadow_training_matrix",
        "ok": ready,
        "promotion_ready": False,
        "e2e_shadow_training_matrix_ready": ready,
        "e2e_shadow_training_matrix_passed": matrix_passed,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "fallback_backend_authoritative": True,
        "native_shadow_updates_original": False,
        "matrix_kind": MATRIX_KIND,
        "optimizer_family": "schedule_free_state_machine",
        "adapter_summary": dict(adapter.get("summary") or {}),
        "matrix_cases": cases,
        "validations": validations,
        "summary": {
            "case_count": len(cases),
            "passed_case_count": sum(1 for case in cases if str(case.get("status")) == "passed"),
            "failed_case_count": sum(1 for case in cases if str(case.get("status")) == "failed"),
            "e2e_shadow_training_matrix_passed": matrix_passed,
            "max_param_diff": _max_case_value(cases, "max_param_diff"),
            "max_state_tensor_diff": _max_case_value(cases, "max_state_tensor_diff"),
            "max_original_shadow_mutation_diff": _max_case_value(cases, "max_original_shadow_mutation_diff"),
            "fallback_backend_authoritative": True,
            "native_shadow_updates_original": False,
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "selected_schedulefree_native_kernel_missing",
                "selected_schedulefree_canary_rollout_policy_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add selected schedule-free explicit canary rollout policy with default off"
            if ready
            else "fix selected schedule-free e2e shadow training matrix blockers"
        ),
        "notes": [
            "Each case keeps the selected pytorch_optimizer plugin update authoritative.",
            "The shadow path uses cloned tensors and never updates the authoritative parameter.",
            "This is an isolated matrix, not user training dispatch.",
        ],
    }


def _run_case(case: Mapping[str, Any]) -> dict[str, Any]:
    name = str(case["optimizer_name"])
    numel = int(case["numel"])
    step_count = int(case.get("shadow_step_count", 3))
    value = torch.linspace(-0.25, 0.35, steps=numel, dtype=torch.float32)
    authoritative = _make_trainer(name, value)
    shadow = _make_trainer(name, value)
    authoritative_optimizer = authoritative._create_optimizer()
    shadow_optimizer = shadow._create_optimizer()

    max_param_diff = 0.0
    max_state_tensor_diff = 0.0
    max_mutation_diff = 0.0
    binding_ready_count = 0
    for step_index in range(step_count):
        grad = _gradient(numel, step_index)
        before_authoritative = authoritative.lora_injector.param.detach().clone()
        _step(authoritative, authoritative_optimizer, grad)

        shadow.lora_injector.param.grad = grad.detach().clone()
        binding = _build_shadow_binding(name, shadow_optimizer, shadow.lora_injector.param)
        binding_ready_count += int(bool(binding["readiness"]["request_shape_ready"]))
        mutation_diff = _max_abs(before_authoritative, shadow.lora_injector.param.detach())
        max_mutation_diff = max(max_mutation_diff, mutation_diff)
        if hasattr(shadow_optimizer, "train"):
            shadow_optimizer.train()
        shadow_optimizer.step()
        shadow_optimizer.zero_grad(set_to_none=True)

        param_diff = _max_abs(authoritative.lora_injector.param.detach(), shadow.lora_injector.param.detach())
        state_compare = _compare_state_dicts(authoritative_optimizer.state_dict(), shadow_optimizer.state_dict())
        max_param_diff = max(max_param_diff, param_diff)
        max_state_tensor_diff = max(max_state_tensor_diff, float(state_compare["max_state_tensor_diff"]))

    final_state_compare = _compare_state_dicts(authoritative_optimizer.state_dict(), shadow_optimizer.state_dict())
    ok = (
        binding_ready_count == step_count
        and max_param_diff <= TOLERANCE
        and max_state_tensor_diff <= TOLERANCE
        and max_mutation_diff <= TOLERANCE
        and bool(final_state_compare["ok"])
    )
    return {
        "schema_version": 1,
        "case": str(case["case"]),
        "optimizer_name": name,
        "optimizer_class": type(authoritative_optimizer).__name__,
        "numel": numel,
        "shadow_step_count": step_count,
        "status": "passed" if ok else "failed",
        "ok": ok,
        "shadow_e2e_ready": ok,
        "binding_ready_count": binding_ready_count,
        "fallback_backend_authoritative": True,
        "native_shadow_updates_original": False,
        "training_path_enabled": False,
        "max_param_diff": max_param_diff,
        "max_state_tensor_diff": max_state_tensor_diff,
        "max_original_shadow_mutation_diff": max_mutation_diff,
        "final_state_compare": final_state_compare,
        "blocked_reasons": [] if ok else _case_blockers(binding_ready_count, step_count, max_param_diff, max_state_tensor_diff, max_mutation_diff, final_state_compare),
    }


def _safe_case(case: Mapping[str, Any], fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = fn()
        payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        return payload
    except Exception as exc:
        return {
            "schema_version": 1,
            "case": str(case.get("case") or "unknown"),
            "optimizer_name": str(case.get("optimizer_name") or "unknown"),
            "numel": int(case.get("numel") or 0),
            "status": "failed",
            "ok": False,
            "shadow_e2e_ready": False,
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": [f"selected_schedulefree_e2e_shadow_case_failed:{type(exc).__name__}"],
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }


def _build_shadow_binding(
    name: str,
    optimizer: torch.optim.Optimizer,
    param: torch.nn.Parameter,
) -> dict[str, Any]:
    if param.grad is None:
        raise ValueError("schedule-free e2e shadow requires a live gradient")
    registry = TurboCoreTensorHandleRegistry(namespace=f"schedulefree_e2e_{name}")
    handles = {
        "param": registry.register(param, role="param").handle_id,
        "grad": registry.register(param.grad.detach().contiguous(), role="grad").handle_id,
    }
    state = _first_live_state(optimizer)
    for key, value in state.items():
        if torch.is_tensor(value) and value.dtype.is_floating_point:
            handles[f"state_{key}"] = registry.register(value.detach().contiguous(), role=f"state_{key}").handle_id
    snapshot = registry.snapshot()
    state_present = bool(_state_contract(optimizer.state_dict()).get("state_present", False)) or not state
    request_shape_ready = bool(
        {"param", "grad"}.issubset(set(handles))
        and int(snapshot.get("handle_count", 0) or 0) >= 2
        and not bool(snapshot.get("pointer_exported", False))
        and not bool(snapshot.get("training_path_enabled", False))
        and state_present
    )
    return {
        "schema_version": 1,
        "schema": "plugin_schedulefree_e2e_shadow_binding_request_v0",
        "selected_optimizer_name": name,
        "reported_roles": sorted(handles),
        "role_handles": handles,
        "registry_snapshot": snapshot,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "readiness": {
            "schema_version": 1,
            "request_shape_ready": request_shape_ready,
            "native_binding_ready": False,
            "performance_test_ready": False,
            "training_path_enabled": False,
            "blocked_reasons": [
                "selected_schedulefree_native_kernel_missing",
                "runtime_dispatch_review_missing",
            ],
        },
    }


def _validations(
    adapter: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    matrix_ready: bool,
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p16_runtime_dispatch_shadow_ready",
            bool(adapter.get("runtime_dispatch_shadow_ready", False)),
            "selected_schedulefree_runtime_dispatch_shadow_missing",
        ),
        _validation(
            "fallback_backend_authoritative",
            bool(adapter.get("fallback_backend_authoritative", False))
            and all(bool(case.get("fallback_backend_authoritative", False)) for case in cases),
            "selected_schedulefree_e2e_shadow_fallback_not_authoritative",
        ),
        _validation(
            "native_shadow_never_updates_original",
            not bool(adapter.get("native_shadow_call_allowed", True))
            and not any(bool(case.get("native_shadow_updates_original", True)) for case in cases),
            "selected_schedulefree_e2e_shadow_mutated_original",
        ),
        _validation(
            "e2e_shadow_training_matrix_ready",
            matrix_ready,
            "selected_schedulefree_e2e_shadow_training_matrix_failed",
        ),
        _validation(
            "runtime_dispatch_still_disabled",
            not bool(adapter.get("runtime_dispatch_ready", True))
            and not bool(adapter.get("native_dispatch_allowed", True))
            and not any(bool(case.get("training_path_enabled", True)) for case in cases),
            "selected_schedulefree_e2e_shadow_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(adapter.get("training_path_enabled", True))
            and not bool(adapter.get("default_behavior_changed", True)),
            "selected_schedulefree_e2e_shadow_changed_default_behavior",
        ),
    ]


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


def _gradient(numel: int, step_index: int) -> torch.Tensor:
    base = torch.linspace(-0.03, 0.05, steps=numel, dtype=torch.float32)
    return base * float(step_index + 1) + 0.001 * float(step_index)


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


def _case_blockers(
    binding_ready_count: int,
    step_count: int,
    max_param_diff: float,
    max_state_tensor_diff: float,
    max_mutation_diff: float,
    state_compare: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if binding_ready_count != step_count:
        blockers.append("selected_schedulefree_e2e_shadow_binding_not_ready")
    if max_param_diff > TOLERANCE:
        blockers.append("selected_schedulefree_e2e_shadow_param_mismatch")
    if max_state_tensor_diff > TOLERANCE or not bool(state_compare.get("ok", False)):
        blockers.append("selected_schedulefree_e2e_shadow_state_mismatch")
    if max_mutation_diff > TOLERANCE:
        blockers.append("selected_schedulefree_e2e_shadow_mutated_authoritative_param")
    return blockers


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _scalar_value(value: Any) -> Any:
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    return repr(value)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _max_case_value(cases: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(case[key]) for case in cases if case.get(key) is not None]
    return max(values) if values else None


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "MATRIX_CASES",
    "MATRIX_KIND",
    "build_plugin_schedulefree_e2e_shadow_training_matrix_scorecard",
]
