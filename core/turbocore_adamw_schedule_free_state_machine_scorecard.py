"""Report-only state-machine scorecard for AdamWScheduleFree.

Schedule-free AdamW has AdamW-like moments, but it also owns train/eval mode
and internal scheduled LR state.  This scorecard proves those contracts before
any native optimizer kernel can be considered.
"""

from __future__ import annotations

import copy
import importlib.util
from typing import Any, Mapping

import torch

from core.configs import OptimizerType, UnifiedTrainingConfig
from core.lulynx_trainer.trainer import LulynxTrainer


TARGET_OPTIMIZER = OptimizerType.ADAMW_SCHEDULE_FREE


class _Injector:
    def __init__(self, value: torch.Tensor) -> None:
        self.param = torch.nn.Parameter(value.detach().clone())
        self.injected_layers = {}

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [self.param]


def build_adamw_schedule_free_state_machine_scorecard() -> dict[str, Any]:
    """Validate AdamWScheduleFree state contracts without native dispatch."""

    if importlib.util.find_spec("schedulefree") is None:
        return _blocked("schedulefree_unavailable")
    cases = [
        _step_requires_train_mode_case(),
        _trainer_request_initializes_train_mode_case(),
        _roundtrip_state_machine_case(),
    ]
    failed = [case for case in cases if not bool(case.get("ok", False))]
    ready = not failed
    blockers = _dedupe(str(reason) for case in failed for reason in case.get("blocked_reasons", []) or [])
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adamw_schedule_free_state_machine_scorecard_v0",
        "gate": "adamw_schedule_free_state_machine_reference",
        "ok": ready,
        "promotion_ready": False,
        "state_machine_reference_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "optimizer_type": TARGET_OPTIMIZER.value,
        "request_contract": {
            "optimizer_type": TARGET_OPTIMIZER.value,
            "optimizer_family": "adamw_schedule_free",
            "external_scheduler_policy": "constant_required",
            "requires_optimizer_train_eval_calls": True,
            "state_machine_owned_by_optimizer": True,
        },
        "state_contract": {
            "param_state_keys": ["z", "exp_avg_sq"],
            "param_group_state_keys": [
                "k",
                "train_mode",
                "weight_sum",
                "lr_max",
                "scheduled_lr",
                "warmup_steps",
                "r",
                "weight_lr_power",
            ],
            "scheduler_coupled": True,
            "decoupled_weight_decay": True,
            "resume_requires_param_group_state": True,
        },
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "passed_case_count": len(cases) - len(failed),
            "mode_toggle_case_count": sum(1 for case in cases if case.get("covers_mode_toggle") is True),
            "resume_case_count": sum(1 for case in cases if case.get("covers_resume") is True),
        },
        "promotion_blockers": blockers + ["native_kernel_missing", "runtime_canary_missing"],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "design native schedule-free launch ABI with explicit train/eval and param_group state"
            if ready
            else "fix AdamWScheduleFree state-machine reference blockers"
        ),
        "notes": [
            "This scorecard is report-only and does not replace the PyTorch/schedulefree optimizer.",
            "AdamWScheduleFree must not share the exact AdamW native kernel until its state machine has a native ABI.",
        ],
    }


def _step_requires_train_mode_case() -> dict[str, Any]:
    import schedulefree

    param = torch.nn.Parameter(torch.linspace(-0.2, 0.3, 4))
    optimizer = schedulefree.AdamWScheduleFree([param], lr=1e-3, weight_decay=0.01)
    param.grad = torch.linspace(0.01, 0.04, 4)
    try:
        optimizer.step()
    except Exception as exc:
        message = str(exc)
        ok = "train mode" in message
        return {
            "schema_version": 1,
            "case": "step_requires_train_mode",
            "ok": ok,
            "covers_mode_toggle": True,
            "error_message": message,
            "blocked_reasons": [] if ok else ["adamw_schedule_free_missing_train_mode_guard"],
        }
    return {
        "schema_version": 1,
        "case": "step_requires_train_mode",
        "ok": False,
        "covers_mode_toggle": True,
        "blocked_reasons": ["adamw_schedule_free_step_allowed_without_train_mode"],
    }


def _trainer_request_initializes_train_mode_case() -> dict[str, Any]:
    trainer = _make_trainer(_initial_tensor())
    optimizer = trainer._create_optimizer()
    group = optimizer.state_dict()["param_groups"][0]
    ok = bool(group.get("train_mode", False))
    return {
        "schema_version": 1,
        "case": "trainer_request_initializes_train_mode",
        "ok": ok,
        "covers_mode_toggle": True,
        "optimizer_class": type(optimizer).__name__,
        "request_fields": {
            "optimizer_type": TARGET_OPTIMIZER.value,
            "scheduler_policy": "constant_required",
            "created_by": "LulynxTrainer._create_optimizer",
        },
        "param_group_train_mode": bool(group.get("train_mode", False)),
        "blocked_reasons": [] if ok else ["trainer_did_not_call_schedule_free_train"],
    }


def _roundtrip_state_machine_case() -> dict[str, Any]:
    value = _initial_tensor()
    grad1 = torch.linspace(0.01, 0.05, steps=value.numel()).reshape_as(value)
    grad2 = torch.linspace(-0.03, 0.02, steps=value.numel()).reshape_as(value)
    trainer = _make_trainer(value)
    optimizer = trainer._create_optimizer()
    param = trainer.lora_injector.param
    _step(param, optimizer, grad1)
    after_step = _state_contract(optimizer.state_dict())
    optimizer.eval()
    after_eval = _state_contract(optimizer.state_dict())
    optimizer.train()
    after_train = _state_contract(optimizer.state_dict())
    saved_state = copy.deepcopy(optimizer.state_dict())
    saved_param = param.detach().clone()

    restored = _make_trainer(saved_param)
    restored_optimizer = restored._create_optimizer()
    restored_param = restored.lora_injector.param
    restored_optimizer.load_state_dict(saved_state)
    _step(param, optimizer, grad2)
    _step(restored_param, restored_optimizer, grad2)
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
        "ok": ok,
        "covers_resume": True,
        "covers_mode_toggle": True,
        "after_step": after_step,
        "after_eval": after_eval,
        "after_train": after_train,
        "max_resume_diff": diff,
        "tolerance": 1e-5,
        "blocked_reasons": [] if ok else ["adamw_schedule_free_state_machine_roundtrip_failed"],
    }


def _make_trainer(value: torch.Tensor) -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = UnifiedTrainingConfig(
        optimizer_type=TARGET_OPTIMIZER,
        learning_rate=1e-3,
        weight_decay=0.01,
        optimizer_args="",
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
    trainer._attach_optimizer_profiles_to_training_loop = lambda: None
    return trainer


def _step(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer, grad: torch.Tensor) -> None:
    param.grad = grad.detach().clone().to(device=param.device, dtype=param.dtype)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _state_contract(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    state = state_dict.get("state", {}) if isinstance(state_dict, Mapping) else {}
    groups = state_dict.get("param_groups", []) if isinstance(state_dict, Mapping) else []
    first_state = next(iter(state.values()), {}) if isinstance(state, Mapping) and state else {}
    group = groups[0] if groups else {}
    state_keys = sorted(str(key) for key in first_state.keys()) if isinstance(first_state, Mapping) else []
    return {
        "param_state_keys": state_keys,
        "has_required_param_state": {"z", "exp_avg_sq"}.issubset(set(state_keys)),
        "train_mode": bool(group.get("train_mode", False)),
        "k": int(group.get("k", 0) or 0),
        "weight_sum": float(group.get("weight_sum", 0.0) or 0.0),
        "scheduled_lr": float(group.get("scheduled_lr", 0.0) or 0.0),
        "lr_max": float(group.get("lr_max", 0.0) or 0.0),
    }


def _initial_tensor() -> torch.Tensor:
    return torch.linspace(-0.25, 0.35, steps=16).reshape(4, 4)


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adamw_schedule_free_state_machine_scorecard_v0",
        "gate": "adamw_schedule_free_state_machine_reference",
        "ok": False,
        "promotion_ready": False,
        "state_machine_reference_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "optimizer_type": TARGET_OPTIMIZER.value,
        "cases": [],
        "promotion_blockers": [reason],
        "blocked_reasons": [reason],
        "recommended_next_step": "install schedulefree before schedule-free native optimizer research",
    }


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["TARGET_OPTIMIZER", "build_adamw_schedule_free_state_machine_scorecard"]
