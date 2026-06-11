"""Report-only selected-optimizer ABI gate for schedule-free plugin routes."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch

from core.configs import SchedulerType
from core.lulynx_trainer.config import LulynxConfig, OptimizerType
from core.lulynx_trainer.optimizer_plugin_support import PLUGIN_SCHEDULE_FREE_OPTIMIZERS
from core.lulynx_trainer.trainer import LulynxTrainer


TARGET_PLUGIN_OPTIMIZERS = tuple(sorted(PLUGIN_SCHEDULE_FREE_OPTIMIZERS))


class _Injector:
    def __init__(self, value: torch.Tensor) -> None:
        self.param = torch.nn.Parameter(value.detach().clone())
        self.injected_layers = {}

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [self.param]


def build_plugin_schedulefree_selected_optimizer_scorecard() -> dict[str, Any]:
    """Validate selected schedule-free plugin optimizer contracts."""

    cases = [_selected_schedulefree_case(name) for name in TARGET_PLUGIN_OPTIMIZERS]
    failed = [case for case in cases if not bool(case.get("ok", False))]
    blockers = _dedupe(str(reason) for case in failed for reason in case.get("blocked_reasons", []) or [])
    ready = not failed and bool(cases)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_schedulefree_selected_optimizer_scorecard_v0",
        "gate": "plugin_schedulefree_selected_optimizer_abi",
        "ok": ready,
        "promotion_ready": False,
        "selected_optimizer_abi_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "selector_type": OptimizerType.PYTORCH_OPTIMIZER.value,
        "selected_optimizer_family": "schedule_free_state_machine",
        "request_contract": {
            "optimizer_type": OptimizerType.PYTORCH_OPTIMIZER.value,
            "selected_optimizer_source": "optimizer_args.name",
            "selected_optimizer_names": list(TARGET_PLUGIN_OPTIMIZERS),
            "external_scheduler_policy": "constant_required",
            "requires_optimizer_train_eval_calls": True,
            "checkpoint_resume_required": True,
            "adamw_kernel_compatible": False,
        },
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "passed_case_count": len(cases) - len(failed),
            "constant_scheduler_case_count": sum(1 for case in cases if case.get("scheduler_class") == "ConstantLR"),
            "resume_case_count": sum(1 for case in cases if case.get("covers_resume") is True),
            "train_eval_case_count": sum(1 for case in cases if case.get("covers_train_eval") is True),
        },
        "promotion_blockers": blockers
        + ["selected_schedulefree_native_abi_missing", "owner_release_hold_missing"],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "design selected schedule-free plugin native ABI with explicit train/eval and group state"
            if ready
            else "fix selected schedule-free plugin ABI blockers"
        ),
        "notes": [
            "This scorecard uses the trainer request path but does not enable native dispatch.",
            "Schedule-free plugin routes must use ConstantLR and optimizer-owned train/eval state.",
            "The exact AdamW native kernel remains blocked for these selected plugin optimizers.",
        ],
    }


def _selected_schedulefree_case(name: str) -> dict[str, Any]:
    try:
        return _run_selected_schedulefree_case(name)
    except Exception as exc:
        return {
            "schema_version": 1,
            "case": f"selected_plugin_{name}",
            "optimizer_name": name,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": [f"selected_plugin_schedulefree_failed:{name}:{type(exc).__name__}"],
        }


def _run_selected_schedulefree_case(name: str) -> dict[str, Any]:
    value = torch.linspace(-0.25, 0.35, steps=16, dtype=torch.float32).reshape(4, 4)
    grad1 = torch.linspace(0.01, 0.05, steps=value.numel(), dtype=torch.float32).reshape_as(value)
    grad2 = torch.linspace(-0.03, 0.02, steps=value.numel(), dtype=torch.float32).reshape_as(value)
    trainer = _make_trainer(name, value)
    optimizer = trainer._create_optimizer()
    scheduler = trainer._create_scheduler(optimizer, total_steps=8)
    has_train_eval = hasattr(optimizer, "train") and hasattr(optimizer, "eval")

    _step(trainer, optimizer, grad1)
    after_step = _state_contract(optimizer.state_dict())
    if has_train_eval:
        optimizer.eval()
    after_eval = _state_contract(optimizer.state_dict())
    if has_train_eval:
        optimizer.train()
    after_train = _state_contract(optimizer.state_dict())
    saved_state = copy.deepcopy(optimizer.state_dict())
    saved_param = trainer.lora_injector.param.detach().clone()

    restored_trainer = _make_trainer(name, saved_param)
    restored_optimizer = restored_trainer._create_optimizer()
    restored_optimizer.load_state_dict(saved_state)
    _step(trainer, optimizer, grad2)
    _step(restored_trainer, restored_optimizer, grad2)
    diff = _max_abs(trainer.lora_injector.param.detach(), restored_trainer.lora_injector.param.detach())
    scheduler_ok = type(scheduler).__name__ == "ConstantLR"
    ok = scheduler_ok and has_train_eval and after_step["state_present"] and diff <= 1e-5
    return {
        "schema_version": 1,
        "case": f"selected_plugin_{name}",
        "optimizer_name": name,
        "optimizer_class": type(optimizer).__name__,
        "ok": ok,
        "covers_resume": True,
        "covers_train_eval": has_train_eval,
        "scheduler_class": type(scheduler).__name__,
        "after_step": after_step,
        "after_eval": after_eval,
        "after_train": after_train,
        "max_resume_diff": diff,
        "tolerance": 1e-5,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "blocked_reasons": [] if ok else [f"selected_plugin_schedulefree_contract_failed:{name}"],
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
        "step_like_value": _first_numeric(first_state, ("step", "k")),
    }


def _first_numeric(state: Mapping[str, Any], keys: tuple[str, ...]) -> int:
    if not isinstance(state, Mapping):
        return 0
    for key in keys:
        value = state.get(key)
        if torch.is_tensor(value) and value.numel() == 1:
            return int(value.detach().cpu().item())
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["TARGET_PLUGIN_OPTIMIZERS", "build_plugin_schedulefree_selected_optimizer_scorecard"]
