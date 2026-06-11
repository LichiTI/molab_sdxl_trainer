"""Selected-optimizer gate for adam-like pytorch_optimizer plugin routes."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch

from core.configs import SchedulerType
from core.lulynx_trainer.config import LulynxConfig, OptimizerType
from core.lulynx_trainer.optimizer_plugin_support import plugin_resume_case
from core.lulynx_trainer.trainer import LulynxTrainer


TARGET_PLUGIN_OPTIMIZERS = (
    "adam",
    "adamax",
    "adamc",
    "adamg",
    "adamod",
    "adamp",
    "adams",
    "adamw",
    "adamwsn",
    "dualadam",
    "exadam",
    "fadam",
    "flashadamw",
    "grokfastadamw",
    "lamb",
    "nadam",
    "novograd",
    "padam",
    "qhadam",
    "radam",
    "ranger",
    "ranger21",
    "ranger25",
    "stableadamw",
    "yogi",
)

ADAMW_STATE_KEYS = {"exp_avg", "exp_avg_sq", "step"}
ADAMW_GROUP_KEYS = {"betas", "eps", "lr", "weight_decay"}


class _Injector:
    def __init__(self, value: torch.Tensor) -> None:
        self.param = torch.nn.Parameter(value.detach().clone())
        self.injected_layers = {}

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [self.param]


def build_plugin_adamlike_selected_optimizer_scorecard() -> dict[str, Any]:
    cases = [_selected_adamlike_case(name) for name in TARGET_PLUGIN_OPTIMIZERS]
    failed = [case for case in cases if not bool(case.get("ok", False))]
    compatible = [case for case in cases if bool(case.get("adamw_native_route_compatible", False))]
    dedicated = [case for case in cases if bool(case.get("dedicated_kernel_required", False))]
    blockers = _dedupe(reason for case in failed for reason in case.get("blocked_reasons", []) or [])
    ready = bool(cases) and not failed and bool(compatible) and len(compatible) + len(dedicated) == len(cases)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_adamlike_selected_optimizer_scorecard_v0",
        "gate": "plugin_adamlike_selected_optimizer_native_compatibility",
        "ok": ready,
        "promotion_ready": False,
        "selected_optimizer_abi_ready": ready,
        "adamw_native_route_candidate_ready": bool(compatible),
        "dedicated_kernel_queue_ready": len(dedicated) == len(cases) - len(compatible),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "selector_type": OptimizerType.PYTORCH_OPTIMIZER.value,
        "selected_optimizer_family": "adam_like_formula",
        "request_contract": {
            "optimizer_type": OptimizerType.PYTORCH_OPTIMIZER.value,
            "selected_optimizer_source": "optimizer_args.name",
            "selected_optimizer_names": list(TARGET_PLUGIN_OPTIMIZERS),
            "adamw_native_route_policy": "allow_only_schema_compatible_selected_adamw",
            "dedicated_kernel_policy": "required_for_formula_or_state_variants",
        },
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "passed_case_count": len(cases) - len(failed),
            "adamw_native_route_compatible_count": len(compatible),
            "dedicated_kernel_required_count": len(dedicated),
            "compatible_optimizer_names": [str(case.get("optimizer_name")) for case in compatible],
            "dedicated_kernel_optimizer_names": [str(case.get("optimizer_name")) for case in dedicated],
        },
        "promotion_blockers": blockers
        + [
            "selected_adamlike_dedicated_kernel_matrix_missing",
            "selected_adamlike_training_tensor_binding_missing",
            "owner_release_hold_missing",
        ],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add selected AdamW plugin TrainingLoop native canary, then batch dedicated kernels for adam-like variants"
            if ready
            else "fix selected adam-like plugin ABI blockers"
        ),
        "notes": [
            "This scorecard uses the trainer request path and real plugin optimizer instances.",
            "Only schema-compatible selected AdamW may reuse the existing AdamW native route.",
            "Adam-like names with different state or formula are queued for dedicated native kernels.",
        ],
    }


def _selected_adamlike_case(name: str) -> dict[str, Any]:
    try:
        return _run_selected_adamlike_case(name)
    except Exception as exc:
        return {
            "schema_version": 1,
            "case": f"selected_plugin_adamlike_{name}",
            "optimizer_name": name,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": [f"selected_plugin_adamlike_failed:{name}:{type(exc).__name__}"],
        }


def _run_selected_adamlike_case(name: str) -> dict[str, Any]:
    value = torch.linspace(-0.25, 0.35, steps=16, dtype=torch.float32).reshape(4, 4)
    grad1 = torch.linspace(0.01, 0.05, steps=value.numel(), dtype=torch.float32).reshape_as(value)
    grad2 = torch.linspace(-0.03, 0.02, steps=value.numel(), dtype=torch.float32).reshape_as(value)
    trainer = _make_trainer(name, value)
    optimizer = trainer._create_optimizer()
    _step(trainer, optimizer, grad1)
    after_step = _state_contract(optimizer.state_dict())
    saved_state = copy.deepcopy(optimizer.state_dict())
    saved_param = trainer.lora_injector.param.detach().clone()
    restored = _make_trainer(name, saved_param)
    restored_optimizer = restored._create_optimizer()
    restored_optimizer.load_state_dict(saved_state)
    _step(trainer, optimizer, grad2)
    _step(restored, restored_optimizer, grad2)
    resume_diff = _max_abs(trainer.lora_injector.param.detach(), restored.lora_injector.param.detach())
    compatibility = _adamw_route_compatibility(name, optimizer, after_step)
    ok = bool(after_step["state_present"] and resume_diff <= 1e-5)
    return {
        "schema_version": 1,
        "case": f"selected_plugin_adamlike_{name}",
        "optimizer_name": name,
        "optimizer_class": type(optimizer).__name__,
        "ok": ok,
        "covers_resume": True,
        "after_step": after_step,
        "max_resume_diff": resume_diff,
        "tolerance": 1e-5,
        "adamw_native_route_compatible": bool(compatibility["adamw_native_route_compatible"]),
        "dedicated_kernel_required": not bool(compatibility["adamw_native_route_compatible"]),
        "compatibility": compatibility,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "blocked_reasons": [] if ok else [f"selected_plugin_adamlike_contract_failed:{name}"],
    }


def _make_trainer(name: str, value: torch.Tensor) -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    case = plugin_resume_case(name)
    trainer.config = LulynxConfig(
        optimizer_type=OptimizerType.PYTORCH_OPTIMIZER,
        learning_rate=1e-3,
        weight_decay=0.01,
        optimizer_args=case.optimizer_args,
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
        "step_like_value": _first_numeric(first_state, group, ("step", "k")),
    }


def _adamw_route_compatibility(name: str, optimizer: torch.optim.Optimizer, state: Mapping[str, Any]) -> dict[str, Any]:
    state_keys = set(str(key) for key in state.get("param_state_keys", []) or [])
    group_keys = set(str(key) for key in state.get("param_group_keys", []) or [])
    class_name = type(optimizer).__name__
    class_compatible = class_name == "AdamW"
    schema_compatible = state_keys == ADAMW_STATE_KEYS and ADAMW_GROUP_KEYS.issubset(group_keys)
    selected_name_compatible = str(name).lower() == "adamw"
    compatible = bool(selected_name_compatible and class_compatible and schema_compatible)
    blockers = []
    if not selected_name_compatible:
        blockers.append("selected_optimizer_formula_not_exact_adamw")
    if not class_compatible:
        blockers.append("selected_optimizer_class_not_adamw")
    if not schema_compatible:
        blockers.append("selected_optimizer_state_schema_not_exact_adamw")
    return {
        "schema_version": 1,
        "adamw_native_route_compatible": compatible,
        "selected_optimizer_name_compatible": selected_name_compatible,
        "optimizer_class_compatible": class_compatible,
        "state_schema_compatible": schema_compatible,
        "native_route": "rust_cuda_adamw_v0" if compatible else "dedicated_kernel_required",
        "blocked_reasons": [] if compatible else blockers,
    }


def _first_numeric(state: Mapping[str, Any], group: Mapping[str, Any], keys: tuple[str, ...]) -> int:
    for source in (state, group):
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            value = source.get(key)
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


__all__ = ["TARGET_PLUGIN_OPTIMIZERS", "build_plugin_adamlike_selected_optimizer_scorecard"]
