"""Report-only checkpoint adapter proof for schedule-free plugin optimizers."""

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


def build_plugin_schedulefree_checkpoint_adapter_scorecard() -> dict[str, Any]:
    """Validate pack/unpack envelope boundaries without trainer integration."""

    cases = [_checkpoint_adapter_case(name) for name in TARGET_PLUGIN_OPTIMIZERS]
    failed = [case for case in cases if not bool(case.get("ok", False))]
    blockers = _dedupe(str(reason) for case in failed for reason in case.get("blocked_reasons", []) or [])
    ready = not failed and bool(cases)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_schedulefree_checkpoint_adapter_scorecard_v0",
        "gate": "plugin_schedulefree_checkpoint_adapter_proof",
        "ok": ready,
        "promotion_ready": False,
        "checkpoint_adapter_proof_ready": ready,
        "runtime_adapter_enabled": False,
        "training_checkpoint_integration_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "adapter_contract": {
            "adapter_kind": "plugin_schedulefree_state_dict_adapter_v0",
            "pack_source": "selected_plugin_optimizer.state_dict",
            "unpack_target": "selected_plugin_optimizer.load_state_dict",
            "selected_optimizer_names": list(TARGET_PLUGIN_OPTIMIZERS),
            "requires_train_mode_restore": True,
            "requires_param_group_state_restore": True,
            "requires_param_state_restore": True,
        },
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "passed_case_count": len(cases) - len(failed),
            "pack_unpack_case_count": sum(1 for case in cases if case.get("pack_unpack_probe") is True),
            "resume_parity_case_count": sum(1 for case in cases if case.get("resume_parity") is True),
        },
        "promotion_blockers": blockers
        + [
            "selected_schedulefree_runtime_checkpoint_adapter_missing",
            "training_checkpoint_integration_missing",
            "owner_release_hold_missing",
        ],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "build report-only training tensor binding canary for selected schedule-free plugin optimizers"
            if ready
            else "fix selected schedule-free checkpoint adapter proof blockers"
        ),
        "notes": [
            "This proof packs and unpacks optimizer state_dict envelopes only.",
            "It does not modify trainer checkpoint save/load or enable native dispatch.",
            "The selected plugin optimizer remains the authoritative update path.",
        ],
    }


def _checkpoint_adapter_case(name: str) -> dict[str, Any]:
    try:
        return _run_checkpoint_adapter_case(name)
    except Exception as exc:
        return {
            "schema_version": 1,
            "case": f"checkpoint_adapter_{name}",
            "optimizer_name": name,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": [f"plugin_schedulefree_checkpoint_adapter_failed:{name}:{type(exc).__name__}"],
        }


def _run_checkpoint_adapter_case(name: str) -> dict[str, Any]:
    value = torch.linspace(-0.25, 0.35, steps=16, dtype=torch.float32).reshape(4, 4)
    grad1 = torch.linspace(0.01, 0.05, steps=value.numel(), dtype=torch.float32).reshape_as(value)
    grad2 = torch.linspace(-0.03, 0.02, steps=value.numel(), dtype=torch.float32).reshape_as(value)
    trainer = _make_trainer(name, value)
    optimizer = trainer._create_optimizer()
    _step(trainer, optimizer, grad1)

    envelope = _pack_checkpoint_envelope(name, optimizer)
    saved_param = trainer.lora_injector.param.detach().clone()
    restored_trainer = _make_trainer(name, saved_param)
    restored_optimizer = restored_trainer._create_optimizer()
    restored_optimizer.load_state_dict(_unpack_checkpoint_envelope(envelope))

    _step(trainer, optimizer, grad2)
    _step(restored_trainer, restored_optimizer, grad2)
    diff = _max_abs(trainer.lora_injector.param.detach(), restored_trainer.lora_injector.param.detach())
    ok = envelope["adapter_contract_ok"] and diff <= 1e-5
    return {
        "schema_version": 1,
        "case": f"checkpoint_adapter_{name}",
        "optimizer_name": name,
        "optimizer_class": type(optimizer).__name__,
        "ok": ok,
        "pack_unpack_probe": True,
        "resume_parity": diff <= 1e-5,
        "envelope_summary": envelope["summary"],
        "max_resume_diff": diff,
        "tolerance": 1e-5,
        "runtime_adapter_enabled": False,
        "training_checkpoint_integration_enabled": False,
        "blocked_reasons": [] if ok else [f"plugin_schedulefree_checkpoint_adapter_contract_failed:{name}"],
    }


def _pack_checkpoint_envelope(name: str, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state_dict = copy.deepcopy(optimizer.state_dict())
    state = state_dict.get("state", {}) if isinstance(state_dict, Mapping) else {}
    groups = state_dict.get("param_groups", []) if isinstance(state_dict, Mapping) else []
    first_state = next(iter(state.values()), {}) if isinstance(state, Mapping) and state else {}
    group = groups[0] if groups else {}
    param_state_keys = sorted(str(key) for key in first_state.keys()) if isinstance(first_state, Mapping) else []
    group_state_keys = sorted(str(key) for key in group.keys() if key != "params") if isinstance(group, Mapping) else []
    envelope = {
        "schema_version": 1,
        "adapter_kind": "plugin_schedulefree_state_dict_adapter_v0",
        "selected_optimizer_name": name,
        "optimizer_class": type(optimizer).__name__,
        "state_dict": state_dict,
        "adapter_contract_ok": bool(param_state_keys) and "train_mode" in set(group_state_keys),
        "summary": {
            "param_group_count": len(groups),
            "state_entry_count": len(state) if isinstance(state, Mapping) else 0,
            "param_state_keys": param_state_keys,
            "group_state_keys": group_state_keys,
            "train_mode_restored_by_group": "train_mode" in set(group_state_keys),
        },
    }
    return envelope


def _unpack_checkpoint_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    if str(envelope.get("adapter_kind", "")) != "plugin_schedulefree_state_dict_adapter_v0":
        raise ValueError("unexpected schedule-free plugin checkpoint adapter kind")
    state_dict = envelope.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("missing schedule-free plugin state_dict")
    return copy.deepcopy(dict(state_dict))


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


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["TARGET_PLUGIN_OPTIMIZERS", "build_plugin_schedulefree_checkpoint_adapter_scorecard"]
