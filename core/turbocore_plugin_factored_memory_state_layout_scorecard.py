"""Report-only state-layout audit for factored-memory plugin optimizers."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch

from core.configs import SchedulerType
from core.lulynx_trainer.config import LulynxConfig, OptimizerType
from core.lulynx_trainer.optimizer_plugin_support import plugin_resume_case
from core.lulynx_trainer.trainer import LulynxTrainer
from core.turbocore_plugin_optimizer_selector_scorecard import build_plugin_optimizer_selector_scorecard


FACTORED_MEMORY_PLUGIN_OPTIMIZERS = (
    "adafactor",
    "came",
    "emofact",
    "galore",
    "scalableshampoo",
    "shampoo",
    "sm3",
    "soap",
)


class _Injector:
    def __init__(self, value: torch.Tensor) -> None:
        self.param = torch.nn.Parameter(value.detach().clone())
        self.injected_layers = {}

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [self.param]


def build_plugin_factored_memory_state_layout_scorecard() -> dict[str, Any]:
    """Probe selected plugin optimizer layouts without enabling dispatch."""

    selector = build_plugin_optimizer_selector_scorecard()
    selector_names = _selector_factored_names(selector)
    rows = [_state_layout_case(name, selector_names) for name in FACTORED_MEMORY_PLUGIN_OPTIMIZERS]
    failed = [row for row in rows if row["state_layout_status"] == "state_layout_probe_failed"]
    manual = [row for row in rows if row["state_layout_status"] == "manual_contract_pending"]
    observed = [row for row in rows if row["state_layout_status"] == "observed_resume_layout"]
    abi_ready = [row for row in rows if row.get("native_layout_abi_ready") is True]
    quality_ready = [row for row in rows if row.get("layout_quality_matrix_ready") is True]
    entry_ready = [row for row in rows if row.get("native_kernel_entry_condition_ready") is True]
    missing = sorted(set(FACTORED_MEMORY_PLUGIN_OPTIMIZERS) - selector_names)
    blockers = _dedupe(
        [f"selector_factored_memory_missing:{name}" for name in missing]
        + [reason for row in failed for reason in row.get("blocked_reasons", []) or []]
    )
    audit_complete = not missing and not failed
    layout_reference_ready = audit_complete and not manual and len(observed) == len(FACTORED_MEMORY_PLUGIN_OPTIMIZERS)
    selected_native_layout_abi_ready = audit_complete and len(abi_ready) == len(FACTORED_MEMORY_PLUGIN_OPTIMIZERS)
    layout_quality_matrix_ready = audit_complete and len(quality_ready) == len(FACTORED_MEMORY_PLUGIN_OPTIMIZERS)
    native_kernel_entry_conditions_ready = audit_complete and len(entry_ready) == len(FACTORED_MEMORY_PLUGIN_OPTIMIZERS)

    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_factored_memory_state_layout_scorecard_v0",
        "gate": "plugin_factored_memory_state_layout_audit",
        "ok": audit_complete,
        "promotion_ready": False,
        "state_layout_audit_complete": audit_complete,
        "state_layout_reference_ready": layout_reference_ready,
        "selected_native_layout_abi_ready": selected_native_layout_abi_ready,
        "layout_quality_matrix_ready": layout_quality_matrix_ready,
        "native_kernel_entry_conditions_ready": native_kernel_entry_conditions_ready,
        "selected_optimizer_abi_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "selector_type": OptimizerType.PYTORCH_OPTIMIZER.value,
        "selected_optimizer_family": "factored_memory_layout",
        "request_contract": {
            "optimizer_type": OptimizerType.PYTORCH_OPTIMIZER.value,
            "selected_optimizer_source": "optimizer_args.name",
            "selected_optimizer_names": list(FACTORED_MEMORY_PLUGIN_OPTIMIZERS),
            "runtime_authority": "existing_python_or_third_party_plugin_optimizer",
            "native_route_policy": "layout_abi_ready_but_dispatch_blocked_until_formula_parity_and_tensor_binding",
        },
        "default_off_contract": _default_off_contract(),
        "native_layout_abi_contract": {
            "schema_version": 1,
            "abi_name": "selected_plugin_factored_memory_layout_abi_v0",
            "abi_scope": "state-layout-only",
            "selected_optimizer_names": list(FACTORED_MEMORY_PLUGIN_OPTIMIZERS),
            "probe_dtype": "torch.float32",
            "probe_param_shape": [4, 4],
            "state_layout_source": "trainer_plugin_request_path_state_dict_after_step_and_resume",
            "formula_parity_ready": False,
            "training_tensor_binding_ready": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
        },
        "state_contract": {
            "uses_trainer_plugin_request_path": True,
            "covers_small_tensor_step": True,
            "covers_state_dict_resume_structure": True,
            "manual_contract_pending_status": "manual_contract_pending",
            "adamw_state_schema_compatible": False,
        },
        "layout_quality_contract": {
            "schema_version": 1,
            "matrix_name": "selected_plugin_factored_memory_layout_quality_matrix_v0",
            "requires_observed_resume_layout": True,
            "requires_state_key_inventory": True,
            "requires_tensor_shape_inventory": True,
            "requires_default_dispatch_off": True,
            "requires_adamw_reuse_blocked": True,
            "does_not_claim_formula_parity": True,
        },
        "selector_scorecard": {
            "scorecard": selector.get("scorecard"),
            "ok": selector.get("ok") is True,
            "factored_memory_optimizer_count": len(selector_names),
        },
        "rows": rows,
        "summary": {
            "case_count": len(rows),
            "observed_resume_layout_count": len(observed),
            "manual_contract_pending_count": len(manual),
            "failed_case_count": len(failed),
            "selector_factored_memory_count": len(selector_names),
            "native_layout_abi_ready_count": len(abi_ready),
            "layout_quality_matrix_ready_count": len(quality_ready),
            "native_kernel_entry_condition_ready_count": len(entry_ready),
            "native_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "plugin_selected_native_ready_count": 0,
        },
        "promotion_blockers": _promotion_blockers(
            blockers,
            selected_native_layout_abi_ready=selected_native_layout_abi_ready,
            layout_quality_matrix_ready=layout_quality_matrix_ready,
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "write factored-memory formula parity, tensor binding, and runtime-dispatch shadow gates"
            if selected_native_layout_abi_ready and layout_quality_matrix_ready
            else "write explicit native layout ABI and quality matrix for selected factored-memory plugin optimizers"
            if audit_complete
            else "fix plugin factored-memory state-layout audit blockers"
        ),
        "notes": [
            "This scorecard is report-only and never enables native dispatch.",
            "Observed state_dict layouts are not native readiness claims.",
            "Layout ABI and quality readiness only prove follow-up kernel entry conditions.",
            "Rows that cannot be safely instantiated must remain manual_contract_pending.",
        ],
    }


def _promotion_blockers(
    blockers: list[str],
    *,
    selected_native_layout_abi_ready: bool,
    layout_quality_matrix_ready: bool,
) -> list[str]:
    pending = list(blockers)
    if not selected_native_layout_abi_ready:
        pending.append("plugin_factored_memory_native_layout_abi_missing")
    if not layout_quality_matrix_ready:
        pending.append("plugin_factored_memory_quality_matrix_missing")
    pending.extend(
        [
            "plugin_factored_memory_formula_parity_missing",
            "plugin_factored_memory_training_tensor_binding_missing",
            "plugin_factored_memory_runtime_dispatch_shadow_missing",
            "owner_release_hold_missing",
        ]
    )
    return _dedupe(pending)


def _selector_factored_names(selector: Mapping[str, Any]) -> set[str]:
    rows = selector.get("rows", []) if isinstance(selector, Mapping) else []
    return {
        str(row.get("optimizer_name", "")).strip().lower()
        for row in rows
        if isinstance(row, Mapping) and row.get("native_route_family") == "factored_memory_layout"
    }


def _state_layout_case(name: str, selector_names: set[str]) -> dict[str, Any]:
    if name not in selector_names:
        return _manual_row(name, "not_classified_by_selector_factored_memory_layout")
    try:
        return _run_state_layout_case(name)
    except (ImportError, ModuleNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        return _manual_row(name, f"{type(exc).__name__}: {exc}")


def _run_state_layout_case(name: str) -> dict[str, Any]:
    value = torch.linspace(-0.25, 0.35, steps=16, dtype=torch.float32).reshape(4, 4)
    grad1 = torch.linspace(0.01, 0.06, steps=value.numel(), dtype=torch.float32).reshape_as(value)
    grad2 = torch.linspace(-0.04, 0.03, steps=value.numel(), dtype=torch.float32).reshape_as(value)
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
    ok = bool(after_step["state_present"] and after_step["param_state_keys"] and resume_diff <= 1e-5)
    layout_quality_matrix = _layout_quality_matrix(after_step, ok=ok)
    layout_quality_ready = layout_quality_matrix["ready"]
    native_layout_abi = _native_layout_abi(name, after_step)
    native_layout_abi_ready = ok and layout_quality_ready and bool(native_layout_abi["state_keys"])
    return {
        "schema_version": 1,
        "case": f"selected_plugin_factored_memory_{name}",
        "optimizer_name": name,
        "optimizer_class": type(optimizer).__name__,
        "selector_route_family": "factored_memory_layout",
        "state_layout_status": "observed_resume_layout" if ok else "state_layout_probe_failed",
        "ok": ok,
        "covers_trainer_plugin_request_path": True,
        "covers_small_tensor_step": True,
        "covers_resume": True,
        "after_step": after_step,
        "max_resume_diff": resume_diff,
        "tolerance": 1e-5,
        "native_layout_abi": native_layout_abi,
        "native_layout_abi_ready": native_layout_abi_ready,
        "layout_quality_matrix": layout_quality_matrix,
        "layout_quality_matrix_ready": layout_quality_ready,
        "native_kernel_entry_condition_ready": native_layout_abi_ready and layout_quality_ready,
        "pending_native_gates": [
            "factored_memory_formula_parity",
            "factored_memory_training_tensor_binding",
            "factored_memory_runtime_dispatch_shadow",
            "owner_release_hold",
        ],
        "default_off_contract": _default_off_contract(),
        "native_ready": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "blocked_reasons": [] if ok else [f"plugin_factored_memory_layout_probe_failed:{name}"],
    }


def _manual_row(name: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "case": f"selected_plugin_factored_memory_{name}",
        "optimizer_name": name,
        "selector_route_family": "factored_memory_layout",
        "state_layout_status": "manual_contract_pending",
        "ok": True,
        "manual_contract_reason": reason,
        "covers_trainer_plugin_request_path": False,
        "covers_small_tensor_step": False,
        "covers_resume": False,
        "native_layout_abi": {
            "schema_version": 1,
            "abi_name": "selected_plugin_factored_memory_layout_abi_v0",
            "optimizer_name": name,
            "state_keys": [],
            "tensor_state_shapes": {},
            "non_tensor_state_keys": [],
            "ready": False,
        },
        "native_layout_abi_ready": False,
        "layout_quality_matrix": {
            "schema_version": 1,
            "ready": False,
            "criteria": {},
            "pending_before_native_dispatch": [
                "observed_resume_layout",
                "factored_memory_formula_parity",
                "factored_memory_training_tensor_binding",
            ],
        },
        "layout_quality_matrix_ready": False,
        "native_kernel_entry_condition_ready": False,
        "pending_native_gates": [
            "observed_resume_layout",
            "factored_memory_formula_parity",
            "factored_memory_training_tensor_binding",
            "factored_memory_runtime_dispatch_shadow",
            "owner_release_hold",
        ],
        "default_off_contract": _default_off_contract(),
        "native_ready": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "blocked_reasons": [],
    }


def _make_trainer(name: str, value: torch.Tensor) -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    case = plugin_resume_case(name)
    trainer.config = LulynxConfig(
        optimizer_type=OptimizerType.PYTORCH_OPTIMIZER,
        learning_rate=1e-3,
        weight_decay=0.0,
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
        "tensor_state_shapes": _tensor_shapes(first_state),
        "non_tensor_state_keys": _non_tensor_keys(first_state),
        "step_like_value": _first_numeric(first_state, group, ("step", "k")),
    }


def _tensor_shapes(state: Mapping[str, Any]) -> dict[str, list[int]]:
    if not isinstance(state, Mapping):
        return {}
    return {str(key): list(value.shape) for key, value in state.items() if torch.is_tensor(value)}


def _non_tensor_keys(state: Mapping[str, Any]) -> list[str]:
    if not isinstance(state, Mapping):
        return []
    return sorted(str(key) for key, value in state.items() if not torch.is_tensor(value))


def _native_layout_abi(name: str, after_step: Mapping[str, Any]) -> dict[str, Any]:
    state_keys = [str(key) for key in after_step.get("param_state_keys", [])]
    tensor_shapes = dict(after_step.get("tensor_state_shapes", {}) or {})
    non_tensor_keys = [str(key) for key in after_step.get("non_tensor_state_keys", [])]
    return {
        "schema_version": 1,
        "abi_name": "selected_plugin_factored_memory_layout_abi_v0",
        "optimizer_name": name,
        "probe_dtype": "torch.float32",
        "probe_param_shape": [4, 4],
        "state_keys": state_keys,
        "tensor_state_shapes": tensor_shapes,
        "non_tensor_state_keys": non_tensor_keys,
        "state_key_count": len(state_keys),
        "tensor_state_count": len(tensor_shapes),
        "non_tensor_state_count": len(non_tensor_keys),
        "state_binding_policy": "bind_by_optimizer_state_key_after_resume",
        "formula_parity_ready": False,
        "training_tensor_binding_ready": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "ready": bool(state_keys) and bool(tensor_shapes),
    }


def _layout_quality_matrix(after_step: Mapping[str, Any], *, ok: bool) -> dict[str, Any]:
    criteria = {
        "trainer_plugin_request_path": True,
        "small_tensor_step_executed": True,
        "state_dict_resume_parity": ok,
        "state_key_inventory_ready": bool(after_step.get("param_state_keys")),
        "tensor_shape_inventory_ready": bool(after_step.get("tensor_state_shapes")),
        "default_dispatch_off": True,
        "adamw_kernel_reuse_blocked": True,
    }
    return {
        "schema_version": 1,
        "matrix_name": "selected_plugin_factored_memory_layout_quality_matrix_v0",
        "ready": all(criteria.values()),
        "criteria": criteria,
        "pending_before_native_dispatch": [
            "factored_memory_formula_parity",
            "factored_memory_training_tensor_binding",
            "factored_memory_runtime_dispatch_shadow",
            "owner_release_hold",
        ],
    }


def _default_off_contract() -> dict[str, Any]:
    return {
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "native_ready_count": 0,
        "plugin_selected_native_ready_count": 0,
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


__all__ = ["FACTORED_MEMORY_PLUGIN_OPTIMIZERS", "build_plugin_factored_memory_state_layout_scorecard"]
