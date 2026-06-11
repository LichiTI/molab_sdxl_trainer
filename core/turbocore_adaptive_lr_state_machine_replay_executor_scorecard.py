"""Trainer-path replay executor for built-in adaptive-LR state machines."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from core.lulynx_trainer.config import LulynxConfig, OptimizerType
from core.lulynx_trainer.trainer import LulynxTrainer
from core.turbocore_adaptive_lr_state_machine_replay_matrix_scorecard import (
    build_adaptive_lr_state_machine_replay_matrix_scorecard,
)


SCORECARD = "turbocore_adaptive_lr_state_machine_replay_executor_scorecard_v0"


@dataclass(frozen=True)
class _ReplayCase:
    optimizer: OptimizerType
    optimizer_args: str = ""
    learning_rate: float = 1e-3
    weight_decay: float = 0.01


TARGET_CASES: tuple[_ReplayCase, ...] = (
    _ReplayCase(OptimizerType.AUTO_PRODIGY, "d0=1e-5,growth_rate=1.01"),
    _ReplayCase(OptimizerType.PRODIGY, "d0=1e-6,d_coef=1.5"),
    _ReplayCase(OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE, "d0=1e-5,d_coef=1.0"),
    _ReplayCase(OptimizerType.DADAPTATION),
    _ReplayCase(OptimizerType.DADAPT_ADAM_PREPRINT),
    _ReplayCase(OptimizerType.DADAPT_ADAGRAD),
    _ReplayCase(OptimizerType.DADAPT_ADAM),
    _ReplayCase(OptimizerType.DADAPT_ADAN),
    _ReplayCase(OptimizerType.DADAPT_ADAN_IP),
    _ReplayCase(OptimizerType.DADAPT_LION),
    _ReplayCase(OptimizerType.DADAPT_SGD),
)


def build_adaptive_lr_state_machine_replay_executor_scorecard(
    *,
    replay_matrix_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run fallback-authority replay cases without enabling native dispatch."""

    replay_matrix = _as_dict(
        replay_matrix_report or build_adaptive_lr_state_machine_replay_matrix_scorecard()
    )
    rows = [_run_case(case, replay_matrix) for case in TARGET_CASES]
    validations = _validations(replay_matrix, rows)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": SCORECARD,
        "gate": "adaptive_lr_state_machine_replay_executor",
        "ok": ready,
        "promotion_ready": False,
        "state_machine_replay_executor_ready": ready,
        "report_only": True,
        "fallback_authority": "existing_python_or_third_party_optimizer",
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_ready_count": 0,
        "rows": rows,
        "replay_matrix_summary": _as_dict(replay_matrix.get("summary")),
        "validations": validations,
        "summary": {
            "target_count": len(rows),
            "reference_replay_executor_ready_count": sum(
                1 for row in rows if row.get("reference_replay_executor_ready") is True
            ),
            "state_machine_replay_matrix_implementation_ready_count": sum(
                1 for row in rows if row.get("state_machine_replay_matrix_implementation_ready") is True
            ),
            "resume_next_step_parity_passed_count": sum(
                1 for row in rows if row.get("resume_next_step_parity_passed") is True
            ),
            "state_machine_abi_implementation_ready_count": 0,
            "native_kernel_preconditions_implementation_ready_count": 0,
            "native_ready_count": 0,
            "product_native_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "adaptive_lr_native_state_machine_abi_implementation_missing",
                "adaptive_lr_cuda_kernel_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "implement adaptive-LR native state-machine ABI preconditions with dispatch still default-off"
            if ready
            else "fix adaptive-LR replay executor blockers"
        ),
        "notes": [
            "This scorecard proves fallback-authority trainer-path replay for adaptive-LR optimizers.",
            "It does not claim native ABI, CUDA kernel, or product dispatch readiness.",
            "Request, schema, UI, runtime dispatch, and default behavior remain unchanged.",
        ],
    }


def _run_case(case: _ReplayCase, replay_matrix: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = _run_resume_parity(case)
        matrix_row = _matrix_rows(replay_matrix).get(case.optimizer.value, {})
        ready = (
            result["max_resume_diff"] <= result["tolerance"]
            and _as_dict(matrix_row.get("state_machine_replay_matrix_artifact")).get("spec_ready") is True
        )
        return {
            "schema_version": 1,
            "optimizer_type": case.optimizer.value,
            "optimizer_class": result["optimizer_class"],
            "state_machine_status": "reference_replay_executor_ready" if ready else "reference_replay_executor_failed",
            "reference_replay_executor_ready": ready,
            "resume_next_step_parity_passed": result["max_resume_diff"] <= result["tolerance"],
            "max_resume_diff": result["max_resume_diff"],
            "tolerance": result["tolerance"],
            "state_dict_top_level_keys": result["state_dict_top_level_keys"],
            "state_dict_param_group_keys": result["state_dict_param_group_keys"],
            "state_entry_count": result["state_entry_count"],
            "matrix_artifact_ready": _as_dict(matrix_row.get("state_machine_replay_matrix_artifact")).get(
                "spec_ready"
            )
            is True,
            "state_machine_replay_matrix_implementation_ready": ready,
            "runtime_authority": "existing_python_or_third_party_optimizer",
            "native_route": "none_report_only",
            "product_native_ready": False,
            "training_path_enabled": False,
            "default_behavior_changed": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "native_kernel_ready": False,
            "next_gate": "adaptive_lr_native_state_machine_abi_preconditions",
            "blocked_reasons": [] if ready else [f"{case.optimizer.value}_adaptive_lr_replay_executor_failed"],
        }
    except Exception as exc:
        return {
            "schema_version": 1,
            "optimizer_type": case.optimizer.value,
            "state_machine_status": "reference_replay_executor_failed",
            "reference_replay_executor_ready": False,
            "resume_next_step_parity_passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "product_native_ready": False,
            "training_path_enabled": False,
            "default_behavior_changed": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "native_kernel_ready": False,
            "blocked_reasons": [f"{case.optimizer.value}_adaptive_lr_replay_executor_exception"],
        }


def _run_resume_parity(case: _ReplayCase) -> dict[str, Any]:
    torch.manual_seed(1234)
    shape = (4, 4)
    initial = torch.linspace(-0.25, 0.35, steps=16, dtype=torch.float32).reshape(shape)
    grad1 = torch.linspace(0.01, 0.05, steps=16, dtype=torch.float32).reshape(shape)
    grad2 = torch.linspace(-0.03, 0.02, steps=16, dtype=torch.float32).reshape(shape)

    trainer = _make_trainer(case, initial)
    optimizer = trainer._create_optimizer()
    param = trainer.lora_injector.param
    optimizer_class = type(optimizer).__name__
    _step(param, optimizer, grad1)
    saved_state = copy.deepcopy(optimizer.state_dict())
    saved_param = param.detach().clone()

    restored_trainer = _make_trainer(case, saved_param)
    restored_optimizer = restored_trainer._create_optimizer()
    restored_param = restored_trainer.lora_injector.param
    restored_optimizer.load_state_dict(saved_state)

    _step(param, optimizer, grad2)
    _step(restored_param, restored_optimizer, grad2)
    max_diff = float((param.detach().float() - restored_param.detach().float()).abs().max().cpu())
    groups = saved_state.get("param_groups", []) if isinstance(saved_state, Mapping) else []
    first_group = groups[0] if groups and isinstance(groups[0], Mapping) else {}
    state = saved_state.get("state", {}) if isinstance(saved_state, Mapping) else {}
    return {
        "optimizer_class": optimizer_class,
        "max_resume_diff": max_diff,
        "tolerance": 1e-6,
        "state_dict_top_level_keys": sorted(str(key) for key in saved_state.keys()),
        "state_dict_param_group_keys": sorted(str(key) for key in first_group.keys() if key != "params"),
        "state_entry_count": len(state) if isinstance(state, Mapping) else 0,
    }


def _make_trainer(case: _ReplayCase, value: torch.Tensor) -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = LulynxConfig(
        optimizer_type=case.optimizer,
        learning_rate=case.learning_rate,
        weight_decay=case.weight_decay,
        optimizer_args=case.optimizer_args,
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


class _Injector:
    def __init__(self, value: torch.Tensor) -> None:
        self.param = torch.nn.Parameter(value.detach().clone())
        self.injected_layers = {}

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [self.param]


def _step(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer, grad: torch.Tensor) -> None:
    param.grad = grad.detach().clone().to(dtype=param.dtype)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _validations(replay_matrix: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = {case.optimizer.value for case in TARGET_CASES}
    present = {str(row.get("optimizer_type") or "") for row in rows}
    return [
        _validation(
            "replay_matrix_ready",
            replay_matrix.get("state_machine_replay_matrix_ready") is True,
            "adaptive_lr_replay_matrix_missing",
        ),
        _validation(
            "optimizer_set_complete",
            present == expected,
            "adaptive_lr_replay_executor_optimizer_set_incomplete",
        ),
        _validation(
            "all_reference_replay_passed",
            all(row.get("reference_replay_executor_ready") is True for row in rows),
            "adaptive_lr_replay_executor_case_failed",
        ),
        _validation(
            "runtime_dispatch_disabled",
            replay_matrix.get("training_path_enabled") is False
            and replay_matrix.get("runtime_dispatch_ready") is False
            and replay_matrix.get("native_dispatch_allowed") is False
            and all(row.get("training_path_enabled") is False for row in rows)
            and all(row.get("runtime_dispatch_ready") is False for row in rows)
            and all(row.get("native_dispatch_allowed") is False for row in rows),
            "adaptive_lr_replay_executor_enabled_dispatch",
        ),
    ]


def _matrix_rows(replay_matrix: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("optimizer_type") or ""): row
        for row in replay_matrix.get("rows", [])
        if isinstance(row, Mapping)
    }


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["SCORECARD", "TARGET_CASES", "build_adaptive_lr_state_machine_replay_executor_scorecard"]
