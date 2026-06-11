# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Trainer-path state/resume matrix for pytorch_optimizer plugin routes.

This is intentionally narrower than plugin discovery. A plugin optimizer only
enters this matrix after it can run through LulynxTrainer._create_optimizer,
step once, save/load state_dict, and produce the same next update after resume.
"""

from __future__ import annotations

import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.configs import SchedulerType
from core.lulynx_trainer.config import LulynxConfig, OptimizerType
from core.lulynx_trainer.optimizer_plugin_support import (
    PLUGIN_PENDING_OR_SPECIAL,
    PLUGIN_RESUME_SMOKE_PASSED,
    plugin_resume_case,
    plugin_resume_cases,
)
from core.lulynx_trainer.optimizer_step_contracts import (
    bind_loss_value_closure,
    bind_step_closure,
    optimizer_requires_create_graph_backward,
    optimizer_requires_step_closure,
    optimizer_step_closure_requires_initial_backward,
    optimizer_uses_fused_backward,
    run_optimizer_fused_backward,
)
from core.lulynx_trainer.trainer import LulynxTrainer


@dataclass(frozen=True)
class PluginResumeCase:
    name: str
    extra_args: str = ""
    shape: tuple[int, ...] = (4, 4)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    scheduler_expected: str = "CosineAnnealingLR"
    include_vector_param: bool = False
    expected_param_group_flags: tuple[bool, ...] = ()

    @property
    def optimizer_args(self) -> str:
        return " ".join(part for part in (f"name={self.name}", self.extra_args) if part)


class _Injector:
    def __init__(self, value: torch.Tensor, vector_value: torch.Tensor | None = None) -> None:
        self.param = torch.nn.Parameter(value.detach().clone())
        self.vector_param = torch.nn.Parameter(vector_value.detach().clone()) if vector_value is not None else None
        self.injected_layers = {}

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        params = [self.param]
        if self.vector_param is not None:
            params.append(self.vector_param)
        return params


def _make_trainer(case: PluginResumeCase, value: torch.Tensor, vector_value: torch.Tensor | None = None) -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = LulynxConfig(
        optimizer_type=OptimizerType.PYTORCH_OPTIMIZER,
        learning_rate=case.learning_rate,
        weight_decay=case.weight_decay,
        optimizer_args=case.optimizer_args,
        lr_scheduler=SchedulerType.COSINE,
        warmup_ratio=0.0,
    )
    trainer.config.semantic_tuner_enabled = False
    trainer.lora_injector = _Injector(value, vector_value)
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


def _set_grads(trainer: LulynxTrainer, grad: torch.Tensor, vector_grad: torch.Tensor | None = None) -> None:
    injector = trainer.lora_injector
    injector.param.grad = grad.detach().clone().to(device=injector.param.device, dtype=injector.param.dtype)
    if injector.vector_param is not None:
        if vector_grad is None:
            raise AssertionError("vector_grad is required for vector param cases")
        injector.vector_param.grad = vector_grad.detach().clone().to(
            device=injector.vector_param.device,
            dtype=injector.vector_param.dtype,
        )


def _step(trainer: LulynxTrainer, optimizer: torch.optim.Optimizer, grad: torch.Tensor, vector_grad: torch.Tensor | None = None) -> None:
    if optimizer_requires_step_closure(optimizer):
        injector = trainer.lora_injector
        torch.manual_seed(20260531)

        def compute_loss() -> torch.Tensor:
            target = grad.detach().clone().to(device=injector.param.device, dtype=injector.param.dtype)
            value = ((injector.param * injector.param) - target).pow(2).mean()
            if injector.vector_param is not None:
                if vector_grad is None:
                    raise AssertionError("vector_grad is required for vector param cases")
                vector_target = vector_grad.detach().clone().to(
                    device=injector.vector_param.device,
                    dtype=injector.vector_param.dtype,
                )
                value = value + ((injector.vector_param * injector.vector_param) - vector_target).pow(2).mean()
            return value

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss()
            loss.backward()
            return loss

        if optimizer_step_closure_requires_initial_backward(optimizer):
            optimizer.zero_grad(set_to_none=True)
            compute_loss().backward()
        bind_step_closure(optimizer, closure)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return

    if optimizer_uses_fused_backward(optimizer):
        optimizer.zero_grad(set_to_none=True)
        injector = trainer.lora_injector
        target = grad.detach().clone().to(device=injector.param.device, dtype=injector.param.dtype)
        loss = ((injector.param * injector.param) - target).pow(2).mean()
        if injector.vector_param is not None:
            if vector_grad is None:
                raise AssertionError("vector_grad is required for vector param cases")
            vector_target = vector_grad.detach().clone().to(
                device=injector.vector_param.device,
                dtype=injector.vector_param.dtype,
            )
            loss = loss + ((injector.vector_param * injector.vector_param) - vector_target).pow(2).mean()
        assert run_optimizer_fused_backward(optimizer, loss, float(optimizer.param_groups[0]["lr"])) is True
        optimizer.zero_grad(set_to_none=True)
        return

    if optimizer_requires_create_graph_backward(optimizer):
        optimizer.zero_grad(set_to_none=True)
        if hasattr(optimizer, "train"):
            optimizer.train()
        injector = trainer.lora_injector
        target = grad.detach().clone().to(device=injector.param.device, dtype=injector.param.dtype)
        loss = ((injector.param * injector.param) - target).pow(2).mean()
        if injector.vector_param is not None:
            if vector_grad is None:
                raise AssertionError("vector_grad is required for vector param cases")
            vector_target = vector_grad.detach().clone().to(
                device=injector.vector_param.device,
                dtype=injector.vector_param.dtype,
            )
            loss = loss + ((injector.vector_param * injector.vector_param) - vector_target).pow(2).mean()
        loss.backward(create_graph=True)
        bind_loss_value_closure(optimizer, float(loss.detach().float().item()))
        torch.manual_seed(20260531)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return

    _set_grads(trainer, grad, vector_grad)
    if hasattr(optimizer, "train"):
        optimizer.train()
    bind_loss_value_closure(optimizer, float(grad.abs().mean().item()))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _assert_param_group_flags(case: PluginResumeCase, optimizer: torch.optim.Optimizer) -> None:
    if not case.expected_param_group_flags:
        return
    flags = tuple(bool(group.get("use_muon")) for group in optimizer.param_groups)
    assert flags == case.expected_param_group_flags, (case.name, flags, optimizer.param_groups)


def _assert_scheduler_policy(case: PluginResumeCase, trainer: LulynxTrainer, optimizer: torch.optim.Optimizer) -> None:
    scheduler = trainer._create_scheduler(optimizer, total_steps=8)
    assert type(scheduler).__name__ == case.scheduler_expected, (
        case.name,
        type(scheduler).__name__,
        case.scheduler_expected,
    )


def _capture_rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state().clone(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    torch_state = state.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state)
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)


def _run_case(case: PluginResumeCase) -> tuple[str, str]:
    torch.manual_seed(20260530)
    numel = max(1, int(torch.tensor(case.shape).prod().item()))
    initial = torch.linspace(-0.25, 0.35, steps=numel).reshape(case.shape)
    grad1 = torch.linspace(0.01, 0.05, steps=numel).reshape(case.shape)
    grad2 = torch.linspace(-0.03, 0.02, steps=numel).reshape(case.shape)

    vector_initial = vector_grad1 = vector_grad2 = None
    if case.include_vector_param or case.expected_param_group_flags:
        vector_initial = torch.linspace(-0.2, 0.1, steps=case.shape[0])
        vector_grad1 = torch.linspace(0.02, 0.04, steps=case.shape[0])
        vector_grad2 = torch.linspace(-0.01, 0.03, steps=case.shape[0])

    trainer = _make_trainer(case, initial, vector_initial)
    optimizer = trainer._create_optimizer()
    optimizer_name = type(optimizer).__name__
    _assert_param_group_flags(case, optimizer)
    _assert_scheduler_policy(case, trainer, optimizer)

    _step(trainer, optimizer, grad1, vector_grad1)
    saved_state = copy.deepcopy(optimizer.state_dict())
    saved_param = trainer.lora_injector.param.detach().clone()
    saved_rng_state = _capture_rng_state()
    saved_vector = None
    if trainer.lora_injector.vector_param is not None:
        saved_vector = trainer.lora_injector.vector_param.detach().clone()

    restored_trainer = _make_trainer(case, saved_param, saved_vector)
    restored_optimizer = restored_trainer._create_optimizer()
    restored_optimizer.load_state_dict(saved_state)

    _restore_rng_state(saved_rng_state)
    _step(trainer, optimizer, grad2, vector_grad2)
    _restore_rng_state(saved_rng_state)
    _step(restored_trainer, restored_optimizer, grad2, vector_grad2)

    original_param = trainer.lora_injector.param.detach()
    restored_param = restored_trainer.lora_injector.param.detach()
    if not torch.allclose(original_param, restored_param, atol=1e-6, rtol=1e-5):
        max_diff = (original_param - restored_param).abs().max().item()
        raise AssertionError(f"{case.name} resume mismatch for {optimizer_name}: max_diff={max_diff}")

    if trainer.lora_injector.vector_param is not None:
        original_vector = trainer.lora_injector.vector_param.detach()
        restored_vector = restored_trainer.lora_injector.vector_param.detach()
        if not torch.allclose(original_vector, restored_vector, atol=1e-6, rtol=1e-5):
            max_diff = (original_vector - restored_vector).abs().max().item()
            raise AssertionError(f"{case.name} vector resume mismatch for {optimizer_name}: max_diff={max_diff}")
    return case.name, optimizer_name


def test_plugin_state_resume_matrix() -> None:
    cases = [PluginResumeCase(**case.__dict__) for case in plugin_resume_cases()]
    results = [_run_case(case) for case in cases]
    assert len(results) == len(cases), results


def test_distributed_muon_single_process_identity_fallback() -> None:
    assert "distributedmuon" in PLUGIN_RESUME_SMOKE_PASSED
    case = PluginResumeCase("DistributedMuon", include_vector_param=True, expected_param_group_flags=(True, False))
    name, optimizer_name = _run_case(case)
    assert name == "DistributedMuon"
    assert optimizer_name == "DistributedMuon"


def test_pending_or_special_routes_are_not_in_resume_allowlist() -> None:
    overlap = set(PLUGIN_PENDING_OR_SPECIAL).intersection(PLUGIN_RESUME_SMOKE_PASSED)
    assert overlap == set(), overlap


def test_key_frontier_routes_are_in_resume_allowlist() -> None:
    for name in (
        "came",
        "stableadamw",
        "scion",
        "ademamix",
        "apollo",
        "apollodqn",
        "bsam",
        "muon",
        "adamuon",
        "adago",
        "schedulefreeadamw",
        "schedulefreeradam",
        "schedulefreesgd",
        "a2grad",
        "adafactor",
        "adahessian",
        "adammini",
        "adalomo",
        "alig",
        "alice",
        "demo",
        "distributedmuon",
        "kron",
        "lbfgs",
        "lomo",
        "spam",
        "sgdsai",
        "spectralsphere",
    ):
        assert name in PLUGIN_RESUME_SMOKE_PASSED, name

    ranger21 = plugin_resume_case("ranger21")
    assert "num_iterations" in ranger21.extra_args
    assert "num_data=1" in plugin_resume_case("bsam").extra_args
    assert "max_iter=1" in plugin_resume_case("lbfgs").extra_args
    assert plugin_resume_case("schedulefreeradam").scheduler_expected == "ConstantLR"


def main() -> int:
    test_plugin_state_resume_matrix()
    test_distributed_muon_single_process_identity_fallback()
    test_pending_or_special_routes_are_not_in_resume_allowlist()
    test_key_frontier_routes_are_in_resume_allowlist()
    print("optimizer_plugin_resume_matrix_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
