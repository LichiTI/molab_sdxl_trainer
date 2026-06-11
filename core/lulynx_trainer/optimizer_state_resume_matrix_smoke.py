# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""CPU-safe optimizer state/resume smoke matrix.

This verifies more than construction: an optimizer route must perform a step,
export state_dict, load that state into a fresh optimizer on cloned parameters,
and produce the same next update as the original optimizer.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.config import LulynxConfig, OptimizerType
from core.lulynx_trainer.trainer import LulynxTrainer


@dataclass(frozen=True)
class OptimizerResumeCase:
    optimizer_type: OptimizerType
    optimizer_args: str = ""
    shape: tuple[int, ...] = (4, 4)
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    config_overrides: dict[str, Any] | None = None
    device: str = "cpu"


class _Injector:
    def __init__(self, value: torch.Tensor) -> None:
        self.param = torch.nn.Parameter(value.detach().clone())
        self.injected_layers = {}

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [self.param]


def _make_trainer(case: OptimizerResumeCase, value: torch.Tensor) -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = LulynxConfig(
        optimizer_type=case.optimizer_type,
        learning_rate=case.learning_rate,
        weight_decay=case.weight_decay,
        optimizer_args=case.optimizer_args,
    )
    trainer.config.semantic_tuner_enabled = False
    for key, value in (case.config_overrides or {}).items():
        setattr(trainer.config, key, value)
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


def _step(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer, grad: torch.Tensor) -> None:
    param.grad = grad.detach().clone().to(device=param.device, dtype=param.dtype)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _run_resume_case(case: OptimizerResumeCase) -> tuple[str, str]:
    torch.manual_seed(1234)
    device = torch.device(case.device)
    initial = torch.linspace(
        -0.25,
        0.35,
        steps=max(1, int(torch.tensor(case.shape).prod().item())),
        device=device,
    ).reshape(case.shape)
    grad1 = torch.linspace(0.01, 0.05, steps=initial.numel()).reshape(case.shape)
    grad2 = torch.linspace(-0.03, 0.02, steps=initial.numel()).reshape(case.shape)

    trainer = _make_trainer(case, initial)
    optimizer = trainer._create_optimizer()
    original_param = trainer.lora_injector.param
    optimizer_name = type(optimizer).__name__

    _step(original_param, optimizer, grad1)
    saved_state = copy.deepcopy(optimizer.state_dict())
    saved_param = original_param.detach().clone()

    restored_trainer = _make_trainer(case, saved_param)
    restored_optimizer = restored_trainer._create_optimizer()
    restored_param = restored_trainer.lora_injector.param
    restored_optimizer.load_state_dict(saved_state)

    _step(original_param, optimizer, grad2)
    _step(restored_param, restored_optimizer, grad2)

    if not torch.allclose(original_param.detach(), restored_param.detach(), atol=1e-6, rtol=1e-5):
        max_diff = (original_param.detach() - restored_param.detach()).abs().max().item()
        raise AssertionError(
            f"{case.optimizer_type.value} resume mismatch for {optimizer_name}: max_diff={max_diff}"
        )
    return case.optimizer_type.value, optimizer_name


def test_cpu_safe_state_resume_matrix() -> None:
    cases = [
        OptimizerResumeCase(OptimizerType.ADAMW),
        OptimizerResumeCase(OptimizerType.SGD_NESTEROV),
        OptimizerResumeCase(OptimizerType.KAHAN_ADAMW_8BIT),
        OptimizerResumeCase(OptimizerType.AUTOMAGIC_PLUS_PLUS),
        OptimizerResumeCase(OptimizerType.AUTO_PRODIGY, optimizer_args="d0=1e-5,growth_rate=1.01"),
        OptimizerResumeCase(OptimizerType.ANIMA_FACTORED_ADAMW, optimizer_args="min_dim=2,min_numel=4"),
        OptimizerResumeCase(OptimizerType.PRODIGY, optimizer_args="d0=1e-6,d_coef=1.5"),
        OptimizerResumeCase(OptimizerType.ADAFACTOR),
        OptimizerResumeCase(OptimizerType.LION),
        OptimizerResumeCase(OptimizerType.DADAPTATION),
        OptimizerResumeCase(OptimizerType.DADAPT_ADAM_PREPRINT),
        OptimizerResumeCase(OptimizerType.DADAPT_ADAGRAD),
        OptimizerResumeCase(OptimizerType.DADAPT_ADAM),
        OptimizerResumeCase(OptimizerType.DADAPT_ADAN),
        OptimizerResumeCase(OptimizerType.DADAPT_ADAN_IP),
        OptimizerResumeCase(OptimizerType.DADAPT_LION),
        OptimizerResumeCase(OptimizerType.DADAPT_SGD),
        OptimizerResumeCase(OptimizerType.ADAMW_SCHEDULE_FREE),
        OptimizerResumeCase(OptimizerType.RADAM_SCHEDULE_FREE),
        OptimizerResumeCase(OptimizerType.SGD_SCHEDULE_FREE),
        OptimizerResumeCase(OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE, optimizer_args="d0=1e-5,d_coef=1.0"),
        OptimizerResumeCase(OptimizerType.GENERIC, optimizer_args="name=Adam"),
        OptimizerResumeCase(OptimizerType.PYTORCH_OPTIMIZER, optimizer_args="name=StableAdamW"),
    ]
    results = [_run_resume_case(case) for case in cases]
    assert len(results) == len(cases), results


def test_cuda_bitsandbytes_state_resume_matrix() -> None:
    if not torch.cuda.is_available():
        return
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return

    cases = [
        OptimizerResumeCase(OptimizerType.ADAMW_8BIT, device="cuda"),
        OptimizerResumeCase(OptimizerType.PAGED_ADAMW, device="cuda"),
        OptimizerResumeCase(OptimizerType.PAGED_ADAMW_32BIT, device="cuda"),
        OptimizerResumeCase(OptimizerType.PAGED_ADAMW_8BIT, device="cuda"),
        OptimizerResumeCase(OptimizerType.PAGED_LION_8BIT, device="cuda"),
        OptimizerResumeCase(OptimizerType.SGD_NESTEROV_8BIT, device="cuda"),
        OptimizerResumeCase(OptimizerType.LION_8BIT, device="cuda"),
    ]
    results = [_run_resume_case(case) for case in cases]
    assert len(results) == len(cases), results


def main() -> int:
    test_cpu_safe_state_resume_matrix()
    test_cuda_bitsandbytes_state_resume_matrix()
    print("optimizer_state_resume_matrix_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
