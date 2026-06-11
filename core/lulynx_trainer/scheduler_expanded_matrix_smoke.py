# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test expanded LR scheduler selection through the native trainer path."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.config import LulynxConfig, OptimizerType, SchedulerType
from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.loss_aware_scheduler import LossAwareCosineScheduler
from core.lulynx_trainer.trainer import LulynxTrainer


def _trainer(*, scheduler: SchedulerType, scheduler_args: str = "", optimizer_type: OptimizerType = OptimizerType.ADAMW) -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = LulynxConfig(
        optimizer_type=optimizer_type,
        lr_scheduler=scheduler,
        lr_scheduler_args=scheduler_args,
        learning_rate=1e-3,
        weight_decay=0.01,
        lr_scheduler_num_cycles=2,
    )
    trainer._log = lambda _msg: None
    return trainer


def _optimizer() -> torch.optim.Optimizer:
    return torch.optim.AdamW([torch.nn.Parameter(torch.ones(2, 2))], lr=1e-3)


def _assert_scheduler(scheduler: SchedulerType, expected_name: str, *, args: str = "") -> object:
    sched = _trainer(scheduler=scheduler, scheduler_args=args)._create_scheduler(_optimizer(), total_steps=12)
    assert type(sched).__name__ == expected_name, (scheduler, type(sched).__name__, expected_name)
    return sched


def test_config_aliases() -> None:
    cases = {
        "cosine": SchedulerType.COSINE,
        "cosine_with_restarts": SchedulerType.COSINE_RESTARTS,
        "cosine_with_min_lr": SchedulerType.COSINE_WITH_MIN_LR,
        "loss_gated_cosine": SchedulerType.LOSS_GATED_COSINE,
        "loss_weighted_annealed_cosine": SchedulerType.LOSS_WEIGHTED_ANNEALED_COSINE,
        "constant": SchedulerType.CONSTANT,
        "constant_with_warmup": SchedulerType.CONSTANT_WARMUP,
        "linear": SchedulerType.LINEAR,
        "polynomial": SchedulerType.POLYNOMIAL,
        "piecewise_constant": SchedulerType.PIECEWISE_CONSTANT,
        "warmup_stable_decay": SchedulerType.TSD,
        "one_cycle": SchedulerType.ONE_CYCLE,
        "inverse_sqrt": SchedulerType.INVERSE_SQRT,
        "adafactor": SchedulerType.ADAFACTOR,
        "restart_linear": SchedulerType.RESTART_LINEAR,
    }
    for raw, expected in cases.items():
        cfg = ConfigAdapter.from_frontend_dict({"schema_id": "sdxl-lora", "lr_scheduler_type": raw})
        assert cfg.lr_scheduler == expected, (raw, cfg.lr_scheduler, expected)


def test_scheduler_matrix() -> None:
    _assert_scheduler(SchedulerType.COSINE, "CosineAnnealingLR", args="eta_min=1e-5")
    _assert_scheduler(SchedulerType.COSINE_RESTARTS, "CosineAnnealingWarmRestarts", args="t_0=3,t_mult=2,eta_min=1e-5")
    _assert_scheduler(SchedulerType.COSINE_WITH_MIN_LR, "LambdaLR", args="min_lr_ratio=0.2,num_cycles=1")
    _assert_scheduler(SchedulerType.LINEAR, "LinearLR", args="start_factor=1.0,end_factor=0.1,total_iters=5")
    _assert_scheduler(SchedulerType.CONSTANT, "ConstantLR")
    _assert_scheduler(SchedulerType.CONSTANT_WARMUP, "ConstantLR")
    _assert_scheduler(SchedulerType.POLYNOMIAL, "PolynomialLR", args="power=2,total_iters=5")
    _assert_scheduler(SchedulerType.PIECEWISE_CONSTANT, "LambdaLR", args="rules=0:1,4:0.5,8:0.1")
    _assert_scheduler(SchedulerType.TSD, "SequentialLR")
    _assert_scheduler(SchedulerType.ONE_CYCLE, "OneCycleLR", args="max_lr=0.003,pct_start=0.2")
    _assert_scheduler(SchedulerType.INVERSE_SQRT, "LambdaLR")
    _assert_scheduler(SchedulerType.ADAFACTOR, "ConstantLR")
    _assert_scheduler(SchedulerType.RESTART_LINEAR, "SequentialLR", args="t_0=4,eta_min=0.1")

    gated = _assert_scheduler(
        SchedulerType.LOSS_GATED_COSINE,
        "LossAwareCosineScheduler",
        args="eta_min=1e-5,patience=2,max_hold_steps=3",
    )
    assert isinstance(gated, LossAwareCosineScheduler)
    assert gated.mode == "gated"

    weighted = _assert_scheduler(
        SchedulerType.LOSS_WEIGHTED_ANNEALED_COSINE,
        "LossAwareCosineScheduler",
        args="eta_min=1e-5,late_loss_gamma=1.5,min_advance_ratio=0.25",
    )
    assert isinstance(weighted, LossAwareCosineScheduler)
    assert weighted.mode == "weighted"


def test_schedule_free_optimizers_force_constant_scheduler() -> None:
    trainer = _trainer(
        scheduler=SchedulerType.COSINE,
        optimizer_type=OptimizerType.ADAMW_SCHEDULE_FREE,
    )
    sched = trainer._create_scheduler(_optimizer(), total_steps=12)
    assert type(sched).__name__ == "ConstantLR"


def main() -> int:
    test_config_aliases()
    test_scheduler_matrix()
    test_schedule_free_optimizers_force_constant_scheduler()
    print("Expanded scheduler matrix smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
