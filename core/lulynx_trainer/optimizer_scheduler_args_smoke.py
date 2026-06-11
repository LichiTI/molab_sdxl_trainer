# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test safe optimizer/scheduler custom argument wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.config import LulynxConfig, OptimizerType, SchedulerType
from core.lulynx_trainer.trainer import LulynxTrainer


class _Injector:
    def __init__(self) -> None:
        self.param = torch.nn.Parameter(torch.ones(2, 2))

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [self.param]


def _make_minimal_trainer(*, optimizer: OptimizerType, optimizer_args: str) -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = LulynxConfig(
        optimizer_type=optimizer,
        lr_scheduler=SchedulerType.POLYNOMIAL,
        learning_rate=1e-3,
        weight_decay=0.01,
        optimizer_args=optimizer_args,
    )
    trainer.config.semantic_tuner_enabled = False
    trainer.lora_injector = _Injector()
    trainer._block_weight_manager = None
    trainer._log = lambda _msg: None
    return trainer


def main() -> int:
    parsed = ConfigAdapter.from_frontend_dict(
        {
            "model_type": "sdxl",
            "optimizer_args_custom": "betas=(0.8, 0.9),eps=1e-7,unsupported=1",
            "lr_scheduler_type": "polynomial",
            "lr_scheduler_args": "power=2.0,total_iters=5",
        }
    )
    assert parsed.optimizer_args == "betas=(0.8, 0.9),eps=1e-7,unsupported=1"
    assert parsed.lr_scheduler == SchedulerType.POLYNOMIAL
    assert parsed.lr_scheduler_args == "power=2.0,total_iters=5"

    plugin_parsed = ConfigAdapter.from_frontend_dict(
        {
            "model_type": "sdxl",
            "optimizer_type": "pytorch_optimizer.CAME",
            "optimizer_args_custom": "eps=1e-8",
        }
    )
    assert plugin_parsed.optimizer == OptimizerType.PYTORCH_OPTIMIZER
    assert "name=CAME" in plugin_parsed.optimizer_args
    assert "eps=1e-8" in plugin_parsed.optimizer_args

    direct_plugin = LulynxConfig(
        optimizer_type="pytorch_optimizer.SCION",
        optimizer_args="",
    )
    assert direct_plugin.optimizer == OptimizerType.PYTORCH_OPTIMIZER
    assert direct_plugin.optimizer_args == "name=SCION"

    for optimizer_name in ("CAME", "StableAdamW", "SCION"):
        plugin_config = LulynxConfig(
            optimizer_type=f"pytorch_optimizer.{optimizer_name}",
            optimizer_args="",
            learning_rate=1e-3,
            weight_decay=0.01,
        )
        assert plugin_config.optimizer == OptimizerType.PYTORCH_OPTIMIZER
        assert plugin_config.optimizer_args == f"name={optimizer_name}"

        plugin_trainer = _make_minimal_trainer(
            optimizer=plugin_config.optimizer,
            optimizer_args=plugin_config.optimizer_args,
        )
        plugin_optimizer = plugin_trainer._create_optimizer()
        assert type(plugin_optimizer).__name__.lower() == optimizer_name.lower()

    trainer = _make_minimal_trainer(
        optimizer=OptimizerType.ADAMW,
        optimizer_args="betas=(0.8, 0.9),eps=1e-7,unsupported=1",
    )
    trainer.config.lr_scheduler_args = "power=2.0,total_iters=5,ignored=1"

    optimizer = trainer._create_optimizer()
    group = optimizer.param_groups[0]
    assert group["betas"] == (0.8, 0.9)
    assert group["eps"] == 1e-7
    assert "unsupported" not in group

    scheduler = trainer._create_scheduler(optimizer, total_steps=10)
    assert getattr(scheduler, "power") == 2.0
    assert getattr(scheduler, "total_iters") == 5

    print("Optimizer/scheduler args smoke passed: safe custom args are parsed and filtered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
