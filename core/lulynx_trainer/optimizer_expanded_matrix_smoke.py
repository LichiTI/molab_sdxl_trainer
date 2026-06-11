# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test expanded optimizer selection through the native trainer path."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.config import LulynxConfig, OptimizerType
from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.optimizer_capabilities import optimizer_capability_report
from core.lulynx_trainer.trainer import LulynxTrainer


class _Injector:
    def __init__(self) -> None:
        self.param = torch.nn.Parameter(torch.ones(2, 2))
        self.injected_layers = {}

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [self.param]


def _make_trainer(*, optimizer_type: str | OptimizerType, optimizer_args: str = "") -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = LulynxConfig(
        optimizer_type=optimizer_type,
        learning_rate=1e-3,
        weight_decay=0.01,
        optimizer_args=optimizer_args,
    )
    trainer.config.semantic_tuner_enabled = False
    trainer.lora_injector = _Injector()
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


def _assert_optimizer(opt_type: str | OptimizerType, expected_names: set[str], *, args: str = "") -> None:
    trainer = _make_trainer(optimizer_type=opt_type, optimizer_args=args)
    optimizer = trainer._create_optimizer()
    assert type(optimizer).__name__ in expected_names, (opt_type, type(optimizer).__name__, trainer._log_messages)
    assert optimizer.param_groups, opt_type


def test_config_aliases() -> None:
    cases = {
        "Automagic++": OptimizerType.AUTOMAGIC_PLUS_PLUS,
        "Automagic": OptimizerType.AUTOMAGIC_PLUS_PLUS,
        "AutoProdigy": OptimizerType.AUTO_PRODIGY,
        "PagedAdamW8bit": OptimizerType.PAGED_ADAMW_8BIT,
        "DAdaptAdamPreprint": OptimizerType.DADAPT_ADAM_PREPRINT,
        "AdamWScheduleFree": OptimizerType.ADAMW_SCHEDULE_FREE,
        "RAdamScheduleFree": OptimizerType.RADAM_SCHEDULE_FREE,
        "SGDScheduleFree": OptimizerType.SGD_SCHEDULE_FREE,
        "prodigyplus.ProdigyPlusScheduleFree": OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE,
        "KahanAdamW8bit": OptimizerType.KAHAN_ADAMW_8BIT,
        "AnimaFactoredAdamW": OptimizerType.ANIMA_FACTORED_ADAMW,
    }
    for raw, expected in cases.items():
        cfg = ConfigAdapter.from_frontend_dict({"schema_id": "sdxl-lora", "optimizer_type": raw})
        assert cfg.optimizer == expected, (raw, cfg.optimizer, expected)

    plugin_cfg = ConfigAdapter.from_frontend_dict(
        {"schema_id": "sdxl-lora", "optimizer_type": "pytorch_optimizer.StableAdamW"}
    )
    assert plugin_cfg.optimizer == OptimizerType.PYTORCH_OPTIMIZER
    assert "name=StableAdamW" in plugin_cfg.optimizer_args

    generic_cfg = ConfigAdapter.from_frontend_dict(
        {"schema_id": "sdxl-lora", "optimizer_type": "bitsandbytes.optim.PagedAdEMAMix8bit"}
    )
    assert generic_cfg.optimizer == OptimizerType.GENERIC
    assert "name=bitsandbytes.optim.PagedAdEMAMix8bit" in generic_cfg.optimizer_args


def test_native_and_safe_fallback_optimizers() -> None:
    _assert_optimizer(OptimizerType.ADAMW, {"AdamW"})
    _assert_optimizer(OptimizerType.ADAMW_8BIT, {"AdamW", "AdamW8bit"})
    _assert_optimizer(OptimizerType.PAGED_ADAMW, {"AdamW", "PagedAdamW"})
    _assert_optimizer(OptimizerType.PAGED_ADAMW_32BIT, {"AdamW", "PagedAdamW32bit"})
    _assert_optimizer(OptimizerType.PAGED_ADAMW_8BIT, {"AdamW", "PagedAdamW8bit"})
    _assert_optimizer(OptimizerType.PAGED_LION_8BIT, {"AdamW", "PagedLion8bit"})
    _assert_optimizer(OptimizerType.SGD_NESTEROV_8BIT, {"AdamW", "SGD8bit"})
    _assert_optimizer(OptimizerType.PRODIGY, {"AdamW", "Prodigy"}, args="d0=1e-6,d_coef=1.5")
    _assert_optimizer(OptimizerType.ADAFACTOR, {"AdamW", "Adafactor"})
    _assert_optimizer(OptimizerType.LION, {"AdamW", "Lion"})
    _assert_optimizer(OptimizerType.LION_8BIT, {"AdamW", "AdamW8bit", "Lion", "Lion8bit"})
    _assert_optimizer(OptimizerType.SGD_NESTEROV, {"SGD"})
    _assert_optimizer(OptimizerType.DADAPTATION, {"AdamW", "DAdaptAdam"})
    _assert_optimizer(OptimizerType.DADAPT_ADAM_PREPRINT, {"AdamW", "DAdaptAdam", "DAdaptAdamPreprint"})
    _assert_optimizer(OptimizerType.DADAPT_ADAGRAD, {"AdamW", "DAdaptAdaGrad"})
    _assert_optimizer(OptimizerType.DADAPT_ADAM, {"AdamW", "DAdaptAdam"})
    _assert_optimizer(OptimizerType.DADAPT_ADAN, {"AdamW", "DAdaptAdan"})
    _assert_optimizer(OptimizerType.DADAPT_ADAN_IP, {"AdamW", "DAdaptAdan", "DAdaptAdanIP"})
    _assert_optimizer(OptimizerType.DADAPT_LION, {"AdamW", "DAdaptLion"})
    _assert_optimizer(OptimizerType.DADAPT_SGD, {"AdamW", "DAdaptSGD"})
    _assert_optimizer(OptimizerType.ADAMW_SCHEDULE_FREE, {"AdamW", "AdamWScheduleFree"})
    _assert_optimizer(OptimizerType.RADAM_SCHEDULE_FREE, {"AdamW", "RAdamScheduleFree"})
    _assert_optimizer(OptimizerType.SGD_SCHEDULE_FREE, {"AdamW", "SGDScheduleFree"})
    _assert_optimizer(OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE, {"AdamW", "ProdigyPlusScheduleFree"})
    _assert_optimizer(OptimizerType.KAHAN_ADAMW_8BIT, {"KahanAdamW8bit"})
    _assert_optimizer(OptimizerType.AUTOMAGIC_PLUS_PLUS, {"AutomagicPlusPlus"})
    _assert_optimizer(OptimizerType.AUTO_PRODIGY, {"AutoProdigy"})
    _assert_optimizer(OptimizerType.ANIMA_FACTORED_ADAMW, {"AnimaFactoredAdamW"})


def test_generic_torch_optimizer_resolution() -> None:
    _assert_optimizer(OptimizerType.GENERIC, {"Adam"}, args="name=Adam,betas=(0.8,0.9)")
    _assert_optimizer(OptimizerType.GENERIC, {"SGD"}, args="name=SGD,momentum=0.9")


def test_optimizer_capability_report_covers_matrix() -> None:
    report = optimizer_capability_report()
    summary = report["summary"]
    assert summary["missing_capability_mappings"] == [], summary
    seen = {str(item["optimizer_type"]) for item in report["optimizers"]}
    for optimizer in OptimizerType:
        assert optimizer.value in seen, (optimizer, summary)


def main() -> int:
    test_config_aliases()
    test_native_and_safe_fallback_optimizers()
    test_generic_torch_optimizer_resolution()
    test_optimizer_capability_report_covers_matrix()
    print("Expanded optimizer matrix smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
