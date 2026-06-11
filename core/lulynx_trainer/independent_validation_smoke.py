# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke checks for independent validation dataset config wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.training_loop import TrainingLoop


def test_validation_path_aliases() -> None:
    parsed = ConfigAdapter.from_frontend_dict(
        {
            "model_type": "sdxl",
            "validation_data_dir": "H:/tmp/lulynx_validation",
            "validation_split": 0.2,
            "validate_every_n_steps": "25",
            "validate_every_n_epochs": "2",
            "eval_batch_size": "3",
            "max_validation_steps": "4",
        }
    )
    assert parsed.eval_data_dir == "H:/tmp/lulynx_validation"
    assert parsed.validation_split == 0.2
    assert parsed.eval_every_n_steps == 25
    assert parsed.eval_every_n_epochs == 2
    assert parsed.eval_batch_size == 3
    assert parsed.max_validation_steps == 4


def test_validate_epoch_respects_max_steps() -> None:
    loop = TrainingLoop.__new__(TrainingLoop)
    loop.unet = torch.nn.Linear(1, 1)
    loop.text_encoder_1 = None
    loop.text_encoder_2 = None
    loop._train_text_encoder_1 = False
    loop._train_text_encoder_2 = False
    loop.max_validation_steps = 2
    calls = {"count": 0}

    def validation_step(_batch) -> float:
        calls["count"] += 1
        return float(calls["count"])

    loop.validation_step = validation_step
    result = loop.validate_epoch([object(), object(), object()], epoch=0)
    assert result["steps"] == 2
    assert result["avg_loss"] == 1.5
    assert calls["count"] == 2


def main() -> int:
    test_validation_path_aliases()
    test_validate_epoch_respects_max_steps()
    print("Independent validation smoke passed: path aliases and max steps are wired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
