# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test initial epoch/step controls without model loading."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.training_loop import TrainingLoop


class _Optimizer:
    param_groups = [{"lr": 0.0}]

    def step(self) -> None:
        pass

    def zero_grad(self, *args: object, **kwargs: object) -> None:
        pass


def main() -> int:
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "model_type": "sdxl",
            "initial_epoch": 2,
            "initial_step": 2,
            "skip_until_initial_step": True,
        }
    )
    assert cfg.initial_epoch == 2
    assert cfg.initial_step == 2
    assert cfg.skip_until_initial_step is True

    loop = TrainingLoop.__new__(TrainingLoop)
    loop.current_epoch = 0
    loop.global_step = 0
    loop.total_steps = 0
    loop.completed_by_step_limit = False
    loop.gradient_accumulation_steps = 1
    loop.initial_step_target = 2
    loop.skip_until_initial_step = True
    loop._should_stop = False
    loop.safeguard = None
    loop.optimizer = _Optimizer()
    loop.lr_scheduler = None
    loop.te_manager = None
    loop.text_encoder_1 = None
    loop.auditor = None
    loop.auditor_interval = 50
    loop._last_loss = 0.0
    loop._maybe_save_safe_state = lambda: None
    loop._get_trainable_params = lambda: []
    trained = {"count": 0}

    def _train_step(_batch: dict, accumulation_steps: int | None = None) -> float:
        trained["count"] += 1
        return 1.0

    loop.train_step = _train_step
    skipped_events: list[dict] = []
    loop.on_step_end = lambda _step, _loss, info: skipped_events.append(info)
    loop.on_epoch_end = None

    result = TrainingLoop.train_epoch(loop, [{"i": 0}, {"i": 1}, {"i": 2}], 2)
    assert loop.global_step == 3
    assert trained["count"] == 1
    assert result["steps"] == 1
    assert len([event for event in skipped_events if event.get("skipped")]) == 2

    print("Training start offset smoke passed: initial step skipping is wired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
