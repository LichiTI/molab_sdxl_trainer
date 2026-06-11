"""Smoke test for AutoController: automatic training parameter tuning."""
from __future__ import annotations

import sys
import os
import importlib.util
from pathlib import Path
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))

# Load auto_controller via importlib from core/training_components/auto_controller.py
_ac_path = os.path.join(_HERE, "..", "..", "core", "training_components", "auto_controller.py")
_ac_spec = importlib.util.spec_from_file_location(
    "core.training_components.auto_controller",
    _ac_path,
)
_ac_mod = importlib.util.module_from_spec(_ac_spec)
sys.modules["core.training_components.auto_controller"] = _ac_mod
_ac_spec.loader.exec_module(_ac_mod)

AutoController = _ac_mod.AutoController
AutoControlConfig = _ac_mod.AutoControlConfig
AutoEvent = _ac_mod.AutoEvent
MetricsTracker = _ac_mod.MetricsTracker

import torch
import torch.nn as nn

_BACKEND_ROOT = Path(_HERE).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.lulynx_trainer.trainer import LulynxTrainer


def test_auto_controller_has_update_and_step():
    """AutoController has update and step methods."""
    cfg = AutoControlConfig(warmup_steps=0)
    ctrl = AutoController(cfg)
    assert hasattr(ctrl, "step"), "AutoController missing step() method"
    assert hasattr(ctrl, "metrics"), "AutoController missing metrics attribute"
    assert hasattr(ctrl.metrics, "update"), "MetricsTracker missing update() method"


def test_lr_decay_on_plateau():
    """Learning rate is reduced when gradient_rank plateaus."""
    cfg = AutoControlConfig(
        warmup_steps=0,
        smart_lr_decay=True,
        gradient_rank_plateau_window=5,
        lr_decay_factor=0.5,
        max_decays=3,
        auto_freeze_te=False,
        smart_early_stop=False,
    )
    ctrl = AutoController(cfg)

    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initial_lr = optimizer.param_groups[0]["lr"]

    # Feed a plateau (same gradient_rank for many steps)
    for step in range(20):
        ctrl.step(
            step=step,
            metrics={"gradient_rank": 5.0, "loss": 0.1},
            optimizer=optimizer,
        )

    new_lr = optimizer.param_groups[0]["lr"]
    assert new_lr < initial_lr, (
        f"LR should have decayed on plateau: {initial_lr} -> {new_lr}"
    )


def test_lr_decay_event_emitted():
    """LR decay triggers an AutoEvent.LR_DECAY event."""
    cfg = AutoControlConfig(
        warmup_steps=0,
        smart_lr_decay=True,
        gradient_rank_plateau_window=5,
        lr_decay_factor=0.5,
        max_decays=3,
        auto_freeze_te=False,
        smart_early_stop=False,
    )
    ctrl = AutoController(cfg)

    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    events_seen = []
    ctrl.register_callback(lambda event, data: events_seen.append(event))

    for step in range(20):
        ctrl.step(
            step=step,
            metrics={"gradient_rank": 5.0},
            optimizer=optimizer,
        )

    assert AutoEvent.LR_DECAY in events_seen, (
        f"Expected LR_DECAY event, got: {[e.value for e in events_seen]}"
    )


def test_warmup_blocks_actions():
    """During warmup, no auto-control actions are taken."""
    cfg = AutoControlConfig(warmup_steps=10, smart_lr_decay=True)
    ctrl = AutoController(cfg)

    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initial_lr = optimizer.param_groups[0]["lr"]

    result = ctrl.step(step=5, metrics={"gradient_rank": 5.0}, optimizer=optimizer)
    assert result.get("warmup") is True, "Should be in warmup at step 5"
    # LR should NOT have changed during warmup
    assert optimizer.param_groups[0]["lr"] == initial_lr, "LR changed during warmup"


def test_metrics_tracker_plateau_detection():
    """MetricsTracker.is_plateau detects flat loss sequences."""
    tracker = MetricsTracker()
    # Feed a flat loss series
    for i in range(30):
        tracker.update(loss=0.5)
    assert tracker.is_plateau("loss", window=20, threshold=0.001), (
        "Flat loss should be detected as plateau"
    )

    # Feed a decreasing loss series
    tracker2 = MetricsTracker()
    for i in range(30):
        tracker2.update(loss=1.0 - i * 0.05)
    assert not tracker2.is_plateau("loss", window=20, threshold=0.001), (
        "Decreasing loss should NOT be detected as plateau"
    )


class _FakeAuditor:
    def __init__(self, metrics):
        self.metrics = dict(metrics)

    def get_last_report(self):
        return {"metrics": dict(self.metrics)}


class _TinySDXLModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unet = nn.Module()
        self.unet.text_encoder = nn.Linear(2, 2)


def _make_minimal_trainer(controller: AutoController, optimizer: torch.optim.Optimizer, auditor: _FakeAuditor):
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer._auto_controller = controller
    trainer.training_loop = SimpleNamespace(
        auditor=auditor,
        optimizer=optimizer,
        _manifold_tracker=None,
    )
    trainer.model = _TinySDXLModel()
    trainer.config = SimpleNamespace(epochs=1, semantic_tuner_enabled=False)
    trainer._dataset = None
    trainer._native_unet_status = None
    trainer._sdxl_lora_low_vram_profile = None
    trainer._data_backend_profile = {}
    trainer._last_vram_status = "ok"
    trainer._ema_tracker = None
    trainer._resource_manager = None
    trainer._dynamic_pruner = None
    trainer._should_stop = False
    trainer._ti_mode = False
    trainer.te_manager = None
    trainer._sampler = None
    trainer._coreset_manager = None
    trainer.on_progress = None
    trainer.on_step = None
    trainer._emit_runtime_event = lambda *_args, **_kwargs: None
    trainer._check_hot_swap = lambda *_args, **_kwargs: None
    trainer._should_emit_step_logging = lambda *_args, **_kwargs: False
    trainer._record_step_logging_overhead = lambda *_args, **_kwargs: None
    trainer._get_current_adapter_state_dict = lambda *_args, **_kwargs: {}
    trainer._log = lambda *_args, **_kwargs: None
    return trainer


def test_sdxl_trainer_step_end_auto_controller_lr_decay():
    """SDXL trainer step-end hook feeds auditor metrics into AutoController."""
    cfg = AutoControlConfig(
        warmup_steps=0,
        smart_lr_decay=True,
        gradient_rank_plateau_window=3,
        lr_decay_factor=0.5,
        max_decays=1,
        auto_freeze_te=False,
        smart_early_stop=False,
    )
    ctrl = AutoController(cfg)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = _make_minimal_trainer(ctrl, optimizer, _FakeAuditor({"gradient_rank": 3.0, "gsnr": 3.0}))

    for step in range(4):
        LulynxTrainer._on_step_end(trainer, step=step, loss=0.2, info={"lr": optimizer.param_groups[0]["lr"]})

    assert optimizer.param_groups[0]["lr"] == 5e-4
    assert ctrl.get_status()["lr_decay_count"] == 1


def test_sdxl_trainer_step_end_auto_controller_early_stop():
    """SDXL trainer step-end hook propagates AutoController early-stop decisions."""
    cfg = AutoControlConfig(
        warmup_steps=0,
        smart_lr_decay=False,
        smart_early_stop=True,
        stable_rank_collapse_threshold=0.5,
        stable_rank_consecutive=2,
        auto_freeze_te=False,
    )
    ctrl = AutoController(cfg)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = _make_minimal_trainer(ctrl, optimizer, _FakeAuditor({"stable_rank": 10.0}))

    LulynxTrainer._on_step_end(trainer, step=0, loss=0.2, info={"lr": 1e-3})
    trainer.training_loop.auditor.metrics["stable_rank"] = 1.0
    LulynxTrainer._on_step_end(trainer, step=1, loss=0.2, info={"lr": 1e-3})
    LulynxTrainer._on_step_end(trainer, step=2, loss=0.2, info={"lr": 1e-3})

    assert trainer._should_stop is True
    assert ctrl.should_stop is True


if __name__ == "__main__":
    print("AutoController Smoke Tests")
    print("=" * 40)
    test_auto_controller_has_update_and_step()
    print("PASS: auto_controller_has_update_and_step")
    test_lr_decay_on_plateau()
    print("PASS: lr_decay_on_plateau")
    test_lr_decay_event_emitted()
    print("PASS: lr_decay_event_emitted")
    test_warmup_blocks_actions()
    print("PASS: warmup_blocks_actions")
    test_metrics_tracker_plateau_detection()
    print("PASS: metrics_tracker_plateau_detection")
    test_sdxl_trainer_step_end_auto_controller_lr_decay()
    print("PASS: sdxl_trainer_step_end_auto_controller_lr_decay")
    test_sdxl_trainer_step_end_auto_controller_early_stop()
    print("PASS: sdxl_trainer_step_end_auto_controller_early_stop")
    print("=" * 40)
    print("All AutoController smoke tests passed!")
