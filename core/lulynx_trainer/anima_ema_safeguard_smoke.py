# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test Anima EMA tracker and SafeGuard wiring.

Proves:
1. EMAStateTracker initializes with correct config and maintains shadow weights.
2. EMA shadow weights diverge from live weights after enough steps.
3. SafeGuard emits STOP on sustained NaN losses.
4. SafeGuard loss spike detection triggers REDUCE_LR.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = BACKEND_ROOT / "core"
TRAINER_ROOT = CORE_ROOT / "lulynx_trainer"


def _ensure_namespace(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name: str, path: Path):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ensure_namespace("core", CORE_ROOT)
_ensure_namespace("core.lulynx_trainer", TRAINER_ROOT)
_load_module("core.constants", CORE_ROOT / "constants.py")
safe_guard_mod = _load_module("core.lulynx_trainer.safe_guard", TRAINER_ROOT / "safe_guard.py")
ema_mod = _load_module("core.lulynx_trainer.ema", TRAINER_ROOT / "ema.py")

EMAStateTracker = ema_mod.EMAStateTracker
SafeGuardAction = safe_guard_mod.SafeGuardAction
SafeGuardConfig = safe_guard_mod.SafeGuardConfig
TrainingSafeGuard = safe_guard_mod.TrainingSafeGuard


def _test_ema_tracker() -> None:
    """EMA shadow weights track live weights with exponential moving average."""
    live = {"weight": torch.ones(4, 4), "bias": torch.zeros(4)}

    tracker = EMAStateTracker(
        initial_state=live,
        decay=0.9,
        update_after_step=0,
        update_every=1,
        use_ema_warmup=False,
    )
    shadow = tracker.get_ema_state_dict()
    assert set(shadow.keys()) == set(live.keys()), "EMA shadow keys should match live keys"
    assert torch.allclose(shadow["weight"], live["weight"]), "EMA shadow should initialize to live weights"

    # Step with changed live weights; shadow should drift toward new values
    live["weight"] = torch.zeros(4, 4)
    tracker.step(live)
    shadow = tracker.get_ema_state_dict()
    # decay=0.9, no warmup: shadow = 0.9 * old + 0.1 * new
    expected = 0.9 * torch.ones(4, 4) + 0.1 * torch.zeros(4, 4)
    assert torch.allclose(shadow["weight"], expected, atol=1e-6), (
        f"EMA shadow drift incorrect: got {shadow['weight'][0,0].item():.4f}, expected 0.9"
    )

    # After many steps with live=0, shadow should approach 0
    for _ in range(100):
        tracker.step(live)
    shadow = tracker.get_ema_state_dict()
    assert shadow["weight"].abs().max() < 0.01, (
        f"EMA shadow should converge to live=0 after 100 steps, max={shadow['weight'].abs().max():.6f}"
    )

    # update_after_step gate: decay=0.0 means shadow copies live (not EMA blending)
    live2 = {"w": torch.zeros(2, 2)}
    tracker2 = EMAStateTracker(
        initial_state=live2,
        decay=0.9,
        update_after_step=5,
        update_every=1,
        use_ema_warmup=False,
    )
    live2["w"] = torch.ones(2, 2) * 99.0
    for _ in range(3):
        tracker2.step(live2)
    shadow2 = tracker2.get_ema_state_dict()
    # With update_after_step=5, get_decay returns 0.0 for step<=5.
    # When decay=0.0, step() copies live to shadow directly.
    # So shadow should track live exactly, not stay at zeros.
    assert torch.allclose(shadow2["w"], torch.ones(2, 2) * 99.0), (
        "EMA with decay=0.0 should copy live weights to shadow"
    )

    # After update_after_step, EMA blending should kick in
    live2["w"] = torch.zeros(2, 2)
    for _ in range(50):
        tracker2.step(live2)
    shadow2 = tracker2.get_ema_state_dict()
    # Now decay > 0, blending with live=0 should converge toward 0
    assert shadow2["w"].abs().max() < 5.0, (
        f"EMA should converge toward live=0 after enough steps, max={shadow2['w'].abs().max():.4f}"
    )


def _test_safeguard_nan_stop() -> None:
    """SafeGuard emits STOP after sustained NaN losses."""
    config = SafeGuardConfig(
        enable_loss_spike_detection=False,
        enable_nan_detection=True,
        nan_check_interval=1,
        max_nan_count=3,
        enable_lr_deadlock_detection=False,
        enable_auto_recovery=False,
        enable_bad_sample_culling=False,
    )
    sg = TrainingSafeGuard(config)

    for i in range(5):
        action = sg.check(step=i, loss=1.0, lr=1e-4, gradients=None, filenames=None)
        assert action == SafeGuardAction.CONTINUE, f"Normal loss at step {i} should not trigger"

    got_stop = False
    for i in range(5, 10):
        action = sg.check(step=i, loss=float("nan"), lr=1e-4, gradients=None, filenames=None)
        if action == SafeGuardAction.STOP:
            got_stop = True
            break
    assert got_stop, "SafeGuard should emit STOP after 3 NaN losses"


def _test_safeguard_spike_detection() -> None:
    """SafeGuard emits REDUCE_LR on loss spike."""
    config = SafeGuardConfig(
        enable_loss_spike_detection=True,
        loss_spike_threshold=2.0,
        loss_window_size=15,
        enable_nan_detection=False,
        enable_lr_deadlock_detection=False,
        enable_auto_recovery=False,
        lr_reduction_factor=0.5,
        enable_bad_sample_culling=False,
    )
    sg = TrainingSafeGuard(config)

    for i in range(15):
        sg.check(step=i, loss=1.0, lr=1e-4, gradients=None, filenames=None)

    action = sg.check(step=15, loss=20.0, lr=1e-4, gradients=None, filenames=None)
    assert action == SafeGuardAction.REDUCE_LR, f"Expected REDUCE_LR on spike, got {action}"


def main() -> int:
    _test_ema_tracker()
    print("  EMA tracker: shadow weights, drift, convergence, update_after_step -- PASS")

    _test_safeguard_nan_stop()
    print("  SafeGuard NaN detection -> STOP -- PASS")

    _test_safeguard_spike_detection()
    print("  SafeGuard loss spike -> REDUCE_LR -- PASS")

    print("Anima EMA/SafeGuard smoke passed: EMA shadow tracking, NaN detection, spike detection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
