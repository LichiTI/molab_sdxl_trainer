# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0-0
"""Smoke-test: trainer runtime consumes seed for process RNG state.

This complements ``newbie_seed_config_smoke.py`` by proving the seed field is
not only preserved by config parsing, but also applied by the trainer before a
run begins.
"""

from __future__ import annotations

import random
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch


def _install_xformers_stub() -> None:
    sys.modules.pop("xformers", None)
    sys.modules.pop("xformers.ops", None)

    xformers_module = ModuleType("xformers")
    ops_module = ModuleType("xformers.ops")

    def _unavailable(*_: object, **__: object) -> object:
        raise RuntimeError("xFormers is unavailable in the Newbie seed runtime smoke")

    ops_module.memory_efficient_attention = _unavailable  # type: ignore[attr-defined]
    ops_module.__spec__ = ModuleSpec("xformers.ops", loader=None)
    xformers_module.ops = ops_module  # type: ignore[attr-defined]
    xformers_module.__spec__ = ModuleSpec("xformers", loader=None)
    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = ops_module


_install_xformers_stub()

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.configs import UnifiedTrainingConfig
from core.lulynx_trainer.trainer import LulynxTrainer


def _capture_rng_snapshot() -> tuple[float, float, float]:
    return (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
    )


def _run_once(seed: int) -> tuple[tuple[float, float, float], list[str]]:
    cfg = UnifiedTrainingConfig(
        model_type="newbie",
        pretrained_model_name_or_path="H:/tmp/newbie-placeholder",
        train_data_dir="H:/tmp/lulynx-newbie-data",
        output_dir="H:/tmp/lulynx-newbie-seed-runtime",
        output_name="newbie_seed_runtime",
        seed=seed,
    )
    trainer = LulynxTrainer(cfg)
    logs: list[str] = []
    trainer.set_callbacks(on_log=logs.append)
    captured: dict[str, tuple[float, float, float]] = {}

    def _fake_run_training(self: LulynxTrainer) -> None:
        captured["snapshot"] = _capture_rng_snapshot()

    with patch.object(LulynxTrainer, "prepare", autospec=True, return_value=None):
        with patch.object(LulynxTrainer, "_apply_gpu_power_limit_if_requested", autospec=True, return_value=None):
            with patch.object(LulynxTrainer, "_initialize_logging_runtime", autospec=True, return_value=None):
                with patch.object(LulynxTrainer, "_finalize_logging_runtime", autospec=True, return_value=None):
                    with patch.object(LulynxTrainer, "_run_training", autospec=True, side_effect=_fake_run_training):
                        ok = trainer.start()
    if not ok:
        raise AssertionError(f"Trainer.start() returned False for seed={seed}")
    snapshot = captured.get("snapshot")
    if snapshot is None:
        raise AssertionError(f"Missing RNG snapshot for seed={seed}")
    return snapshot, logs


def _test_same_seed_is_deterministic() -> None:
    a, logs_a = _run_once(42)
    b, logs_b = _run_once(42)
    assert a == b, f"Expected identical RNG snapshots for same seed, got {a} vs {b}"
    assert any("Seed applied: 42" in line for line in logs_a), logs_a
    assert any("Seed applied: 42" in line for line in logs_b), logs_b


def _test_zero_seed_is_consumed() -> None:
    snap, logs = _run_once(0)
    assert isinstance(snap, tuple) and len(snap) == 3, snap
    assert any("Seed applied: 0" in line for line in logs), logs


def _test_negative_seed_skips_override() -> None:
    random.seed(777)
    np.random.seed(777)
    torch.manual_seed(777)
    expected = _capture_rng_snapshot()

    random.seed(777)
    np.random.seed(777)
    torch.manual_seed(777)
    actual, logs = _run_once(-1)

    assert actual == expected, f"Negative seed should not override RNG state: {actual} vs {expected}"
    assert any("negative sentinel requested" in line for line in logs), logs


def main() -> int:
    tests = [
        ("same_seed_is_deterministic", _test_same_seed_is_deterministic),
        ("zero_seed_is_consumed", _test_zero_seed_is_consumed),
        ("negative_seed_skips_override", _test_negative_seed_skips_override),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        print("FAIL -- trainer runtime seed consumption NOT fully proved")
        return 1
    print("PASS -- trainer runtime consumes seed and honors -1 sentinel without overriding RNG state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
