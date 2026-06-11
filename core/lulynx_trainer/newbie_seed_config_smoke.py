# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0-0
"""Smoke-test: Newbie seed field round-trips through config pipeline.

Proves that the `seed` field survives ConfigAdapter normalization and
is accessible as a UnifiedTrainingConfig attribute. Documents that the
trainer does NOT currently consume the seed (no `torch.manual_seed`
call on startup), so this is config closure only, not determinism closure.
"""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import torch


def _install_xformers_stub() -> None:
    sys.modules.pop("xformers", None)
    sys.modules.pop("xformers.ops", None)

    xformers_module = ModuleType("xformers")
    ops_module = ModuleType("xformers.ops")

    def _unavailable(*_: object, **__: object) -> object:
        raise RuntimeError("xFormers is unavailable in the Newbie seed smoke")

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
from core.lulynx_trainer.config_adapter import ConfigAdapter


def _test_seed_survives_config_adapter() -> None:
    """ConfigAdapter.from_frontend_dict preserves explicit seed value."""
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "newbie-lora",
            "pretrained_model_name_or_path": "H:/tmp/newbie-placeholder",
            "train_data_dir": "H:/tmp/lulynx-newbie-data",
            "seed": 42,
        }
    )
    assert cfg.seed == 42, f"Expected seed=42, got {cfg.seed}"


def _test_seed_default_value() -> None:
    """ConfigAdapter uses default seed when not specified."""
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "newbie-lora",
            "pretrained_model_name_or_path": "H:/tmp/newbie-placeholder",
            "train_data_dir": "H:/tmp/lulynx-newbie-data",
        }
    )
    assert cfg.seed == 1337, f"Expected default seed=1337, got {cfg.seed}"


def _test_seed_minus_one_round_trip() -> None:
    """ConfigAdapter preserves seed=-1 (random seed sentinel)."""
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "newbie-lora",
            "pretrained_model_name_or_path": "H:/tmp/newbie-placeholder",
            "train_data_dir": "H:/tmp/lulynx-newbie-data",
            "seed": -1,
        }
    )
    assert cfg.seed == -1, f"Expected seed=-1, got {cfg.seed}"


def _test_seed_is_int() -> None:
    """ConfigAdapter coerces seed to int."""
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "newbie-lora",
            "pretrained_model_name_or_path": "H:/tmp/newbie-placeholder",
            "train_data_dir": "H:/tmp/lulynx-newbie-data",
            "seed": "99",
        }
    )
    assert isinstance(cfg.seed, int), f"Expected seed to be int, got {type(cfg.seed)}"
    assert cfg.seed == 99, f"Expected seed=99, got {cfg.seed}"


def _test_seed_preserves_zero() -> None:
    """ConfigAdapter preserves seed=0 (deterministic zero seed)."""
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "newbie-lora",
            "pretrained_model_name_or_path": "H:/tmp/newbie-placeholder",
            "train_data_dir": "H:/tmp/lulynx-newbie-data",
            "seed": 0,
        }
    )
    assert cfg.seed == 0, f"Expected seed=0, got {cfg.seed}"


def main() -> int:
    tests = [
        ("seed_survives_config_adapter", _test_seed_survives_config_adapter),
        ("seed_default_value", _test_seed_default_value),
        ("seed_minus_one_round_trip", _test_seed_minus_one_round_trip),
        ("seed_is_int", _test_seed_is_int),
        ("seed_preserves_zero", _test_seed_preserves_zero),
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
        print("FAIL -- seed config round-trip NOT fully proved")
        return 1
    print(
        "PASS -- seed config round-trips through ConfigAdapter. "
        "NOTE: trainer does NOT consume seed for determinism; "
        "this is config-closure only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
