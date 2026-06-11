# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test data transfer profiling modes without starting a trainer."""

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
training_loop_mod = _load_module("core.lulynx_trainer.training_loop", TRAINER_ROOT / "training_loop.py")

TrainingLoop = training_loop_mod.TrainingLoop


def _make_loop(mode: str = "event", enabled: bool = True):
    loop = TrainingLoop.__new__(TrainingLoop)
    loop.device = "cpu"
    loop.data_transfer_non_blocking = True
    loop.data_transfer_profile_enabled = enabled
    loop.data_transfer_profile_mode = TrainingLoop._normalize_data_transfer_profile_mode(mode)
    loop.data_transfer_profile_window = 1
    loop._transfer_profile_steps = 0
    loop._transfer_profile_step_seconds = 0.0
    loop._transfer_profile_seconds = 0.0
    loop._transfer_profile_bytes = 0
    loop._transfer_profile_ops = 0
    loop._transfer_profile_by_label = {}
    loop._transfer_profile_pending_events = []
    loop._last_transfer_profile_snapshot = None
    return loop


def _test_mode_normalization() -> None:
    assert TrainingLoop._normalize_data_transfer_profile_mode("cuda_event") == "event"
    assert TrainingLoop._normalize_data_transfer_profile_mode("legacy") == "sync"
    assert TrainingLoop._normalize_data_transfer_profile_mode("disabled") == "off"
    assert TrainingLoop._normalize_data_transfer_profile_mode("unknown") == "event"


def _test_cpu_profile_sample() -> None:
    loop = _make_loop("event")
    source = torch.ones(2, 3)
    moved = loop._profiled_to(source, label="cpu_tensor", device="cpu", dtype=torch.float32)
    assert moved.shape == source.shape
    assert loop._transfer_profile_ops == 1
    assert loop._transfer_profile_bytes == source.numel() * source.element_size()
    assert "cpu_tensor" in loop._transfer_profile_by_label
    loop._record_transfer_profile_step(0.1)
    assert loop._transfer_profile_ops == 0
    assert loop._transfer_profile_by_label == {}


def _test_profile_off() -> None:
    loop = _make_loop("off")
    source = torch.ones(1)
    loop._profiled_to(source, label="off", device="cpu")
    assert loop._transfer_profile_ops == 0
    assert loop._transfer_profile_bytes == 0


def _test_cuda_event_path_if_available() -> None:
    if not torch.cuda.is_available():
        return
    loop = _make_loop("event")
    loop.device = "cuda"
    source = torch.ones(8, pin_memory=True)
    moved = loop._profiled_to(source, label="cuda_tensor", device="cuda")
    assert moved.is_cuda
    assert len(loop._transfer_profile_pending_events) == 1
    snapshot = loop._record_transfer_profile_step(0.1)
    assert snapshot is not None
    assert snapshot["ops"] == 1
    assert loop._transfer_profile_pending_events == []


def main() -> int:
    _test_mode_normalization()
    print("  data transfer profile mode normalization -- PASS")
    _test_cpu_profile_sample()
    print("  data transfer profile CPU sample/window reset -- PASS")
    _test_profile_off()
    print("  data transfer profile off mode -- PASS")
    _test_cuda_event_path_if_available()
    print("  data transfer profile CUDA event path -- PASS/SKIP")
    print("Data transfer profile smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
