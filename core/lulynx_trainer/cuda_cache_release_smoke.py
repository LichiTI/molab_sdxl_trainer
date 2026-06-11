# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test CUDA cache release strategy normalization and gating."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

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
_load_module("core.lulynx_trainer.safe_guard", TRAINER_ROOT / "safe_guard.py")
_load_module("core.lulynx_trainer.model_family", TRAINER_ROOT / "model_family.py")
TrainingLoop = _load_module(
    "core.lulynx_trainer.training_loop",
    TRAINER_ROOT / "training_loop.py",
).TrainingLoop


def _build_loop(strategy: str) -> TrainingLoop:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=1e-3)
    return TrainingLoop(
        unet=nn.Identity(),
        text_encoder_1=nn.Identity(),
        text_encoder_2=None,
        vae=nn.Identity(),
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)),
        lora_injector=SimpleNamespace(get_trainable_params=lambda: []),
        optimizer=optimizer,
        lr_scheduler=None,
        device="cpu",
        dtype=torch.float32,
        cuda_cache_release_strategy=strategy,
        cuda_cache_release_interval=1,
    )


def _patch_cuda(loop: TrainingLoop):
    return patch.multiple(
        "torch.cuda",
        is_available=lambda: True,
        empty_cache=lambda: None,
    ), patch.object(loop, "_cuda_memory_snapshot", side_effect=[{"reserved_mb": 8.0, "allocated_mb": 4.0}, {"reserved_mb": 2.0, "allocated_mb": 4.0}] * 8)


def test_strategy_alias_normalization() -> None:
    loop = _build_loop("every_step")
    assert loop._cuda_cache_release_strategy == "aggressive", loop._cuda_cache_release_strategy


def test_phase_boundary_dedupes_per_step() -> None:
    loop = _build_loop("phase_boundary")
    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.empty_cache") as empty_mock, patch.object(
        loop,
        "_cuda_memory_snapshot",
        side_effect=[{"reserved_mb": 8.0, "allocated_mb": 4.0}, {"reserved_mb": 2.0, "allocated_mb": 4.0}] * 8,
    ):
        first = loop._maybe_release_cuda_cache("phase_boundary", 1)
        second = loop._maybe_release_cuda_cache("phase_boundary", 1)
        third = loop._maybe_release_cuda_cache("phase_boundary", 2)
    assert first.get("ok") is True, first
    assert second == {}, second
    assert third.get("ok") is True, third
    assert empty_mock.call_count == 2, empty_mock.call_count


def test_after_optimizer_scope_and_forced_oom_release() -> None:
    loop = _build_loop("after_optimizer")
    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.empty_cache") as empty_mock, patch.object(
        loop,
        "_cuda_memory_snapshot",
        side_effect=[{"reserved_mb": 8.0, "allocated_mb": 4.0}, {"reserved_mb": 2.0, "allocated_mb": 4.0}] * 8,
    ):
        ignored = loop._maybe_release_cuda_cache("phase_boundary", 1)
        released = loop._maybe_release_cuda_cache("after_optimizer", 1)
    assert ignored == {}, ignored
    assert released.get("ok") is True, released
    assert empty_mock.call_count == 1, empty_mock.call_count

    off_loop = _build_loop("off")
    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.empty_cache") as empty_mock, patch.object(
        off_loop,
        "_cuda_memory_snapshot",
        side_effect=[{"reserved_mb": 8.0, "allocated_mb": 4.0}, {"reserved_mb": 2.0, "allocated_mb": 4.0}] * 8,
    ):
        forced = off_loop._maybe_release_cuda_cache("oom_recovery", 3, force=True)
    assert forced.get("ok") is True, forced
    assert forced.get("forced") is True, forced
    assert empty_mock.call_count == 1, empty_mock.call_count


def main() -> int:
    test_strategy_alias_normalization()
    print("[PASS] CUDA cache strategy alias normalization")
    test_phase_boundary_dedupes_per_step()
    print("[PASS] CUDA cache phase-boundary dedupe")
    test_after_optimizer_scope_and_forced_oom_release()
    print("[PASS] CUDA cache after-optimizer / forced OOM release")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
