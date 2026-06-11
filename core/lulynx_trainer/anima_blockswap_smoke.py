# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test Anima/native DiT block discovery for BlockSwap."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
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
_load_module("core.lulynx_trainer.memory_optimizations", TRAINER_ROOT / "memory_optimizations.py")
TrainingLoop = _load_module(
    "core.lulynx_trainer.training_loop",
    TRAINER_ROOT / "training_loop.py",
).TrainingLoop


class _TinyAnimaBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)


class _NetBlocksAnimaRoute(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Module()
        self.net.blocks = nn.ModuleList([_TinyAnimaBlock(), _TinyAnimaBlock()])


class _BlockModulesAnimaRoute(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._blocks = nn.ModuleList([_TinyAnimaBlock(), _TinyAnimaBlock()])

    def _block_modules(self):
        return list(self._blocks)


def _build_loop(unet: nn.Module) -> TrainingLoop:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=1e-3)
    return TrainingLoop(
        unet=unet,
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
        blocks_to_swap=1,
        model_arch="anima",
    )


def _run_case(case_name: str, unet: nn.Module) -> dict[str, object]:
    captured: dict[str, object] = {"case": case_name}

    class _FakeOffloader:
        def __init__(self, *, blocks, blocks_to_swap, device, enable_backward=True, should_swap=None, strategy="auto", **kwargs):
            block_list = list(blocks)
            captured["block_count"] = len(block_list)
            captured["blocks_to_swap"] = blocks_to_swap
            captured["device"] = str(device)
            captured["enable_backward"] = enable_backward
            captured["strategy"] = strategy
            captured["should_swap_result"] = (
                should_swap(SimpleNamespace(_lora_leaf=False), "probe") if should_swap else None
            )
            self._captured = captured

        def prepare_before_forward(self):
            self._captured["prepared"] = True

        def install_forward_hooks(self, model):
            self._captured["hooks_installed"] = True

        def strategy_state(self):
            return {
                "block_swap_strategy": "sync",
                "requested_block_swap_strategy": self._captured["strategy"],
                "block_swap_strategy_fallback_reason": "smoke",
            }

    with patch("core.lulynx_trainer.memory_optimizations.BlockSwapOffloader", _FakeOffloader):
        loop = _build_loop(unet)

    assert loop._block_offloader is not None, f"BlockSwapOffloader should initialize for {case_name}"
    assert captured.get("block_count") == 2, captured
    assert captured.get("blocks_to_swap") == 1, captured
    assert captured.get("prepared") is True, captured
    assert captured.get("hooks_installed") is True, captured
    assert captured.get("should_swap_result") is True, captured
    return captured


def main() -> int:
    net_blocks_case = _run_case("net.blocks", _NetBlocksAnimaRoute())
    block_modules_case = _run_case("_block_modules", _BlockModulesAnimaRoute())

    assert net_blocks_case["device"] == "cpu", net_blocks_case
    assert block_modules_case["device"] == "cpu", block_modules_case

    print(
        "Anima blockswap smoke passed: "
        f"net.blocks={net_blocks_case['block_count']} and "
        f"_block_modules={block_modules_case['block_count']} "
        "reach BlockSwapOffloader.prepare_before_forward() on CPU"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
