# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test Newbie/DiT block discovery for BlockSwap."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.training_loop import TrainingLoop


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])


class _TinyNewbieUnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = _TinyNet()


def main() -> int:
    captured: dict[str, object] = {}

    class _FakeOffloader:
        def __init__(self, *, blocks, blocks_to_swap, device, enable_backward=True, should_swap=None, strategy="auto", **kwargs):
            captured["block_count"] = len(list(blocks))
            captured["blocks_to_swap"] = blocks_to_swap
            captured["device"] = str(device)
            captured["enable_backward"] = enable_backward
            captured["strategy"] = strategy
            captured["should_swap_result"] = should_swap(SimpleNamespace(_lora_leaf=False), "probe") if should_swap else None
            self.prepared = False

        def prepare_before_forward(self):
            self.prepared = True
            captured["prepared"] = True

        def install_forward_hooks(self, model):
            captured["hooks_installed"] = True

        def strategy_state(self):
            return {
                "block_swap_strategy": "sync",
                "requested_block_swap_strategy": captured["strategy"],
                "block_swap_strategy_fallback_reason": "smoke",
            }

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=1e-3)

    with patch("core.lulynx_trainer.memory_optimizations.BlockSwapOffloader", _FakeOffloader):
        loop = TrainingLoop(
            unet=_TinyNewbieUnet(),
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
            blocks_to_swap=2,
            model_arch="newbie",
        )

    assert loop._block_offloader is not None, "BlockSwapOffloader should be initialized for Newbie blocks"
    assert captured.get("block_count") == 3, captured
    assert captured.get("blocks_to_swap") == 2, captured
    assert captured.get("prepared") is True, captured
    assert captured.get("hooks_installed") is True, captured
    assert captured.get("should_swap_result") is True, captured

    print("Newbie blockswap smoke passed: DiT-style net.blocks are discovered and prepared for BlockSwap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
