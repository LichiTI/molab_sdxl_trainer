# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Real Newbie native-transformer LoRA injector smoke.

Verifies that the Warehouse `_NextDiTWrapper` reconstructed from the local
Newbie transformer bundle exposes real `nn.Linear` leaves for `attention.qkv`
and `attention.out`, accepts LoRA injection on those leaves, and propagates a
finite backward signal into injected LoRA parameters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.lora_injector import LoRAInjector
from core.lulynx_trainer.newbie_loader import _load_transformer_native
from core.lulynx_trainer.newbie_smoke import run_newbie_transformer_smoke


def main(argv: list[str] | None = None) -> int:
    del argv

    transformer, notes = _load_transformer_native(
        "H:/lulynx-trainer/models/newbie/transformer",
        torch.float32,
        "cpu",
    )
    if transformer is None:
        raise RuntimeError(f"Failed to load Newbie transformer: {notes}")

    injector = LoRAInjector(
        rank=2,
        alpha=2,
        dropout=0.1,
        model_arch="newbie",
    )
    injected = injector.inject_unet(transformer)
    if len(injected) != 80:
        raise AssertionError(f"Expected 80 injected Newbie LoRA layers, got {len(injected)}")

    for layer in injector.injected_layers.values():
        for param in layer.lora.parameters():
            param.requires_grad_(True)

    state = injector.get_lora_state_dict()
    if not any("attention_qkv" in key for key in state):
        raise AssertionError("Injected LoRA state is missing attention_qkv keys")
    if not any("attention_out" in key for key in state):
        raise AssertionError("Injected LoRA state is missing attention_out keys")

    smoke = run_newbie_transformer_smoke(
        transformer,
        target_modules=("attention.qkv", "attention.out"),
        latent_size=8,
    )
    if not smoke.passed:
        raise RuntimeError(f"Injected Newbie transformer smoke failed: {smoke.reason}")

    print(
        "Newbie native injector smoke passed: "
        f"injected={len(injected)}, "
        f"grad_targets={list(smoke.gradient_targets)[:6]}, "
        f"state_keys={len(state)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

