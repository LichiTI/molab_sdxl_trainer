# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test wavelet loss math and TrainingLoop config plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from backend.core.lulynx_trainer.config_adapter import ConfigAdapter
from backend.core.lulynx_trainer.training_loop import TrainingLoop
from backend.core.lulynx_trainer.wavelet_loss import wavelet_loss


def test_wavelet_loss_supports_multiple_levels() -> None:
    pred = torch.zeros((2, 1, 8, 8))
    target = torch.ones_like(pred)

    scalar = wavelet_loss(pred, target, levels=2, high_freq_weight=1.5, reduction="mean")
    per_sample = wavelet_loss(pred, target, levels=2, high_freq_weight=1.5, reduction="none")

    assert scalar.ndim == 0
    assert per_sample.shape == (2,)
    assert torch.isfinite(scalar)
    assert torch.isfinite(per_sample).all()


def test_wavelet_approx_weight_affects_low_frequency_loss() -> None:
    pred = torch.zeros((1, 1, 8, 8))
    target = torch.ones_like(pred)

    without_approx = wavelet_loss(pred, target, levels=2, high_freq_weight=0.0, approx_weight=0.0)
    with_approx = wavelet_loss(pred, target, levels=2, high_freq_weight=0.0, approx_weight=0.5)

    assert with_approx > without_approx


def test_config_adapter_wavelet_aliases_survive() -> None:
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "wavelet_loss_enabled": "true",
            "wavelet_loss_weight": 0.25,
            "wavelet_loss_levels": 3,
            "wavelet_loss_approx_weight": 0.1,
        }
    )

    assert cfg.wavelet_loss_enabled is True
    assert cfg.wavelet_loss_high_freq_weight == 0.25
    assert cfg.wavelet_loss_levels == 3
    assert cfg.wavelet_loss_approx_weight == 0.1


def test_training_loop_stores_wavelet_knobs() -> None:
    loop = TrainingLoop.__new__(TrainingLoop)
    TrainingLoop.__init__(
        loop,
        unet=None,
        text_encoder_1=None,
        text_encoder_2=None,
        vae=None,
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=None,
        lora_injector=None,
        optimizer=None,
        lr_scheduler=None,
        device="cpu",
        dtype=torch.float32,
        wavelet_loss_enabled=True,
        wavelet_loss_levels=3,
        wavelet_loss_high_freq_weight=0.25,
        wavelet_loss_approx_weight=0.1,
        wavelet_loss_base_loss="l1",
    )

    assert loop.wavelet_loss_enabled is True
    assert loop.wavelet_loss_levels == 3
    assert loop.wavelet_loss_high_freq_weight == 0.25
    assert loop.wavelet_loss_approx_weight == 0.1
    assert loop.wavelet_loss_base_loss == "l1"


def main() -> int:
    test_wavelet_loss_supports_multiple_levels()
    test_wavelet_approx_weight_affects_low_frequency_loss()
    test_config_adapter_wavelet_aliases_survive()
    test_training_loop_stores_wavelet_knobs()
    print("wavelet_loss_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
