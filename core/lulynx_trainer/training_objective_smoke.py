"""Smoke tests for SDXL diffusion objective and noise/loss knobs.

These tests avoid model loading.  They exercise the Warehouse math helpers used
by ``TrainingLoop`` so UI fields are not merely preserved in config.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.training_loop import TrainingLoop
from core.training_components.noise_utils import apply_ip_noise


class _Scheduler:
    def __init__(self) -> None:
        self.alphas_cumprod = torch.linspace(0.95, 0.05, 10)

        class _Config:
            num_train_timesteps = 10

        self.config = _Config()

    def get_velocity(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha = self.alphas_cumprod[timesteps].to(device=latents.device, dtype=latents.dtype)
        while alpha.dim() < latents.dim():
            alpha = alpha.unsqueeze(-1)
        return alpha.sqrt() * noise - (1.0 - alpha).sqrt() * latents


def _loop() -> TrainingLoop:
    loop = TrainingLoop.__new__(TrainingLoop)
    loop.noise_scheduler = _Scheduler()
    loop.huber_c = 0.2
    loop.huber_scale = 1.5
    loop.huber_schedule = "constant"
    loop.loss_type = "huber"
    loop.scale_v_pred_loss_like_noise_pred = True
    loop.masked_loss = True
    loop.alpha_mask = True
    loop.device = "cpu"
    return loop


def main() -> int:
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "model_type": "sdxl",
            "v_parameterization": True,
            "scale_v_pred_loss_like_noise_pred": True,
            "zero_terminal_snr": True,
            "noise_offset_random_strength": True,
            "ip_noise_gamma_random_strength": True,
            "huber_schedule": "snr",
            "huber_scale": 1.25,
        }
    )
    assert cfg.v_parameterization is True
    assert cfg.scale_v_pred_loss_like_noise_pred is True
    assert cfg.zero_terminal_snr is True
    assert cfg.noise_offset_random_strength is True
    assert cfg.ip_noise_gamma_random_strength is True
    assert cfg.huber_schedule == "snr"
    assert cfg.huber_scale == 1.25

    loop = _loop()
    latents = torch.ones((2, 4, 2, 2))
    noise = torch.full_like(latents, 0.5)
    timesteps = torch.tensor([0, 9], dtype=torch.long)

    velocity = loop._velocity_target(latents, noise, timesteps)
    expected = loop.noise_scheduler.get_velocity(latents, noise, timesteps)
    assert torch.allclose(velocity, expected)

    loss = torch.ones((2, 4, 2, 2))
    scaled = loop._scale_v_prediction_loss(loss, timesteps)
    assert scaled.shape == loss.shape
    assert torch.all(scaled <= loss)
    assert not torch.allclose(scaled[0], scaled[1])

    loop.huber_schedule = "constant"
    huber = loop._compute_diffusion_loss(torch.ones_like(latents), torch.zeros_like(latents), reduction="none", timesteps=timesteps)
    assert huber.shape == latents.shape
    assert torch.isfinite(huber).all()

    loop.huber_schedule = "snr"
    snr_delta = loop._huber_delta(timesteps, latents)
    assert snr_delta.shape == (2, 1, 1, 1)
    assert torch.isfinite(snr_delta).all()

    gamma = torch.tensor([0.0, 0.25])
    ip_noise = apply_ip_noise(torch.zeros_like(latents), gamma, timesteps, loop.noise_scheduler.alphas_cumprod)
    assert ip_noise.shape == latents.shape
    assert torch.allclose(ip_noise[0], torch.zeros_like(ip_noise[0]))
    assert not torch.allclose(ip_noise[1], torch.zeros_like(ip_noise[1]))

    strength = loop._sample_strength(0.5, True, (16,), latents)
    assert torch.all(strength >= 0)
    assert torch.all(strength <= 0.5)

    elementwise = torch.ones((2, 1, 2, 2))
    elementwise[0, :, 0, 0] = 10.0
    masked_batch = {
        "loss_masks": torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]),
        "caption_weights": torch.tensor([2.0, 1.0]),
    }
    per_sample = loop._loss_to_per_sample(elementwise, masked_batch)
    assert torch.allclose(per_sample, torch.tensor([10.0, 1.0]))
    weighted = loop._weighted_mean_loss(per_sample, masked_batch)
    assert torch.allclose(weighted, torch.tensor(7.0))

    print("Training objective smoke passed: v-pred, random noise, Huber, masks, and caption weights are wired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

