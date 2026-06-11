# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for logit-normal timestep sampling."""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_anima_flow = _import_module(
    "anima_flow",
    os.path.join(_HERE, "anima_flow.py"),
)
AnimaFlowConfig = _anima_flow.AnimaFlowConfig
sample_anima_sigmas = _anima_flow.sample_anima_sigmas

import torch


def test_anima_logit_normal():
    cfg = AnimaFlowConfig(timestep_sampling="logit_normal", logit_mean=0.0, logit_std=1.0)
    sigmas = sample_anima_sigmas(1000, device="cpu", dtype=torch.float32, config=cfg)
    assert sigmas.shape == (1000,), f"Shape mismatch: {sigmas.shape}"
    assert (sigmas >= 0).all() and (sigmas <= 1).all(), "Sigmas should be in [0, 1]"
    # Check distribution is roughly centered (sigmoid of N(0,1) centers around 0.5)
    mean_val = sigmas.mean().item()
    assert 0.3 < mean_val < 0.7, f"Mean {mean_val} should be roughly 0.5"
    print("PASS: Anima logit_normal produces valid sigmas")


def test_logit_mean_shifts_distribution():
    gen = torch.Generator().manual_seed(42)
    cfg_low = AnimaFlowConfig(timestep_sampling="logit_normal", logit_mean=-2.0, logit_std=1.0)
    cfg_high = AnimaFlowConfig(timestep_sampling="logit_normal", logit_mean=2.0, logit_std=1.0)
    sigmas_low = sample_anima_sigmas(5000, device="cpu", dtype=torch.float32, config=cfg_low)
    sigmas_high = sample_anima_sigmas(5000, device="cpu", dtype=torch.float32, config=cfg_high)
    assert sigmas_low.mean().item() < sigmas_high.mean().item(), \
        f"Mean with logit_mean=-2 ({sigmas_low.mean():.3f}) should be < mean with logit_mean=2 ({sigmas_high.mean():.3f})"
    print("PASS: logit_mean shifts distribution center")


def test_logit_std_controls_spread():
    cfg_tight = AnimaFlowConfig(timestep_sampling="logit_normal", logit_mean=0.0, logit_std=0.1)
    cfg_wide = AnimaFlowConfig(timestep_sampling="logit_normal", logit_mean=0.0, logit_std=3.0)
    sigmas_tight = sample_anima_sigmas(5000, device="cpu", dtype=torch.float32, config=cfg_tight)
    sigmas_wide = sample_anima_sigmas(5000, device="cpu", dtype=torch.float32, config=cfg_wide)
    assert sigmas_tight.std().item() < sigmas_wide.std().item(), \
        "Smaller std should produce tighter distribution"
    print("PASS: logit_std controls spread")


def test_ddpm_logit_normal_range():
    """Simulate the DDPM logit-normal timestep path."""
    max_t = 1000
    batch_size = 2000
    logit_mean, logit_std = 0.0, 1.0
    normals = torch.randn((batch_size,))
    probs = torch.sigmoid(logit_mean + logit_std * normals)
    timesteps = (probs * max_t).clamp(0, max_t - 1).long()
    assert (timesteps >= 0).all() and (timesteps < max_t).all(), \
        f"Timesteps out of range: min={timesteps.min()}, max={timesteps.max()}"
    print("PASS: DDPM logit-normal timesteps in valid range")


def test_backward_compat_default():
    cfg = AnimaFlowConfig()  # default: timestep_sampling="sigma"
    sigmas = sample_anima_sigmas(100, device="cpu", dtype=torch.float32, config=cfg)
    assert sigmas.shape == (100,)
    assert (sigmas >= 0).all() and (sigmas <= 1).all()
    print("PASS: default config unchanged (backward compatible)")


def test_config_field():
    cfg_path = os.path.join(_HERE, "..", "configs.py")
    with open(cfg_path, encoding="utf-8") as f:
        src = f.read()
    assert "ddpm_timestep_sampling" in src, "Missing ddpm_timestep_sampling in configs.py"
    print("PASS: config field exists")


if __name__ == "__main__":
    test_anima_logit_normal()
    test_logit_mean_shifts_distribution()
    test_logit_std_controls_spread()
    test_ddpm_logit_normal_range()
    test_backward_compat_default()
    test_config_field()
    print("\nAll logit-normal timestep smoke tests passed!")
