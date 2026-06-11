"""Smoke test for Anima noise scheduling fields: sigma shapes, timestep distributions, eps/dt noise shapes."""
from __future__ import annotations

import sys
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))

# Load anima_flow via importlib to avoid heavy import chains
_af_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.anima_flow",
    os.path.join(_HERE, "anima_flow.py"),
)
_af_mod = importlib.util.module_from_spec(_af_spec)
sys.modules["core.lulynx_trainer.anima_flow"] = _af_mod
_af_spec.loader.exec_module(_af_mod)

sample_anima_sigmas = _af_mod.sample_anima_sigmas
build_anima_flow_inputs = _af_mod.build_anima_flow_inputs
AnimaFlowConfig = _af_mod.AnimaFlowConfig

import torch


def test_sigma_shape_batch():
    """Sigma scheduling functions produce (B,) shape."""
    B = 8
    sigmas = sample_anima_sigmas(
        B, device=torch.device("cpu"), dtype=torch.float32, config=AnimaFlowConfig()
    )
    assert sigmas.shape == (B,), f"Expected ({B},), got {sigmas.shape}"


def test_sigma_shape_expandable():
    """Sigmas can be viewed as (B, 1, 1, 1) for broadcasting with latent shapes."""
    B = 4
    sigmas = sample_anima_sigmas(
        B, device=torch.device("cpu"), dtype=torch.float32, config=AnimaFlowConfig()
    )
    # The canonical expand pattern used in build_anima_flow_inputs
    view_shape = (B, 1, 1, 1)
    sigma_view = sigmas.view(view_shape)
    assert sigma_view.shape == view_shape, f"Expected {view_shape}, got {sigma_view.shape}"


def test_different_timestep_distributions_different_sigmas():
    """Different timestep_sampling modes produce different sigma distributions."""
    B = 64
    torch.manual_seed(42)
    sigmas_uniform = sample_anima_sigmas(
        B, device=torch.device("cpu"), dtype=torch.float32,
        config=AnimaFlowConfig(timestep_sampling="uniform"),
    )
    torch.manual_seed(42)
    sigmas_sigmoid = sample_anima_sigmas(
        B, device=torch.device("cpu"), dtype=torch.float32,
        config=AnimaFlowConfig(timestep_sampling="sigmoid", sigmoid_scale=3.0),
    )

    # Same seed but different distribution should yield different means
    # (sigmoid with high scale pushes toward extremes)
    mean_diff = (sigmas_uniform.mean() - sigmas_sigmoid.mean()).abs()
    assert mean_diff > 0.01, (
        f"Uniform and sigmoid sigmas should differ in distribution: "
        f"uniform_mean={sigmas_uniform.mean():.4f}, sigmoid_mean={sigmas_sigmoid.mean():.4f}"
    )


def test_eps_noise_shape_matches_latents():
    """eps/dt noise shapes match the latent shapes fed to build_anima_flow_inputs."""
    B, C, H, W = 2, 16, 8, 8
    latents = torch.randn(B, C, H, W)
    noise = torch.randn(B, C, H, W)
    sigmas = sample_anima_sigmas(
        B, device=torch.device("cpu"), dtype=torch.float32, config=AnimaFlowConfig()
    )

    noisy, target, timesteps = build_anima_flow_inputs(latents, noise, sigmas)

    # noisy latents should match input shape
    assert noisy.shape == latents.shape, f"noisy shape {noisy.shape} != latent shape {latents.shape}"
    # target (velocity) should match latent shape
    assert target.shape == latents.shape, f"target shape {target.shape} != latent shape {latents.shape}"
    # timesteps should match batch size
    assert timesteps.shape == (B,), f"timesteps shape {timesteps.shape} != ({B},)"


def test_sigmas_in_unit_range():
    """All sigma values should be in [0, 1] after clamping."""
    B = 128
    for mode in ("sigma", "uniform", "sigmoid"):
        sigmas = sample_anima_sigmas(
            B, device=torch.device("cpu"), dtype=torch.float32,
            config=AnimaFlowConfig(timestep_sampling=mode),
        )
        assert sigmas.min() >= 0.0, f"{mode}: sigma min {sigmas.min()} < 0"
        assert sigmas.max() <= 1.0, f"{mode}: sigma max {sigmas.max()} > 1"


def test_shift_modifies_distribution():
    """Shift sampling produces different mean than plain uniform sampling."""
    B = 256
    torch.manual_seed(0)
    sigmas_uniform = sample_anima_sigmas(
        B, device=torch.device("cpu"), dtype=torch.float32,
        config=AnimaFlowConfig(timestep_sampling="uniform"),
    )
    torch.manual_seed(0)
    sigmas_shift = sample_anima_sigmas(
        B, device=torch.device("cpu"), dtype=torch.float32,
        config=AnimaFlowConfig(timestep_sampling="shift", discrete_flow_shift=3.0),
    )
    # Shift > 1 should push mean upward
    assert sigmas_shift.mean() > sigmas_uniform.mean(), (
        f"Shift should increase mean: uniform={sigmas_uniform.mean():.4f}, "
        f"shift={sigmas_shift.mean():.4f}"
    )


if __name__ == "__main__":
    print("Anima Noise Fields Smoke Tests")
    print("=" * 40)
    test_sigma_shape_batch()
    print("PASS: sigma_shape_batch")
    test_sigma_shape_expandable()
    print("PASS: sigma_shape_expandable")
    test_different_timestep_distributions_different_sigmas()
    print("PASS: different_timestep_distributions_different_sigmas")
    test_eps_noise_shape_matches_latents()
    print("PASS: eps_noise_shape_matches_latents")
    test_sigmas_in_unit_range()
    print("PASS: sigmas_in_unit_range")
    test_shift_modifies_distribution()
    print("PASS: shift_modifies_distribution")
    print("=" * 40)
    print("All Anima noise fields smoke tests passed!")
