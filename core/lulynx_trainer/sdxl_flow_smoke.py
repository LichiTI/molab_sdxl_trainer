"""Smoke test for SDXL Flow Matching engine.

Validates that:
1. SDXLFlowConfig defaults match the Phase 1.1 spec
2. sample_sdxl_flow_sigmas produces valid sigmas for each sampling strategy
3. build_sdxl_flow_inputs produces correct targets for epsilon/velocity/sample
4. compute_sdxl_loss_weighting produces correct weight shapes for all schemes
5. apply_flow_shift works with the same formula as Anima
6. generator parameter produces reproducible results
"""
from __future__ import annotations

import sys
import os
import math
import importlib.util

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.sdxl_flow",
    os.path.join(_HERE, "sdxl_flow.py"),
)
_sdxl_flow = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.sdxl_flow"] = _sdxl_flow
_spec.loader.exec_module(_sdxl_flow)
SDXLFlowConfig = _sdxl_flow.SDXLFlowConfig
sample_sdxl_flow_sigmas = _sdxl_flow.sample_sdxl_flow_sigmas
build_sdxl_flow_inputs = _sdxl_flow.build_sdxl_flow_inputs
compute_sdxl_loss_weighting = _sdxl_flow.compute_sdxl_loss_weighting
apply_flow_shift = _sdxl_flow.apply_flow_shift


def test_sdxl_flow_config_defaults():
    """SDXLFlowConfig has correct Phase 1.1 defaults."""

    cfg = SDXLFlowConfig()
    assert cfg.timestep_sampling == "uniform"
    assert cfg.sigmoid_scale == 1.0
    assert cfg.discrete_flow_shift == 1.0
    assert cfg.weighting_scheme == "none"
    assert cfg.model_prediction_type == "epsilon"
    assert cfg.logit_mean == 0.0
    assert cfg.logit_std == 1.0
    print("PASS: test_sdxl_flow_config_defaults")
    return True


def test_sample_sigmas_uniform():
    """Uniform sampling produces sigmas in [0, 1]."""

    cfg = SDXLFlowConfig(timestep_sampling="uniform")
    sigmas = sample_sdxl_flow_sigmas(8, device=torch.device("cpu"), dtype=torch.float32, config=cfg)
    assert sigmas.shape == (8,), f"Expected (8,), got {sigmas.shape}"
    assert sigmas.min() >= 1e-5, f"Sigmas too low: {sigmas.min()}"
    assert sigmas.max() <= 1.0 - 1e-5, f"Sigmas too high: {sigmas.max()}"
    print("PASS: test_sample_sigmas_uniform")
    return True


def test_sample_sigmas_sigma_alias():
    """'sigma' is an alias for 'uniform' sampling."""

    cfg = SDXLFlowConfig(timestep_sampling="sigma")
    sigmas = sample_sdxl_flow_sigmas(8, device=torch.device("cpu"), dtype=torch.float32, config=cfg)
    assert sigmas.shape == (8,)
    assert sigmas.min() >= 1e-5
    assert sigmas.max() <= 1.0 - 1e-5
    print("PASS: test_sample_sigmas_sigma_alias")
    return True


def test_sample_sigmas_sigmoid():
    """Sigmoid sampling produces sigmas in [0, 1]."""

    cfg = SDXLFlowConfig(timestep_sampling="sigmoid", sigmoid_scale=1.5)
    sigmas = sample_sdxl_flow_sigmas(16, device=torch.device("cpu"), dtype=torch.float32, config=cfg)
    assert sigmas.shape == (16,)
    assert sigmas.min() >= 1e-5
    assert sigmas.max() <= 1.0 - 1e-5
    print("PASS: test_sample_sigmas_sigmoid")
    return True


def test_sample_sigmas_logit_normal():
    """Logit-normal sampling produces sigmas in [0, 1]."""

    cfg = SDXLFlowConfig(timestep_sampling="logit_normal", logit_mean=0.0, logit_std=1.0)
    sigmas = sample_sdxl_flow_sigmas(16, device=torch.device("cpu"), dtype=torch.float32, config=cfg)
    assert sigmas.shape == (16,)
    assert sigmas.min() >= 1e-5
    assert sigmas.max() <= 1.0 - 1e-5
    print("PASS: test_sample_sigmas_logit_normal")
    return True


def test_sample_sigmas_shift():
    """Shift sampling applies flow shift transformation."""

    cfg = SDXLFlowConfig(timestep_sampling="shift", discrete_flow_shift=3.0)
    sigmas = sample_sdxl_flow_sigmas(64, device=torch.device("cpu"), dtype=torch.float32, config=cfg)
    assert sigmas.shape == (64,)
    # With shift=3.0, mean should be biased upward
    mean = sigmas.mean().item()
    assert mean > 0.4, f"Shift sampling with shift=3.0 should bias mean upward, got {mean}"
    print("PASS: test_sample_sigmas_shift")
    return True


def test_sample_sigmas_invalid_mode():
    """Invalid timestep_sampling raises ValueError."""

    cfg = SDXLFlowConfig(timestep_sampling="invalid_mode")
    try:
        sample_sdxl_flow_sigmas(4, device=torch.device("cpu"), dtype=torch.float32, config=cfg)
        print("FAIL: test_sample_sigmas_invalid_mode — no ValueError raised")
        return False
    except ValueError:
        pass
    print("PASS: test_sample_sigmas_invalid_mode")
    return True


def test_sample_sigmas_generator():
    """Generator parameter produces reproducible results."""

    cfg = SDXLFlowConfig(timestep_sampling="uniform")
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    sigmas1 = sample_sdxl_flow_sigmas(16, device=torch.device("cpu"), dtype=torch.float32, config=cfg, generator=gen1)
    sigmas2 = sample_sdxl_flow_sigmas(16, device=torch.device("cpu"), dtype=torch.float32, config=cfg, generator=gen2)
    assert torch.allclose(sigmas1, sigmas2), "Seeded generators should produce identical sigmas"
    print("PASS: test_sample_sigmas_generator")
    return True


def test_build_flow_inputs_epsilon():
    """build_sdxl_flow_inputs with epsilon target: target = noise."""

    latents = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(latents)
    sigmas = torch.tensor([0.0, 1.0])

    noisy, target, timesteps = build_sdxl_flow_inputs(
        latents, noise, sigmas, num_train_timesteps=1000, model_prediction_type="epsilon"
    )
    # At sigma=0: noisy ≈ latents
    assert torch.allclose(noisy[0], latents[0], atol=1e-5), "At sigma=0, noisy should equal latents"
    # At sigma=1: noisy ≈ noise
    assert torch.allclose(noisy[1], noise[1], atol=1e-5), "At sigma=1, noisy should equal noise"
    # Target is noise (epsilon prediction)
    assert torch.allclose(target, noise, atol=1e-5), "Epsilon target should equal noise"
    # Timesteps scale correctly
    assert abs(timesteps[0].item()) < 1e-3
    assert abs(timesteps[1].item() - 1000.0) < 1e-3
    print("PASS: test_build_flow_inputs_epsilon")
    return True


def test_build_flow_inputs_velocity():
    """build_sdxl_flow_inputs with velocity target: target = noise - latents."""

    latents = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(latents)
    sigmas = torch.tensor([0.0, 1.0])

    noisy, target, timesteps = build_sdxl_flow_inputs(
        latents, noise, sigmas, num_train_timesteps=1000, model_prediction_type="velocity"
    )
    # Target is velocity = noise - latents
    assert torch.allclose(target, noise - latents, atol=1e-5), "Velocity target should be noise - latents"
    print("PASS: test_build_flow_inputs_velocity")
    return True


def test_build_flow_inputs_sample():
    """build_sdxl_flow_inputs with sample target: target = latents."""

    latents = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(latents)
    sigmas = torch.tensor([0.0, 1.0])

    noisy, target, timesteps = build_sdxl_flow_inputs(
        latents, noise, sigmas, num_train_timesteps=1000, model_prediction_type="sample"
    )
    # Target is latents (predict clean sample)
    assert torch.allclose(target, latents, atol=1e-5), "Sample target should equal latents"
    print("PASS: test_build_flow_inputs_sample")
    return True


def test_build_flow_inputs_invalid_type():
    """Invalid model_prediction_type raises ValueError."""

    latents = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(latents)
    sigmas = torch.tensor([0.5, 0.5])
    try:
        build_sdxl_flow_inputs(latents, noise, sigmas, model_prediction_type="bad_type")
        print("FAIL: test_build_flow_inputs_invalid_type — no ValueError raised")
        return False
    except ValueError:
        pass
    print("PASS: test_build_flow_inputs_invalid_type")
    return True


def test_flow_midpoint_interpolation():
    """At sigma=0.5, interpolation is exactly the midpoint."""

    latents = torch.ones(1, 4, 4, 4)
    noise = torch.zeros(1, 4, 4, 4)
    sigmas = torch.tensor([0.5])

    noisy, _, _ = build_sdxl_flow_inputs(latents, noise, sigmas, model_prediction_type="epsilon")
    # At sigma=0.5: noisy = 0.5*latents + 0.5*noise = 0.5*ones
    assert torch.allclose(noisy, 0.5 * torch.ones_like(latents), atol=1e-5)
    print("PASS: test_flow_midpoint_interpolation")
    return True


def test_apply_flow_shift():
    """apply_flow_shift matches the Anima formula."""

    sigmas = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    # With shift=1.0, output should be identical
    shifted = apply_flow_shift(sigmas, 1.0)
    assert torch.allclose(shifted, sigmas, atol=1e-6), "shift=1.0 should be identity"

    # With shift=2.0, check formula: (s * 2) / (1 + s)
    expected = (sigmas * 2.0) / (1.0 + sigmas)
    shifted = apply_flow_shift(sigmas, 2.0)
    assert torch.allclose(shifted, expected, atol=1e-6), "shift=2.0 should match formula"

    # shift<=0 should be identity
    shifted_zero = apply_flow_shift(sigmas, 0.0)
    assert torch.allclose(shifted_zero, sigmas, atol=1e-6)
    print("PASS: test_apply_flow_shift")
    return True


def test_compute_loss_weighting():
    """compute_sdxl_loss_weighting produces correct shapes for all schemes."""

    sigmas = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    for scheme in ("none", "sigma_sqrt", "cosmap", "logit_normal"):
        weights = compute_sdxl_loss_weighting(sigmas, scheme=scheme)
        assert weights.shape == sigmas.shape, f"Scheme {scheme}: shape mismatch {weights.shape} vs {sigmas.shape}"
        assert torch.isfinite(weights).all(), f"Scheme {scheme}: non-finite weights"
        if scheme != "none":
            assert (weights > 0).all(), f"Scheme {scheme}: non-positive weights"
    print("PASS: test_compute_loss_weighting")
    return True


def test_compute_loss_weighting_logit_normal_params():
    """compute_sdxl_loss_weighting logit_normal respects mean/std params."""

    sigmas = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    w1 = compute_sdxl_loss_weighting(sigmas, scheme="logit_normal", logit_mean=0.0, logit_std=1.0)
    w2 = compute_sdxl_loss_weighting(sigmas, scheme="logit_normal", logit_mean=2.0, logit_std=0.5)
    # Different params should produce different weights
    assert not torch.allclose(w1, w2, atol=1e-4), "Different logit_normal params should produce different weights"
    print("PASS: test_compute_loss_weighting_logit_normal_params")
    return True


def test_compute_loss_weighting_invalid_scheme():
    """Invalid weighting scheme raises ValueError."""

    sigmas = torch.tensor([0.5])
    try:
        compute_sdxl_loss_weighting(sigmas, scheme="invalid_scheme")
        print("FAIL: test_compute_loss_weighting_invalid_scheme — no ValueError raised")
        return False
    except ValueError:
        pass
    print("PASS: test_compute_loss_weighting_invalid_scheme")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = []
    tests = [
        test_sdxl_flow_config_defaults,
        test_sample_sigmas_uniform,
        test_sample_sigmas_sigma_alias,
        test_sample_sigmas_sigmoid,
        test_sample_sigmas_logit_normal,
        test_sample_sigmas_shift,
        test_sample_sigmas_invalid_mode,
        test_sample_sigmas_generator,
        test_build_flow_inputs_epsilon,
        test_build_flow_inputs_velocity,
        test_build_flow_inputs_sample,
        test_build_flow_inputs_invalid_type,
        test_flow_midpoint_interpolation,
        test_apply_flow_shift,
        test_compute_loss_weighting,
        test_compute_loss_weighting_logit_normal_params,
        test_compute_loss_weighting_invalid_scheme,
    ]

    for test_fn in tests:
        try:
            ok = test_fn()
            results.append((test_fn.__name__, ok))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAIL: {test_fn.__name__} — {e}")
            results.append((test_fn.__name__, False))

    print("\n" + "=" * 60)
    print("SDXL Flow Matching Smoke Test Results")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")
    print(f"\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
