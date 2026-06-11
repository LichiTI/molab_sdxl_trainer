# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test Anima flow-matching parameter branches.

Exercises every non-default branch of:
- ``sample_anima_sigmas``: 5 timestep_sampling modes x sigmoid_scale variants
- ``compute_anima_loss_weighting``: 5 weighting schemes
- ``build_anima_flow_inputs``: interpolation, velocity target, timestep mapping
- ``apply_anima_flow_shift``: shift > 1, shift < 1, shift == 1
"""

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
anima_flow = _load_module("core.lulynx_trainer.anima_flow", TRAINER_ROOT / "anima_flow.py")

AnimaFlowConfig = anima_flow.AnimaFlowConfig
sample_anima_sigmas = anima_flow.sample_anima_sigmas
apply_anima_flow_shift = anima_flow.apply_anima_flow_shift
build_anima_flow_inputs = anima_flow.build_anima_flow_inputs
compute_anima_loss_weighting = anima_flow.compute_anima_loss_weighting


def _assert_sigma_valid(sigmas: torch.Tensor, label: str, *, low: float = 0.0, high: float = 1.0) -> None:
    assert sigmas.dim() == 1, f"{label}: expected 1-D, got {sigmas.shape}"
    assert torch.isfinite(sigmas).all(), f"{label}: non-finite values"
    assert sigmas.min() >= low - 1e-6, f"{label}: min {sigmas.min()} < {low}"
    assert sigmas.max() <= high + 1e-6, f"{label}: max {sigmas.max()} > {high}"


def main() -> int:
    device = "cpu"
    dtype = torch.float32
    batch = 32
    gen = torch.Generator(device).manual_seed(42)

    # -- 1. All 5 timestep_sampling modes produce valid sigmas --
    modes = ["sigma", "uniform", "sigmoid", "shift", "flux_shift"]
    for mode in modes:
        cfg = AnimaFlowConfig(timestep_sampling=mode, sigmoid_scale=1.0, discrete_flow_shift=2.0)
        sigmas = sample_anima_sigmas(batch, device=device, dtype=dtype, config=cfg, generator=gen)
        _assert_sigma_valid(sigmas, f"mode={mode}")

    # -- 2. Sigmoid scale variants shape the distribution --
    cfg_tight = AnimaFlowConfig(timestep_sampling="sigmoid", sigmoid_scale=5.0)
    cfg_flat = AnimaFlowConfig(timestep_sampling="sigmoid", sigmoid_scale=0.2)
    sig_tight = sample_anima_sigmas(256, device=device, dtype=dtype, config=cfg_tight, generator=gen)
    sig_flat = sample_anima_sigmas(256, device=device, dtype=dtype, config=cfg_flat, generator=gen)
    tight_spread = (sig_tight - 0.5).abs().mean()
    flat_spread = (sig_flat - 0.5).abs().mean()
    assert tight_spread > flat_spread, (
        f"sigmoid_scale=5 should spread more than 0.2: tight={tight_spread:.4f} flat={flat_spread:.4f}"
    )

    # -- 3. Flow shift: shift > 1 biases toward higher sigma, shift < 1 toward lower --
    cfg_shift_high = AnimaFlowConfig(timestep_sampling="shift", sigmoid_scale=1.0, discrete_flow_shift=5.0)
    cfg_shift_low = AnimaFlowConfig(timestep_sampling="shift", sigmoid_scale=1.0, discrete_flow_shift=0.2)
    sig_high = sample_anima_sigmas(256, device=device, dtype=dtype, config=cfg_shift_high, generator=gen)
    sig_low = sample_anima_sigmas(256, device=device, dtype=dtype, config=cfg_shift_low, generator=gen)
    assert sig_high.mean() > sig_low.mean(), (
        f"shift=5 should bias higher than shift=0.2: high={sig_high.mean():.4f} low={sig_low.mean():.4f}"
    )

    # -- 4. apply_anima_flow_shift monotonicity and identity at shift=1 --
    test_sigmas = torch.linspace(0.05, 0.95, 20)
    identity = apply_anima_flow_shift(test_sigmas, 1.0)
    assert torch.allclose(identity, test_sigmas, atol=1e-6), "shift=1 should be identity"

    shifted_up = apply_anima_flow_shift(test_sigmas, 5.0)
    shifted_down = apply_anima_flow_shift(test_sigmas, 0.2)
    assert (shifted_up >= test_sigmas - 1e-6).all(), "shift>1 should increase sigmas"
    assert (shifted_down <= test_sigmas + 1e-6).all(), "shift<1 should decrease sigmas"

    no_shift = apply_anima_flow_shift(test_sigmas, -1.0)
    assert torch.allclose(no_shift, test_sigmas, atol=1e-6), "shift<=0 should be identity"

    # -- 5. All 5 weighting schemes produce valid positive weights --
    sigma_range = torch.linspace(0.01, 0.99, 20)
    for scheme in ["none", "sigma_sqrt", "cosmap", "mode", "logit_normal"]:
        weights = compute_anima_loss_weighting(sigma_range, scheme=scheme)
        assert weights.dim() == 1, f"scheme={scheme}: expected 1-D"
        assert torch.isfinite(weights).all(), f"scheme={scheme}: non-finite weights"
        assert (weights >= 0).all(), f"scheme={scheme}: negative weights"

    none_weights = compute_anima_loss_weighting(sigma_range, scheme="none")
    assert torch.allclose(none_weights, torch.ones_like(sigma_range)), "none scheme should be all ones"

    # -- 6. build_anima_flow_inputs correctness --
    latents = torch.randn(2, 16, 4, 4)
    noise = torch.randn(2, 16, 4, 4)
    sigmas = torch.tensor([0.3, 0.7])
    noisy, target, timesteps = build_anima_flow_inputs(latents, noise, sigmas, num_train_timesteps=1000)

    assert noisy.shape == latents.shape, f"noisy shape {noisy.shape} != latents {latents.shape}"
    assert target.shape == latents.shape, f"target shape {target.shape} != latents {latents.shape}"
    assert timesteps.shape == (2,), f"timesteps shape {timesteps.shape}"

    expected_noisy_0 = 0.7 * latents[0] + 0.3 * noise[0]
    assert torch.allclose(noisy[0], expected_noisy_0, atol=1e-6), "noisy interpolation incorrect"

    expected_target = noise - latents
    assert torch.allclose(target, expected_target, atol=1e-6), "velocity target should be noise - latents"
    assert torch.allclose(timesteps, sigmas * 1000.0, atol=1e-4), "timesteps should be sigmas * 1000"

    # -- 7. ValueError on unsupported mode --
    try:
        bad_cfg = AnimaFlowConfig(timestep_sampling="invalid_mode")
        sample_anima_sigmas(4, device=device, dtype=dtype, config=bad_cfg)
        raise AssertionError("Should have raised ValueError for invalid mode")
    except ValueError:
        pass

    try:
        compute_anima_loss_weighting(sigma_range, scheme="invalid_scheme")
        raise AssertionError("Should have raised ValueError for invalid scheme")
    except ValueError:
        pass

    try:
        sample_anima_sigmas(0, device=device, dtype=dtype)
        raise AssertionError("Should have raised ValueError for batch_size=0")
    except ValueError:
        pass

    print(
        "Anima flow-params smoke passed: all 5 sampling modes, 5 weighting schemes, "
        "flow shift monotonicity, interpolation/velocity/timestep correctness, and error paths"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
