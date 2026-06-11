"""Smoke test for FasterDiT SNR integration."""

import sys
from pathlib import Path

import torch

core_path = Path(__file__).parent.parent
if str(core_path) not in sys.path:
    sys.path.insert(0, str(core_path))


def test_faster_dit_snr_integration():
    """Test that FasterDiT SNR is properly integrated into training loop."""
    from lulynx_trainer.faster_dit_snr import FasterDiTSNRConfig, FasterDiTSNRWeighter

    # Create config and weighter
    config = FasterDiTSNRConfig(
        mode='sqrt',
        gamma=5.0,
        timestep_sampling='low_snr_bias',
        low_snr_weight=1.5,
    )
    weighter = FasterDiTSNRWeighter(config)

    # Simulate training loop usage
    batch_size = 8
    num_timesteps = 1000
    alphas_cumprod = torch.linspace(0.999, 0.001, num_timesteps)

    # Compute SNR for random timesteps
    timesteps = torch.randint(0, num_timesteps, (batch_size,))
    alpha = alphas_cumprod[timesteps]
    snr = alpha / (1.0 - alpha + 1e-8)

    # Get loss weights
    weights = weighter(snr, v_parameterization=False)

    assert weights.shape == (batch_size,)
    assert (weights > 0).all()
    print(f"Loss weights: {weights.tolist()}")

    # Test timestep sampling
    from lulynx_trainer.faster_dit_snr import sample_timesteps_with_snr_bias

    sampled_timesteps = sample_timesteps_with_snr_bias(
        batch_size=batch_size,
        num_train_timesteps=num_timesteps,
        alphas_cumprod=alphas_cumprod,
        device=torch.device('cpu'),
        strategy='low_snr_bias',
        bias_strength=1.5,
    )

    assert sampled_timesteps.shape == (batch_size,)
    print(f"Sampled timesteps: {sampled_timesteps.tolist()}")


def test_training_loop_mock():
    """Mock training loop to test loss computation."""
    from lulynx_trainer.faster_dit_snr import FasterDiTSNRConfig, FasterDiTSNRWeighter

    config = FasterDiTSNRConfig(mode='sqrt', gamma=5.0)
    weighter = FasterDiTSNRWeighter(config)

    # Mock a training step
    batch_size = 4
    num_timesteps = 1000
    alphas_cumprod = torch.linspace(0.999, 0.001, num_timesteps)

    # Sample timesteps
    timesteps = torch.randint(0, num_timesteps, (batch_size,))

    # Compute SNR
    alpha = alphas_cumprod[timesteps]
    snr = alpha / (1.0 - alpha + 1e-8)

    # Simulate loss
    pred = torch.randn(batch_size, 4, 32, 32)
    target = torch.randn(batch_size, 4, 32, 32)
    loss = (pred - target).pow(2).mean(dim=[1, 2, 3])

    # Apply SNR weighting
    weights = weighter(snr)
    weighted_loss = loss * weights

    print(f"Unweighted loss: {loss.mean().item():.4f}")
    print(f"Weighted loss: {weighted_loss.mean().item():.4f}")
    print(f"SNR values: {snr.tolist()}")
    print(f"Weight values: {weights.tolist()}")

    assert weighted_loss.shape == (batch_size,)


def test_config_validation():
    """Test config validation and scorecard."""
    from lulynx_trainer.faster_dit_snr import (
        FasterDiTSNRConfig,
        build_faster_dit_snr_scorecard,
    )

    # Valid config
    config = FasterDiTSNRConfig(mode='sqrt', gamma=5.0)
    scorecard = build_faster_dit_snr_scorecard(
        config=config,
        weighting_tested=True,
        sampling_tested=True,
        convergence_improved=True,
    )

    assert scorecard["ok"]
    assert scorecard["optimization_ready"]
    assert scorecard["mode"] == "sqrt"
    assert scorecard["gamma"] == 5.0
    print(f"Scorecard: {scorecard}")


if __name__ == "__main__":
    print("=== Test 1: Integration Test ===")
    test_faster_dit_snr_integration()

    print("\n=== Test 2: Training Loop Mock ===")
    test_training_loop_mock()

    print("\n=== Test 3: Config Validation ===")
    test_config_validation()

    print("\nAll smoke tests passed!")
