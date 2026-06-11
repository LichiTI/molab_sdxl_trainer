"""Smoke test for the clean-room CNS colored-noise primitive."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.cns_sampling import (  # noqa: E402
    CNSCalibration,
    CNSRecolorer,
    build_cns_recolorer,
    radial_frequency_bins,
)


def _calibration() -> CNSCalibration:
    gamma = torch.zeros(1, 2, 8)
    gamma[:, :, :3] = 0.90
    gamma[:, :, 3:] = 0.05
    return CNSCalibration.from_arrays(
        gamma=gamma,
        aspects=torch.tensor([[32.0, 32.0]]),
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
    )


def _spatial_std(x: torch.Tensor) -> torch.Tensor:
    centered = x - x.mean(dim=(-2, -1), keepdim=True)
    return centered.square().mean(dim=(-2, -1), keepdim=True).sqrt()


def _low_high_ratio(x: torch.Tensor) -> float:
    bins = radial_frequency_bins(x.shape[-2], x.shape[-1], 8, device=x.device)
    power = torch.fft.fft2(x.float(), dim=(-2, -1)).abs().square().mean(dim=(0, 1))
    low = power[bins < 3].mean()
    high = power[bins >= 3].mean()
    return float((low / high.clamp_min(1e-12)).item())


def test_strength_zero_is_pass_through() -> None:
    torch.manual_seed(7)
    white = torch.randn(2, 4, 32, 32, dtype=torch.float32)
    recolorer = CNSRecolorer(_calibration(), strength=0.0)
    out = recolorer.recolor(white, sigma=0.75)
    assert torch.equal(out, white)


def test_recolor_preserves_shape_dtype_and_variance() -> None:
    torch.manual_seed(11)
    white = torch.randn(2, 4, 32, 32, dtype=torch.float32)
    recolorer = CNSRecolorer(_calibration(), strength=1.0)
    out = recolorer.recolor(white, sigma=0.75)
    assert out.shape == white.shape
    assert out.dtype == white.dtype
    assert torch.allclose(_spatial_std(out), _spatial_std(white), rtol=2e-3, atol=2e-3)
    assert _low_high_ratio(out) < _low_high_ratio(white)


def test_recolor_supports_fake_5d_latents() -> None:
    torch.manual_seed(13)
    white = torch.randn(1, 3, 1, 16, 24, dtype=torch.float32)
    recolorer = CNSRecolorer(_calibration(), strength=0.5)
    out = recolorer.recolor(white, sigma=0.5)
    assert out.shape == white.shape
    assert out.dtype == white.dtype


def test_auto_calibration_is_explicitly_unavailable() -> None:
    try:
        build_cns_recolorer(gamma_path="auto")
    except FileNotFoundError:
        return
    raise AssertionError("Expected CNS auto calibration to raise until a bundled resolver exists")


def main() -> int:
    tests = (
        test_strength_zero_is_pass_through,
        test_recolor_preserves_shape_dtype_and_variance,
        test_recolor_supports_fake_5d_latents,
        test_auto_calibration_is_explicitly_unavailable,
    )
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("CNS sampling smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
