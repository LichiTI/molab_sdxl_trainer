"""Clean-room CNS colored-noise primitive.

CNS is only meaningful for stochastic samplers that inject fresh noise at each
denoise step.  This module provides the math primitive and calibration contract;
runtime samplers must opt in explicitly at their noise-injection seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class CNSCalibration:
    """Completion-matrix payload for colored noise sampling.

    Shapes:
    - gamma: (A, T, F), completion per aspect, step, radial frequency bin.
    - aspects: (A, 2), pixel or latent height/width used for calibration.
    - sigmas: (T + 1,), decreasing calibration sigma schedule.
    """

    gamma: torch.Tensor
    aspects: torch.Tensor
    sigmas: torch.Tensor

    @classmethod
    def from_arrays(cls, gamma, aspects, sigmas) -> "CNSCalibration":
        gamma_t = torch.as_tensor(gamma, dtype=torch.float32)
        aspects_t = torch.as_tensor(aspects, dtype=torch.float32).reshape(-1, 2)
        sigmas_t = torch.as_tensor(sigmas, dtype=torch.float32).flatten()
        if gamma_t.ndim != 3:
            raise ValueError(f"CNS gamma must be shaped (A, T, F), got {tuple(gamma_t.shape)}")
        if aspects_t.ndim != 2 or aspects_t.shape[1] != 2:
            raise ValueError(f"CNS aspects must be shaped (A, 2), got {tuple(aspects_t.shape)}")
        if aspects_t.shape[0] != gamma_t.shape[0]:
            raise ValueError("CNS aspects count must match gamma aspect count")
        if sigmas_t.numel() != gamma_t.shape[1] + 1:
            raise ValueError("CNS sigmas length must equal gamma step count + 1")
        return cls(gamma=gamma_t.clamp(0.0, 1.0), aspects=aspects_t, sigmas=sigmas_t)

    @classmethod
    def from_npz(cls, path: str | Path) -> "CNSCalibration":
        import numpy as np

        data = np.load(str(path))
        return cls.from_arrays(data["gamma"], data["aspects"], data["sigmas"])


def radial_frequency_bins(
    height: int,
    width: int,
    bins: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a (height, width) radial FFT bin map in [0, bins)."""
    height = int(height)
    width = int(width)
    bins = int(bins)
    if height < 1 or width < 1 or bins < 1:
        raise ValueError("height, width, and bins must be positive")
    fy = torch.fft.fftfreq(height, device=device).view(height, 1)
    fx = torch.fft.fftfreq(width, device=device).view(1, width)
    radius = torch.sqrt(fy.square() + fx.square())
    max_radius = radius.max().clamp_min(1e-12)
    normalized = radius / max_radius
    return torch.clamp((normalized * bins).long(), min=0, max=bins - 1)


class CNSRecolorer:
    """Recolor white noise according to a CNS completion matrix."""

    def __init__(self, calibration: CNSCalibration, *, strength: float = 1.0) -> None:
        self.calibration = calibration
        self.strength = float(max(0.0, min(1.0, strength)))
        self._bin_cache: Dict[Tuple[int, int, int, torch.device], torch.Tensor] = {}
        sigmas = calibration.sigmas[:-1].float()
        self._sigma_order = torch.argsort(sigmas)
        self._sigmas_asc = sigmas[self._sigma_order]

    @classmethod
    def from_npz(cls, path: str | Path, *, strength: float = 1.0) -> "CNSRecolorer":
        return cls(CNSCalibration.from_npz(path), strength=strength)

    @property
    def frequency_bins(self) -> int:
        return int(self.calibration.gamma.shape[-1])

    def _select_aspect_index(self, height: int, width: int) -> int:
        aspects = self.calibration.aspects.float()
        target_ar = float(width) / max(float(height), 1.0)
        aspect_ar = aspects[:, 1] / aspects[:, 0].clamp_min(1.0)
        return int(torch.argmin((aspect_ar - target_ar).abs()).item())

    def _interpolate_gamma(self, aspect_index: int, sigma: float, device: torch.device) -> torch.Tensor:
        gamma = self.calibration.gamma[aspect_index].float()[self._sigma_order]
        sigmas = self._sigmas_asc
        sigma_t = torch.tensor(float(sigma), dtype=torch.float32)
        if sigma_t <= sigmas[0]:
            row = gamma[0]
        elif sigma_t >= sigmas[-1]:
            row = gamma[-1]
        else:
            upper = int(torch.searchsorted(sigmas, sigma_t, right=False).item())
            lower = max(upper - 1, 0)
            span = (sigmas[upper] - sigmas[lower]).clamp_min(1e-12)
            weight = (sigma_t - sigmas[lower]) / span
            row = gamma[lower] * (1.0 - weight) + gamma[upper] * weight
        return row.to(device=device)

    def _bin_map(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (int(height), int(width), self.frequency_bins, device)
        cached = self._bin_cache.get(key)
        if cached is None:
            cached = radial_frequency_bins(height, width, self.frequency_bins, device=device)
            self._bin_cache[key] = cached
        return cached

    @staticmethod
    def _spatial_std(x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=(-2, -1), keepdim=True)
        return centered.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)

    def recolor(self, white: torch.Tensor, *, sigma: float, height: Optional[int] = None, width: Optional[int] = None) -> torch.Tensor:
        """Return colored noise with the same shape, dtype, and variance budget."""
        if self.strength <= 0.0:
            return white
        if white.ndim < 4:
            raise ValueError("CNS white noise must have at least 4 dims ending in (H, W)")

        h = int(height or white.shape[-2])
        w = int(width or white.shape[-1])
        if h != int(white.shape[-2]) or w != int(white.shape[-1]):
            raise ValueError("Explicit CNS height/width must match white noise trailing dimensions")

        work = white.float()
        original_std = self._spatial_std(work)
        aspect_index = self._select_aspect_index(h, w)
        gamma_row = self._interpolate_gamma(aspect_index, float(sigma), work.device).clamp(0.0, 1.0)
        scale_vec = torch.sqrt((1.0 - gamma_row).clamp_min(0.0))
        scale_map = scale_vec[self._bin_map(h, w, work.device)]

        colored_fft = torch.fft.fft2(work, dim=(-2, -1)) * scale_map
        colored = torch.fft.ifft2(colored_fft, dim=(-2, -1)).real
        colored = colored - colored.mean(dim=(-2, -1), keepdim=True)
        colored = colored / self._spatial_std(colored) * original_std

        if self.strength < 1.0:
            mixed = (1.0 - self.strength) * work + self.strength * colored
            mixed = mixed - mixed.mean(dim=(-2, -1), keepdim=True)
            colored = mixed / self._spatial_std(mixed) * original_std

        return colored.to(dtype=white.dtype)


def build_cns_recolorer(
    *,
    gamma_path: str = "",
    strength: float = 1.0,
    calibration: Optional[CNSCalibration] = None,
) -> Optional[CNSRecolorer]:
    """Build a recolorer from an explicit calibration source.

    Empty ``gamma_path`` means CNS is disabled.  The literal ``auto`` is reserved
    for a future bundled calibration resolver and currently raises a clear
    error instead of silently guessing.
    """
    if calibration is not None:
        return CNSRecolorer(calibration, strength=strength)
    gamma_path = str(gamma_path or "").strip()
    if not gamma_path:
        return None
    if gamma_path.lower() == "auto":
        raise FileNotFoundError("CNS auto calibration is not bundled in Lulynx yet")
    return CNSRecolorer.from_npz(gamma_path, strength=strength)
