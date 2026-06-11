"""Wavelet-based frequency-aware loss for diffusion training.

Decomposes prediction error into frequency sub-bands using a Haar DWT and
computes a weighted loss that can emphasize high-frequency details.  This
helps the model learn fine textures and sharp edges that pure L2/L1 loss
tends to smooth out.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional


def _haar_wavelet_decompose_2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-level 2D Haar DWT.

    Parameters
    ----------
    x : torch.Tensor
        Shape (B, C, H, W).  H and W must be even.

    Returns
    -------
    (LL, LH, HL, HH) : tuple of tensors, each shape (B, C, H/2, W/2).
    """
    # Split into quadrants via shift-and-diff
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x00 = x[:, :, 0::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = (x00 + x10 + x01 + x11) * 0.5
    hl = (x00 - x10 + x01 - x11) * 0.5
    lh = (x00 + x10 - x01 - x11) * 0.5
    hh = (x00 - x10 - x01 + x11) * 0.5
    return ll, lh, hl, hh


def wavelet_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    levels: int = 2,
    high_freq_weight: float = 2.0,
    approx_weight: float = 0.0,
    base_loss: str = "l2",
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute wavelet-frequency-aware loss.

    Parameters
    ----------
    pred : torch.Tensor
        Model prediction, shape (B, C, H, W).
    target : torch.Tensor
        Ground truth, shape (B, C, H, W).
    levels : int
        Number of DWT decomposition levels.
    high_freq_weight : float
        Weight multiplier for high-frequency sub-bands (LH, HL, HH).
        1.0 = standard loss (no frequency emphasis).
    approx_weight : float
        Optional extra weight for the final low-frequency LL approximation.
    base_loss : str
        Base loss function: ``"l2"`` or ``"l1"``.
    reduction : str
        ``"mean"`` or ``"none"``.

    Returns
    -------
    torch.Tensor
        Scalar loss (or per-sample if reduction="none").
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

    residual = pred - target

    loss_fn = F.mse_loss if base_loss == "l2" else F.l1_loss

    if reduction not in {"mean", "none"}:
        raise ValueError(f"Unsupported reduction={reduction!r}; expected 'mean' or 'none'")

    def _band_reduce(loss: torch.Tensor) -> torch.Tensor:
        if reduction == "mean":
            return loss.mean()
        return loss.flatten(1).mean(dim=1)

    total_loss: torch.Tensor | None = None
    current = residual
    used_levels = 0

    for _ in range(levels):
        h, w = current.shape[-2], current.shape[-1]
        if h < 2 or w < 2:
            break
        ll, lh, hl, hh = _haar_wavelet_decompose_2d(current)
        band_loss = (
            _band_reduce(loss_fn(lh, torch.zeros_like(lh), reduction="none"))
            + _band_reduce(loss_fn(hl, torch.zeros_like(hl), reduction="none"))
            + _band_reduce(loss_fn(hh, torch.zeros_like(hh), reduction="none"))
        ) * high_freq_weight
        total_loss = band_loss if total_loss is None else total_loss + band_loss
        current = ll
        used_levels += 1

    if used_levels == 0:
        base = loss_fn(residual, torch.zeros_like(residual), reduction="none")
        return _band_reduce(base)

    if approx_weight > 0:
        approx_loss = _band_reduce(loss_fn(current, torch.zeros_like(current), reduction="none")) * approx_weight
        total_loss = approx_loss if total_loss is None else total_loss + approx_loss

    if total_loss is None:
        base = loss_fn(residual, torch.zeros_like(residual), reduction="none")
        return _band_reduce(base)
    return total_loss
