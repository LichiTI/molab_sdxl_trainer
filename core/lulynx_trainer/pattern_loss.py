# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Pattern Loss — per-frequency-band loss with configurable loss functions.

Decomposes predictions and targets into frequency bands via Haar DWT,
then applies different loss functions to each band. This allows
independent optimization of low-frequency structure and high-frequency detail.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _band_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "l2",
    huber_c: float = 0.1,
) -> torch.Tensor:
    """Compute loss for a single frequency band."""
    if loss_type == "l1":
        return F.l1_loss(pred, target, reduction="none")
    if loss_type in ("huber", "smooth_l1"):
        return F.huber_loss(pred, target, delta=huber_c, reduction="none")
    return F.mse_loss(pred, target, reduction="none")


def _haar_dwt_2d(x: torch.Tensor):
    """Single-level Haar DWT — returns (LL, LH, HL, HH)."""
    # Import from wavelet_loss if available, else inline
    try:
        from .wavelet_loss import _haar_wavelet_decompose_2d
        return _haar_wavelet_decompose_2d(x)
    except ImportError:
        pass
    # Inline fallback: standard Haar wavelet
    x0 = x[:, :, 0::2, :]
    x1 = x[:, :, 1::2, :]
    ll = (x0[:, :, :, 0::2] + x0[:, :, :, 1::2] + x1[:, :, :, 0::2] + x1[:, :, :, 1::2]) * 0.25
    lh = (x0[:, :, :, 0::2] + x0[:, :, :, 1::2] - x1[:, :, :, 0::2] - x1[:, :, :, 1::2]) * 0.25
    hl = (x0[:, :, :, 0::2] - x0[:, :, :, 1::2] + x1[:, :, :, 0::2] - x1[:, :, :, 1::2]) * 0.25
    hh = (x0[:, :, :, 0::2] - x0[:, :, :, 1::2] - x1[:, :, :, 0::2] + x1[:, :, :, 1::2]) * 0.25
    return ll, lh, hl, hh


def pattern_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    levels: int = 1,
    ll_type: str = "l2",
    ll_weight: float = 1.0,
    high_type: str = "huber",
    high_weight: float = 2.0,
    high_huber_c: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """Per-frequency-band loss with configurable loss functions per band.

    Decomposes *pred* and *target* separately, then applies different
    loss functions to the low-frequency (LL) and high-frequency (LH/HL/HH) bands.
    """
    if pred.dim() != 4 or target.dim() != 4:
        return F.mse_loss(pred, target, reduction=reduction)

    total = torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)
    current_pred = pred.float()
    current_target = target.float()
    band_count = 0

    for _ in range(levels):
        _, _, H, W = current_pred.shape
        if H < 2 or W < 2:
            break
        # Ensure even dimensions
        if H % 2 != 0:
            current_pred = current_pred[:, :, :H - 1, :]
            current_target = current_target[:, :, :H - 1, :]
        if W % 2 != 0:
            current_pred = current_pred[:, :, :, :W - 1]
            current_target = current_target[:, :, :, :W - 1]

        pred_ll, pred_lh, pred_hl, pred_hh = _haar_dwt_2d(current_pred)
        tgt_ll, tgt_lh, tgt_hl, tgt_hh = _haar_dwt_2d(current_target)

        for pb, tb in [(pred_lh, tgt_lh), (pred_hl, tgt_hl), (pred_hh, tgt_hh)]:
            bl = _band_loss(pb, tb, high_type, high_huber_c)
            total = total + bl.mean(dim=(1, 2, 3)) * high_weight
            band_count += 1

        bl = _band_loss(pred_ll, tgt_ll, ll_type)
        total = total + bl.mean(dim=(1, 2, 3)) * ll_weight
        band_count += 1

        current_pred = pred_ll
        current_target = tgt_ll

    if band_count == 0:
        return F.mse_loss(pred, target, reduction=reduction)

    total = total / band_count
    if reduction == "mean":
        return total.mean()
    if reduction == "sum":
        return total.sum()
    return total
