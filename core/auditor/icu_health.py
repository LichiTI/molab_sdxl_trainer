# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""ICU Health Indicator — aggregate LoRAAuditor metrics into a 0-100 score."""

from __future__ import annotations

from typing import Optional

from .types import AuditMetrics


def compute_icu_score(
    metrics: AuditMetrics,
    ema_loss: Optional[float] = None,
    prev_ema_loss: Optional[float] = None,
) -> int:
    """Compute a 0-100 health score from existing audit metrics.

    Each dimension contributes a penalty (0-20 points).  The final score is
    ``100 - sum(penalties)``, clamped to [0, 100].  Only non-None metrics
    contribute; missing metrics are treated as healthy (no penalty).
    """
    penalty = 0.0

    if metrics.stable_rank is not None:
        sr = metrics.stable_rank
        if sr < 2.0:
            penalty += min(20.0, (2.0 - sr) * 10.0)
        elif sr > 64.0:
            penalty += min(20.0, (sr - 64.0) * 0.5)

    if metrics.svd_entropy is not None:
        if metrics.svd_entropy < 0.3:
            penalty += min(20.0, (0.3 - metrics.svd_entropy) * 66.7)

    if metrics.dead_neuron_rate is not None:
        if metrics.dead_neuron_rate > 0.05:
            penalty += min(20.0, (metrics.dead_neuron_rate - 0.05) * 200.0)

    if metrics.spectral_smoothness is not None:
        if metrics.spectral_smoothness > 0.5:
            penalty += min(20.0, (metrics.spectral_smoothness - 0.5) * 40.0)

    if metrics.gsnr is not None:
        if metrics.gsnr < 1.0:
            penalty += min(20.0, (1.0 - metrics.gsnr) * 20.0)

    if ema_loss is not None and prev_ema_loss is not None and prev_ema_loss > 0:
        loss_ratio = ema_loss / prev_ema_loss
        if loss_ratio > 1.05:
            penalty += min(15.0, (loss_ratio - 1.05) * 150.0)

    return max(0, min(100, int(round(100.0 - penalty))))
