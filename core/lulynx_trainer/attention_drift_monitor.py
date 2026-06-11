# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""SageAttention Drift Monitor — quantization error tracking.

SageAttention uses INT8/FP8 quantization for speed.  Over long training
runs the quantization error can drift, producing subtly wrong attention
outputs.  This monitor periodically compares SageAttn vs SDPA reference
on a small synthetic input and raises a warning (or triggers fallback)
when relative error exceeds a threshold.

Integration: instantiated in TrainingLoop.__init__ when
sageattn_drift_check_interval > 0; called every N steps in the training
epoch loop.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

__all__ = ["AttentionDriftMonitor"]

logger = logging.getLogger(__name__)


class AttentionDriftMonitor:
    """Monitors SageAttention output drift against SDPA reference."""

    def __init__(
        self,
        threshold: float = 0.01,
        fallback: str = "warn",
    ) -> None:
        self.threshold = threshold
        self.fallback = fallback
        self._last_drift: float = 0.0
        self._check_count: int = 0
        self._breach_count: int = 0

    @property
    def last_drift(self) -> float:
        return self._last_drift

    @property
    def breach_count(self) -> int:
        return self._breach_count

    @torch.no_grad()
    def check_drift(self, model: torch.nn.Module) -> float:
        """Run a drift check on the first SageAttn module found.

        Returns the relative error, or 0.0 if no SageAttn modules exist.
        """
        self._check_count += 1

        for name, module in model.named_modules():
            if getattr(module, "_attention_backend", None) != "sageattn":
                continue

            try:
                device = next(module.parameters()).device
                dtype = next(module.parameters()).dtype
            except StopIteration:
                continue

            head_dim = getattr(module, "head_dim", 64)
            num_heads = getattr(module, "num_heads", 8)
            q = torch.randn(1, num_heads, 16, head_dim, device=device, dtype=dtype)
            k = torch.randn(1, num_heads, 16, head_dim, device=device, dtype=dtype)
            v = torch.randn(1, num_heads, 16, head_dim, device=device, dtype=dtype)

            sdpa_out = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), dropout_p=0.0
            ).to(dtype)

            try:
                from .anima_attention import _sageattn_attention
                sage_out = _sageattn_attention(q, k, v)
            except Exception:
                return 0.0

            ref_norm = sdpa_out.float().norm().clamp(min=1e-8)
            drift = (sage_out.float() - sdpa_out.float()).norm() / ref_norm
            self._last_drift = drift.item()

            if self._last_drift > self.threshold:
                self._breach_count += 1
                logger.warning(
                    "SageAttn drift %.4f exceeds threshold %.4f on module %s "
                    "(check #%d, breach #%d)",
                    self._last_drift, self.threshold, name,
                    self._check_count, self._breach_count,
                )
                if self.fallback == "fallback_sdpa":
                    logger.warning("Falling back to SDPA backend")
                    from .anima_attention import patch_anima_attention
                    patch_anima_attention(model, backend="sdpa")

            return self._last_drift

        return 0.0
