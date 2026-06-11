"""Sliding-mode CFG correction primitive.

SMC-CFG modifies the cond/uncond CFG combine in velocity space.  It is
training-free, stateful across denoise steps, and default-off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class SMCCFGConfig:
    enabled: bool = False
    lam: float = 5.0
    alpha: float = 0.2

    def normalized(self) -> "SMCCFGConfig":
        return SMCCFGConfig(
            enabled=bool(self.enabled),
            lam=max(float(self.lam), 0.0),
            alpha=max(float(self.alpha), 0.0),
        )


class SMCCFGState:
    """Stateful alpha-adaptive sliding-mode CFG combiner."""

    def __init__(self, *, lam: float = 5.0, alpha: float = 0.2) -> None:
        self.lam = max(float(lam), 0.0)
        self.alpha = max(float(alpha), 0.0)
        self._e_prev: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._e_prev = None

    def combine(
        self,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Return SMC-corrected CFG velocity.

        ``cond`` and ``uncond`` are model predictions in the same velocity space.
        With ``alpha=0`` this is exactly standard CFG.
        """
        e = cond - uncond
        e_prev = e if self._e_prev is None else self._e_prev
        sliding_surface = (e - e_prev) + self.lam * e_prev
        gain = self.alpha * e.detach().abs().mean().clamp_min(1e-12)
        correction = -gain * torch.sign(sliding_surface)
        self._e_prev = e.detach()
        return uncond + float(guidance_scale) * (e + correction)


def standard_cfg(cond: torch.Tensor, uncond: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    return uncond + float(guidance_scale) * (cond - uncond)


def build_smc_cfg_state(config: Optional[SMCCFGConfig]) -> Optional[SMCCFGState]:
    if config is None:
        return None
    normalized = config.normalized()
    if not normalized.enabled or normalized.alpha <= 0.0:
        return None
    return SMCCFGState(lam=normalized.lam, alpha=normalized.alpha)
