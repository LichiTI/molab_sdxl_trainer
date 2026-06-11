# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Activation drift tracker via forward hooks.

Captures activation statistics (mean, std, abs_max) on anchor layers
and computes drift relative to a baseline captured on the first step.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LayerStats:
    mean: float
    std: float
    abs_max: float


class ActivationDriftTracker:
    """Track activation drift on selected anchor layers."""

    def __init__(
        self,
        model: nn.Module,
        anchor_layers: Optional[List[str]] = None,
    ) -> None:
        self._model = model
        self._user_anchors = anchor_layers
        self._hooks: list = []
        self._features: Dict[str, torch.Tensor] = {}
        self._baseline: Dict[str, LayerStats] = {}
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        for name, module in self._model.named_modules():
            if name and self._should_track(name):
                handle = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(handle)
        self._installed = True

    def _should_track(self, name: str) -> bool:
        if self._user_anchors:
            return any(anchor in name for anchor in self._user_anchors)
        lower = name.lower()
        if "mid_block" in lower:
            return True
        if "double" in lower:
            for idx in (4, 10):
                if f".{idx}." in name or name.endswith(f".{idx}"):
                    return True
        if "single" in lower and (".0." in name or name.endswith(".0")):
            return True
        return False

    def _make_hook(self, name: str):
        def _hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = output
            if isinstance(output, tuple) and output:
                tensor = output[0]
            if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2:
                self._features[name] = tensor.detach()
        return _hook

    @staticmethod
    def _compute_stats(tensor: torch.Tensor) -> LayerStats:
        with torch.no_grad():
            flat = tensor.float().flatten()
            return LayerStats(
                mean=float(flat.mean().item()),
                std=float(flat.std().item()),
                abs_max=float(flat.abs().max().item()),
            )

    def capture_baseline(self) -> None:
        for name, tensor in self._features.items():
            self._baseline[name] = self._compute_stats(tensor)

    @property
    def has_baseline(self) -> bool:
        return len(self._baseline) > 0

    @property
    def tracked_layers(self) -> List[str]:
        return list(self._baseline.keys())

    def compute_drift(self) -> Dict[str, Dict[str, float]]:
        drift: Dict[str, Dict[str, float]] = {}
        for name, tensor in self._features.items():
            baseline = self._baseline.get(name)
            if baseline is None:
                continue
            current = self._compute_stats(tensor)
            drift[name] = {
                "mean_drift": current.mean - baseline.mean,
                "std_drift": current.std - baseline.std,
                "max_drift": current.abs_max - baseline.abs_max,
                "mean_drift_pct": (
                    (current.mean - baseline.mean) / (abs(baseline.mean) + 1e-8) * 100
                ),
            }
        return drift

    def clear_features(self) -> None:
        self._features.clear()

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._installed = False
