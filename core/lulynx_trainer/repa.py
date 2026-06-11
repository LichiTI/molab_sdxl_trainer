# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""REPA-style hidden feature alignment primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class REPALossConfig:
    loss_type: str = "cosine"
    weight: float = 1.0
    projection_dim: int = 0
    stop_grad_target: bool = True


@dataclass
class SoftREPAConfig:
    enabled: bool = False
    schedule: str = "linear"
    min_weight: float = 0.0
    max_weight: float = 1.0
    sigma_window: Tuple[float, float] = (0.0, 1.0)


def softrepa_weight(step: int, total_steps: int, sigma: Optional[torch.Tensor], config: SoftREPAConfig) -> float:
    if not config.enabled:
        return 0.0
    sigma_min, sigma_max = config.sigma_window
    if sigma is not None:
        sigma_value = float(sigma.detach().float().mean().cpu())
        if sigma_value < float(sigma_min) or sigma_value > float(sigma_max):
            return 0.0
    progress = 1.0 if total_steps <= 1 else max(0.0, min(float(step) / float(total_steps - 1), 1.0))
    schedule = str(config.schedule or "linear").lower()
    if schedule == "constant":
        scale = 1.0
    elif schedule == "cosine":
        scale = 0.5 - 0.5 * torch.cos(torch.tensor(progress * torch.pi)).item()
    else:
        scale = progress
    return float(config.min_weight) + (float(config.max_weight) - float(config.min_weight)) * float(scale)


class REPAFeatureProjector(nn.Module):
    def __init__(self, hidden_dim: int, projection_dim: int = 0) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.projection_dim = int(projection_dim or hidden_dim)
        if self.projection_dim <= 0 or self.projection_dim == self.hidden_dim:
            self.proj = nn.Identity()
        elif self.hidden_dim <= 0:
            self.proj = nn.LazyLinear(self.projection_dim, bias=False)
        else:
            self.proj = nn.Linear(self.hidden_dim, self.projection_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def repa_alignment_loss(
    hidden: torch.Tensor,
    target: torch.Tensor,
    config: Optional[REPALossConfig] = None,
    projector: Optional[nn.Module] = None,
) -> torch.Tensor:
    cfg = config or REPALossConfig()
    if cfg.stop_grad_target:
        target = target.detach()
    if projector is not None:
        hidden = projector(hidden)
    if hidden.shape != target.shape:
        raise ValueError(f"REPA hidden/target shape mismatch: {tuple(hidden.shape)} vs {tuple(target.shape)}")
    loss_type = str(cfg.loss_type or "cosine").lower()
    if loss_type == "l2":
        loss = F.mse_loss(hidden, target)
    elif loss_type == "l1":
        loss = F.l1_loss(hidden, target)
    else:
        hidden_f = hidden.flatten(0, -2)
        target_f = target.flatten(0, -2)
        loss = 1.0 - F.cosine_similarity(hidden_f.float(), target_f.float(), dim=-1).mean()
    return loss * float(cfg.weight)


class REPAFeatureCapture:
    def __init__(self, model: nn.Module, target_modules: Iterable[str]) -> None:
        self.model = model
        self.target_modules = list(target_modules)
        self.features: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def install(self) -> "REPAFeatureCapture":
        modules = dict(self.model.named_modules())
        for name in self.target_modules:
            module = modules.get(name)
            if module is None:
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def clear(self) -> None:
        self.features.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()

    def _make_hook(self, name: str):
        def _hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) and output else output
            if isinstance(value, torch.Tensor):
                self.features[name] = value
        return _hook
