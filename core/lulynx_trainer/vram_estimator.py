# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Pre-training VRAM breakdown estimator.

Walks model components (UNet/DiT, VAE, text encoders, LoRA adapters) and
estimates per-component GPU memory usage based on parameter counts, dtypes,
optimizer state multipliers, and activation heuristics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_BYTES_PER_MB = 1024 * 1024


@dataclass
class ComponentVRAM:
    name: str
    params_mb: float
    gradients_mb: float
    optimizer_states_mb: float

    @property
    def total_mb(self) -> float:
        return self.params_mb + self.gradients_mb + self.optimizer_states_mb

    def as_dict(self) -> Dict[str, float]:
        return {
            "name": self.name,
            "params_mb": round(self.params_mb, 1),
            "gradients_mb": round(self.gradients_mb, 1),
            "optimizer_states_mb": round(self.optimizer_states_mb, 1),
            "total_mb": round(self.total_mb, 1),
        }


@dataclass
class VRAMBreakdown:
    components: list
    activation_estimate_mb: float
    total_estimate_mb: float
    available_mb: float
    safety_rating: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "components": [c.as_dict() for c in self.components],
            "activation_estimate_mb": round(self.activation_estimate_mb, 1),
            "total_estimate_mb": round(self.total_estimate_mb, 1),
            "available_mb": round(self.available_mb, 1),
            "safety_rating": self.safety_rating,
        }

    def summary_lines(self) -> list:
        lines = []
        for c in self.components:
            lines.append(
                f"  {c.name}: {c.total_mb:.0f} MB "
                f"(params={c.params_mb:.0f} grad={c.gradients_mb:.0f} "
                f"opt={c.optimizer_states_mb:.0f})"
            )
        lines.append(f"  Activations (est.): {self.activation_estimate_mb:.0f} MB")
        lines.append(f"  Total estimate: {self.total_estimate_mb:.0f} MB")
        lines.append(f"  Available VRAM: {self.available_mb:.0f} MB")
        lines.append(f"  Safety: {self.safety_rating}")
        return lines


def _module_param_bytes(module: nn.Module, trainable_only: bool = False) -> int:
    total = 0
    for p in module.parameters():
        if trainable_only and not p.requires_grad:
            continue
        total += p.numel() * p.element_size()
    return total


def _params_bytes_from_iter(params: Iterator[nn.Parameter]) -> int:
    total = 0
    for p in params:
        total += p.numel() * p.element_size()
    return total


def _normalise_optimizer_name(optimizer_type: str) -> str:
    return str(optimizer_type or "").strip().lower().replace("-", "_").replace(" ", "")


def _optimizer_state_multiplier(optimizer_type: str) -> float:
    """Return optimizer-state memory multiplier relative to trainable params.

    These are planning estimates for the implementations wired by the trainer,
    not generic optimizer-family labels. ``prodigyopt.Prodigy`` keeps full-size
    first/second moments plus Prodigy p0/s state. ``ProdigyPlusScheduleFree``
    defaults to factored second moments, so large matrix-like LoRA tensors are
    much closer to ~1.2x in practice.
    """
    opt = _normalise_optimizer_name(optimizer_type)
    if opt in ("sgd", "sgd_nesterov"):
        return 0.0
    if opt in ("adafactor",):
        return 1.0
    if opt in {
        "prodigyplus.prodigyplusschedulefree",
        "prodigyplusschedulefree",
        "prodigyschedulefree",
        "prodigy_schedule_free",
    }:
        return 1.25
    if opt in ("prodigy", "prodigyopt.prodigy"):
        return 4.0
    # Adam, AdamW, AdamW8bit, etc. store m + v.
    return 2.0


def _estimate_activation_mb(
    resolution: int,
    batch_size: int,
    model_channels: int = 320,
    gradient_checkpointing: bool = False,
) -> float:
    megapixels = (resolution / 1024.0) ** 2
    base_mb = megapixels * batch_size * model_channels * 4 / _BYTES_PER_MB * 1024
    if gradient_checkpointing:
        base_mb *= 0.3
    return base_mb


def _component_vram(
    name: str,
    module: Optional[nn.Module],
    optimizer_multiplier: float,
) -> ComponentVRAM:
    if module is None:
        return ComponentVRAM(name=name, params_mb=0.0, gradients_mb=0.0, optimizer_states_mb=0.0)

    all_bytes = _module_param_bytes(module, trainable_only=False)
    trainable_bytes = _module_param_bytes(module, trainable_only=True)

    params_mb = all_bytes / _BYTES_PER_MB
    gradients_mb = trainable_bytes / _BYTES_PER_MB
    optimizer_states_mb = trainable_bytes * optimizer_multiplier / _BYTES_PER_MB

    return ComponentVRAM(
        name=name,
        params_mb=params_mb,
        gradients_mb=gradients_mb,
        optimizer_states_mb=optimizer_states_mb,
    )


def estimate_vram_breakdown(
    model: Any,
    config: Any,
    lora_params: Optional[Iterator[nn.Parameter]] = None,
) -> VRAMBreakdown:
    """Estimate per-component VRAM usage from loaded model and config."""
    opt_type = str(getattr(config, "optimizer_type", "adamw"))
    opt_mult = _optimizer_state_multiplier(opt_type)

    components = []

    unet = getattr(model, "unet", None)
    components.append(_component_vram("UNet/DiT", unet, opt_mult))

    vae = getattr(model, "vae", None)
    components.append(_component_vram("VAE", vae, 0.0))

    te1 = getattr(model, "text_encoder_1", None)
    components.append(_component_vram("TextEncoder1", te1, opt_mult))

    te2 = getattr(model, "text_encoder_2", None)
    if te2 is not None:
        components.append(_component_vram("TextEncoder2", te2, opt_mult))

    if lora_params is not None:
        lora_list = list(lora_params)
        if lora_list:
            lora_bytes = _params_bytes_from_iter(iter(lora_list))
            lora_mb = lora_bytes / _BYTES_PER_MB
            components.append(ComponentVRAM(
                name="LoRA Adapter",
                params_mb=lora_mb,
                gradients_mb=lora_mb,
                optimizer_states_mb=lora_mb * opt_mult,
            ))

    resolution = int(getattr(config, "resolution", 1024) or 1024)
    batch_size = int(getattr(config, "train_batch_size", 1) or 1)
    gc = bool(getattr(config, "gradient_checkpointing", False))
    activation_mb = _estimate_activation_mb(resolution, batch_size, gradient_checkpointing=gc)

    component_total = sum(c.total_mb for c in components)
    total = component_total + activation_mb

    available_mb = 0.0
    if torch.cuda.is_available():
        try:
            available_mb = torch.cuda.get_device_properties(0).total_mem / _BYTES_PER_MB
        except Exception:
            pass

    utilization = total / available_mb if available_mb > 0 else 1.0
    if utilization < 0.7:
        safety = "safe"
    elif utilization < 0.85:
        safety = "watch"
    elif utilization < 0.95:
        safety = "tight"
    else:
        safety = "danger"

    return VRAMBreakdown(
        components=components,
        activation_estimate_mb=activation_mb,
        total_estimate_mb=total,
        available_mb=available_mb,
        safety_rating=safety,
    )
