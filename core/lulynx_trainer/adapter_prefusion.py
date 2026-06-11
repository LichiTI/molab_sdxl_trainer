# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Adapter Pre-Fusion — merge an existing LoRA into base model weights.

Before injecting a *new* LoRA for training, this module loads an existing
LoRA checkpoint and fuses its delta weights (up @ down * scale) directly
into the base model's linear layers.  The result is a model that
"remembers" the old LoRA's knowledge as its starting point, letting the
new training build on top of it.

Integration: called in trainer.py after base model load but before
network_weights_path loading and new LoRA injection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["prefuse_adapter_into_model"]

logger = logging.getLogger(__name__)


def prefuse_adapter_into_model(
    model: nn.Module,
    adapter_path: str,
    scale: float = 1.0,
) -> int:
    """Merge a LoRA checkpoint into base model weights in-place.

    Parameters
    ----------
    model : the base model (UNet or full pipeline wrapper).
    adapter_path : path to a safetensors or .pt LoRA checkpoint.
    scale : multiplier for the LoRA delta (1.0 = full strength).

    Returns
    -------
    Number of layers successfully fused.
    """
    state_dict = _load_adapter_state_dict(adapter_path)
    if not state_dict:
        logger.warning("Empty or unreadable adapter at %s", adapter_path)
        return 0

    pairs = _extract_lora_pairs(state_dict)
    if not pairs:
        logger.warning("No lora_down/lora_up pairs found in %s", adapter_path)
        return 0

    unet = getattr(model, "unet", model)
    named_modules = dict(unet.named_modules())

    fused = 0
    for module_key, (down_w, up_w, alpha_val) in pairs.items():
        target = _find_module(named_modules, module_key)
        if target is None or not isinstance(target, nn.Linear):
            continue

        rank = down_w.shape[0]
        lora_scale = (alpha_val / rank) * scale if alpha_val else scale
        delta = (up_w.float() @ down_w.float()) * lora_scale
        target.weight.data.add_(delta.to(target.weight.device, target.weight.dtype))
        fused += 1

    logger.info("Pre-fused %d/%d adapter layers from %s (scale=%.2f)", fused, len(pairs), adapter_path, scale)
    return fused


def _load_adapter_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load a LoRA state dict from safetensors or .pt file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Adapter file not found: {path}")

    if p.suffix in (".safetensors",):
        try:
            from safetensors.torch import load_file
            return load_file(str(p))
        except ImportError:
            logger.warning("safetensors not available, falling back to torch.load")

    return torch.load(str(p), map_location="cpu", weights_only=True)


def _extract_lora_pairs(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[float]]]:
    """Parse lora_down/lora_up weight pairs from state dict keys.

    Handles common naming conventions:
      - ``module.lora_down.weight`` / ``module.lora_up.weight``
      - ``module.lora_A.weight`` / ``module.lora_B.weight``
    """
    pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[float]]] = {}

    for key in state_dict:
        if "weight" not in key:
            continue

        down_key = None
        up_key = None
        base_key = None

        if "lora_down" in key:
            down_key = key
            up_key = key.replace("lora_down", "lora_up")
            base_key = key.replace(".lora_down.weight", "")
        elif "lora_A" in key:
            down_key = key
            up_key = key.replace("lora_A", "lora_B")
            base_key = key.replace(".lora_A.weight", "")

        if down_key is None or up_key not in state_dict:
            continue
        if base_key in pairs:
            continue

        alpha = None
        for suffix in (".alpha", ".lora_alpha"):
            alpha_key = base_key + suffix
            if alpha_key in state_dict:
                alpha = float(state_dict[alpha_key].item())
                break

        pairs[base_key] = (state_dict[down_key], state_dict[up_key], alpha)

    return pairs


def _find_module(
    named_modules: Dict[str, nn.Module],
    key: str,
) -> Optional[nn.Module]:
    """Find a module by a LoRA state dict key, trying common naming transforms."""
    if key in named_modules:
        return named_modules[key]

    dotted = key.replace("_", ".")
    if dotted in named_modules:
        return named_modules[dotted]

    parts = key.split(".")
    for prefix in ("lora_unet_", "lora_te1_", "lora_te2_", "lora_te_"):
        if key.startswith(prefix):
            trimmed = key[len(prefix):].replace("_", ".")
            if trimmed in named_modules:
                return named_modules[trimmed]

    return None
