"""Frozen compression companion helpers.

The first supported mode is conservative: load a LoRA-style companion into the
already-injected adapter slots, merge its delta into each wrapped base Linear,
then reset the adapter slots so normal training can proceed with fresh trainable
adapter weights.  This gives compression a compensated base without adding a
second trainable adapter stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class CompressionCompanionResult:
    enabled: bool
    path: str = ""
    type: str = "lora"
    mode: str = "merge_into_base"
    scale: float = 1.0
    merged_layers: int = 0
    skipped_layers: int = 0
    reset_layers: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": self.path,
            "type": self.type,
            "mode": self.mode,
            "scale": self.scale,
            "merged_layers": self.merged_layers,
            "skipped_layers": self.skipped_layers,
            "reset_layers": self.reset_layers,
            "warnings": list(self.warnings),
        }


def normalize_compression_companion_type(value: Any) -> str:
    item = str(value or "lora").strip().lower().replace("-", "_")
    aliases = {"lycoris": "lora", "locon": "lora", "loha": "lora", "lokr": "lora"}
    return aliases.get(item, item)


def normalize_compression_companion_mode(value: Any) -> str:
    item = str(value or "merge_into_base").strip().lower().replace("-", "_")
    aliases = {"merge": "merge_into_base", "bake": "merge_into_base", "bake_into_base": "merge_into_base"}
    return aliases.get(item, item)


def _reset_lora_adapter(adapter: nn.Module) -> bool:
    down = getattr(adapter, "lora_down", None)
    up = getattr(adapter, "lora_up", None)
    if not isinstance(down, nn.Linear) or not isinstance(up, nn.Linear):
        return False
    nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
    nn.init.zeros_(up.weight)
    for param in adapter.parameters():
        param.requires_grad = True
    return True


def _merge_lora_layer_into_base(layer: nn.Module, *, scale: float) -> bool:
    original = getattr(layer, "original", None)
    adapter = getattr(layer, "lora", None)
    if not isinstance(original, nn.Linear) or adapter is None:
        return False
    down = getattr(adapter, "lora_down", None)
    up = getattr(adapter, "lora_up", None)
    if not isinstance(down, nn.Linear) or not isinstance(up, nn.Linear):
        return False
    if down.weight.ndim != 2 or up.weight.ndim != 2:
        return False
    delta = up.weight.detach().float() @ down.weight.detach().float()
    scaling = float(getattr(adapter, "scaling", 1.0) or 1.0) * float(scale)
    if delta.shape != original.weight.shape:
        return False
    with torch.no_grad():
        merged = original.weight.detach().float() + delta * scaling
        original.weight.data.copy_(merged.to(device=original.weight.device, dtype=original.weight.dtype))
    return True


def apply_compression_companion(
    lora_injector: Any,
    *,
    path: str,
    companion_type: str = "lora",
    mode: str = "merge_into_base",
    scale: float = 1.0,
    disable_mmap: bool = False,
) -> CompressionCompanionResult:
    ctype = normalize_compression_companion_type(companion_type)
    cmode = normalize_compression_companion_mode(mode)
    result = CompressionCompanionResult(
        enabled=True,
        path=str(path or ""),
        type=ctype,
        mode=cmode,
        scale=float(scale or 1.0),
    )
    if lora_injector is None:
        result.warnings.append("compression companion skipped: no adapter injector is active")
        return result
    if ctype != "lora":
        result.warnings.append(f"compression companion type {ctype!r} is not supported yet")
        return result
    if cmode != "merge_into_base":
        result.warnings.append(f"compression companion mode {cmode!r} is not supported yet")
        return result
    companion_path = Path(result.path)
    if not companion_path.is_file():
        result.warnings.append(f"compression companion file not found: {result.path}")
        return result
    load_lora = getattr(lora_injector, "load_lora", None)
    if not callable(load_lora):
        result.warnings.append("compression companion skipped: injector cannot load LoRA weights")
        return result

    try:
        load_lora(str(companion_path), disable_mmap=disable_mmap)
    except TypeError:
        load_lora(str(companion_path))
    except Exception as exc:
        result.warnings.append(f"compression companion load failed: {exc}")
        return result

    injected = getattr(lora_injector, "injected_layers", {})
    if not isinstance(injected, dict) or not injected:
        result.warnings.append("compression companion skipped: no injected adapter layers found")
        return result

    for layer in injected.values():
        if _merge_lora_layer_into_base(layer, scale=result.scale):
            result.merged_layers += 1
        else:
            result.skipped_layers += 1
        adapter = getattr(layer, "lora", layer)
        if isinstance(adapter, nn.Module) and _reset_lora_adapter(adapter):
            result.reset_layers += 1

    if result.merged_layers == 0:
        result.warnings.append("compression companion did not match any mergeable standard LoRA Linear layers")
    return result


__all__ = [
    "CompressionCompanionResult",
    "apply_compression_companion",
    "normalize_compression_companion_mode",
    "normalize_compression_companion_type",
]
