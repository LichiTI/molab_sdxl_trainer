# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Load and merge base LoRA weights before training resumes or diff-trains."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

from .dim_from_weights import _load_state_dict


@dataclass(frozen=True)
class BaseLoRAWeightResolution:
    requested_paths: tuple[str, ...]
    multipliers: tuple[float, ...]
    loaded_paths: tuple[str, ...]
    loaded_keys: int
    expected_keys: int
    mode: str
    applied: bool


def parse_base_lora_weight_request(config: Any) -> tuple[list[str], list[float]]:
    """Parse comma-separated base LoRA paths and multiplier values."""
    base_weight_path = str(getattr(config, "base_weight_path", "") or "").strip()
    paths = [part.strip() for part in base_weight_path.split(",") if part.strip()]
    if not paths:
        return [], []

    raw_multipliers = str(getattr(config, "base_weights_multiplier", "") or "").strip()
    multipliers: list[float] = []
    if raw_multipliers:
        for part in raw_multipliers.split(","):
            value = part.strip()
            if value:
                multipliers.append(float(value))

    if not multipliers:
        multipliers = [1.0] * len(paths)
    elif len(multipliers) < len(paths):
        multipliers.extend([1.0] * (len(paths) - len(multipliers)))
    else:
        multipliers = multipliers[: len(paths)]
    return paths, multipliers


def merge_base_lora_state_dicts(
    paths: list[str],
    multipliers: list[float],
    *,
    log_fn: Optional[Callable[[str], None]] = None,
) -> tuple[Dict[str, torch.Tensor], tuple[str, ...]]:
    """Load base LoRA files and sum matching tensors with per-file multipliers."""
    merged_state: Dict[str, torch.Tensor] = {}
    loaded_paths: list[str] = []
    for path_str, multiplier in zip(paths, multipliers):
        base_path = Path(path_str)
        if not base_path.is_file():
            if log_fn is not None:
                log_fn(f"Base weight path {path_str} not found, skipping")
            continue

        state_dict = _load_state_dict(str(base_path))
        if state_dict is None:
            if log_fn is not None:
                log_fn(f"Failed to load base weights from {path_str}, skipping")
            continue

        loaded_paths.append(path_str)
        if log_fn is not None:
            log_fn(f"Loading base LoRA from {path_str} (multiplier={multiplier})")

        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            scaled = value * float(multiplier)
            if key in merged_state:
                merged_state[key] = merged_state[key] + scaled
            else:
                merged_state[key] = scaled

    return merged_state, tuple(loaded_paths)


def load_base_lora_weights(
    config: Any,
    injector: Any,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
) -> BaseLoRAWeightResolution:
    """Apply configured base LoRA weights to an already-created injector."""
    paths, multipliers = parse_base_lora_weight_request(config)
    if not paths:
        return BaseLoRAWeightResolution((), (), (), 0, 0, "off", False)

    if len(paths) == 1 and len(multipliers) == 1 and multipliers[0] == 1.0:
        base_path = Path(paths[0])
        if base_path.is_file() and hasattr(injector, "load_lora"):
            if log_fn is not None:
                log_fn(f"Loading base LoRA weights from {base_path}")
            injector.load_lora(str(base_path))
            return BaseLoRAWeightResolution(tuple(paths), tuple(multipliers), tuple(paths), 0, 0, "single", True)
        if log_fn is not None:
            log_fn(f"Base weight path {paths[0]} not found or injector has no load_lora, skipping")
        return BaseLoRAWeightResolution(tuple(paths), tuple(multipliers), (), 0, 0, "single", False)

    merged_state, loaded_paths = merge_base_lora_state_dicts(paths, multipliers, log_fn=log_fn)
    if merged_state and hasattr(injector, "load_lora_state_dict"):
        loaded_keys, expected_keys = injector.load_lora_state_dict(merged_state)
        if log_fn is not None:
            log_fn(f"Merged {len(loaded_paths)} base weight files")
        return BaseLoRAWeightResolution(
            tuple(paths),
            tuple(multipliers),
            loaded_paths,
            int(loaded_keys),
            int(expected_keys),
            "merged",
            True,
        )

    return BaseLoRAWeightResolution(tuple(paths), tuple(multipliers), loaded_paths, 0, 0, "merged", False)
