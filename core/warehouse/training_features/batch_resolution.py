"""Batch resolution utilities for distributed training.

Provides validation and computation of per-device batch sizes,
gradient accumulation steps, and effective batch sizes.

This module is a Warehouse design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BatchResolution:
    """Resolved batch configuration."""
    per_device_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    effective_batch_size: int
    global_batch_size: int
    warnings: tuple[str, ...] = ()


def resolve_batch_config(
    global_batch_size: int,
    world_size: int = 1,
    gradient_accumulation_steps: int = 1,
    max_per_device: int = 0,
) -> BatchResolution:
    """Compute per-device batch size from a global target.

    Parameters
    ----------
    global_batch_size:
        Total samples per optimization step.
    world_size:
        Number of distributed processes.
    gradient_accumulation_steps:
        Micro-batch accumulations per step.
    max_per_device:
        If >0, cap per-device batch size at this value.

    Returns
    -------
    BatchResolution with computed values and any warnings.
    """
    warnings: list[str] = []

    if global_batch_size < 1:
        warnings.append("global_batch_size < 1, clamping to 1")
        global_batch_size = 1
    if world_size < 1:
        world_size = 1
    if gradient_accumulation_steps < 1:
        gradient_accumulation_steps = 1

    effective = world_size * gradient_accumulation_steps
    per_device = global_batch_size // effective

    if per_device < 1:
        per_device = 1
        warnings.append(
            f"global_batch_size ({global_batch_size}) < world_size * accum "
            f"({effective}); per_device clamped to 1"
        )

    if max_per_device > 0 and per_device > max_per_device:
        warnings.append(
            f"per_device ({per_device}) exceeds max ({max_per_device}); "
            f"consider increasing gradient_accumulation_steps"
        )

    actual_effective = per_device * effective
    if actual_effective != global_batch_size:
        warnings.append(
            f"effective batch ({actual_effective}) differs from "
            f"requested ({global_batch_size})"
        )

    if gradient_accumulation_steps > 128:
        warnings.append(
            f"gradient_accumulation_steps={gradient_accumulation_steps} "
            f"is very high; training may be unstable"
        )

    return BatchResolution(
        per_device_batch_size=per_device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        world_size=world_size,
        effective_batch_size=actual_effective,
        global_batch_size=global_batch_size,
        warnings=tuple(warnings),
    )


def validate_batch_config(
    per_device: int,
    accumulation: int,
    world_size: int,
    *,
    min_effective: int = 1,
    max_effective: int = 0,
) -> list[str]:
    """Validate a batch configuration and return error messages."""
    errors: list[str] = []
    if per_device < 1:
        errors.append("per_device_batch_size must be >= 1")
    if accumulation < 1:
        errors.append("gradient_accumulation_steps must be >= 1")
    if world_size < 1:
        errors.append("world_size must be >= 1")
    if errors:
        return errors
    effective = per_device * accumulation * world_size
    if effective < min_effective:
        errors.append(
            f"effective batch ({effective}) < minimum ({min_effective})"
        )
    if max_effective > 0 and effective > max_effective:
        errors.append(
            f"effective batch ({effective}) > maximum ({max_effective})"
        )
    return errors

