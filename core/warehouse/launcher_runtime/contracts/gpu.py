"""GPU information contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    """Description of a single detected GPU."""

    name: str
    vendor: str  # "nvidia" | "intel" | "amd" | "unknown"
    driver_version: str = ""
    vram_mb: int = 0


@dataclass(frozen=True)
class GpuStats:
    """Live GPU utilization snapshot."""

    gpu_util_percent: int = 0
    mem_used_mb: int = 0
    mem_total_mb: int = 0
    temperature_c: int = 0
