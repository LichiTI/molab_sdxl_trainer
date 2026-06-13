"""Hardware runtime detection via environment variables.

Probes the current environment to determine which hardware backend
(NVIDIA CUDA, AMD ROCm, Intel XPU) should be used for training launch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareBackend:
    """Detected hardware backend configuration."""

    backend: str
    backend_label: str
    env_source: str


def detect_hardware_backend() -> HardwareBackend:
    """Detect the preferred hardware backend from environment variables.

    Checks (in order):
    1. ``LULYNX_PREFERRED_RUNTIME`` for explicit override
    2. ROCm AMD flags (``LULYNX_ROCM_AMD_STARTUP``, ``LULYNX_AMD_EXPERIMENTAL``)
    3. Intel XPU flags (``LULYNX_INTEL_XPU_STARTUP``, etc.)
    4. Falls back to ``nvidia-cuda``

    Returns a ``HardwareBackend`` with the resolved backend name.
    """
    preferred = os.environ.get("LULYNX_PREFERRED_RUNTIME", "").strip().lower()

    if preferred == "rocm-amd":
        return HardwareBackend("rocm-amd", "AMD ROCm", "LULYNX_PREFERRED_RUNTIME")
    if preferred in {"intel-xpu", "intel-xpu-sage"}:
        return HardwareBackend(preferred, "Intel XPU", "LULYNX_PREFERRED_RUNTIME")

    # AMD ROCm detection
    if os.environ.get("LULYNX_ROCM_AMD_STARTUP", "").strip() == "1":
        return HardwareBackend("rocm-amd", "AMD ROCm", "LULYNX_ROCM_AMD_STARTUP")
    if os.environ.get("LULYNX_AMD_EXPERIMENTAL", "").strip() == "1":
        return HardwareBackend("rocm-amd", "AMD ROCm", "LULYNX_AMD_EXPERIMENTAL")

    # Intel XPU detection
    if os.environ.get("LULYNX_INTEL_XPU_STARTUP", "").strip() == "1":
        return HardwareBackend("intel-xpu", "Intel XPU", "LULYNX_INTEL_XPU_STARTUP")
    if os.environ.get("LULYNX_INTEL_XPU_SAGE_STARTUP", "").strip() == "1":
        return HardwareBackend("intel-xpu-sage", "Intel XPU + Sage", "LULYNX_INTEL_XPU_SAGE_STARTUP")
    if os.environ.get("LULYNX_INTEL_XPU_EXPERIMENTAL", "").strip() == "1":
        return HardwareBackend("intel-xpu", "Intel XPU", "LULYNX_INTEL_XPU_EXPERIMENTAL")
    if os.environ.get("LULYNX_INTEL_XPU_SAGE_EXPERIMENTAL", "").strip() == "1":
        return HardwareBackend("intel-xpu-sage", "Intel XPU + Sage", "LULYNX_INTEL_XPU_SAGE_EXPERIMENTAL")

    return HardwareBackend("nvidia-cuda", "NVIDIA CUDA", "default")
