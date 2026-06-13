"""Runtime environment detection and memory configuration.

Provides hardware accelerator detection, attention-backend probing,
and memory allocator configuration for the lulynx training system.

This module is a Warehouse design.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from typing import Any


class VendorTag(str, enum.Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    UNKNOWN = "unknown"


class AttentionBackendTag(str, enum.Enum):
    SDPA = "sdpa"
    XFORMERS = "xformers"
    FLASH_ATTN = "flash_attn"
    SAGE_ATTN = "sageattn"
    ROCM = "rocm"
    XPU = "xpu"


@dataclass(frozen=True)
class AcceleratorInfo:
    """Detected hardware accelerator."""
    vendor: VendorTag
    device_name: str
    vram_mb: int = 0


@dataclass(frozen=True)
class MemoryConfig:
    """Memory allocator settings."""
    cuda_expandable_segments: bool = False
    cuda_max_split_size_mb: int = 0
    pytorch_cuda_alloc_conf: str = ""

    def as_env_vars(self) -> dict[str, str]:
        """Return environment variables to set for memory configuration."""
        env: dict[str, str] = {}
        if self.cuda_expandable_segments:
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        elif self.cuda_max_split_size_mb > 0:
            env["PYTORCH_CUDA_ALLOC_CONF"] = (
                f"max_split_size_mb:{self.cuda_max_split_size_mb}"
            )
        elif self.pytorch_cuda_alloc_conf:
            env["PYTORCH_CUDA_ALLOC_CONF"] = self.pytorch_cuda_alloc_conf
        return env


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Complete point-in-time runtime environment snapshot."""
    accelerators: tuple[AcceleratorInfo, ...] = ()
    available_backends: tuple[AttentionBackendTag, ...] = ()
    preferred_backend: AttentionBackendTag | None = None
    memory: MemoryConfig = MemoryConfig()
    environment: dict[str, str] = field(default_factory=dict)


def _detect_vendor(device_name: str) -> VendorTag:
    lower = device_name.lower()
    if "nvidia" in lower or "geforce" in lower or "rtx" in lower or "gtx" in lower:
        return VendorTag.NVIDIA
    if "amd" in lower or "radeon" in lower or "mi" in lower:
        return VendorTag.AMD
    if "intel" in lower or "arc" in lower or "xpu" in lower:
        return VendorTag.INTEL
    return VendorTag.UNKNOWN


def _probe_backends(vendor: VendorTag) -> list[AttentionBackendTag]:
    backends: list[AttentionBackendTag] = [AttentionBackendTag.SDPA]
    if vendor == VendorTag.AMD:
        backends.append(AttentionBackendTag.ROCM)
    if vendor == VendorTag.INTEL:
        backends.append(AttentionBackendTag.XPU)
    try:
        import xformers  # noqa: F401
        backends.append(AttentionBackendTag.XFORMERS)
    except ImportError:
        pass
    try:
        import flash_attn  # noqa: F401
        backends.append(AttentionBackendTag.FLASH_ATTN)
    except ImportError:
        pass
    try:
        import sageattention  # noqa: F401
        backends.append(AttentionBackendTag.SAGE_ATTN)
    except ImportError:
        pass
    return backends


def _resolve_preferred(
    backends: list[AttentionBackendTag],
    vendor: VendorTag,
) -> AttentionBackendTag | None:
    env_pref = os.environ.get("LULYNX_PREFERRED_BACKEND", "").strip().lower()
    if env_pref:
        for b in backends:
            if b.value == env_pref:
                return b
    if AttentionBackendTag.XFORMERS in backends:
        return AttentionBackendTag.XFORMERS
    if vendor == VendorTag.AMD and AttentionBackendTag.ROCM in backends:
        return AttentionBackendTag.ROCM
    if vendor == VendorTag.INTEL and AttentionBackendTag.XPU in backends:
        return AttentionBackendTag.XPU
    return AttentionBackendTag.SDPA


def _read_memory_config(low_vram: bool = False) -> MemoryConfig:
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    expandable = "expandable_segments:True" in alloc_conf
    max_split = 0
    if "max_split_size_mb:" in alloc_conf:
        try:
            part = alloc_conf.split("max_split_size_mb:")[1].split(",")[0]
            max_split = int(part.strip())
        except (IndexError, ValueError):
            pass
    if low_vram and not alloc_conf:
        expandable = True
    return MemoryConfig(
        cuda_expandable_segments=expandable,
        cuda_max_split_size_mb=max_split,
        pytorch_cuda_alloc_conf=alloc_conf,
    )


class RuntimeDetector:
    """Detect runtime hardware, backends, and memory configuration.

    Example::

        det = RuntimeDetector()
        snap = det.detect()
        print(snap.preferred_backend)
    """

    def __init__(
        self,
        *,
        device_names: list[str] | None = None,
        low_vram: bool = False,
    ) -> None:
        self._device_names = device_names or []
        self._low_vram = low_vram

    def detect(self) -> RuntimeSnapshot:
        accels: list[AcceleratorInfo] = []
        for name in self._device_names:
            vendor = _detect_vendor(name)
            accels.append(AcceleratorInfo(vendor=vendor, device_name=name))

        primary_vendor = accels[0].vendor if accels else VendorTag.UNKNOWN
        backends = _probe_backends(primary_vendor)
        preferred = _resolve_preferred(backends, primary_vendor)
        mem_cfg = _read_memory_config(self._low_vram)
        env_snapshot = {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES", ""),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        }

        return RuntimeSnapshot(
            accelerators=tuple(accels),
            available_backends=tuple(backends),
            preferred_backend=preferred,
            memory=mem_cfg,
            environment=env_snapshot,
        )

