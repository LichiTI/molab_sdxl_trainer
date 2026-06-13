"""Resource Check — lightweight system resource probes for preflight.

Checks disk free space and attempts to detect GPU information via
``nvidia-smi`` or PyTorch if available.  Pure-stdlib imports only;
optional integrations degrade gracefully.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import MessageBag


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiskStatus:
    """Disk space information for a single path."""

    path: str
    free_bytes: int
    total_bytes: int
    free_gb: float
    total_gb: float


@dataclass(frozen=True)
class GpuStatus:
    """Detected GPU information (best-effort)."""

    name: str
    vram_total_mb: int
    vram_free_mb: int
    driver_version: str = ""
    cuda_version: str = ""


@dataclass
class ResourceReport:
    """Consolidated resource check result."""

    disk: DiskStatus | None = None
    gpus: list[GpuStatus] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------

_GB = 1024 ** 3


def _check_disk(path: Path) -> DiskStatus | None:
    """Return disk usage for *path*, or None on failure."""
    try:
        usage = shutil.disk_usage(path)
        return DiskStatus(
            path=str(path),
            free_bytes=usage.free,
            total_bytes=usage.total,
            free_gb=round(usage.free / _GB, 2),
            total_gb=round(usage.total / _GB, 2),
        )
    except OSError:
        return None


# ---------------------------------------------------------------------------
# GPU (best-effort)
# ---------------------------------------------------------------------------

def _try_nvidia_smi() -> list[GpuStatus]:
    """Parse ``nvidia-smi`` output for basic GPU info."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
    except (OSError, subprocess.TimeoutExpired):
        return []

    gpus: list[GpuStatus] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(GpuStatus(
                name=parts[0],
                vram_total_mb=int(float(parts[1])),
                vram_free_mb=int(float(parts[2])),
                driver_version=parts[3],
            ))
        except (ValueError, IndexError):
            continue
    return gpus


def _try_torch_gpu() -> list[GpuStatus]:
    """Try to detect GPUs via PyTorch (if importable)."""
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    gpus: list[GpuStatus] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        gpus.append(GpuStatus(
            name=props.name,
            vram_total_mb=total // (1024 * 1024),
            vram_free_mb=free // (1024 * 1024),
            cuda_version=torch.version.cuda or "",
        ))
    return gpus


def _detect_gpus() -> list[GpuStatus]:
    """Best-effort GPU detection, trying multiple methods."""
    gpus = _try_nvidia_smi()
    if not gpus:
        gpus = _try_torch_gpu()
    return gpus


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ResourceChecker:
    """Probes system resources relevant to training readiness.

    Stateless.  Call :meth:`check` to obtain a :class:`ResourceReport`.

    Example::

        checker = ResourceChecker()
        report = checker.check(
            data_dir=Path("/data/training"),
            min_disk_free_gb=50.0,
            min_gpu_vram_mb=8000,
        )
    """

    def check(
        self,
        *,
        data_dir: Path | None = None,
        min_disk_free_gb: float = 0.0,
        min_gpu_vram_mb: int = 0,
    ) -> ResourceReport:
        """Run resource checks and return a consolidated report.

        Parameters
        ----------
        data_dir:
            Path whose containing disk is checked for free space.
        min_disk_free_gb:
            Minimum acceptable free disk space in GiB.  A warning is
            emitted if actual free space is below this threshold.
        min_gpu_vram_mb:
            Minimum acceptable per-GPU VRAM in MiB.  A warning is
            emitted for any GPU below this threshold.
        """
        bag = MessageBag()
        disk: DiskStatus | None = None

        if data_dir is not None:
            disk = _check_disk(Path(data_dir))
            if disk is None:
                bag.add_warning(f"Could not read disk usage for {data_dir}")
            elif min_disk_free_gb > 0 and disk.free_gb < min_disk_free_gb:
                bag.add_warning(
                    f"Low disk space: {disk.free_gb:.1f} GiB free "
                    f"(minimum {min_disk_free_gb:.1f} GiB)"
                )
            elif disk is not None:
                bag.add_note(f"Disk free: {disk.free_gb:.1f} / {disk.total_gb:.1f} GiB")

        gpus = _detect_gpus()
        if not gpus:
            bag.add_note("No GPUs detected (nvidia-smi and PyTorch both unavailable or no GPU)")
        else:
            bag.add_note(f"Detected {len(gpus)} GPU(s)")
            if min_gpu_vram_mb > 0:
                for gpu in gpus:
                    if gpu.vram_total_mb < min_gpu_vram_mb:
                        bag.add_warning(
                            f"GPU '{gpu.name}' has {gpu.vram_total_mb} MiB VRAM "
                            f"(minimum {min_gpu_vram_mb} MiB)"
                        )

        return ResourceReport(
            disk=disk,
            gpus=gpus,
            errors=list(bag.errors),
            warnings=list(bag.warnings),
            notes=list(bag.notes),
        )
