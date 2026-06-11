"""GPU capability detection and gating for Lulynx fused Triton kernels.

Clean-room implementation using Lulynx naming. Detection is cached so the
hot training path can call the gate cheaply on every layer-wrap decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass(frozen=True)
class LulynxGPUInfo:
    """Immutable snapshot of a CUDA device's relevant capabilities."""

    name: str
    vram_gb: float
    capability: tuple[int, int]
    sm_count: int
    smem_bytes_per_block: int
    threads_per_sm: int

    @property
    def smem_kb(self) -> int:
        return self.smem_bytes_per_block // 1024

    @property
    def is_ada_or_newer(self) -> bool:
        """Ada Lovelace (SM 8.9, RTX 40xx) or newer — the tuned target."""
        return self.capability >= (8, 9)

    @property
    def supports_fused_bf16(self) -> bool:
        """bf16 tensor cores require Ampere (SM 8.0) or newer."""
        return self.capability >= (8, 0)


@lru_cache(maxsize=8)
def detect_gpu(device_index: int = 0) -> LulynxGPUInfo:
    """Return a :class:`LulynxGPUInfo` for ``device_index``.

    Raises ``RuntimeError`` when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; Lulynx Triton ops require a CUDA GPU")

    props = torch.cuda.get_device_properties(device_index)
    # ``shared_memory_per_block_optin`` is the opt-in dynamic smem ceiling
    # (larger than the static default on Ampere+). Fall back when absent.
    smem = getattr(props, "shared_memory_per_block_optin", 0) or props.shared_memory_per_block
    return LulynxGPUInfo(
        name=props.name,
        vram_gb=props.total_memory / (1024 ** 3),
        capability=(props.major, props.minor),
        sm_count=props.multi_processor_count,
        smem_bytes_per_block=int(smem),
        threads_per_sm=props.max_threads_per_multi_processor,
    )


@lru_cache(maxsize=1)
def triton_available() -> bool:
    """True when Triton is importable and a CUDA device is present."""
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except Exception:
        return False
    return True


def can_run_fused_bf16(device_index: int = 0) -> bool:
    """Gate for the fused bf16 path: Triton present and bf16 tensor cores."""
    if not triton_available():
        return False
    try:
        return detect_gpu(device_index).supports_fused_bf16
    except RuntimeError:
        return False


def describe_gpu(info: LulynxGPUInfo) -> str:
    """One-line human-readable GPU summary for logs."""
    return (
        f"{info.name} | SM {info.capability[0]}.{info.capability[1]} | "
        f"{info.sm_count} SMs | {info.smem_kb} KB smem/block | {info.vram_gb:.1f} GB"
    )


# Upper bound on launched program instances for grid-stride kernels: keeps
# SM occupancy high without over-subscribing small problems.
MAX_GRID_BLOCKS = 256
