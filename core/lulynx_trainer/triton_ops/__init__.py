"""Lulynx Triton-accelerated fused training kernels (default-off).

Importing this package never imports Triton itself: kernel modules under
``triton_ops.lora`` pull in ``triton`` lazily, so environments without
Triton/CUDA can still import ``lulynx_trainer`` without error.

The acceleration strategy is a "split design": cuDNN owns the
compute-bound frozen-base GEMM while a single Triton kernel fuses the
memory-bound LoRA path (down-projection, up-projection, scale, and the
residual add) so the low-rank hidden state never leaves SRAM. This is a
clean-room Lulynx implementation; it shares no source with any reference.
"""

from __future__ import annotations

from .config import (  # noqa: F401
    LulynxGPUInfo,
    can_run_fused_bf16,
    describe_gpu,
    detect_gpu,
    triton_available,
)

__all__ = [
    "LulynxGPUInfo",
    "can_run_fused_bf16",
    "describe_gpu",
    "detect_gpu",
    "triton_available",
]

__version__ = "0.1.0"
