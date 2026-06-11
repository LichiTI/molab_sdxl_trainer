# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Offloaded Gradient Checkpointing — pinned memory + async CUDA stream activation offload.

An alternative to PyTorch's ``torch.autograd.graph.save_on_cpu`` that uses:
1. A pre-allocated pinned memory pool for fast CPU↔GPU transfers
2. A dedicated CUDA stream for async host↔device copies
3. Manual pack/unpack hooks on autograd to intercept saved tensors

## Comparison with standard ``save_on_cpu``

| Feature                 | save_on_cpu (standard)      | pinned_async (this)           |
|-------------------------|-----------------------------|-------------------------------|
| Memory allocation       | Per-tensor malloc            | Pre-allocated pool             |
| Transfer mode           | Synchronous                  | Async via dedicated stream     |
| Pin memory              | Optional                     | Always (from pool)             |
| Overlap compute/copy    | No                           | Yes (separate stream)          |
| Pool size control       | None                         | Configurable pool_gb           |

The pooled approach avoids repeated ``cudaMallocHost`` / ``cudaFreeHost`` calls
which are expensive system calls.  The async stream allows the GPU to continue
computing while activations are being transferred to CPU.

Warehouse implementation using only public PyTorch APIs.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "OffloadedCheckpointContext",
    "offloaded_checkpoint_forward",
]


class PinnedMemoryPool:
    """Pre-allocated pinned memory pool for CPU↔GPU activation transfers.

    Allocates a contiguous block of pinned CPU memory and hands out slices.
    When the pool is exhausted, falls back to standard ``torch.empty(..., pin_memory=True)``.
    """

    def __init__(self, pool_gb: float = 1.0) -> None:
        self._pool_bytes = int(pool_gb * 1024 * 1024 * 1024)
        self._buffer: Optional[torch.Tensor] = None
        self._offset = 0
        self._lock = threading.Lock()
        self._fallback_count = 0
        self._alloc_count = 0

    def _ensure_buffer(self) -> None:
        if self._buffer is None:
            try:
                self._buffer = torch.empty(
                    self._pool_bytes, dtype=torch.uint8, pin_memory=True
                )
                logger.info(
                    "PinnedMemoryPool: allocated %.2f GB pinned memory",
                    self._pool_bytes / (1024**3),
                )
            except RuntimeError:
                logger.warning(
                    "PinnedMemoryPool: failed to allocate %.2f GB pinned memory, "
                    "falling back to per-tensor allocation",
                    self._pool_bytes / (1024**3),
                )
                self._buffer = None

    def allocate(self, byte_size: int) -> torch.Tensor:
        """Allocate a slice from the pool, or fall back to fresh pinned memory."""
        self._alloc_count += 1
        byte_size = max(int(byte_size), 0)

        with self._lock:
            self._ensure_buffer()

            if self._buffer is not None:
                aligned = (byte_size + 63) & ~63
                if self._offset + aligned <= self._pool_bytes:
                    t = self._buffer[self._offset : self._offset + byte_size]
                    self._offset += aligned
                    return t

        self._fallback_count += 1
        try:
            return torch.empty(byte_size, dtype=torch.uint8, pin_memory=True)
        except RuntimeError:
            logger.debug(
                "PinnedMemoryPool: per-tensor pinned allocation failed; "
                "falling back to regular CPU memory",
                exc_info=True,
            )
            return torch.empty(byte_size, dtype=torch.uint8)

    def reset(self) -> None:
        """Reset the pool offset for reuse in the next forward pass."""
        with self._lock:
            self._offset = 0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "pool_gb": self._pool_bytes / (1024**3),
            "used_mb": self._offset / (1024**2),
            "alloc_count": self._alloc_count,
            "fallback_count": self._fallback_count,
        }

    def cleanup(self) -> None:
        with self._lock:
            self._buffer = None
            self._offset = 0


class OffloadedCheckpointContext:
    """Manages activation offloading with pinned memory pool + async stream.

    Usage::

        ctx = OffloadedCheckpointContext(pool_gb=1.0)

        # In training loop:
        with ctx.offload_context():
            output = model(input)  # activations offloaded to CPU
        output.backward()  # activations restored from CPU async

    """

    def __init__(self, pool_gb: float = 1.0, device: str = "cuda") -> None:
        self._pool = PinnedMemoryPool(pool_gb)
        self._device = torch.device(device)
        self._transfer_stream: Optional[torch.cuda.Stream] = None
        self._saved_tensor_events: Dict[int, torch.cuda.Event] = {}
        self._offloaded_count = 0
        self._restored_count = 0
        self._total_bytes_offloaded = 0

    @staticmethod
    def _tensor_view_from_byte_buffer(
        byte_buffer: torch.Tensor,
        shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create a CPU tensor view backed directly by a byte buffer."""
        numel = 1
        for dim in shape:
            numel *= int(dim)
        needed_bytes = numel * torch.empty((), dtype=dtype).element_size()
        if byte_buffer.numel() < needed_bytes:
            raise RuntimeError(
                f"byte buffer too small for tensor view: {byte_buffer.numel()} < {needed_bytes}"
            )
        if needed_bytes == 0:
            return torch.empty(tuple(shape), dtype=dtype, device="cpu")
        typed = byte_buffer[:needed_bytes].view(dtype)
        return typed[:numel].view(tuple(shape))

    def _ensure_stream(self) -> None:
        if self._transfer_stream is None and self._device.type == "cuda":
            self._transfer_stream = torch.cuda.Stream(device=self._device)

    def _pack_hook(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Size, torch.dtype, torch.device]:
        """Pack hook: move activation to CPU pinned memory asynchronously."""
        if not tensor.is_cuda:
            return (tensor, tensor.shape, tensor.dtype, tensor.device)

        self._ensure_stream()

        byte_size = tensor.nelement() * tensor.element_size()
        cpu_buffer = self._pool.allocate(byte_size)
        try:
            cpu_tensor = self._tensor_view_from_byte_buffer(
                cpu_buffer, tensor.shape, tensor.dtype
            )
        except Exception:
            logger.debug(
                "OffloadedCheckpointContext: failed to create tensor view from "
                "pinned byte buffer; falling back to regular CPU tensor",
                exc_info=True,
            )
            cpu_tensor = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")

        if self._transfer_stream is not None:
            with torch.cuda.stream(self._transfer_stream):
                cpu_tensor.copy_(tensor, non_blocking=True)
            event = self._transfer_stream.record_event()
            self._saved_tensor_events[id(cpu_tensor)] = event
        else:
            cpu_tensor.copy_(tensor)

        self._offloaded_count += 1
        self._total_bytes_offloaded += byte_size

        return (cpu_tensor, tensor.shape, tensor.dtype, tensor.device)

    def _unpack_hook(
        self, packed: Tuple[torch.Tensor, torch.Size, torch.dtype, torch.device]
    ) -> torch.Tensor:
        """Unpack hook: restore activation from CPU to GPU."""
        cpu_tensor, shape, dtype, orig_device = packed

        if not orig_device.type == "cuda":
            return cpu_tensor

        event = self._saved_tensor_events.pop(id(cpu_tensor), None)
        if event is not None:
            event.synchronize()

        gpu_tensor = torch.empty(shape, dtype=dtype, device=orig_device)
        if self._transfer_stream is not None:
            with torch.cuda.stream(self._transfer_stream):
                gpu_tensor.copy_(cpu_tensor, non_blocking=True)
            torch.cuda.current_stream(orig_device).wait_stream(self._transfer_stream)
        else:
            gpu_tensor.copy_(cpu_tensor)

        self._restored_count += 1
        return gpu_tensor

    @contextmanager
    def offload_context(self):
        """Context manager that installs pack/unpack hooks for activation offloading."""
        self._pool.reset()
        self._offloaded_count = 0
        self._restored_count = 0
        self._total_bytes_offloaded = 0
        self._saved_tensor_events.clear()

        with torch.autograd.graph.saved_tensors_hooks(
            self._pack_hook, self._unpack_hook
        ):
            yield

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "offloaded_count": self._offloaded_count,
            "restored_count": self._restored_count,
            "total_mb_offloaded": self._total_bytes_offloaded / (1024 * 1024),
            "pool": self._pool.stats,
        }

    def cleanup(self) -> None:
        self._pool.cleanup()
        self._saved_tensor_events.clear()
        self._transfer_stream = None


def offloaded_checkpoint_forward(
    run_function,
    *args,
    ctx: Optional[OffloadedCheckpointContext] = None,
    **kwargs,
):
    """Run a function under gradient checkpointing with offloaded activations.

    Drop-in replacement for the ``cpu_offload_checkpoint`` helper in
    ``memory_optimizations.py``.  When ``ctx`` is provided, uses the pooled
    async offload; otherwise falls back to standard ``save_on_cpu``.
    """
    if ctx is not None:
        with ctx.offload_context():
            return torch.utils.checkpoint.checkpoint(
                run_function, *args, use_reentrant=False, **kwargs
            )

    save_on_cpu = getattr(torch.autograd.graph, "save_on_cpu", None)
    if save_on_cpu is None:
        return run_function(*args, **kwargs)
    with save_on_cpu(pin_memory=torch.cuda.is_available()):
        return run_function(*args, **kwargs)

