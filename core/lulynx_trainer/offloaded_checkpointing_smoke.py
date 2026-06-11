# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for offloaded checkpointing pinned-pool tensor views."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

try:
    from .offloaded_checkpointing import OffloadedCheckpointContext, PinnedMemoryPool
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from offloaded_checkpointing import OffloadedCheckpointContext, PinnedMemoryPool


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_tensor_view_uses_byte_buffer() -> None:
    buffer = torch.empty(4 * 6, dtype=torch.uint8)
    view = OffloadedCheckpointContext._tensor_view_from_byte_buffer(
        buffer, torch.Size((2, 3)), torch.float32
    )
    _assert(view.shape == (2, 3), "view shape mismatch")
    _assert(view.dtype == torch.float32, "view dtype mismatch")
    _assert(view.data_ptr() == buffer.data_ptr(), "view does not share byte buffer storage")

    view.fill_(3.5)
    roundtrip = buffer.view(torch.float32).view(2, 3)
    _assert(torch.equal(roundtrip, view), "byte buffer does not reflect tensor writes")


def test_pool_regular_cpu_fallback_view() -> None:
    pool = PinnedMemoryPool(pool_gb=0.0)
    buffer = pool.allocate(16)
    view = OffloadedCheckpointContext._tensor_view_from_byte_buffer(
        buffer, torch.Size((4,)), torch.float32
    )
    _assert(view.numel() == 4, "fallback view numel mismatch")
    _assert(view.data_ptr() == buffer.data_ptr(), "fallback view does not share storage")
    pool.cleanup()


def test_non_cuda_pack_passthrough() -> None:
    ctx = OffloadedCheckpointContext(pool_gb=0.0, device="cpu")
    tensor = torch.randn(2, 3)
    packed = ctx._pack_hook(tensor)
    restored = ctx._unpack_hook(packed)
    _assert(restored is tensor, "CPU tensor should pass through unchanged")


def test_cuda_pack_unpack_if_available() -> None:
    if not torch.cuda.is_available():
        return

    ctx = OffloadedCheckpointContext(pool_gb=0.01, device="cuda")
    source = torch.randn(2, 3, device="cuda", dtype=torch.float32)
    packed = ctx._pack_hook(source)
    cpu_tensor = packed[0]
    _assert(cpu_tensor.device.type == "cpu", "packed tensor should be on CPU")
    restored = ctx._unpack_hook(packed)
    torch.cuda.synchronize()
    _assert(restored.device.type == "cuda", "restored tensor should be on CUDA")
    _assert(torch.allclose(restored, source), "restored tensor mismatch")
    _assert(ctx.stats["offloaded_count"] == 1, "offload count mismatch")
    _assert(ctx.stats["restored_count"] == 1, "restore count mismatch")
    ctx.cleanup()


def main() -> None:
    test_tensor_view_uses_byte_buffer()
    test_pool_regular_cpu_fallback_view()
    test_non_cuda_pack_passthrough()
    test_cuda_pack_unpack_if_available()
    print("[PASS] offloaded checkpointing pinned-pool smoke passed")


if __name__ == "__main__":
    main()
