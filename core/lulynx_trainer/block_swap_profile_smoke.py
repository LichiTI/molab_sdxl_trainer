"""Smoke-test BlockSwap profiling snapshots.

This intentionally runs on CPU so it can validate strategy resolution and
profile fields without requiring a CUDA device.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from memory_optimizations import BlockSwapOffloader  # noqa: E402


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def main() -> int:
    blocks = [TinyBlock(), TinyBlock(), TinyBlock()]
    offloader = BlockSwapOffloader(
        blocks=blocks,
        blocks_to_swap=1,
        device=torch.device("cpu"),
        enable_backward=False,
        strategy="pipeline",
    )
    try:
        initial = offloader.profile_state()
        assert initial["requested_strategy"] == "pipeline", initial
        assert initial["resolved_strategy"] == "sync", initial
        assert "requires CUDA" in initial["fallback_reason"], initial
        assert initial["pipeline_requested"] is True, initial
        assert initial["pipeline_active"] is False, initial
        assert initial["prefetch_enabled"] is False, initial
        capability = initial["pipeline_stream_capability"]
        assert capability["requested"] is True, capability
        assert capability["available"] is False, capability
        assert "CUDA" in capability["reason"], capability

        offloader.prepare_before_forward()
        after_prepare = offloader.profile_state()
        assert after_prepare["stats"]["prepare_count"] == 1, after_prepare
        assert after_prepare["stats"]["total_prepare_ms"] >= 0.0, after_prepare
        assert after_prepare["stats"]["avg_prepare_ms"] >= 0.0, after_prepare

        offloader.ensure_block_on_device(2)
        after_direct = offloader.profile_state()
        assert after_direct["stats"]["direct_move_count"] == 1, after_direct
        assert after_direct["stats"]["total_direct_move_ms"] >= 0.0, after_direct
    finally:
        offloader.cleanup()

    if torch.cuda.is_available():
        cuda_blocks = [TinyBlock().cuda(), TinyBlock().cuda(), TinyBlock().cuda()]
        cuda_offloader = BlockSwapOffloader(
            blocks=cuda_blocks,
            blocks_to_swap=1,
            device=torch.device("cuda"),
            enable_backward=False,
            strategy="pipeline",
        )
        try:
            cuda_initial = cuda_offloader.profile_state()
            assert cuda_initial["requested_strategy"] == "pipeline", cuda_initial
            assert cuda_initial["resolved_strategy"] == "pipeline", cuda_initial
            assert cuda_initial["pipeline_active"] is True, cuda_initial
            assert cuda_initial["pipeline_stream_capability"]["available"] is True, cuda_initial
            assert cuda_initial["pipeline_stream_capability"]["stream_count"] >= 1, cuda_initial

            cuda_offloader.prepare_before_forward()
            cuda_offloader.prefetch_next(1)
            cuda_mid = cuda_offloader.profile_state()
            assert cuda_mid["pending_futures"] >= 0, cuda_mid
            cuda_offloader.ensure_block_on_device(2)
            cuda_after = cuda_offloader.profile_state()
            assert cuda_after["stats"]["prefetch_count"] >= 1, cuda_after
            assert cuda_after["stats"]["pipeline_enqueue_count"] >= 1, cuda_after
            assert cuda_after["stats"]["pipeline_event_wait_count"] >= 1, cuda_after
        finally:
            cuda_offloader.cleanup()

    print("block_swap_profile_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
