"""Small BlockSwap pipeline stream probe.

The script is intentionally optional and CPU-safe: without CUDA it only
verifies that requesting pipeline falls back cleanly.  With CUDA it compares
the existing async prefetch path with the experimental stream/event pipeline
path on tiny Linear blocks and prints the collected profile snapshots.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from memory_optimizations import BlockSwapOffloader  # noqa: E402


class TinyBlock(nn.Module):
    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x) + x)


def _run_probe(strategy: str, rounds: int = 5) -> dict[str, object]:
    device = torch.device("cuda")
    blocks = [TinyBlock().to(device) for _ in range(4)]
    offloader = BlockSwapOffloader(
        blocks=blocks,
        blocks_to_swap=1,
        device=device,
        enable_backward=False,
        strategy=strategy,
    )
    try:
        offloader.prepare_before_forward()
        started = time.perf_counter()
        for _ in range(max(int(rounds), 1)):
            offloader.prefetch_next(2)
            offloader.ensure_block_on_device(3)
            offloader._move_unit_to_device(3, torch.device("cpu"))
        torch.cuda.synchronize()
        profile = offloader.profile_state()
        profile["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        return profile
    finally:
        offloader.cleanup()


def main() -> int:
    if not torch.cuda.is_available():
        blocks = [TinyBlock(), TinyBlock(), TinyBlock()]
        offloader = BlockSwapOffloader(
            blocks=blocks,
            blocks_to_swap=1,
            device=torch.device("cpu"),
            enable_backward=False,
            strategy="pipeline",
        )
        try:
            profile = offloader.profile_state()
            assert profile["resolved_strategy"] == "sync", profile
            print("CUDA not available; pipeline fallback profile:")
            print(json.dumps(profile, indent=2, sort_keys=True))
        finally:
            offloader.cleanup()
        return 0

    results = {
        "async": _run_probe("async"),
        "pipeline": _run_probe("pipeline"),
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
