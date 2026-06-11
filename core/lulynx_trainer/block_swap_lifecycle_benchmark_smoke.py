"""CPU-safe smoke test for block_swap_lifecycle_benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.block_swap_lifecycle_benchmark import run_strategy  # noqa: E402


def main() -> int:
    import torch

    device = torch.device("cpu")
    result = run_strategy(
        strategy="pipeline",
        device=device,
        blocks=4,
        dim=16,
        batch=1,
        tokens=4,
        swap_count=1,
        warmup=0,
        repeats=1,
        enable_backward_hooks=True,
    )
    assert result["requested_strategy"] == "pipeline", result
    assert result["resolved_strategy"] == "sync", result
    assert "CUDA" in result["fallback_reason"], result
    assert result["timing_ms"]["step_mean"] >= 0.0, result
    profile = result["offloader_profile"]
    assert profile["stats"]["prepare_count"] >= 1, profile
    assert profile["stats"]["prefetch_count"] >= 1, profile
    assert profile["pipeline_requested"] is True, profile
    assert profile["pipeline_active"] is False, profile
    print("block_swap_lifecycle_benchmark_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

