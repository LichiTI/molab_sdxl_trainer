"""Smoke test for the TurboCore LoRA V3 paired benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lulynx_trainer.turbocore_lora_v3_dispatch_pair_benchmark import build_v3_dispatch_pair_benchmark  # noqa: E402


def test_cpu_fallback_pair_smoke() -> None:
    payload = build_v3_dispatch_pair_benchmark(
        presets=["dit_short"],
        ranks=[4],
        dtype=torch.float32,
        device=torch.device("cpu"),
        iters=1,
        warmup=0,
    )
    assert payload["benchmark"] == "turbocore_lora_v3_dispatch_pair"
    assert payload["summary"]["case_count"] == 3
    assert payload["summary"]["ready_for_training_activation"] is False
    assert payload["summary"]["smoke_only"] is True
    measured = [row for row in payload["results"] if not row["skipped"]]
    assert len(measured) == 3
    assert all(row["candidate_route"]["path"] == "pytorch_explicit" for row in measured)
    assert all(row["max_abs_error"] == 0.0 for row in measured)


def test_cuda_route_pair_smoke() -> None:
    if not torch.cuda.is_available():
        return
    small_payload = build_v3_dispatch_pair_benchmark(
        presets=["sdxl_short"],
        ranks=[16],
        dtype=torch.float16,
        device=torch.device("cuda"),
        iters=1,
        warmup=0,
    )
    small_routes = [row["candidate_route"]["path"] for row in small_payload["results"]]
    assert "triton_lora_delta_v1" in small_routes

    large_payload = build_v3_dispatch_pair_benchmark(
        presets=["dit_short"],
        ranks=[8],
        dtype=torch.float16,
        device=torch.device("cuda"),
        iters=1,
        warmup=0,
    )
    large_routes = [row["candidate_route"]["path"] for row in large_payload["results"]]
    assert "pytorch_explicit" in large_routes
    assert "triton_lora_delta_v2_tc" not in large_routes
    assert small_payload["summary"]["ready_for_training_activation"] is False
    assert large_payload["summary"]["ready_for_training_activation"] is False


if __name__ == "__main__":
    test_cpu_fallback_pair_smoke()
    test_cuda_route_pair_smoke()
    print("turbocore_lora_v3_dispatch_pair_benchmark_smoke: ok")
