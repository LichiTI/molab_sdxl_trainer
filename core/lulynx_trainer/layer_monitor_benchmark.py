"""Benchmark the optional LoRA layer monitor collector.

This does not load SDXL/Anima/Newbie models. It isolates the collector overhead
with synthetic LoRA-like layers so we can estimate the cost of the six displayed
metrics before enabling them during real training.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

try:
    from .layer_monitor import collect_lora_layer_stats
except ImportError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from layer_monitor import collect_lora_layer_stats


class _Injector:
    def __init__(self, layers: int, dim: int, rank: int, device: torch.device) -> None:
        self.injected_layers = {
            f"lora_block_{idx:03d}": torch.nn.Sequential(
                torch.nn.Linear(dim, rank, bias=False),
                torch.nn.Linear(rank, dim, bias=False),
            ).to(device)
            for idx in range(layers)
        }


def _make_grads(injector: _Injector) -> None:
    for module in injector.injected_layers.values():
        for param in module.parameters():
            param.grad = torch.randn_like(param)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(
    layers: int,
    dim: int,
    rank: int,
    max_layers: int,
    repeats: int,
    device_name: str,
    mode: str,
    sample_size: int,
    include_payload: bool,
) -> dict:
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    injector = _Injector(layers=layers, dim=dim, rank=rank, device=device)
    params = [p for module in injector.injected_layers.values() for p in module.parameters()]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    _make_grads(injector)

    # Baseline measures the loop/timing overhead without tensor reductions.
    baseline_times = []
    collect_times = []
    for _ in range(max(repeats, 1)):
        _sync(device)
        start = time.perf_counter()
        _sync(device)
        baseline_times.append(time.perf_counter() - start)

        _make_grads(injector)
        _sync(device)
        start = time.perf_counter()
        result = collect_lora_layer_stats(
            injector,
            optimizer,
            max_layers=max_layers,
            mode=mode,
            sample_size=sample_size,
        )
        _sync(device)
        collect_times.append(time.perf_counter() - start)

    baseline_mean = statistics.mean(baseline_times)
    collect_mean = statistics.mean(collect_times)
    overhead = max(collect_mean - baseline_mean, 0.0)
    payload = asdict(result)
    if not include_payload:
        payload["layers"] = payload["layers"][:1]
    return {
        "device": str(device),
        "layers_total": layers,
        "layers_sampled": max_layers if max_layers > 0 else layers,
        "dim": dim,
        "rank": rank,
        "mode": mode,
        "sample_size": sample_size,
        "repeats": repeats,
        "baseline_ms_mean": baseline_mean * 1000.0,
        "collect_ms_mean": collect_mean * 1000.0,
        "overhead_ms_mean": overhead * 1000.0,
        "collect_ms_median": statistics.median(collect_times) * 1000.0,
        "sample_payload": payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=120)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-layers", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=["sampled", "exact"], default="sampled")
    parser.add_argument("--sample-size", type=int, default=4096)
    parser.add_argument("--include-payload", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(
        layers=args.layers,
        dim=args.dim,
        rank=args.rank,
        max_layers=args.max_layers,
        repeats=args.repeats,
        device_name=args.device,
        mode=args.mode,
        sample_size=args.sample_size,
        include_payload=args.include_payload,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
