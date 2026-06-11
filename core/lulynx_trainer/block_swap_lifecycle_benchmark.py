"""BlockSwap forward/backward lifecycle benchmark.

This probe is intentionally outside the trainer.  It exercises the same
forward hooks and optional backward hooks used by BlockSwapOffloader on a tiny
sequential model, then writes a JSON profile that can be compared across
``sync``, ``async`` and experimental ``pipeline`` strategies.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from memory_optimizations import BlockSwapOffloader  # noqa: E402


class TinyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.linear(x)) + x)


class TinySequentialModel(nn.Module):
    def __init__(self, block_count: int, dim: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(dim) for _ in range(block_count)])
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.out(x)


def _resolve_device(requested: str) -> torch.device:
    value = str(requested or "auto").lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_snapshot(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {"peak_allocated_mb": 0.0, "peak_reserved_mb": 0.0}
    return {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 2),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024 / 1024, 2),
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(float(statistics.mean(values)), 3) if values else 0.0


def run_strategy(
    *,
    strategy: str,
    device: torch.device,
    blocks: int,
    dim: int,
    batch: int,
    tokens: int,
    swap_count: int,
    warmup: int,
    repeats: int,
    enable_backward_hooks: bool,
) -> Dict[str, Any]:
    torch.manual_seed(1234)
    model = TinySequentialModel(blocks, dim).to(device)
    offloader = BlockSwapOffloader(
        blocks=model.blocks,
        blocks_to_swap=swap_count,
        device=device,
        enable_backward=enable_backward_hooks,
        strategy=strategy,
    )
    offloader.install_forward_hooks(model)

    x = torch.randn(batch, tokens, dim, device=device)
    target = torch.randn(batch, tokens, dim, device=device)
    criterion = nn.MSELoss()
    timings: List[Dict[str, float]] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        for step in range(max(warmup + repeats, 1)):
            model.zero_grad(set_to_none=True)

            started = time.perf_counter()
            offloader.prepare_before_forward()
            _sync(device)
            prepare_ms = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            output = model(x)
            loss = criterion(output, target)
            _sync(device)
            forward_ms = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            loss.backward()
            _sync(device)
            backward_ms = (time.perf_counter() - started) * 1000.0

            if step >= warmup:
                timings.append(
                    {
                        "prepare_ms": prepare_ms,
                        "forward_ms": forward_ms,
                        "backward_ms": backward_ms,
                        "step_ms": prepare_ms + forward_ms + backward_ms,
                    }
                )
    finally:
        profile = offloader.profile_state()
        offloader.cleanup()

    result: Dict[str, Any] = {
        "requested_strategy": strategy,
        "success": True,
        "resolved_strategy": profile.get("resolved_strategy"),
        "fallback_reason": profile.get("fallback_reason", ""),
        "device": str(device),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "timing_ms": {
            "prepare_mean": _mean(item["prepare_ms"] for item in timings),
            "forward_mean": _mean(item["forward_ms"] for item in timings),
            "backward_mean": _mean(item["backward_ms"] for item in timings),
            "step_mean": _mean(item["step_ms"] for item in timings),
        },
        "memory": _memory_snapshot(device),
        "offloader_profile": profile,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--strategies", default="sync,async,pipeline")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--swap-count", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--disable-backward-hooks", action="store_true")
    parser.add_argument("--output", default="temp/block_swap_lifecycle_benchmark.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = _resolve_device(args.device)
    strategies = [item.strip() for item in str(args.strategies).split(",") if item.strip()]
    swap_count = max(0, min(int(args.swap_count), max(int(args.blocks) - 1, 0)))

    results = {
        "probe": "block_swap_lifecycle_benchmark",
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "config": {
            "blocks": int(args.blocks),
            "dim": int(args.dim),
            "batch": int(args.batch),
            "tokens": int(args.tokens),
            "swap_count": int(swap_count),
            "enable_backward_hooks": not bool(args.disable_backward_hooks),
        },
        "strategies": {},
    }

    for strategy in strategies:
        try:
            results["strategies"][strategy] = run_strategy(
                strategy=strategy,
                device=device,
                blocks=int(args.blocks),
                dim=int(args.dim),
                batch=int(args.batch),
                tokens=int(args.tokens),
                swap_count=swap_count,
                warmup=max(int(args.warmup), 0),
                repeats=max(int(args.repeats), 1),
                enable_backward_hooks=not bool(args.disable_backward_hooks),
            )
        except Exception as exc:
            results["strategies"][strategy] = {
                "requested_strategy": strategy,
                "resolved_strategy": "error",
                "device": str(device),
                "success": False,
                "failed_reason": f"{type(exc).__name__}: {exc}",
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



