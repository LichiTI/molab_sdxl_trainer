"""Microbenchmark LyCORIS materialized vs no-materialize math paths."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.lycoris_layers import LoHaLayer, LoKrLayer


@dataclass
class BenchResult:
    adapter: str
    mode: str
    device: str
    dtype: str
    steps: int
    warmup: int
    mean_step_ms: float
    median_step_ms: float
    peak_vram_mb: float


def _dtype(name: str, device: torch.device) -> torch.dtype:
    normalized = name.strip().lower()
    if normalized == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _seed_module(module: nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn_like(param) * 0.02)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_vram_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _materialized_forward(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, layer.get_delta_weight())


def _run(
    *,
    adapter: str,
    mode: str,
    layer: nn.Module,
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    target: torch.Tensor,
    steps: int,
    warmup: int,
    device: torch.device,
    dtype_name: str,
) -> BenchResult:
    times: list[float] = []
    layer.train()

    for index in range(warmup + steps):
        layer.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        _sync(device)
        start = time.perf_counter()
        out = forward_fn(layer, x)
        loss = (out * target).float().mean()
        loss.backward()
        _sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if index == warmup - 1:
            _reset_peak(device)
        if index >= warmup:
            times.append(elapsed_ms)

    return BenchResult(
        adapter=adapter,
        mode=mode,
        device=str(device),
        dtype=dtype_name,
        steps=steps,
        warmup=warmup,
        mean_step_ms=statistics.fmean(times),
        median_step_ms=statistics.median(times),
        peak_vram_mb=_peak_vram_mb(device),
    )


def _compare_outputs(layer: nn.Module, x: torch.Tensor, dtype: torch.dtype) -> None:
    with torch.no_grad():
        optimized = layer(x)
        materialized = _materialized_forward(layer, x)
    if dtype in (torch.float16, torch.bfloat16):
        rtol, atol = 5e-2, 5e-2
    else:
        rtol, atol = 1e-4, 1e-5
    torch.testing.assert_close(optimized, materialized, rtol=rtol, atol=atol)


def _decision(materialized: BenchResult, optimized: BenchResult) -> str:
    speedup = materialized.mean_step_ms / max(optimized.mean_step_ms, 1e-9)
    vram_delta = optimized.peak_vram_mb - materialized.peak_vram_mb
    if speedup >= 0.98 and vram_delta <= 1.0:
        return "pareto_default"
    if speedup >= 0.95 and vram_delta <= 0.0:
        return "near_pareto_default_candidate"
    return "experimental_only"


def _make_layers(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> list[tuple[str, nn.Module]]:
    if args.adapter in {"lokr", "both"}:
        yield (
            "lokr",
            LoKrLayer(
                args.in_features,
                args.out_features,
                rank=args.rank,
                alpha=args.alpha,
                factor=args.factor,
                decompose_both=args.decompose_both,
                no_materialize_forward=True,
            ).to(device=device, dtype=dtype),
        )
    if args.adapter in {"loha", "both"}:
        yield (
            "loha",
            LoHaLayer(
                args.in_features,
                args.out_features,
                rank=args.rank,
                alpha=args.alpha,
            ).to(device=device, dtype=dtype),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=["lokr", "loha", "both"], default="both")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--in-features", type=int, default=2048)
    parser.add_argument("--out-features", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--factor", type=int, default=16)
    parser.add_argument("--decompose-both", action="store_true")
    parser.add_argument("--output-dir", default="temp/lycoris_no_materialize_benchmark")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype(args.dtype, device)
    dtype_name = str(dtype).replace("torch.", "")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(1234)
    x = torch.randn(args.batch, args.tokens, args.in_features, device=device, dtype=dtype, requires_grad=True)
    target = torch.randn(args.batch, args.tokens, args.out_features, device=device, dtype=dtype)

    results: list[BenchResult] = []
    for adapter, layer in _make_layers(args, device, dtype):
        _seed_module(layer, seed=100 if adapter == "lokr" else 200)
        _compare_outputs(layer, x, dtype)

        materialized = _run(
            adapter=adapter,
            mode="materialized",
            layer=layer,
            forward_fn=_materialized_forward,
            x=x,
            target=target,
            steps=args.steps,
            warmup=args.warmup,
            device=device,
            dtype_name=dtype_name,
        )
        optimized = _run(
            adapter=adapter,
            mode="no_materialize",
            layer=layer,
            forward_fn=lambda module, value: module(value),
            x=x,
            target=target,
            steps=args.steps,
            warmup=args.warmup,
            device=device,
            dtype_name=dtype_name,
        )
        results.extend([materialized, optimized])

        speedup = materialized.mean_step_ms / max(optimized.mean_step_ms, 1e-9)
        vram_delta = optimized.peak_vram_mb - materialized.peak_vram_mb
        print(
            f"[benchmark] adapter={adapter} mode=materialized "
            f"avg_step_ms={materialized.mean_step_ms:.2f} median_step_ms={materialized.median_step_ms:.2f} "
            f"peak_vram_mb={materialized.peak_vram_mb:.1f}"
        )
        print(
            f"[benchmark] adapter={adapter} mode=no_materialize "
            f"avg_step_ms={optimized.mean_step_ms:.2f} median_step_ms={optimized.median_step_ms:.2f} "
            f"peak_vram_mb={optimized.peak_vram_mb:.1f}"
        )
        print(
            f"[benchmark] adapter={adapter} speedup={speedup:.2f}x "
            f"vram_delta_mb={vram_delta:.1f} decision={_decision(materialized, optimized)} "
            f"steps={args.steps} warmup={args.warmup}"
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    print(f"[benchmark] summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
