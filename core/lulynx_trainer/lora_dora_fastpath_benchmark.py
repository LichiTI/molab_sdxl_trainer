"""Microbenchmark standard LoRA fast path and DoRA optimized weight build."""

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

from core.lulynx.dora_layer import DoRALinear
from core.lulynx_trainer.lora_injector import LoRALayer, LoRALinear


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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_vram_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def _seed_module(module: nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            if param.requires_grad:
                param.copy_(torch.randn_like(param) * 0.02)


def _reference_lora(layer: LoRALayer, x: torch.Tensor) -> torch.Tensor:
    return layer.lora_up(layer.dropout(layer.lora_down(x))) * layer.scaling


def _reference_dora_weight(layer: DoRALinear) -> torch.Tensor:
    lora_weight = layer.lora_B @ layer.lora_A
    weight_eff = layer.base_weight + layer.scaling * lora_weight
    norm = torch.linalg.norm(weight_eff, dim=1, keepdim=True)
    return layer.m.unsqueeze(1) * (weight_eff / (norm + 1e-6))


def _reference_dora(layer: DoRALinear, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, _reference_dora_weight(layer), layer.base_bias)


def _reference_dora_wrapper(layer: LoRALinear, x: torch.Tensor) -> torch.Tensor:
    dora = layer.lora
    return F.linear(x, _reference_dora_weight(dora), dora.base_bias)


def _legacy_dora_wrapper(layer: LoRALinear, x: torch.Tensor) -> torch.Tensor:
    _ = layer.original(x)
    return layer.lora(x)


def _assert_close(adapter: str, optimized: torch.Tensor, reference: torch.Tensor, dtype: torch.dtype) -> None:
    if dtype in (torch.float16, torch.bfloat16):
        rtol, atol = 5e-2, 5e-2
    elif adapter == "dora":
        rtol, atol = 1e-4, 1e-5
    else:
        rtol, atol = 1e-5, 1e-6
    torch.testing.assert_close(optimized, reference, rtol=rtol, atol=atol)


def _run(
    *,
    adapter: str,
    mode: str,
    layer: nn.Module,
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    dtype_name: str,
) -> BenchResult:
    times: list[float] = []
    layer.train()

    for index in range(args.warmup + args.steps):
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
        if index == args.warmup - 1:
            _reset_peak(device)
        if index >= args.warmup:
            times.append(elapsed_ms)

    return BenchResult(
        adapter=adapter,
        mode=mode,
        device=str(device),
        dtype=dtype_name,
        steps=args.steps,
        warmup=args.warmup,
        mean_step_ms=statistics.fmean(times),
        median_step_ms=statistics.median(times),
        peak_vram_mb=_peak_vram_mb(device),
    )


def _decision(reference: BenchResult, optimized: BenchResult) -> str:
    speedup = reference.mean_step_ms / max(optimized.mean_step_ms, 1e-9)
    vram_delta = optimized.peak_vram_mb - reference.peak_vram_mb
    if speedup >= 0.98 and vram_delta <= 1.0:
        return "pareto_default"
    if speedup >= 0.95 and vram_delta <= 0.0:
        return "near_pareto_default_candidate"
    return "experimental_only"


def _bench_lora(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> list[BenchResult]:
    layer = LoRALayer(args.in_features, args.out_features, rank=args.rank, alpha=args.alpha, dropout=0.0)
    layer = layer.to(device=device, dtype=dtype)
    _seed_module(layer, 100)
    return _bench_pair(
        adapter="lora",
        layer=layer,
        reference_fn=_reference_lora,
        optimized_fn=lambda module, value: module(value),
        args=args,
        device=device,
        dtype=dtype,
    )


def _bench_dora(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> list[BenchResult]:
    base = nn.Linear(args.in_features, args.out_features, bias=True)
    layer = DoRALinear(base, rank=args.rank, alpha=args.alpha).to(device=device, dtype=dtype)
    _seed_module(layer, 200)
    with torch.no_grad():
        layer.m.copy_(torch.rand_like(layer.m) + 0.5)
    return _bench_pair(
        adapter="dora",
        layer=layer,
        reference_fn=_reference_dora,
        optimized_fn=lambda module, value: module(value),
        args=args,
        device=device,
        dtype=dtype,
    )


def _bench_dora_wrapper(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> list[BenchResult]:
    base = nn.Linear(args.in_features, args.out_features, bias=True)
    layer = LoRALinear(base, rank=args.rank, alpha=args.alpha, use_dora=True).to(device=device, dtype=dtype)
    _seed_module(layer, 210)
    with torch.no_grad():
        layer.lora.m.copy_(torch.rand_like(layer.lora.m) + 0.5)
    return _bench_pair(
        adapter="dora_wrapper",
        layer=layer,
        reference_fn=_legacy_dora_wrapper,
        optimized_fn=lambda module, value: module(value),
        correctness_fn=_reference_dora_wrapper,
        args=args,
        device=device,
        dtype=dtype,
    )


def _bench_pair(
    *,
    adapter: str,
    layer: nn.Module,
    reference_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    optimized_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    correctness_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> list[BenchResult]:
    dtype_name = str(dtype).replace("torch.", "")
    torch.manual_seed(300)
    x = torch.randn(args.batch, args.tokens, args.in_features, device=device, dtype=dtype, requires_grad=True)
    target = torch.randn(args.batch, args.tokens, args.out_features, device=device, dtype=dtype)

    with torch.no_grad():
        optimized = optimized_fn(layer, x)
        reference = (correctness_fn or reference_fn)(layer, x)
    _assert_close(adapter, optimized, reference, dtype)

    reference_result = _run(
        adapter=adapter,
        mode="reference",
        layer=layer,
        forward_fn=reference_fn,
        x=x,
        target=target,
        args=args,
        device=device,
        dtype_name=dtype_name,
    )
    optimized_result = _run(
        adapter=adapter,
        mode="optimized",
        layer=layer,
        forward_fn=optimized_fn,
        x=x,
        target=target,
        args=args,
        device=device,
        dtype_name=dtype_name,
    )

    speedup = reference_result.mean_step_ms / max(optimized_result.mean_step_ms, 1e-9)
    vram_delta = optimized_result.peak_vram_mb - reference_result.peak_vram_mb
    print(
        f"[benchmark] adapter={adapter} mode=reference "
        f"avg_step_ms={reference_result.mean_step_ms:.2f} median_step_ms={reference_result.median_step_ms:.2f} "
        f"peak_vram_mb={reference_result.peak_vram_mb:.1f}"
    )
    print(
        f"[benchmark] adapter={adapter} mode=optimized "
        f"avg_step_ms={optimized_result.mean_step_ms:.2f} median_step_ms={optimized_result.median_step_ms:.2f} "
        f"peak_vram_mb={optimized_result.peak_vram_mb:.1f}"
    )
    print(
        f"[benchmark] adapter={adapter} speedup={speedup:.2f}x "
        f"vram_delta_mb={vram_delta:.1f} decision={_decision(reference_result, optimized_result)}"
    )
    return [reference_result, optimized_result]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=["lora", "dora", "dora_wrapper", "both"], default="both")
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
    parser.add_argument("--output-dir", default="temp/lora_dora_fastpath_benchmark")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype(args.dtype, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[BenchResult] = []
    if args.adapter in {"lora", "both"}:
        results.extend(_bench_lora(args, device, dtype))
    if args.adapter in {"dora", "both"}:
        results.extend(_bench_dora(args, device, dtype))
    if args.adapter in {"dora_wrapper", "both"}:
        results.extend(_bench_dora_wrapper(args, device, dtype))

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    print(f"[benchmark] summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
