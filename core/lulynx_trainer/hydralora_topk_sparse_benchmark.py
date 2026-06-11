"""Microbenchmark HydraLoRA dense-reference top-k vs sparse top-k."""

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

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.hydralora import HydraLoRAConfig, HydraLoRALinear


@dataclass
class BenchResult:
    mode: str
    device: str
    dtype: str
    num_experts: int
    top_k: int
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


def _seed_layer(layer: HydraLoRALinear) -> None:
    torch.manual_seed(123)
    with torch.no_grad():
        layer.lora_down.normal_(std=0.02)
        layer.lora_up.normal_(std=0.02)
        layer.gate.weight.normal_(std=0.05)


def _dense_topk_reference(layer: HydraLoRALinear, x: torch.Tensor) -> torch.Tensor:
    base_out = layer.original(x)
    x_d = layer.dropout(x)
    logits = layer.gate(x)
    weights = layer._top_k_weights(logits)
    proj = torch.einsum("...i,eri->...er", x_d, layer.lora_down)
    deltas = torch.einsum("...er,eor->...eo", proj, layer.lora_up)
    mixed = (weights.unsqueeze(-1) * deltas * layer.scaling).sum(dim=-2)
    return base_out + mixed


def _run(
    *,
    mode: str,
    layer: HydraLoRALinear,
    forward_fn: Callable[[HydraLoRALinear, torch.Tensor], torch.Tensor],
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
        mode=mode,
        device=str(device),
        dtype=dtype_name,
        num_experts=args.num_experts,
        top_k=args.top_k,
        steps=args.steps,
        warmup=args.warmup,
        mean_step_ms=statistics.fmean(times),
        median_step_ms=statistics.median(times),
        peak_vram_mb=_peak_vram_mb(device),
    )


def _decision(dense: BenchResult, sparse: BenchResult) -> str:
    speedup = dense.mean_step_ms / max(sparse.mean_step_ms, 1e-9)
    vram_delta = sparse.peak_vram_mb - dense.peak_vram_mb
    if speedup >= 0.98 and vram_delta <= 1.0:
        return "pareto_default"
    if speedup >= 0.95 and vram_delta <= 0.0:
        return "near_pareto_default_candidate"
    return "experimental_only"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output-dir", default="temp/hydralora_topk_sparse_benchmark")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype(args.dtype, device)
    dtype_name = str(dtype).replace("torch.", "")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = nn.Linear(args.in_features, args.out_features, bias=False)
    layer = HydraLoRALinear(
        base,
        HydraLoRAConfig(
            num_experts=args.num_experts,
            rank=args.rank,
            alpha=args.alpha,
            routing="top_k",
            top_k=args.top_k,
            sparse_top_k=True,
        ),
    ).to(device=device, dtype=dtype)
    _seed_layer(layer)

    torch.manual_seed(456)
    x = torch.randn(args.batch, args.tokens, args.in_features, device=device, dtype=dtype, requires_grad=True)
    target = torch.randn(args.batch, args.tokens, args.out_features, device=device, dtype=dtype)

    with torch.no_grad():
        sparse = layer(x)
        dense = _dense_topk_reference(layer, x)
    tolerance = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    torch.testing.assert_close(sparse, dense, rtol=tolerance, atol=tolerance)

    dense_result = _run(
        mode="dense_reference",
        layer=layer,
        forward_fn=_dense_topk_reference,
        x=x,
        target=target,
        args=args,
        device=device,
        dtype_name=dtype_name,
    )
    sparse_result = _run(
        mode="sparse_top_k",
        layer=layer,
        forward_fn=lambda module, value: module(value),
        x=x,
        target=target,
        args=args,
        device=device,
        dtype_name=dtype_name,
    )

    speedup = dense_result.mean_step_ms / max(sparse_result.mean_step_ms, 1e-9)
    vram_delta = sparse_result.peak_vram_mb - dense_result.peak_vram_mb
    print(
        f"[benchmark] adapter=hydralora mode=dense_reference "
        f"avg_step_ms={dense_result.mean_step_ms:.2f} median_step_ms={dense_result.median_step_ms:.2f} "
        f"peak_vram_mb={dense_result.peak_vram_mb:.1f}"
    )
    print(
        f"[benchmark] adapter=hydralora mode=sparse_top_k "
        f"avg_step_ms={sparse_result.mean_step_ms:.2f} median_step_ms={sparse_result.median_step_ms:.2f} "
        f"peak_vram_mb={sparse_result.peak_vram_mb:.1f}"
    )
    print(
        f"[benchmark] adapter=hydralora speedup={speedup:.2f}x "
        f"vram_delta_mb={vram_delta:.1f} decision={_decision(dense_result, sparse_result)} "
        f"experts={args.num_experts} top_k={args.top_k}"
    )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps([asdict(dense_result), asdict(sparse_result)], indent=2),
        encoding="utf-8",
    )
    print(f"[benchmark] summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
