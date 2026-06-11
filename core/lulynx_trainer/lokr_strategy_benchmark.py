"""Research harness for LoKr strategy comparisons.

Benchmarks materialized LoKr against multiple no-materialize implementations
under eager and optional torch.compile execution. The goal is not to change
runtime defaults directly, but to provide a repeatable way to understand where
LoKr becomes memory-efficient without giving back too much throughput.
"""

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
import torch.nn.functional as F

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.lycoris_layers import LoKrLayer


@dataclass
class BenchResult:
    runtime_label: str
    features: int
    factor: int
    strategy: str
    resolved_strategy: str
    compiled: bool
    device: str
    dtype: str
    steps: int
    warmup: int
    mean_step_ms: float
    median_step_ms: float
    peak_vram_mb: float
    speedup_vs_materialized: float
    vram_delta_vs_materialized: float


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


def _parse_int_list(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        stripped = item.strip()
        if stripped:
            values.append(int(stripped))
    if not values:
        raise ValueError(f"Expected at least one integer in {text!r}")
    return values


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


def _mark_compile_step_begin() -> None:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "cudagraph_mark_step_begin"):
        compiler.cudagraph_mark_step_begin()


def _seed_layer(layer: LoKrLayer, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in layer.parameters():
            param.copy_(torch.randn_like(param) * 0.02)


def _materialized_forward(layer: LoKrLayer, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, layer.get_delta_weight())


def _legacy_no_materialize_forward(layer: LoKrLayer, x: torch.Tensor) -> torch.Tensor:
    w1 = layer._materialize_w1()
    w2 = layer._materialize_w2()
    x_view = x.reshape(*x.shape[:-1], layer.in_a, layer.in_b)
    x_w2 = F.linear(x_view, w2)
    w1_in = x_w2.transpose(-1, -2)
    out = F.linear(w1_in, w1).transpose(-1, -2)
    out = out.reshape(*x.shape[:-1], layer.out_features)
    if layer.scaling != 1.0:
        out = out * layer.scaling
    return out


def _matmul_no_materialize_forward(layer: LoKrLayer, x: torch.Tensor) -> torch.Tensor:
    w1 = layer._materialize_w1()
    w2 = layer._materialize_w2()
    flat_x = x.reshape(-1, layer.in_a, layer.in_b)
    tmp = torch.matmul(flat_x, w2.t())
    out = torch.matmul(tmp.transpose(-1, -2), w1.t()).transpose(-1, -2)
    out = out.reshape(*x.shape[:-1], layer.out_features)
    if layer.scaling != 1.0:
        out = out * layer.scaling
    return out


def _auto_no_materialize_forward(layer: LoKrLayer, x: torch.Tensor) -> torch.Tensor:
    return layer._forward_no_materialize(x)


def _auto_resolved_strategy(features: int, factor: int) -> str:
    if features >= 2048 and factor >= 16:
        return "matmul"
    return "legacy"


def _assert_close(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    if dtype in (torch.float16, torch.bfloat16):
        rtol, atol = 5e-2, 5e-2
    else:
        rtol, atol = 1e-4, 1e-5
    torch.testing.assert_close(reference, candidate, rtol=rtol, atol=atol)


def _maybe_compile(
    fn: Callable[[LoKrLayer, torch.Tensor], torch.Tensor],
    *,
    enabled: bool,
) -> Callable[[LoKrLayer, torch.Tensor], torch.Tensor]:
    if not enabled or not hasattr(torch, "compile"):
        return fn
    return torch.compile(fn, mode="reduce-overhead")


def _run(
    *,
    layer: LoKrLayer,
    fn: Callable[[LoKrLayer, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    target: torch.Tensor,
    steps: int,
    warmup: int,
    device: torch.device,
    compiled: bool,
) -> tuple[float, float, float]:
    times: list[float] = []
    layer.train()

    for index in range(warmup + steps):
        layer.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        if compiled:
            _mark_compile_step_begin()
        _sync(device)
        start = time.perf_counter()
        out = fn(layer, x)
        loss = (out * target).float().mean()
        loss.backward()
        _sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if index == warmup - 1:
            _reset_peak(device)
        if index >= warmup:
            times.append(elapsed_ms)

    return (
        statistics.fmean(times),
        statistics.median(times),
        _peak_vram_mb(device),
    )


def _strategy_results(
    *,
    layer: LoKrLayer,
    x: torch.Tensor,
    target: torch.Tensor,
    features: int,
    factor: int,
    runtime_label: str,
    steps: int,
    warmup: int,
    device: torch.device,
    dtype_name: str,
    enable_compile: bool,
) -> list[BenchResult]:
    eager_strategies = [
        ("materialized", _materialized_forward, False),
        ("legacy_no_materialize", _legacy_no_materialize_forward, False),
        ("matmul_no_materialize", _matmul_no_materialize_forward, False),
    ]
    strategies = list(eager_strategies)
    if enable_compile:
        strategies.extend(
            (name, _maybe_compile(fn, enabled=True), True)
            for name, fn, _ in eager_strategies
        )

    with torch.no_grad():
        reference = _materialized_forward(layer, x.detach())
        _assert_close(reference, _legacy_no_materialize_forward(layer, x.detach()), x.dtype)
        _assert_close(reference, _matmul_no_materialize_forward(layer, x.detach()), x.dtype)
        _assert_close(reference, _auto_no_materialize_forward(layer, x.detach()), x.dtype)
    if enable_compile:
        compare_x = x.detach().clone().requires_grad_(bool(x.requires_grad))
        compiled_functions = {
            name: fn for name, fn, compiled in strategies if compiled
        }
        _mark_compile_step_begin()
        compiled_reference = compiled_functions["materialized"](layer, compare_x).clone()
        _mark_compile_step_begin()
        compiled_legacy = compiled_functions["legacy_no_materialize"](layer, compare_x).clone()
        _mark_compile_step_begin()
        compiled_matmul = compiled_functions["matmul_no_materialize"](layer, compare_x).clone()
        _assert_close(reference, compiled_reference, x.dtype)
        _assert_close(reference, compiled_legacy, x.dtype)
        _assert_close(reference, compiled_matmul, x.dtype)

    raw_results: list[tuple[str, bool, float, float, float]] = []
    for strategy_name, fn, compiled in strategies:
        mean_ms, median_ms, peak_mb = _run(
            layer=layer,
            fn=fn,
            x=x,
            target=target,
            steps=steps,
            warmup=warmup,
            device=device,
            compiled=compiled,
        )
        raw_results.append((strategy_name, compiled, mean_ms, median_ms, peak_mb))

    baseline_by_compiled: dict[bool, tuple[float, float]] = {}
    for strategy_name, compiled, mean_ms, _, peak_mb in raw_results:
        if strategy_name == "materialized":
            baseline_by_compiled[compiled] = (mean_ms, peak_mb)

    results: list[BenchResult] = []
    for strategy_name, compiled, mean_ms, median_ms, peak_mb in raw_results:
        baseline_mean, baseline_peak = baseline_by_compiled[compiled]
        speedup = baseline_mean / max(mean_ms, 1e-9)
        vram_delta = peak_mb - baseline_peak
        if strategy_name == "materialized":
            resolved_strategy = "materialized"
        elif "matmul" in strategy_name:
            resolved_strategy = "matmul"
        else:
            resolved_strategy = "legacy"
        results.append(
            BenchResult(
                runtime_label=runtime_label,
                features=features,
                factor=factor,
                strategy=strategy_name,
                resolved_strategy=resolved_strategy,
                compiled=compiled,
                device=str(device),
                dtype=dtype_name,
                steps=steps,
                warmup=warmup,
                mean_step_ms=mean_ms,
                median_step_ms=median_ms,
                peak_vram_mb=peak_mb,
                speedup_vs_materialized=speedup,
                vram_delta_vs_materialized=vram_delta,
            )
        )
    return results


def _print_group_summary(results: list[BenchResult]) -> None:
    runtime_label = results[0].runtime_label
    features = results[0].features
    factor = results[0].factor
    for compiled in (False, True):
        group = [result for result in results if result.compiled == compiled]
        if not group:
            continue
        mode_label = "compiled" if compiled else "eager"
        for result in group:
            print(
                f"[benchmark] runtime={runtime_label} features={features} factor={factor} mode={mode_label} "
                f"strategy={result.strategy} resolved={result.resolved_strategy} avg_step_ms={result.mean_step_ms:.4f} "
                f"peak_vram_mb={result.peak_vram_mb:.1f} "
                f"speedup_vs_materialized={result.speedup_vs_materialized:.3f}x "
                f"vram_delta_mb={result.vram_delta_vs_materialized:.1f}"
            )
        best = min(group, key=lambda item: item.mean_step_ms)
        print(
            f"[benchmark] runtime={runtime_label} features={features} factor={factor} mode={mode_label} "
            f"best_strategy={best.strategy} best_avg_step_ms={best.mean_step_ms:.4f}"
        )
        no_materialize_group = [result for result in group if result.strategy != "materialized"]
        best_no_materialize = min(no_materialize_group, key=lambda item: item.mean_step_ms)
        auto_resolved = _auto_resolved_strategy(features, factor)
        auto_matches_best = auto_resolved == best_no_materialize.resolved_strategy
        print(
            f"[strategy-opt] runtime={runtime_label} features={features} factor={factor} mode={mode_label} "
            f"best_no_materialize={best_no_materialize.resolved_strategy} "
            f"source={best_no_materialize.strategy} avg_step_ms={best_no_materialize.mean_step_ms:.4f} "
            f"auto_resolved={auto_resolved} auto_matches_best={auto_matches_best}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--features", default="1024,2048")
    parser.add_argument("--factors", default="8,16,32")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--runtime-label", default="flashattention2_locked")
    parser.add_argument("--output-dir", default="temp/lokr_strategy_benchmark")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype(args.dtype, device)
    dtype_name = str(dtype).replace("torch.", "")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_list = _parse_int_list(args.features)
    factor_list = _parse_int_list(args.factors)
    all_results: list[BenchResult] = []

    for features in feature_list:
        for factor in factor_list:
            layer = LoKrLayer(
                features,
                features,
                rank=args.rank,
                alpha=args.alpha,
                factor=factor,
                no_materialize_forward=True,
                no_materialize_strategy="auto",
            ).to(device=device, dtype=dtype)
            _seed_layer(layer, seed=features * 10 + factor)
            x = torch.randn(
                args.batch,
                args.tokens,
                features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            target = torch.randn(
                args.batch,
                args.tokens,
                features,
                device=device,
                dtype=dtype,
            )
            group_results = _strategy_results(
                layer=layer,
                x=x,
                target=target,
                features=features,
                factor=factor,
                runtime_label=args.runtime_label,
                steps=args.steps,
                warmup=args.warmup,
                device=device,
                dtype_name=dtype_name,
                enable_compile=not args.disable_compile and hasattr(torch, "compile"),
            )
            _print_group_summary(group_results)
            all_results.extend(group_results)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps([asdict(result) for result in all_results], indent=2),
        encoding="utf-8",
    )
    print(f"[benchmark] summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
