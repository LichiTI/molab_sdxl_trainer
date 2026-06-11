"""Fair paired benchmark for the TurboCore LoRA V3 research dispatcher.

The regular matrix benchmark runs each candidate independently, which is useful
for broad smoke coverage but can mix candidate timing with per-run random input
variance.  This probe compares PyTorch explicit math and the V3 dispatcher on
the same generated tensors for each shape/rank case.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lulynx_trainer.turbocore_lora_fused_benchmark import SHAPE_PRESETS  # noqa: E402
from core.turbocore_triton_lora import (  # noqa: E402
    triton_lora_delta_available,
    triton_lora_delta_unavailable_reason,
    triton_lora_delta_v3_decision_for_shape,
    triton_lora_delta_v3_dispatch_candidate,
)


def _device(value: str) -> torch.device:
    requested = str(value or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _dtype(value: str, device: torch.device) -> torch.dtype:
    normalized = str(value or "float16").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if device.type == "cpu":
        return torch.float32
    return torch.float16


def _parse_csv(value: str, default: list[str]) -> list[str]:
    items = [part.strip() for part in str(value or "").split(",") if part.strip()]
    return items or default


def _parse_int_csv(value: str, default: list[int]) -> list[int]:
    items = [max(int(part.strip()), 1) for part in str(value or "").split(",") if part.strip()]
    return items or default


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_fn(fn: Callable[[], torch.Tensor], *, device: torch.device, iters: int, warmup: int) -> tuple[float, torch.Tensor]:
    last = fn()
    for _ in range(max(warmup, 0)):
        last = fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(max(iters, 1)):
        last = fn()
    _sync(device)
    return ((time.perf_counter() - start) * 1000.0) / max(iters, 1), last


def _explicit_fn(x: torch.Tensor, down: torch.Tensor, up: torch.Tensor, base: torch.Tensor, scale: float) -> torch.Tensor:
    return base + F.linear(F.linear(x, down), up) * float(scale)


def _can_run_v3_route(route: dict[str, Any], *, device: torch.device) -> tuple[bool, str]:
    path = str(route.get("path") or "pytorch_explicit")
    if path == "pytorch_explicit":
        return True, "ok"
    if device.type != "cuda":
        return False, "v3_triton_route_requires_cuda"
    if not triton_lora_delta_available():
        return False, triton_lora_delta_unavailable_reason()
    return True, "ok"


def _bench_case(
    *,
    preset: str,
    batch: int,
    tokens: int,
    width: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    x = torch.randn(batch, tokens, width, dtype=dtype, device=device)
    down = torch.randn(rank, width, dtype=dtype, device=device)
    up = torch.randn(width, rank, dtype=dtype, device=device)
    base = torch.randn(batch, tokens, width, dtype=dtype, device=device)
    scale = 1.0 / max(rank, 1)
    route = triton_lora_delta_v3_decision_for_shape(dtype=dtype, out_features=width, rank=rank)
    can_run, skip_reason = _can_run_v3_route(route, device=device)

    reference_ms, reference = _time_fn(
        lambda: _explicit_fn(x, down, up, base, scale),
        device=device,
        iters=iters,
        warmup=warmup,
    )
    row: dict[str, Any] = {
        "preset": preset,
        "batch": int(batch),
        "tokens": int(tokens),
        "width": int(width),
        "rank": int(rank),
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "reference_ms": round(reference_ms, 4),
        "candidate_route": route,
    }
    if not can_run:
        row.update({
            "skipped": True,
            "skip_reason": skip_reason,
            "candidate_ms": None,
            "speedup_vs_reference": None,
            "max_abs_error": None,
        })
        return row

    candidate_ms, candidate = _time_fn(
        lambda: triton_lora_delta_v3_dispatch_candidate(x, down, up, base, scale),
        device=device,
        iters=iters,
        warmup=warmup,
    )
    row.update({
        "skipped": False,
        "candidate_ms": round(candidate_ms, 4),
        "speedup_vs_reference": round(reference_ms / candidate_ms, 4) if candidate_ms > 0 else 0.0,
        "max_abs_error": float((reference.float() - candidate.float()).abs().max().detach().cpu()),
    })
    return row


def build_v3_dispatch_pair_benchmark(
    *,
    presets: list[str],
    ranks: list[int],
    dtype: torch.dtype,
    device: torch.device,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for preset in presets:
        shapes = SHAPE_PRESETS.get(preset)
        if not shapes:
            raise ValueError(f"unknown preset {preset!r}; available={sorted(SHAPE_PRESETS)}")
        for batch, tokens, width in shapes:
            for rank in ranks:
                rows.append(
                    _bench_case(
                        preset=preset,
                        batch=batch,
                        tokens=tokens,
                        width=width,
                        rank=rank,
                        dtype=dtype,
                        device=device,
                        iters=max(int(iters), 1),
                        warmup=max(int(warmup), 0),
                    )
                )
    return {
        "schema_version": 1,
        "benchmark": "turbocore_lora_v3_dispatch_pair",
        "ok": True,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "presets": presets,
        "ranks": ranks,
        "iters": int(iters),
        "warmup": int(warmup),
        "summary": _summarize(rows, iters=iters, warmup=warmup),
        "results": rows,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def _summarize(rows: list[dict[str, Any]], *, iters: int, warmup: int) -> dict[str, Any]:
    measured = [row for row in rows if not row.get("skipped")]
    skipped = [row for row in rows if row.get("skipped")]
    speeds = [float(row["speedup_vs_reference"]) for row in measured if row.get("speedup_vs_reference") is not None]
    route_counts: dict[str, int] = {}
    route_speedups: dict[str, list[float]] = {}
    for row in rows:
        path = str((row.get("candidate_route") or {}).get("path") or "unknown")
        route_counts[path] = route_counts.get(path, 0) + 1
        if not row.get("skipped") and row.get("speedup_vs_reference") is not None:
            route_speedups.setdefault(path, []).append(float(row["speedup_vs_reference"]))

    route_summaries = []
    for path, count in sorted(route_counts.items()):
        values = route_speedups.get(path, [])
        route_summaries.append({
            "path": path,
            "case_count": count,
            "measured_count": len(values),
            "avg_speedup_vs_reference": round(sum(values) / len(values), 4) if values else None,
            "win_count": sum(1 for value in values if value > 1.05),
            "loss_count": sum(1 for value in values if value < 0.95),
        })

    return {
        "case_count": len(rows),
        "measured_count": len(measured),
        "skipped_count": len(skipped),
        "avg_speedup_vs_reference": round(sum(speeds) / len(speeds), 4) if speeds else None,
        "win_count": sum(1 for value in speeds if value > 1.05),
        "loss_count": sum(1 for value in speeds if value < 0.95),
        "route_summaries": route_summaries,
        "smoke_only": int(iters) < 5 or int(warmup) < 1 or len(measured) < 8,
        "ready_for_training_activation": False,
        "recommended_next_step": "repeat paired V3 benchmark on real route shapes before any training dispatcher",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fair paired benchmark for TurboCore LoRA V3 dispatcher")
    parser.add_argument("--presets", default="tiny,sdxl_short,dit_short")
    parser.add_argument("--ranks", default="4,8,16")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    device = _device(args.device)
    dtype = _dtype(args.dtype, device)
    payload = build_v3_dispatch_pair_benchmark(
        presets=_parse_csv(args.presets, ["tiny"]),
        ranks=_parse_int_csv(args.ranks, [4]),
        dtype=dtype,
        device=device,
        iters=max(int(args.iters), 1),
        warmup=max(int(args.warmup), 0),
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
