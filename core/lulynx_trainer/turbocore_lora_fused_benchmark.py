"""Benchmark harness for the future TurboCore LoRA fused delta kernel.

This script does not use native TurboCore kernels yet.  It measures the current
PyTorch reference path for ``delta = linear(linear(x, down), up)`` so future
Rust/CUDA implementations have a repeatable parity and performance baseline.
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

from core.turbocore_phase1 import lora_delta_reference  # noqa: E402
from core.turbocore_candidates import get_turbocore_candidate  # noqa: E402
try:  # noqa: E402
    from core.turbocore_triton_lora import (  # type: ignore
        triton_lora_delta_v2_config_for_shape,
        triton_lora_delta_v3_decision_for_shape,
    )
except Exception:  # pragma: no cover - Triton is optional
    triton_lora_delta_v2_config_for_shape = None  # type: ignore[assignment]
    triton_lora_delta_v3_decision_for_shape = None  # type: ignore[assignment]


SHAPE_PRESETS: dict[str, list[tuple[int, int, int]]] = {
    "tiny": [(2, 64, 320), (2, 128, 768)],
    "sdxl_short": [(2, 256, 320), (2, 128, 640), (1, 64, 1280)],
    "dit_short": [(1, 256, 1152), (1, 512, 1536), (1, 1024, 3072)],
}


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
    if device.type == "cpu" and normalized in {"fp16", "float16", "half"}:
        return torch.float32
    return torch.float16


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
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms / max(iters, 1), last


def _reference_fn(x: torch.Tensor, down: torch.Tensor, up: torch.Tensor, base: torch.Tensor, scale: float) -> torch.Tensor:
    return lora_delta_reference(x, down, up, scale=scale, base_output=base)


def _explicit_fn(x: torch.Tensor, down: torch.Tensor, up: torch.Tensor, base: torch.Tensor, scale: float) -> torch.Tensor:
    return base + F.linear(F.linear(x, down), up) * scale


def _bench_case(
    *,
    batch: int,
    tokens: int,
    in_features: int,
    rank: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
    iters: int,
    warmup: int,
    compile_candidate: bool,
    candidate_name: str = "",
) -> dict[str, Any]:
    x = torch.randn(batch, tokens, in_features, dtype=dtype, device=device)
    down = torch.randn(rank, in_features, dtype=dtype, device=device)
    up = torch.randn(out_features, rank, dtype=dtype, device=device)
    base = torch.randn(batch, tokens, out_features, dtype=dtype, device=device)
    scale = 1.0 / max(rank, 1)

    ref_ms, ref = _time_fn(lambda: _reference_fn(x, down, up, base, scale), device=device, iters=iters, warmup=warmup)
    registered = get_turbocore_candidate("lora_fused", candidate_name or None)
    candidate_callable = registered.callable if registered is not None else _explicit_fn
    candidate_label = registered.name if registered is not None else "pytorch_explicit"
    candidate_ms, candidate_out = _time_fn(lambda: candidate_callable(x, down, up, base, scale), device=device, iters=iters, warmup=warmup)
    max_abs = float((ref.float() - candidate_out.float()).abs().max().detach().cpu())

    row: dict[str, Any] = {
        "batch": batch,
        "tokens": tokens,
        "in_features": in_features,
        "out_features": out_features,
        "rank": rank,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "reference_ms": round(ref_ms, 4),
        "candidate_ms": round(candidate_ms, 4),
        "explicit_ms": round(candidate_ms, 4),
        "max_abs_error": max_abs,
        "candidate": candidate_label,
        "native_kernel_present": bool(registered.native) if registered is not None else False,
    }
    if candidate_label in {"triton_lora_delta_v2", "triton_lora_delta_v2_tc"} and triton_lora_delta_v2_config_for_shape is not None:
        row["candidate_config"] = triton_lora_delta_v2_config_for_shape(
            out_features=out_features,
            rank=rank,
        )
    if candidate_label == "triton_lora_delta_v3_dispatch" and triton_lora_delta_v3_decision_for_shape is not None:
        row["candidate_route"] = triton_lora_delta_v3_decision_for_shape(
            dtype=dtype,
            out_features=out_features,
            rank=rank,
        )

    if compile_candidate and hasattr(torch, "compile"):
        try:
            compiled = torch.compile(_explicit_fn, dynamic=False, fullgraph=False)
            compiled_ms, compiled_out = _time_fn(lambda: compiled(x, down, up, base, scale), device=device, iters=iters, warmup=warmup)
            row["compiled_ms"] = round(compiled_ms, 4)
            row["compiled_max_abs_error"] = float((ref.float() - compiled_out.float()).abs().max().detach().cpu())
        except Exception as exc:
            row["compiled_error"] = f"{type(exc).__name__}: {exc}"

    return row


def run_benchmark(
    *,
    preset: str,
    ranks: list[int],
    dtype: torch.dtype,
    device: torch.device,
    iters: int,
    warmup: int,
    compile_candidate: bool = False,
    candidate_name: str = "",
    shape_filter: Callable[[int, int, int, int], bool] | None = None,
) -> dict[str, Any]:
    shapes = SHAPE_PRESETS.get(preset)
    if not shapes:
        raise ValueError(f"unknown preset {preset!r}; available={sorted(SHAPE_PRESETS)}")
    rows = []
    for batch, tokens, width in shapes:
        out_features = width
        for rank in ranks:
            if shape_filter is not None and not shape_filter(batch, tokens, width, rank):
                continue
            rows.append(
                _bench_case(
                    batch=batch,
                    tokens=tokens,
                    in_features=width,
                    rank=rank,
                    out_features=out_features,
                    dtype=dtype,
                    device=device,
                    iters=iters,
                    warmup=warmup,
                    compile_candidate=compile_candidate,
                    candidate_name=candidate_name,
                )
            )
    best = min(rows, key=lambda item: float(item.get("candidate_ms", item.get("explicit_ms", 1e30)))) if rows else {}
    return {
        "schema_version": 1,
        "benchmark": "turbocore_lora_fused_delta_reference",
        "preset": preset,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "iters": int(iters),
        "warmup": int(warmup),
        "summary": {
            "native_kernel_present": False,
            "best_reference_ms": best.get("reference_ms"),
            "best_candidate_ms": best.get("candidate_ms"),
            "recommended_next_step": "implement and compare a fused delta candidate against this baseline",
        },
        "results": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TurboCore LoRA fused delta benchmark harness")
    parser.add_argument("--preset", default="tiny", choices=sorted(SHAPE_PRESETS))
    parser.add_argument("--ranks", default="4,8,16,32")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--compile", action="store_true", dest="compile_candidate")
    parser.add_argument("--candidate", default="", help="Registered lora_fused candidate name")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    device = _device(args.device)
    dtype = _dtype(args.dtype, device)
    ranks = [max(int(part.strip()), 1) for part in str(args.ranks).split(",") if part.strip()]
    payload = run_benchmark(
        preset=args.preset,
        ranks=ranks,
        dtype=dtype,
        device=device,
        iters=max(int(args.iters), 1),
        warmup=max(int(args.warmup), 0),
        compile_candidate=bool(args.compile_candidate),
        candidate_name=str(args.candidate or ""),
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
