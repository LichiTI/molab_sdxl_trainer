"""Config sweep for the TurboCore Triton LoRA v2_tc research candidate."""

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
    triton_lora_delta_v2_tc_candidate_with_config,
    triton_lora_delta_v2_tc_config_candidates_for_shape,
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
    items: list[int] = []
    for part in str(value or "").split(","):
        text = part.strip()
        if text:
            items.append(max(int(text), 1))
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


def _bench_config(
    *,
    preset: str,
    batch: int,
    tokens: int,
    width: int,
    rank: int,
    config: dict[str, Any],
    x: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    base: torch.Tensor,
    reference: torch.Tensor,
    reference_ms: float,
    scale: float,
    dtype: torch.dtype,
    device: torch.device,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    candidate_ms, candidate = _time_fn(
        lambda: triton_lora_delta_v2_tc_candidate_with_config(
            x,
            down,
            up,
            base,
            scale,
            launch_config=config,
        ),
        device=device,
        iters=iters,
        warmup=warmup,
    )
    speedup = reference_ms / candidate_ms if candidate_ms > 0 else 0.0
    return {
        "preset": preset,
        "batch": int(batch),
        "tokens": int(tokens),
        "width": int(width),
        "rank": int(rank),
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "config": config,
        "reference_ms": round(reference_ms, 4),
        "candidate_ms": round(candidate_ms, 4),
        "speedup_vs_reference": round(speedup, 4),
        "max_abs_error": float((reference.float() - candidate.float()).abs().max().detach().cpu()),
    }


def build_v2_tc_config_sweep(
    *,
    presets: list[str],
    ranks: list[int],
    dtype: torch.dtype,
    device: torch.device,
    iters: int,
    warmup: int,
    min_width: int = 1024,
    max_configs: int = 0,
) -> dict[str, Any]:
    started = time.perf_counter()
    if device.type != "cuda" or not triton_lora_delta_available():
        return {
            "schema_version": 1,
            "benchmark": "turbocore_lora_v2_tc_config_sweep",
            "ok": True,
            "skipped": True,
            "reason": triton_lora_delta_unavailable_reason(),
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }
    if dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("v2_tc config sweep targets float16/bfloat16 only")

    rows: list[dict[str, Any]] = []
    for preset in presets:
        shapes = SHAPE_PRESETS.get(preset)
        if not shapes:
            raise ValueError(f"unknown preset {preset!r}; available={sorted(SHAPE_PRESETS)}")
        for batch, tokens, width in shapes:
            if int(width) < int(min_width):
                continue
            for rank in ranks:
                x = torch.randn(batch, tokens, width, dtype=dtype, device=device)
                down = torch.randn(rank, width, dtype=dtype, device=device)
                up = torch.randn(width, rank, dtype=dtype, device=device)
                base = torch.randn(batch, tokens, width, dtype=dtype, device=device)
                scale = 1.0 / max(rank, 1)
                reference_ms, reference = _time_fn(
                    lambda: _explicit_fn(x, down, up, base, scale),
                    device=device,
                    iters=max(int(iters), 1),
                    warmup=max(int(warmup), 0),
                )
                configs = triton_lora_delta_v2_tc_config_candidates_for_shape(out_features=width, rank=rank)
                if max_configs > 0:
                    configs = configs[:max_configs]
                for config in configs:
                    rows.append(
                        _bench_config(
                            preset=preset,
                            batch=batch,
                            tokens=tokens,
                            width=width,
                            rank=rank,
                            config=config,
                            x=x,
                            down=down,
                            up=up,
                            base=base,
                            reference=reference,
                            reference_ms=reference_ms,
                            scale=scale,
                            dtype=dtype,
                            device=device,
                            iters=max(int(iters), 1),
                            warmup=max(int(warmup), 0),
                        )
                    )

    return {
        "schema_version": 1,
        "benchmark": "turbocore_lora_v2_tc_config_sweep",
        "ok": True,
        "skipped": False,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "presets": presets,
        "ranks": ranks,
        "iters": int(iters),
        "warmup": int(warmup),
        "min_width": int(min_width),
        "summary": _summarize(rows),
        "results": rows,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    by_config: dict[str, list[float]] = {}
    for row in rows:
        case_key = f"{row['preset']}:{row['width']}:r{row['rank']}"
        by_case.setdefault(case_key, []).append(row)
        config_name = str((row.get("config") or {}).get("name", "unknown"))
        by_config.setdefault(config_name, []).append(float(row.get("speedup_vs_reference") or 0.0))

    best_cases = []
    for case_key, case_rows in sorted(by_case.items()):
        best = max(case_rows, key=lambda item: float(item.get("speedup_vs_reference") or 0.0))
        best_cases.append({
            "case": case_key,
            "best_config": (best.get("config") or {}).get("name", "unknown"),
            "best_speedup_vs_reference": best.get("speedup_vs_reference"),
            "candidate_ms": best.get("candidate_ms"),
            "reference_ms": best.get("reference_ms"),
            "max_abs_error": best.get("max_abs_error"),
        })

    config_summaries = []
    for name, values in sorted(by_config.items()):
        config_summaries.append({
            "config": name,
            "case_count": len(values),
            "avg_speedup_vs_reference": round(sum(values) / len(values), 4) if values else None,
            "best_speedup_vs_reference": round(max(values), 4) if values else None,
            "win_count": sum(1 for value in values if value > 1.05),
            "loss_count": sum(1 for value in values if value < 0.95),
        })
    ranked = sorted(
        config_summaries,
        key=lambda item: float(item.get("avg_speedup_vs_reference") or 0.0),
        reverse=True,
    )
    return {
        "case_count": len(by_case),
        "measurement_count": len(rows),
        "best_config_by_average": ranked[0]["config"] if ranked else "",
        "config_summaries": config_summaries,
        "best_cases": best_cases,
        "ready_for_training_activation": False,
        "recommended_next_step": "promote only configs that repeat across real route shapes and stability checks",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TurboCore LoRA v2_tc config sweep")
    parser.add_argument("--presets", default="sdxl_short,dit_short")
    parser.add_argument("--ranks", default="4,8,16")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--min-width", type=int, default=1024)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    device = _device(args.device)
    dtype = _dtype(args.dtype, device)
    payload = build_v2_tc_config_sweep(
        presets=_parse_csv(args.presets, ["sdxl_short"]),
        ranks=_parse_int_csv(args.ranks, [4]),
        dtype=dtype,
        device=device,
        iters=max(int(args.iters), 1),
        warmup=max(int(args.warmup), 0),
        min_width=max(int(args.min_width), 1),
        max_configs=max(int(args.max_configs), 0),
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
