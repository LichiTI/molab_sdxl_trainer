"""Benchmark harness for the future TurboCore native optimizer path.

This is a StandardCore/PyTorch baseline for LoRA-sized parameter groups.  It
does not enable native optimizer kernels; it measures the current update-stage
work so a later Rust/CUDA implementation has parity and performance targets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_candidates import get_turbocore_candidate  # noqa: E402


PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"layers": 8, "in_features": 320, "out_features": 320},
    "sdxl_lora_short": {"layers": 24, "in_features": 640, "out_features": 640},
    "dit_lora_short": {"layers": 32, "in_features": 1536, "out_features": 1536},
}


def _device(value: str) -> torch.device:
    requested = str(value or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _dtype(value: str, device: torch.device) -> torch.dtype:
    normalized = str(value or "float32").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"} and device.type != "cpu":
        return torch.float16
    return torch.float32


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_lora_params(
    *,
    layers: int,
    in_features: int,
    out_features: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for _ in range(max(layers, 1)):
        down = torch.nn.Parameter(torch.randn(rank, in_features, dtype=dtype, device=device) * 0.01)
        up = torch.nn.Parameter(torch.randn(out_features, rank, dtype=dtype, device=device) * 0.01)
        params.extend([down, up])
    return params


def _seed_grads(params: list[torch.nn.Parameter]) -> None:
    for param in params:
        param.grad = torch.randn_like(param)


def _finite_check(params: list[torch.nn.Parameter]) -> bool:
    for param in params:
        grad = param.grad
        if grad is not None and not bool(torch.isfinite(grad).all().item()):
            return False
    return True


def _total_grad_norm(params: list[torch.nn.Parameter]) -> torch.Tensor:
    grads = [param.grad.detach().float().norm(2) for param in params if param.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(grads), ord=2)


def _time(fn, *, device: torch.device, iters: int, warmup: int) -> float:
    for _ in range(max(warmup, 0)):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(max(iters, 1)):
        fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0 / max(iters, 1)


def _clone_values(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [param.detach().clone() for param in params]


def _restore_values(params: list[torch.nn.Parameter], values: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, value in zip(params, values):
            param.copy_(value)


def run_benchmark(
    *,
    preset: str,
    ranks: list[int],
    dtype: torch.dtype,
    device: torch.device,
    iters: int,
    warmup: int,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    candidate_name: str = "",
) -> dict[str, Any]:
    shape = PRESETS.get(preset)
    if not shape:
        raise ValueError(f"unknown preset {preset!r}; available={sorted(PRESETS)}")
    rows: list[dict[str, Any]] = []
    for rank in ranks:
        params = _make_lora_params(
            layers=int(shape["layers"]),
            in_features=int(shape["in_features"]),
            out_features=int(shape["out_features"]),
            rank=max(int(rank), 1),
            dtype=dtype,
            device=device,
        )
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        initial_values = _clone_values(params)

        def seed() -> None:
            _restore_values(params, initial_values)
            _seed_grads(params)

        seed()
        registered = get_turbocore_candidate("native_optimizer", candidate_name or None)
        candidate_callable = registered.callable if registered is not None else None
        candidate_label = registered.name if registered is not None else "pytorch_adamw"

        finite_ms = _time(lambda: _finite_check(params), device=device, iters=iters, warmup=warmup)
        norm_ms = _time(lambda: _total_grad_norm(params), device=device, iters=iters, warmup=warmup)

        def clip_once() -> None:
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

        seed()
        clip_ms = _time(clip_once, device=device, iters=iters, warmup=warmup)

        def step_once() -> None:
            seed()
            if candidate_callable is None:
                optimizer.step()
            else:
                candidate_callable(params, lr, weight_decay, max_grad_norm)

        step_ms = _time(step_once, device=device, iters=iters, warmup=warmup)

        def zero_once() -> None:
            _seed_grads(params)
            optimizer.zero_grad(set_to_none=True)

        zero_ms = _time(zero_once, device=device, iters=iters, warmup=warmup)

        param_count = sum(param.numel() for param in params)
        rows.append(
            {
                "rank": int(rank),
                "layers": int(shape["layers"]),
                "parameter_tensors": len(params),
                "parameter_count": int(param_count),
                "dtype": str(dtype).replace("torch.", ""),
                "device": str(device),
                "finite_check_ms": round(finite_ms, 4),
                "grad_norm_ms": round(norm_ms, 4),
                "clip_grad_norm_ms": round(clip_ms, 4),
                "adamw_step_ms": round(step_ms, 4),
                "candidate_step_ms": round(step_ms, 4),
                "zero_grad_ms": round(zero_ms, 4),
                "candidate": candidate_label,
                "native_kernel_present": bool(registered.native) if registered is not None else False,
            }
        )
    best = min(rows, key=lambda row: float(row["adamw_step_ms"])) if rows else {}
    return {
        "schema_version": 1,
        "benchmark": "turbocore_native_optimizer_reference",
        "preset": preset,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "iters": int(iters),
        "warmup": int(warmup),
        "summary": {
            "native_kernel_present": False,
            "best_adamw_step_ms": best.get("adamw_step_ms"),
            "recommended_next_step": "compare native LoRA AdamW/grad-norm candidates against this baseline",
        },
        "results": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TurboCore native optimizer benchmark harness")
    parser.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    parser.add_argument("--ranks", default="4,8,16,32")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--candidate", default="", help="Registered native_optimizer candidate name")
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
