"""Flat-buffer benchmark for the Triton AdamW v0 research kernel.

This probe measures the ideal contiguous-buffer optimizer layout before any
training runtime integration. It does not flatten model parameters and does not
enable native optimizer dispatch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    project_root = backend_root.parent
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.turbocore_optimizer_performance_gate import evaluate_optimizer_performance_gate  # noqa: E402
from core.turbocore_triton_optimizer import (  # noqa: E402
    triton_adamw_flat_available,
    triton_adamw_flat_metadata,
    triton_adamw_flat_unavailable_reason,
    triton_adamw_flat_v0_step_,
)


PRESETS: dict[str, list[int]] = {
    "tiny": [262_144, 1_048_576],
    "dit_block_short": [4_063_232],
    "optimizer_short": [4_194_304, 16_777_216],
}


def _device(value: str) -> torch.device:
    requested = str(value or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_numels(value: str, default: list[int]) -> list[int]:
    items = [int(part.strip()) for part in str(value or "").split(",") if part.strip()]
    return [max(item, 1) for item in items] or list(default)


def _parse_block_sizes(value: str, default: list[int]) -> list[int]:
    items = [int(part.strip()) for part in str(value or "").split(",") if part.strip()]
    return [max(item, 128) for item in items] or list(default)


def _make_flat_inputs(numel: int, *, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    param = torch.randn(int(numel), generator=generator, dtype=torch.float32) * 0.01
    grad = torch.randn(int(numel), generator=generator, dtype=torch.float32) * 0.001
    return param.to(device=device).contiguous(), grad.to(device=device).contiguous()


def _time_torch_fused(
    param_value: torch.Tensor,
    grad_value: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    iters: int,
    warmup: int,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    param = torch.nn.Parameter(param_value.detach().clone())
    param.grad = grad_value.detach().clone()
    optimizer = torch.optim.AdamW([param], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=(device.type == "cuda"))
    for _ in range(max(int(warmup), 0)):
        optimizer.step()
        param.grad = grad_value
    _sync(device)
    started = time.perf_counter()
    for _ in range(max(int(iters), 1)):
        optimizer.step()
        param.grad = grad_value
    _sync(device)
    return (time.perf_counter() - started) * 1000.0 / max(int(iters), 1), param.detach().clone()


def _time_triton_v0(
    param_value: torch.Tensor,
    grad_value: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    iters: int,
    warmup: int,
    block_size: int,
    device: torch.device,
) -> tuple[float | None, torch.Tensor | None, str]:
    if not triton_adamw_flat_available():
        return None, None, triton_adamw_flat_unavailable_reason()
    param = param_value.detach().clone().contiguous()
    grad = grad_value.detach().clone().contiguous()
    exp_avg = torch.zeros_like(param)
    exp_avg_sq = torch.zeros_like(param)
    step = 0
    try:
        for _ in range(max(int(warmup), 0)):
            step += 1
            triton_adamw_flat_v0_step_(
                param,
                grad,
                exp_avg,
                exp_avg_sq,
                step=step,
                lr=lr,
                beta1=betas[0],
                beta2=betas[1],
                eps=eps,
                weight_decay=weight_decay,
                block_size=block_size,
            )
        _sync(device)
        started = time.perf_counter()
        for _ in range(max(int(iters), 1)):
            step += 1
            triton_adamw_flat_v0_step_(
                param,
                grad,
                exp_avg,
                exp_avg_sq,
                step=step,
                lr=lr,
                beta1=betas[0],
                beta2=betas[1],
                eps=eps,
                weight_decay=weight_decay,
                block_size=block_size,
            )
        _sync(device)
    except Exception as exc:  # pragma: no cover - host compiler/toolchain dependent
        return None, None, f"triton_launch_failed: {type(exc).__name__}: {exc}"
    return (time.perf_counter() - started) * 1000.0 / max(int(iters), 1), param.detach().clone(), ""


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().detach().cpu().item()) if diff.numel() else 0.0
    denom = expected.float().abs().clamp_min(1e-8)
    max_rel = float((diff / denom).max().detach().cpu().item()) if diff.numel() else 0.0
    return max_abs, max_rel


def _bench_one(
    *,
    numel: int,
    device: torch.device,
    seed: int,
    iters: int,
    warmup: int,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    block_sizes: list[int],
) -> dict[str, Any]:
    param, grad = _make_flat_inputs(numel, device=device, seed=seed)
    torch_ms, torch_out = _time_torch_fused(
        param,
        grad,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        iters=iters,
        warmup=warmup,
        device=device,
    )
    triton_runs: list[dict[str, Any]] = []
    best_triton: tuple[float | None, torch.Tensor | None, str, int] = (None, None, "", int(block_sizes[0]))
    for block_size in block_sizes:
        triton_ms, triton_out, reason = _time_triton_v0(
            param,
            grad,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            iters=iters,
            warmup=warmup,
            block_size=int(block_size),
            device=device,
        )
        triton_runs.append({
            "block_size": int(block_size),
            "triton_adamw_flat_v0_ms": round(float(triton_ms), 4) if triton_ms is not None else None,
            "speedup_vs_torch_adamw_fused": round(float(torch_ms) / float(triton_ms), 4) if triton_ms else None,
            "skip_reason": reason,
        })
        if triton_ms is not None and (best_triton[0] is None or float(triton_ms) < float(best_triton[0])):
            best_triton = (triton_ms, triton_out, reason, int(block_size))
    triton_ms, triton_out, reason, best_block_size = best_triton
    row: dict[str, Any] = {
        "numel": int(numel),
        "param_mb": round(float(numel) * 4.0 / 1024 / 1024, 3),
        "state_mb": round(float(numel) * 8.0 / 1024 / 1024, 3),
        "torch_adamw_fused_ms": round(float(torch_ms), 4),
        "triton_adamw_flat_v0_ms": round(float(triton_ms), 4) if triton_ms is not None else None,
        "speedup_vs_torch_adamw_fused": round(float(torch_ms) / float(triton_ms), 4) if triton_ms else None,
        "native_kernel_present": triton_ms is not None,
        "triton_skip_reason": reason,
        "best_block_size": int(best_block_size),
        "block_size_results": triton_runs,
    }
    if triton_out is not None:
        max_abs, max_rel = _max_errors(triton_out, torch_out)
        row.update({
            "parity_max_abs_diff": max_abs,
            "parity_max_rel_diff": max_rel,
        })
    else:
        row.update({
            "parity_max_abs_diff": None,
            "parity_max_rel_diff": None,
        })
    return row


def _as_performance_gate_payload(rows: list[dict[str, Any]], *, iters: int, warmup: int) -> dict[str, Any]:
    if not rows:
        return {"iters": iters, "warmup": warmup, "stateful_abi_gate": {"ok": False}, "results": []}
    measured = [row for row in rows if row.get("triton_adamw_flat_v0_ms") is not None]
    gate_row = (
        max(measured, key=lambda row: float(row.get("speedup_vs_torch_adamw_fused") or 0.0))
        if measured
        else min(rows, key=lambda row: float(row.get("torch_adamw_fused_ms") or float("inf")))
    )
    results: list[dict[str, Any]] = [
        {
            "optimizer": "torch_adamw_fused",
            "success": True,
            "step_ms": gate_row.get("torch_adamw_fused_ms"),
            "state_mb": gate_row.get("state_mb"),
            "parameter_mb": gate_row.get("param_mb"),
            "exact_adamw_candidate": True,
            "native_kernel_present": False,
            "parity_max_abs_diff": 0.0,
            "parity_max_rel_diff": 0.0,
        }
    ]
    if gate_row.get("triton_adamw_flat_v0_ms") is not None:
        results.append(
            {
                "optimizer": "triton_adamw_flat_v0",
                "success": True,
                "step_ms": gate_row.get("triton_adamw_flat_v0_ms"),
                "state_mb": gate_row.get("state_mb"),
                "parameter_mb": gate_row.get("param_mb"),
                "exact_adamw_candidate": True,
                "native_kernel_present": True,
                "parity_max_abs_diff": gate_row.get("parity_max_abs_diff"),
                "parity_max_rel_diff": gate_row.get("parity_max_rel_diff"),
            }
        )
    return {
        "iters": int(iters),
        "warmup": int(warmup),
        "stateful_abi_gate": {"ok": True, "source": "flat_benchmark_reference_state_buffers"},
        "results": results,
    }


def build_triton_adamw_flat_benchmark(
    *,
    numels: list[int],
    device: torch.device,
    iters: int,
    warmup: int,
    seed: int = 1234,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    block_sizes: list[int] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for index, numel in enumerate(numels):
        rows.append(
            _bench_one(
                numel=max(int(numel), 1),
                device=device,
                seed=int(seed) + index,
                iters=max(int(iters), 1),
                warmup=max(int(warmup), 0),
                lr=float(lr),
                weight_decay=float(weight_decay),
                betas=(float(beta1), float(beta2)),
                eps=float(eps),
                block_sizes=[max(int(item), 128) for item in (block_sizes or [1024])],
            )
        )
    gate_payload = _as_performance_gate_payload(rows, iters=max(int(iters), 1), warmup=max(int(warmup), 0))
    performance_gate = evaluate_optimizer_performance_gate(gate_payload)
    measured = [row for row in rows if row.get("speedup_vs_torch_adamw_fused") is not None]
    best = max(measured, key=lambda row: float(row.get("speedup_vs_torch_adamw_fused") or 0.0)) if measured else None
    return {
        "schema_version": 1,
        "benchmark": "turbocore_triton_adamw_flat_v0",
        "ok": True,
        "device": str(device),
        "dtype": "float32",
        "iters": int(max(int(iters), 1)),
        "warmup": int(max(int(warmup), 0)),
        "config": {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "betas": [float(beta1), float(beta2)],
            "eps": float(eps),
            "block_sizes": [int(item) for item in (block_sizes or [1024])],
        },
        "candidate_metadata": triton_adamw_flat_metadata(),
        "summary": {
            "available": triton_adamw_flat_available(),
            "unavailable_reason": triton_adamw_flat_unavailable_reason(),
            "best_numel": best.get("numel") if best else None,
            "best_speedup_vs_torch_adamw_fused": best.get("speedup_vs_torch_adamw_fused") if best else None,
            "performance_gate_ok": bool(performance_gate.get("ok", False)),
            "performance_gate_status": performance_gate.get("status"),
            "training_activation_allowed": False,
            "recommended_next_step": "optimize or discard this flat kernel based on repeated speedup vs PyTorch fused AdamW",
        },
        "performance_gate": performance_gate,
        "results": rows,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Triton flat AdamW v0 against PyTorch fused AdamW")
    parser.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    parser.add_argument("--numels", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--block-sizes", default="", help="Comma-separated block sizes for Triton sweep; overrides --block-size")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    device = _device(args.device)
    if device.type != "cuda":
        payload = {
            "schema_version": 1,
            "benchmark": "turbocore_triton_adamw_flat_v0",
            "ok": False,
            "device": str(device),
            "error": "triton_adamw_flat_v0_requires_cuda",
            "candidate_metadata": triton_adamw_flat_metadata(),
        }
    else:
        payload = build_triton_adamw_flat_benchmark(
            numels=_parse_numels(args.numels, PRESETS[str(args.preset)]),
            device=device,
            iters=max(int(args.iters), 1),
            warmup=max(int(args.warmup), 0),
            seed=int(args.seed),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            beta1=float(args.beta1),
            beta2=float(args.beta2),
            eps=float(args.eps),
            block_sizes=_parse_block_sizes(args.block_sizes, [max(int(args.block_size), 128)]),
        )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
