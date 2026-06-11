"""Layout-cost probe for the TurboCore flat AdamW research kernel.

The flat Triton AdamW kernel is fast only if parameters, gradients, and optimizer
state already live in contiguous buffers. This probe measures whether per-step
gather/scatter would erase that win before any runtime integration work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

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


PRESETS: dict[str, dict[str, int]] = {
    "micro": {"blocks": 1, "hidden": 64, "mlp": 128},
    "tiny": {"blocks": 2, "hidden": 64, "mlp": 128},
    "dit_block_short": {"blocks": 2, "hidden": 256, "mlp": 1024},
}


def _device(value: str) -> torch.device:
    requested = str(value or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dit_like_shapes(*, blocks: int, hidden: int, mlp: int) -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = []
    shapes.extend([(hidden, hidden), (hidden,), (hidden * 4, hidden), (hidden, hidden * 4)])
    for _ in range(max(int(blocks), 1)):
        for _attn in range(2):
            shapes.extend([(hidden, hidden), (hidden, hidden), (hidden, hidden), (hidden, hidden)])
        shapes.extend([(mlp, hidden), (hidden, mlp)])
        shapes.extend([(hidden * 3, hidden), (hidden * 3, hidden), (hidden * 3, hidden)])
        shapes.extend([(hidden,), (hidden,)])
    shapes.extend([(hidden * 2, hidden), (hidden, hidden), (hidden,)])
    return shapes


def _make_param_values(
    shapes: Iterable[tuple[int, ...]],
    *,
    device: torch.device,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    values: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    for shape in shapes:
        value = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.01
        grad = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.001
        values.append(value.to(device=device).contiguous())
        grads.append(grad.to(device=device).contiguous())
    return values, grads


def _make_params(values: list[torch.Tensor], grads: list[torch.Tensor]) -> list[torch.nn.Parameter]:
    params = [torch.nn.Parameter(value.detach().clone()) for value in values]
    for param, grad in zip(params, grads):
        param.grad = grad.detach().clone()
    return params


def _concat_flat(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.detach().reshape(-1) for tensor in tensors]).contiguous()


def _copy_flat_to_tensors(flat: torch.Tensor, tensors: list[torch.Tensor]) -> None:
    offset = 0
    for tensor in tensors:
        count = int(tensor.numel())
        tensor.copy_(flat[offset : offset + count].view_as(tensor))
        offset += count


def _copy_params_from_flat(flat: torch.Tensor, params: list[torch.nn.Parameter]) -> None:
    offset = 0
    with torch.no_grad():
        for param in params:
            count = int(param.numel())
            param.copy_(flat[offset : offset + count].view_as(param))
            offset += count


def _copy_grads_to_flat(params: list[torch.nn.Parameter], grad_flat: torch.Tensor) -> None:
    offset = 0
    for param in params:
        count = int(param.numel())
        grad = param.grad
        if grad is None:
            grad_flat[offset : offset + count].zero_()
        else:
            grad_flat[offset : offset + count].copy_(grad.reshape(-1))
        offset += count


def _time_torch_fused_params(
    values: list[torch.Tensor],
    grads: list[torch.Tensor],
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    iters: int,
    warmup: int,
    device: torch.device,
) -> tuple[float, list[torch.Tensor]]:
    params = _make_params(values, grads)
    optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=(device.type == "cuda"))
    for _ in range(max(int(warmup), 0)):
        optimizer.step()
    _sync(device)
    started = time.perf_counter()
    for _ in range(max(int(iters), 1)):
        optimizer.step()
    _sync(device)
    return (time.perf_counter() - started) * 1000.0 / max(int(iters), 1), [param.detach().clone() for param in params]


def _time_flat_kernel(
    param_flat: torch.Tensor,
    grad_flat: torch.Tensor,
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
    param = param_flat.detach().clone().contiguous()
    grad = grad_flat.detach().clone().contiguous()
    exp_avg = torch.zeros_like(param)
    exp_avg_sq = torch.zeros_like(param)
    step = 0
    try:
        for _ in range(max(int(warmup), 0)):
            step += 1
            triton_adamw_flat_v0_step_(param, grad, exp_avg, exp_avg_sq, step=step, lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay, block_size=block_size)
        _sync(device)
        started = time.perf_counter()
        for _ in range(max(int(iters), 1)):
            step += 1
            triton_adamw_flat_v0_step_(param, grad, exp_avg, exp_avg_sq, step=step, lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay, block_size=block_size)
        _sync(device)
    except Exception as exc:  # pragma: no cover - host compiler/toolchain dependent
        return None, None, f"triton_launch_failed: {type(exc).__name__}: {exc}"
    return (time.perf_counter() - started) * 1000.0 / max(int(iters), 1), param.detach().clone(), ""


def _time_gather_scatter(
    values: list[torch.Tensor],
    grads: list[torch.Tensor],
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    iters: int,
    warmup: int,
    block_size: int,
    device: torch.device,
) -> tuple[float | None, float | None, float | None, float | None, torch.Tensor | None, str]:
    if not triton_adamw_flat_available():
        return None, None, None, None, None, triton_adamw_flat_unavailable_reason()
    params = _make_params(values, grads)
    grad_flat = torch.empty(sum(param.numel() for param in params), device=device, dtype=torch.float32)
    param_flat = _concat_flat([param.detach() for param in params])
    exp_avg = torch.zeros_like(param_flat)
    exp_avg_sq = torch.zeros_like(param_flat)
    step = 0
    try:
        for _ in range(max(int(warmup), 0)):
            step += 1
            _copy_grads_to_flat(params, grad_flat)
            triton_adamw_flat_v0_step_(param_flat, grad_flat, exp_avg, exp_avg_sq, step=step, lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay, block_size=block_size)
            _copy_params_from_flat(param_flat, params)

        _sync(device)
        started = time.perf_counter()
        for _ in range(max(int(iters), 1)):
            step += 1
            _copy_grads_to_flat(params, grad_flat)
            triton_adamw_flat_v0_step_(param_flat, grad_flat, exp_avg, exp_avg_sq, step=step, lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay, block_size=block_size)
            _copy_params_from_flat(param_flat, params)
        _sync(device)
        total_ms = (time.perf_counter() - started) * 1000.0 / max(int(iters), 1)
    except Exception as exc:  # pragma: no cover - host compiler/toolchain dependent
        return None, None, None, None, None, f"triton_layout_failed: {type(exc).__name__}: {exc}"
    denom = max(int(iters), 1)
    final_param_flat = param_flat.detach().clone()
    gather_ms, kernel_ms, scatter_ms = _measure_layout_phase_breakdown(
        params,
        param_flat,
        grad_flat,
        exp_avg,
        exp_avg_sq,
        step=step,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        block_size=block_size,
        device=device,
        iters=min(3, denom),
    )
    return (
        total_ms,
        gather_ms,
        kernel_ms,
        scatter_ms,
        final_param_flat,
        "",
    )


def _measure_layout_phase_breakdown(
    params: list[torch.nn.Parameter],
    param_flat: torch.Tensor,
    grad_flat: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    step: int,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    block_size: int,
    device: torch.device,
    iters: int,
) -> tuple[float, float, float]:
    gather_total = 0.0
    kernel_total = 0.0
    scatter_total = 0.0
    denom = max(int(iters), 1)
    for index in range(denom):
        gather_start = time.perf_counter()
        _copy_grads_to_flat(params, grad_flat)
        _sync(device)
        gather_total += time.perf_counter() - gather_start

        kernel_start = time.perf_counter()
        triton_adamw_flat_v0_step_(param_flat, grad_flat, exp_avg, exp_avg_sq, step=int(step) + index + 1, lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay, block_size=block_size)
        _sync(device)
        kernel_total += time.perf_counter() - kernel_start

        scatter_start = time.perf_counter()
        _copy_params_from_flat(param_flat, params)
        _sync(device)
        scatter_total += time.perf_counter() - scatter_start
    return gather_total * 1000.0 / denom, kernel_total * 1000.0 / denom, scatter_total * 1000.0 / denom


def _max_errors_flat(actual_flat: torch.Tensor | None, expected_tensors: list[torch.Tensor]) -> tuple[float | None, float | None]:
    if actual_flat is None:
        return None, None
    expected = _concat_flat(expected_tensors)
    diff = (actual_flat.float() - expected.float()).abs()
    max_abs = float(diff.max().detach().cpu().item()) if diff.numel() else 0.0
    denom = expected.float().abs().clamp_min(1e-8)
    max_rel = float((diff / denom).max().detach().cpu().item()) if diff.numel() else 0.0
    return max_abs, max_rel


def _gate_for_row(row: dict[str, Any], *, iters: int, warmup: int, candidate_key: str) -> dict[str, Any]:
    candidate_ms = row.get(candidate_key)
    results: list[dict[str, Any]] = [
        {
            "optimizer": "torch_adamw_fused",
            "success": True,
            "step_ms": row.get("torch_adamw_fused_ms"),
            "state_mb": row.get("state_mb"),
            "parameter_mb": row.get("param_mb"),
            "exact_adamw_candidate": True,
            "native_kernel_present": False,
            "parity_max_abs_diff": 0.0,
            "parity_max_rel_diff": 0.0,
        }
    ]
    if candidate_ms is not None:
        results.append(
            {
                "optimizer": candidate_key.replace("_ms", ""),
                "success": True,
                "step_ms": candidate_ms,
                "state_mb": row.get("state_mb"),
                "parameter_mb": row.get("param_mb"),
                "exact_adamw_candidate": True,
                "native_kernel_present": True,
                "parity_max_abs_diff": row.get(f"{candidate_key}_parity_max_abs_diff"),
                "parity_max_rel_diff": row.get(f"{candidate_key}_parity_max_rel_diff"),
            }
        )
    return evaluate_optimizer_performance_gate(
        {
            "iters": int(iters),
            "warmup": int(warmup),
            "stateful_abi_gate": {"ok": True, "source": "layout_cost_probe"},
            "results": results,
        }
    )


def run_adamw_layout_probe(
    *,
    preset: str = "tiny",
    device: str = "auto",
    iters: int = 20,
    warmup: int = 5,
    seed: int = 1234,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    block_size: int = 1024,
) -> dict[str, Any]:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; available={sorted(PRESETS)}")
    torch_device = _device(device)
    if torch_device.type != "cuda":
        return {
            "schema_version": 1,
            "probe": "turbocore_adamw_layout_probe",
            "ok": False,
            "device": str(torch_device),
            "error": "layout_probe_requires_cuda",
            "candidate_metadata": triton_adamw_flat_metadata(),
        }
    started = time.perf_counter()
    shapes = _dit_like_shapes(**PRESETS[preset])
    values, grads = _make_param_values(shapes, device=torch_device, seed=int(seed))
    param_flat = _concat_flat(values)
    grad_flat = _concat_flat(grads)
    param_count = int(param_flat.numel())
    torch_ms, torch_out = _time_torch_fused_params(
        values,
        grads,
        lr=float(lr),
        weight_decay=float(weight_decay),
        betas=(float(beta1), float(beta2)),
        eps=float(eps),
        iters=max(int(iters), 1),
        warmup=max(int(warmup), 0),
        device=torch_device,
    )
    flat_ms, flat_out, flat_reason = _time_flat_kernel(
        param_flat,
        grad_flat,
        lr=float(lr),
        weight_decay=float(weight_decay),
        betas=(float(beta1), float(beta2)),
        eps=float(eps),
        iters=max(int(iters), 1),
        warmup=max(int(warmup), 0),
        block_size=max(int(block_size), 128),
        device=torch_device,
    )
    layout_ms, gather_ms, layout_kernel_ms, scatter_ms, layout_out, layout_reason = _time_gather_scatter(
        values,
        grads,
        lr=float(lr),
        weight_decay=float(weight_decay),
        betas=(float(beta1), float(beta2)),
        eps=float(eps),
        iters=max(int(iters), 1),
        warmup=max(int(warmup), 0),
        block_size=max(int(block_size), 128),
        device=torch_device,
    )
    flat_abs, flat_rel = _max_errors_flat(flat_out, torch_out)
    layout_abs, layout_rel = _max_errors_flat(layout_out, torch_out)
    row = {
        "preset": preset,
        "parameter_tensors": len(values),
        "parameter_count": param_count,
        "param_mb": round(param_count * 4.0 / 1024 / 1024, 3),
        "state_mb": round(param_count * 8.0 / 1024 / 1024, 3),
        "torch_adamw_fused_ms": round(float(torch_ms), 4),
        "triton_flat_persistent_ms": round(float(flat_ms), 4) if flat_ms is not None else None,
        "triton_flat_persistent_speedup": round(float(torch_ms) / float(flat_ms), 4) if flat_ms else None,
        "triton_flat_persistent_ms_parity_max_abs_diff": flat_abs,
        "triton_flat_persistent_ms_parity_max_rel_diff": flat_rel,
        "triton_flat_persistent_skip_reason": flat_reason,
        "triton_with_gather_scatter_ms": round(float(layout_ms), 4) if layout_ms is not None else None,
        "triton_with_gather_scatter_speedup": round(float(torch_ms) / float(layout_ms), 4) if layout_ms else None,
        "triton_with_gather_scatter_ms_parity_max_abs_diff": layout_abs,
        "triton_with_gather_scatter_ms_parity_max_rel_diff": layout_rel,
        "gather_grad_ms": round(float(gather_ms), 4) if gather_ms is not None else None,
        "layout_kernel_ms": round(float(layout_kernel_ms), 4) if layout_kernel_ms is not None else None,
        "scatter_param_ms": round(float(scatter_ms), 4) if scatter_ms is not None else None,
        "triton_with_gather_scatter_skip_reason": layout_reason,
        "block_size": max(int(block_size), 128),
    }
    flat_gate = _gate_for_row(row, iters=max(int(iters), 1), warmup=max(int(warmup), 0), candidate_key="triton_flat_persistent_ms")
    layout_gate = _gate_for_row(row, iters=max(int(iters), 1), warmup=max(int(warmup), 0), candidate_key="triton_with_gather_scatter_ms")
    return {
        "schema_version": 1,
        "probe": "turbocore_adamw_layout_probe",
        "ok": True,
        "preset": preset,
        "shape_config": dict(PRESETS[preset]),
        "device": str(torch_device),
        "dtype": "float32",
        "iters": int(max(int(iters), 1)),
        "warmup": int(max(int(warmup), 0)),
        "candidate_metadata": triton_adamw_flat_metadata(),
        "summary": {
            "flat_kernel_gate_ok": bool(flat_gate.get("ok", False)),
            "layout_including_gather_scatter_gate_ok": bool(layout_gate.get("ok", False)),
            "flat_kernel_speedup": row.get("triton_flat_persistent_speedup"),
            "layout_including_gather_scatter_speedup": row.get("triton_with_gather_scatter_speedup"),
            "layout_tax_ms": round(float(layout_ms - flat_ms), 4) if layout_ms is not None and flat_ms is not None else None,
            "training_activation_allowed": False,
            "recommendation": _recommend(row, flat_gate=flat_gate, layout_gate=layout_gate),
        },
        "performance_gates": {
            "flat_persistent": flat_gate,
            "with_gather_scatter": layout_gate,
        },
        "results": [row],
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def _recommend(row: dict[str, Any], *, flat_gate: dict[str, Any], layout_gate: dict[str, Any]) -> str:
    if not bool(flat_gate.get("ok", False)):
        return "discard_or_optimize_flat_kernel_before_layout_work"
    if bool(layout_gate.get("ok", False)):
        return "layout_cost_still_passes_gate_try_route_level_probe"
    flat_speedup = float(row.get("triton_flat_persistent_speedup") or 0.0)
    layout_speedup = float(row.get("triton_with_gather_scatter_speedup") or 0.0)
    if flat_speedup > 1.2 and layout_speedup < 1.1:
        return "persistent_flat_buffers_required_gather_scatter_erases_kernel_win"
    return "repeat_with_larger_shapes_and_profile_layout_cost"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure layout cost for TurboCore flat AdamW v0")
    parser.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
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
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    payload = run_adamw_layout_probe(
        preset=str(args.preset),
        device=str(args.device),
        iters=max(int(args.iters), 1),
        warmup=max(int(args.warmup), 0),
        seed=int(args.seed),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        eps=float(args.eps),
        block_size=max(int(args.block_size), 128),
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
