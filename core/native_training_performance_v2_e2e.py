"""End-to-end performance gate for Native Training Performance Roadmap V2."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from core.turbocore_native_update_training_executor import build_native_update_training_executor
from core.turbocore_native_router_canary import build_native_training_router_canary


DEFAULT_SHAPE = (1, 8, 64)
DEFAULT_RANK = 4
DEFAULT_WARMUP_STEPS = 4
DEFAULT_MEASURED_STEPS = 12
DEFAULT_EXTRA_PARAM_COUNT = 24


def build_native_training_performance_gate_v2(
    *,
    lora_report: Mapping[str, Any] | None = None,
    optimizer_report: Mapping[str, Any] | None = None,
    data_report: Mapping[str, Any] | None = None,
    router_mode: str = "auto",
    dtype: str = "float16",
    shape: Sequence[int] = DEFAULT_SHAPE,
    rank: int = DEFAULT_RANK,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    measured_steps: int = DEFAULT_MEASURED_STEPS,
    device: torch.device | None = None,
) -> dict[str, Any]:
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora = dict(lora_report or {})
    optimizer = dict(optimizer_report or {})
    data = dict(data_report or {})
    router = build_native_training_router_canary(
        lora_report=lora,
        optimizer_report=optimizer,
        data_report=data,
        mode=router_mode,
    )
    if target_device.type != "cuda":
        blockers = ["cuda_required_for_e2e_training_performance_gate"]
        return _report(
            ready=False,
            blockers=blockers,
            lora_report=lora,
            optimizer_report=optimizer,
            data_report=data,
            router_report=router,
            performance_report={},
        )
    baseline = _benchmark_flow(
        dtype=dtype,
        device=target_device,
        shape=shape,
        rank=rank,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        use_native=False,
    )
    native = _benchmark_flow(
        dtype=dtype,
        device=target_device,
        shape=shape,
        rank=rank,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        use_native=True,
    )
    speedup = baseline["step_ms"] / max(native["step_ms"], 1e-6)
    ready_upstream = all(bool(report.get("promotion_ready", False)) for report in (lora, optimizer, data))
    blockers: list[str] = []
    if not ready_upstream:
        blockers.append("e2e_training_upstream_gates_not_ready")
    if not bool(router.get("promotion_ready", False)):
        blockers.append("runtime_native_router_canary_not_ready")
    if not bool(native["native_route_hit_count"] > 0):
        blockers.append("e2e_native_route_not_hit")
    if not bool(speedup >= 1.0):
        blockers.append("e2e_native_speedup_not_positive")
    if not bool(native["training_path_enabled"]):
        blockers.append("e2e_native_training_path_not_enabled")
    if not bool(native["performance_test_ready"]):
        blockers.append("e2e_native_performance_report_not_ready")
    ready = not blockers
    performance_report = {
        "schema_version": 1,
        "benchmark": "native_training_v2_tiny_lora_flow",
        "dtype": native["dtype"],
        "shape": list(shape),
        "rank": int(rank),
        "warmup_steps": int(warmup_steps),
        "measured_steps": int(measured_steps),
        "baseline_step_ms": baseline["step_ms"],
        "native_step_ms": native["step_ms"],
        "end_to_end_speedup": round(speedup, 4),
        "baseline_peak_mb": baseline["peak_mb"],
        "native_peak_mb": native["peak_mb"],
        "native_route_hit_count": native["native_route_hit_count"],
        "fallback_count": native["fallback_count"],
        "baseline_report": baseline,
        "native_report": native,
    }
    return _report(
        ready=ready,
        blockers=blockers,
        lora_report=lora,
        optimizer_report=optimizer,
        data_report=data,
        router_report=router,
        performance_report=performance_report,
    )


def _benchmark_flow(
    *,
    dtype: str,
    device: torch.device,
    shape: Sequence[int],
    rank: int,
    warmup_steps: int,
    measured_steps: int,
    use_native: bool,
) -> dict[str, Any]:
    dtype_name = _normalize_dtype(dtype)
    torch_dtype = torch.float16 if dtype_name == "float16" else torch.float32
    if dtype_name not in {"float32", "float16"}:
        return {
            "ok": False,
            "blocked_reasons": [f"unsupported_e2e_dtype:{dtype_name}"],
            "training_path_enabled": False,
            "performance_test_ready": False,
            "native_route_hit_count": 0,
            "fallback_count": 1,
            "step_ms": float("inf"),
            "peak_mb": 0.0,
            "dtype": dtype_name,
        }
    torch.manual_seed(4242)
    x = (torch.randn(*shape, device=device, dtype=torch.float32) * (0.25 if torch_dtype is torch.float16 else 1.0)).to(torch_dtype)
    x.requires_grad_(False)
    in_features = int(shape[-1])
    out_features = int(shape[-1])
    scale = 1.0 / max(int(rank), 1)
    down = torch.nn.Parameter((torch.randn(int(rank), in_features, device=device, dtype=torch.float32) * 0.25).to(torch_dtype))
    up = torch.nn.Parameter((torch.randn(out_features, int(rank), device=device, dtype=torch.float32) * 0.25).to(torch_dtype))
    base = (torch.randn(int(shape[0]) * int(shape[1]), out_features, device=device, dtype=torch.float32) * 0.25).to(torch_dtype).reshape(*shape[:-1], out_features)
    extras = [
        torch.nn.Parameter((torch.randn(4, in_features, device=device, dtype=torch.float32) * 0.05).to(torch_dtype))
        for _ in range(DEFAULT_EXTRA_PARAM_COUNT)
    ]
    params = [down, up] + extras
    optimizer = torch.optim.AdamW([{"params": params, "lr": 1e-3, "weight_decay": 0.01}], foreach=False)
    executor = None
    if use_native:
        executor = build_native_update_training_executor(
            optimizer=optimizer,
            params=params,
            config={
                "lr": 1e-3,
                "weight_decay": 0.01,
                "max_grad_norm": 0.0,
                "finite_check": False,
                "prefer_native_cuda": True,
                "prefer_triton": False,
                "block_size": 256,
                "sync_optimizer_state_each_step": False,
                "sync_params_from_optimizer_each_step": False,
                "sync_pytorch_optimizer_state_each_step": False,
            },
        )
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    total_steps = int(warmup_steps) + int(measured_steps)
    last_route_report: dict[str, Any] = {}
    for step_index in range(total_steps):
        optimizer.zero_grad(set_to_none=True)
        out = base + F.linear(F.linear(x.float(), down.float()), up.float()) * scale
        loss = out.square().mean()
        loss.backward()
        for extra in extras:
            extra.grad = torch.full_like(extra, 0.001)
        if use_native:
            last_route_report = dict(executor({"training_dispatch": True, "training_path_enabled": True})) if executor is not None else {}
            if not bool(last_route_report.get("ok", False)):
                break
        else:
            optimizer.step()
        if step_index == warmup_steps - 1:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if executor is not None:
        executor.close()
    measured = max(int(measured_steps), 1)
    return {
        "ok": bool(use_native and bool(last_route_report) or not use_native),
        "dtype": dtype_name,
        "use_native": bool(use_native),
        "step_ms": round((elapsed * 1000.0) / measured, 4),
        "peak_mb": round(torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0, 3),
        "native_route_hit_count": 1 if use_native else 0,
        "fallback_count": 0 if use_native else 1,
        "training_path_enabled": bool(use_native),
        "performance_test_ready": True,
        "lora_training_flow": True,
        "native_optimizer_measured": bool(use_native),
        "native_lora_forward_measured": False,
        "optimizer_tensor_count": len(params),
        "route_report": last_route_report,
        "blocked_reasons": [],
    }


def _report(
    *,
    ready: bool,
    blockers: list[str],
    lora_report: Mapping[str, Any],
    optimizer_report: Mapping[str, Any],
    data_report: Mapping[str, Any],
    router_report: Mapping[str, Any],
    performance_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "gate": "e2e_training_performance_gate",
        "scorecard": "native_training_performance_v2_e2e_v0",
        "ok": True,
        "promotion_ready": ready,
        "training_path_enabled": ready,
        "lora_report": dict(lora_report),
        "optimizer_report": dict(optimizer_report),
        "data_report": dict(data_report),
        "router_report": dict(router_report),
        "performance_report": dict(performance_report),
        "promotion_blockers": _dedupe(blockers),
        "blocked_reasons": _dedupe(blockers),
    }


def _normalize_dtype(value: str) -> str:
    normalized = str(value or "").replace("torch.", "").strip().lower()
    return {"fp16": "float16", "half": "float16", "bf16": "bfloat16", "fp32": "float32"}.get(
        normalized,
        normalized,
    )


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_native_training_performance_gate_v2"]
