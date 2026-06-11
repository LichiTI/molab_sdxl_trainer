"""P6 attention route scorecard for native training performance follow-up."""

from __future__ import annotations

import importlib.util
import time
from types import SimpleNamespace
from typing import Any, Sequence

import torch

from core.lulynx_trainer.attention_kernel_adapters import (
    forward_only_attention_bhnd,
    sdpa_attention_bhnd,
    torch_attention_bhnd,
)
from core.lulynx_trainer.attention_runtime_profile import build_attention_runtime_profile


DEFAULT_SHAPE = (1, 8, 256, 64)
DEFAULT_WARMUP = 3
DEFAULT_ITERS = 12
MIN_SDPA_SPEEDUP = 1.0


def build_attention_route_scorecard(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | str | None = None,
    shape: Sequence[int] = DEFAULT_SHAPE,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERS,
) -> dict[str, Any]:
    """Build a report-only scorecard for the next attention performance lane."""

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_dtype = _resolve_dtype(dtype, target_device)
    capabilities = _attention_capabilities(target_device)
    parity = _parity_case(device=target_device, dtype=target_dtype, shape=shape)
    backward = _backward_case(device=target_device, dtype=target_dtype, shape=shape)
    benchmark = _benchmark_case(
        device=target_device,
        dtype=target_dtype,
        shape=shape,
        warmup=warmup,
        iterations=iterations,
    )
    profile = _runtime_profile()
    blockers: list[str] = []
    if target_device.type != "cuda":
        blockers.append("cuda_required_for_attention_route_performance")
    if not bool(parity.get("parity_ok", False)):
        blockers.append("attention_route_sdpa_parity_failed")
    if not bool(backward.get("backward_parity_ok", False)):
        blockers.append("attention_route_recompute_backward_failed")
    if not bool(profile.get("profile_active", False)):
        blockers.append("attention_route_runtime_profile_inactive")
    speedup = _float_or_none(benchmark.get("speedup_vs_torch_attention"))
    if target_device.type == "cuda" and (speedup is None or speedup < MIN_SDPA_SPEEDUP):
        blockers.append("attention_route_sdpa_speedup_below_1x")
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_attention_route_scorecard_v0",
        "gate": "p6_attention_native_route",
        "ok": True,
        "promotion_ready": ready,
        "training_path_enabled": ready,
        "experimental_only": True,
        "default_behavior_changed": False,
        "device": str(target_device),
        "dtype": str(target_dtype).replace("torch.", ""),
        "shape": [int(item) for item in shape],
        "capabilities": capabilities,
        "parity": parity,
        "backward": backward,
        "benchmark": benchmark,
        "runtime_profile": profile,
        "promotion_blockers": _dedupe(blockers),
        "blocked_reasons": _dedupe(blockers),
    }


def _parity_case(*, device: torch.device, dtype: torch.dtype, shape: Sequence[int]) -> dict[str, Any]:
    q, k, v = _make_qkv(device=device, dtype=dtype, shape=shape, seed=6031)
    try:
        sdpa = sdpa_attention_bhnd(q, k, v)
        torch_out = torch_attention_bhnd(q, k, v)
        max_diff = _max_abs_diff(sdpa.float(), torch_out.float())
        tolerance = _tolerance(dtype)
        return {
            "ok": True,
            "backend": "sdpa_vs_torch_bhnd",
            "parity_ok": bool(max_diff <= tolerance),
            "max_abs_diff": max_diff,
            "tolerance": tolerance,
        }
    except Exception as exc:
        return _case_error("sdpa_parity_case", exc, parity_ok=False)


def _backward_case(*, device: torch.device, dtype: torch.dtype, shape: Sequence[int]) -> dict[str, Any]:
    q0, k0, v0 = _make_qkv(device=device, dtype=dtype, shape=shape, seed=6032)
    direct = [tensor.detach().clone().requires_grad_(True) for tensor in (q0, k0, v0)]
    shim = [tensor.detach().clone().requires_grad_(True) for tensor in (q0, k0, v0)]
    try:
        direct_out = sdpa_attention_bhnd(direct[0], direct[1], direct[2])
        direct_out.square().mean().backward()

        def forward_fn(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return sdpa_attention_bhnd(q_in, k_in, v_in)

        shim_out = forward_only_attention_bhnd(shim[0], shim[1], shim[2], forward_fn=forward_fn)
        shim_out.square().mean().backward()
        grad_diffs = {
            name: _max_abs_diff(left.grad.float(), right.grad.float())
            for name, left, right in zip(("q", "k", "v"), direct, shim)
            if left.grad is not None and right.grad is not None
        }
        max_grad_diff = max(grad_diffs.values()) if grad_diffs else float("inf")
        tolerance = _tolerance(dtype) * 2.0
        finite = all(_grad_finite(tensor) for tensor in shim)
        return {
            "ok": True,
            "backend": "forward_only_recompute_backward",
            "backward_parity_ok": bool(finite and max_grad_diff <= tolerance),
            "finite_gradients": finite,
            "max_abs_grad_diff": max_grad_diff,
            "grad_diffs": grad_diffs,
            "tolerance": tolerance,
        }
    except Exception as exc:
        return _case_error("recompute_backward_case", exc, backward_parity_ok=False)


def _benchmark_case(
    *,
    device: torch.device,
    dtype: torch.dtype,
    shape: Sequence[int],
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "ok": False,
            "skipped": True,
            "reason": "cuda_required_for_attention_benchmark",
            "speedup_vs_torch_attention": None,
        }
    q, k, v = _make_qkv(device=device, dtype=dtype, shape=shape, seed=6033)
    try:
        torch_ms = _time_attention(torch_attention_bhnd, q, k, v, warmup=warmup, iterations=iterations)
        sdpa_ms = _time_attention(sdpa_attention_bhnd, q, k, v, warmup=warmup, iterations=iterations)
        return {
            "ok": True,
            "benchmark": "attention_sdpa_vs_torch_bhnd_v0",
            "warmup": int(warmup),
            "iterations": int(iterations),
            "torch_attention_step_ms": torch_ms,
            "sdpa_step_ms": sdpa_ms,
            "speedup_vs_torch_attention": round(torch_ms / max(sdpa_ms, 1e-6), 4),
        }
    except Exception as exc:
        return _case_error("attention_benchmark_case", exc, speedup_vs_torch_attention=None)


def _runtime_profile() -> dict[str, Any]:
    config = SimpleNamespace(
        model_arch="anima",
        attention_backend="sdpa",
        experimental_attention_profile_enabled=True,
        experimental_attention_profile_window=128,
        experimental_attention_profile_backend="auto",
        experimental_attention_profile_torch_max_tokens=1024,
    )
    profile = SimpleNamespace(
        enabled=True,
        window_size=128,
        backend="auto",
        torch_fallback_max_tokens=1024,
        launcher_attention_backend="sdpa",
        flex_runtime_active=True,
    )
    plan = SimpleNamespace(
        requested_attention_backend="sdpa",
        attention_backend="sdpa",
        sdpa_backend_policy="cutlass",
        attention_split_chunks=0,
        attention_early_deletion=False,
        amd_sdpa_slice_trigger_gb=0.0,
        amd_sdpa_slice_target_gb=0.0,
        warnings=[],
        reasons=["p6_attention_route_scorecard_profile"],
    )
    return build_attention_runtime_profile(
        config=config,
        runtime_plan=plan,
        model_arch="anima",
        route="anima",
        profile=profile,
        patched=0,
        applied=False,
        source="p6_attention_route_scorecard",
    )


def _make_qkv(
    *,
    device: torch.device,
    dtype: torch.dtype,
    shape: Sequence[int],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(int(seed))
    tensors = [
        (torch.randn(*shape, device=device, dtype=torch.float32) * 0.25).to(dtype)
        for _ in range(3)
    ]
    return tensors[0].contiguous(), tensors[1].contiguous(), tensors[2].contiguous()


def _time_attention(fn: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, warmup: int, iterations: int) -> float:
    for _ in range(max(int(warmup), 0)):
        fn(q, k, v)
    torch.cuda.synchronize(q.device)
    started = time.perf_counter()
    for _ in range(max(int(iterations), 1)):
        fn(q, k, v)
    torch.cuda.synchronize(q.device)
    return round((time.perf_counter() - started) * 1000.0 / max(int(iterations), 1), 4)


def _attention_capabilities(device: torch.device) -> dict[str, Any]:
    return {
        "cuda": bool(device.type == "cuda" and torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "sdpa": True,
        "flash2": _module_available("flash_attn"),
        "sageattention": _module_available("sageattention"),
        "xformers": _module_available("xformers"),
        "flex_attention": _flex_attention_available(),
    }


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _flex_attention_available() -> bool:
    try:
        from torch.nn.attention.flex_attention import flex_attention

        return callable(flex_attention)
    except Exception:
        return False


def _resolve_dtype(value: torch.dtype | str | None, device: torch.device) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    normalized = str(value or "auto").strip().lower().replace("torch.", "")
    if normalized == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if normalized in {"fp16", "float16", "half"} and device.type == "cuda":
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def _tolerance(dtype: torch.dtype) -> float:
    if dtype is torch.float16:
        return 5e-2
    if dtype is torch.bfloat16:
        return 8e-2
    return 1e-4


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().cpu().item())


def _grad_finite(tensor: torch.Tensor) -> bool:
    return tensor.grad is not None and bool(torch.isfinite(tensor.grad).all().detach().cpu().item())


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _case_error(case: str, exc: Exception, **extra: Any) -> dict[str, Any]:
    payload = {
        "ok": False,
        "case": case,
        "error": f"{type(exc).__name__}: {exc}",
        "blocked_reasons": [f"{case}_failed:{type(exc).__name__}"],
    }
    payload.update(extra)
    return payload


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


__all__ = ["build_attention_route_scorecard"]
