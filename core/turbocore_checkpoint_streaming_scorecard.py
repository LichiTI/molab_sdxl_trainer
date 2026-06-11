"""P6 checkpoint streaming/offloaded activation scorecard."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint_mod

from core.lulynx_trainer.offloaded_checkpoint_runtime_profile import (
    build_offloaded_checkpoint_runtime_profile,
)
from core.lulynx_trainer.offloaded_checkpointing import (
    OffloadedCheckpointContext,
    offloaded_checkpoint_forward,
)


DEFAULT_SHAPE = (2, 64, 96)
DEFAULT_HIDDEN_MULTIPLIER = 2
DEFAULT_WARMUP = 1
DEFAULT_ITERS = 5


class _CheckpointToyBlock(nn.Module):
    def __init__(self, width: int, hidden_width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, hidden_width)
        self.gate = nn.Linear(width, hidden_width)
        self.down = nn.Linear(hidden_width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = torch.nn.functional.silu(self.up(x)) * self.gate(x)
        return self.down(x) + residual


def build_checkpoint_streaming_scorecard(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | str | None = None,
    shape: Sequence[int] = DEFAULT_SHAPE,
    hidden_multiplier: int = DEFAULT_HIDDEN_MULTIPLIER,
    pool_gb: float = 0.01,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERS,
) -> dict[str, Any]:
    """Build a report-only scorecard for pinned async activation offload."""

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_dtype = _resolve_dtype(dtype, target_device)
    resolved_shape = _shape3(shape)
    hidden_width = max(int(resolved_shape[2]) * max(int(hidden_multiplier), 1), int(resolved_shape[2]))
    parity = _parity_case(
        device=target_device,
        dtype=target_dtype,
        shape=resolved_shape,
        hidden_width=hidden_width,
        pool_gb=pool_gb,
    )
    benchmark = _benchmark_case(
        device=target_device,
        dtype=target_dtype,
        shape=resolved_shape,
        hidden_width=hidden_width,
        pool_gb=pool_gb,
        warmup=warmup,
        iterations=iterations,
    )
    runtime_profile = _runtime_profile(pool_gb=pool_gb, case=_as_dict(parity.get("pinned_async")))
    blockers: list[str] = []
    if target_device.type != "cuda":
        blockers.append("cuda_required_for_checkpoint_streaming_performance")
    if not bool(parity.get("parity_ok", False)):
        blockers.append("checkpoint_streaming_parity_failed")
    if target_device.type == "cuda" and not bool(parity.get("offload_operational", False)):
        blockers.append("checkpoint_streaming_offload_not_observed")
    if target_device.type == "cuda" and not bool(runtime_profile.get("pinned_async_active", False)):
        blockers.append("checkpoint_streaming_runtime_profile_inactive")
    if target_device.type == "cuda" and not bool(benchmark.get("ok", False)):
        blockers.append("checkpoint_streaming_benchmark_failed")
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_checkpoint_streaming_scorecard_v0",
        "gate": "p6_checkpoint_streaming_route",
        "ok": True,
        "promotion_ready": ready,
        "training_path_enabled": ready,
        "experimental_only": True,
        "default_behavior_changed": False,
        "device": str(target_device),
        "dtype": str(target_dtype).replace("torch.", ""),
        "shape": list(resolved_shape),
        "hidden_width": int(hidden_width),
        "pool_gb": float(pool_gb),
        "capabilities": _capabilities(target_device),
        "parity": parity,
        "benchmark": benchmark,
        "runtime_profile": runtime_profile,
        "promotion_blockers": _dedupe(blockers),
        "blocked_reasons": _dedupe(blockers),
    }


def _parity_case(
    *,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    hidden_width: int,
    pool_gb: float,
) -> dict[str, Any]:
    state = _base_state(width=shape[2], hidden_width=hidden_width, seed=7011)
    try:
        standard = _run_training_case(
            "standard_checkpoint",
            state=state,
            device=device,
            dtype=dtype,
            shape=shape,
            hidden_width=hidden_width,
            pool_gb=pool_gb,
            seed=7012,
        )
        save_on_cpu = _run_training_case(
            "save_on_cpu",
            state=state,
            device=device,
            dtype=dtype,
            shape=shape,
            hidden_width=hidden_width,
            pool_gb=pool_gb,
            seed=7012,
        )
        pinned_async = _run_training_case(
            "pinned_async",
            state=state,
            device=device,
            dtype=dtype,
            shape=shape,
            hidden_width=hidden_width,
            pool_gb=pool_gb,
            seed=7012,
        )
        standard_grads = _as_dict(standard.get("grads"))
        pinned_grads = _as_dict(pinned_async.get("grads"))
        save_grads = _as_dict(save_on_cpu.get("grads"))
        pinned_diffs = _diff_summary(standard_grads, pinned_grads)
        save_diffs = _diff_summary(standard_grads, save_grads)
        tolerance = _tolerance(dtype)
        pinned_stats = _as_dict(pinned_async.get("ctx_stats"))
        parity_ok = bool(
            standard.get("ok", False)
            and save_on_cpu.get("ok", False)
            and pinned_async.get("ok", False)
            and pinned_diffs["max_abs_diff"] <= tolerance
            and save_diffs["max_abs_diff"] <= tolerance
            and standard.get("finite_gradients", False)
            and save_on_cpu.get("finite_gradients", False)
            and pinned_async.get("finite_gradients", False)
        )
        offload_operational = bool(
            device.type == "cuda"
            and int(pinned_stats.get("offloaded_count", 0) or 0) > 0
            and int(pinned_stats.get("restored_count", 0) or 0) > 0
            and float(pinned_stats.get("total_mb_offloaded", 0.0) or 0.0) > 0.0
        )
        return {
            "ok": True,
            "case": "checkpoint_streaming_forward_backward_parity",
            "parity_ok": parity_ok,
            "offload_operational": offload_operational,
            "finite_gradients": bool(
                standard.get("finite_gradients", False)
                and save_on_cpu.get("finite_gradients", False)
                and pinned_async.get("finite_gradients", False)
            ),
            "tolerance": tolerance,
            "pinned_async_vs_standard": pinned_diffs,
            "save_on_cpu_vs_standard": save_diffs,
            "standard_checkpoint": _compact_case(standard),
            "save_on_cpu": _compact_case(save_on_cpu),
            "pinned_async": _compact_case(pinned_async),
            "blocked_reasons": [] if parity_ok else ["checkpoint_streaming_forward_backward_parity_failed"],
        }
    except Exception as exc:
        return _case_error("checkpoint_streaming_parity_case", exc, parity_ok=False)


def _benchmark_case(
    *,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    hidden_width: int,
    pool_gb: float,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "ok": False,
            "skipped": True,
            "reason": "cuda_required_for_checkpoint_streaming_benchmark",
            "pinned_async_step_ms": None,
            "standard_checkpoint_step_ms": None,
            "save_on_cpu_step_ms": None,
        }
    state = _base_state(width=shape[2], hidden_width=hidden_width, seed=7021)
    try:
        standard_ms = _time_route(
            "standard_checkpoint",
            state=state,
            device=device,
            dtype=dtype,
            shape=shape,
            hidden_width=hidden_width,
            pool_gb=pool_gb,
            warmup=warmup,
            iterations=iterations,
        )
        save_on_cpu_ms = _time_route(
            "save_on_cpu",
            state=state,
            device=device,
            dtype=dtype,
            shape=shape,
            hidden_width=hidden_width,
            pool_gb=pool_gb,
            warmup=warmup,
            iterations=iterations,
        )
        pinned_ms = _time_route(
            "pinned_async",
            state=state,
            device=device,
            dtype=dtype,
            shape=shape,
            hidden_width=hidden_width,
            pool_gb=pool_gb,
            warmup=warmup,
            iterations=iterations,
        )
        return {
            "ok": True,
            "benchmark": "checkpoint_streaming_routes_v0",
            "warmup": int(warmup),
            "iterations": int(iterations),
            "standard_checkpoint_step_ms": standard_ms,
            "save_on_cpu_step_ms": save_on_cpu_ms,
            "pinned_async_step_ms": pinned_ms,
            "pinned_async_vs_standard_ratio": round(pinned_ms / max(standard_ms, 1e-6), 4),
            "pinned_async_vs_save_on_cpu_ratio": round(pinned_ms / max(save_on_cpu_ms, 1e-6), 4),
        }
    except Exception as exc:
        return _case_error("checkpoint_streaming_benchmark_case", exc)


def _run_training_case(
    route: str,
    *,
    state: Mapping[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    hidden_width: int,
    pool_gb: float,
    seed: int,
) -> dict[str, Any]:
    model = _load_model(state, width=shape[2], hidden_width=hidden_width, device=device, dtype=dtype)
    x, target = _input_target(shape=shape, device=device, dtype=dtype, seed=seed)
    ctx: OffloadedCheckpointContext | None = None
    peak_mb = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    try:
        if route == "standard_checkpoint":
            out = checkpoint_mod.checkpoint(model, x, use_reentrant=False)
        elif route == "save_on_cpu":
            out = offloaded_checkpoint_forward(model, x, ctx=None)
        elif route == "pinned_async":
            ctx = OffloadedCheckpointContext(pool_gb=pool_gb, device=str(device))
            out = offloaded_checkpoint_forward(model, x, ctx=ctx)
        else:
            raise ValueError(f"unknown checkpoint streaming route: {route}")
        loss = (out.float() - target.float()).square().mean()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        grads = _collect_grads(model, x)
        ctx_stats = dict(ctx.stats) if ctx is not None else {}
        return {
            "ok": True,
            "route": route,
            "loss": float(loss.detach().float().cpu().item()),
            "output": out.detach().float().cpu(),
            "grads": grads,
            "finite_gradients": _all_finite(grads),
            "ctx_stats": ctx_stats,
            "peak_allocated_mb": round(peak_mb, 4),
        }
    finally:
        if ctx is not None:
            ctx.cleanup()


def _time_route(
    route: str,
    *,
    state: Mapping[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    hidden_width: int,
    pool_gb: float,
    warmup: int,
    iterations: int,
) -> float:
    for index in range(max(int(warmup), 0)):
        _run_training_case(
            route,
            state=state,
            device=device,
            dtype=dtype,
            shape=shape,
            hidden_width=hidden_width,
            pool_gb=pool_gb,
            seed=7030 + index,
        )
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for index in range(max(int(iterations), 1)):
        _run_training_case(
            route,
            state=state,
            device=device,
            dtype=dtype,
            shape=shape,
            hidden_width=hidden_width,
            pool_gb=pool_gb,
            seed=7040 + index,
        )
    torch.cuda.synchronize(device)
    return round((time.perf_counter() - started) * 1000.0 / max(int(iterations), 1), 4)


def _runtime_profile(*, pool_gb: float, case: Mapping[str, Any]) -> dict[str, Any]:
    profile = build_offloaded_checkpoint_runtime_profile(
        requested=True,
        mode="pinned_async",
        pool_gb=float(pool_gb),
        context=None,
    )
    stats = dict(case.get("ctx_stats", {}) or {})
    profile["pinned_async_active"] = bool(stats)
    profile["stats"] = stats
    profile["source"] = "p6_checkpoint_streaming_scorecard"
    if not stats:
        profile["disabled_reason"] = "scorecard_context_unavailable"
    elif "disabled_reason" in profile:
        profile.pop("disabled_reason", None)
    return profile


def _base_state(*, width: int, hidden_width: int, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(int(seed))
    model = _CheckpointToyBlock(width=width, hidden_width=hidden_width)
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _load_model(
    state: Mapping[str, torch.Tensor],
    *,
    width: int,
    hidden_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> _CheckpointToyBlock:
    model = _CheckpointToyBlock(width=width, hidden_width=hidden_width)
    model.load_state_dict({name: tensor.detach().clone() for name, tensor in state.items()})
    model.to(device=device, dtype=dtype)
    model.train()
    return model


def _input_target(
    *,
    shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(int(seed))
    x = (torch.randn(*shape, device=device, dtype=torch.float32) * 0.2).to(dtype)
    target = (torch.randn(*shape, device=device, dtype=torch.float32) * 0.1).to(dtype)
    return x.detach().clone().requires_grad_(True), target.contiguous()


def _collect_grads(model: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    if x.grad is not None:
        grads["input"] = x.grad.detach().float().cpu()
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().float().cpu()
    return grads


def _diff_summary(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    keys = sorted(set(left) & set(right))
    diffs = {
        key: _max_abs_diff(left[key], right[key])
        for key in keys
    }
    return {
        "compared_tensor_count": len(keys),
        "missing_left": sorted(set(right) - set(left)),
        "missing_right": sorted(set(left) - set(right)),
        "max_abs_diff": max(diffs.values()) if diffs else float("inf"),
        "diffs": diffs,
    }


def _compact_case(case: Mapping[str, Any]) -> dict[str, Any]:
    stats = dict(case.get("ctx_stats", {}) or {})
    return {
        "ok": bool(case.get("ok", False)),
        "route": str(case.get("route", "")),
        "loss": case.get("loss"),
        "finite_gradients": bool(case.get("finite_gradients", False)),
        "peak_allocated_mb": case.get("peak_allocated_mb", 0.0),
        "ctx_stats": stats,
    }


def _capabilities(device: torch.device) -> dict[str, Any]:
    return {
        "cuda": bool(device.type == "cuda" and torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "saved_tensors_hooks": hasattr(torch.autograd.graph, "saved_tensors_hooks"),
        "save_on_cpu": hasattr(torch.autograd.graph, "save_on_cpu"),
        "torch_checkpoint": hasattr(checkpoint_mod, "checkpoint"),
    }


def _shape3(shape: Sequence[int]) -> tuple[int, int, int]:
    values = [int(item) for item in list(shape)]
    if len(values) != 3:
        raise ValueError(f"checkpoint streaming shape must be 3D, got {values}")
    return tuple(max(value, 1) for value in values)  # type: ignore[return-value]


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


def _all_finite(values: Mapping[str, torch.Tensor]) -> bool:
    return bool(values) and all(bool(torch.isfinite(tensor).all().item()) for tensor in values.values())


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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


__all__ = ["build_checkpoint_streaming_scorecard"]
