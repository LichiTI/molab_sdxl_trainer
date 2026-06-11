"""Short benchmark smoke for dev-only cached CUDA AdamW runtime sessions."""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
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


def _inject_native_artifact_dir_from_env() -> None:
    raw = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    if not raw:
        return
    path = Path(raw).expanduser()
    if path.is_dir():
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _make_buffers(numel: int) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    return {
        "param_flat": (torch.arange(numel, dtype=torch.float32, device=device) * 0.0005 + 0.5).contiguous(),
        "grad_flat": (torch.arange(numel, dtype=torch.float32, device=device) * 0.00001 + 0.01).contiguous(),
        "exp_avg": torch.zeros(numel, dtype=torch.float32, device=device).contiguous(),
        "exp_avg_sq": torch.zeros(numel, dtype=torch.float32, device=device).contiguous(),
    }


def _reference_step(buffers: dict[str, torch.Tensor], step_index: int, config: dict[str, Any]) -> None:
    param = buffers["param_flat"]
    grad = buffers["grad_flat"]
    exp_avg = buffers["exp_avg"]
    exp_avg_sq = buffers["exp_avg_sq"]
    beta1, beta2 = [float(item) for item in config["betas"]]
    lr = float(config["lr"])
    eps = float(config["eps"])
    weight_decay = float(config["weight_decay"])
    step_number = int(step_index) + 1
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    if weight_decay:
        param.mul_(1.0 - lr * weight_decay)
    step_size = lr / (1.0 - beta1**step_number)
    denom = exp_avg_sq.sqrt().div_((1.0 - beta2**step_number) ** 0.5).add_(eps)
    param.addcdiv_(exp_avg, denom, value=-step_size)


def _run_torch_adamw_baseline(
    initial: dict[str, torch.Tensor],
    *,
    iterations: int,
    warmup: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    param = torch.nn.Parameter(initial["param_flat"].detach().clone().contiguous())
    param.grad = initial["grad_flat"].detach().clone().contiguous()
    beta1, beta2 = [float(item) for item in config["betas"]]
    kwargs = {
        "lr": float(config["lr"]),
        "betas": (beta1, beta2),
        "eps": float(config["eps"]),
        "weight_decay": float(config["weight_decay"]),
    }
    provider = "torch_adamw_fused"
    try:
        optimizer = torch.optim.AdamW([param], fused=True, **kwargs)
        for _ in range(max(int(warmup), 0)):
            optimizer.step()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(max(int(iterations), 1)):
            optimizer.step()
        torch.cuda.synchronize()
    except Exception:
        provider = "torch_adamw_standard_fallback"
        param = torch.nn.Parameter(initial["param_flat"].detach().clone().contiguous())
        param.grad = initial["grad_flat"].detach().clone().contiguous()
        optimizer = torch.optim.AdamW([param], **kwargs)
        for _ in range(max(int(warmup), 0)):
            optimizer.step()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(max(int(iterations), 1)):
            optimizer.step()
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    state = optimizer.state[param]
    return {
        "provider": provider,
        "elapsed_ms": elapsed_ms,
        "param_flat": param.detach(),
        "exp_avg": state["exp_avg"].detach(),
        "exp_avg_sq": state["exp_avg_sq"].detach(),
    }


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _max_rel_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    diff = (left.detach() - right.detach()).abs()
    denom = right.detach().abs().clamp_min(1e-12)
    return float((diff / denom).max().item())


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "min": ordered[0],
        "median": float(statistics.median(ordered)),
        "max": ordered[-1],
    }


def _case_label(numel: int) -> str:
    if numel <= 8192:
        return "lora_tiny"
    if numel <= 65536:
        return "lora_small"
    if numel <= 262144:
        return "adapter_mid"
    if numel <= 1048576:
        return "dit_block_flat"
    return "large_flat"


def _benchmark_once(lulynx_native: Any, *, numel: int, iterations: int, warmup: int) -> dict[str, Any]:
    base_config = {
        "lr": 1e-3,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "step_index": 0,
        "block_size": 256,
        "max_numel": max(numel, 4096),
    }
    native_buffers = _make_buffers(numel)
    torch_baseline = _run_torch_adamw_baseline(
        native_buffers,
        iterations=iterations,
        warmup=warmup,
        config=base_config,
    )

    runtime = lulynx_native.create_adamw_cuda_kernel_runtime_session_py(str(PROJECT_ROOT), "compute_89")
    assert runtime["ok"] is True, runtime
    runtime_id = int(runtime["runtime_session_id"])
    try:
        bench_config = dict(base_config)
        bench_config.update({"iterations": int(iterations), "warmup": int(warmup), "use_cuda_events": True})
        native = lulynx_native.benchmark_adamw_cuda_kernel_runtime_session_py(
            runtime_id,
            native_buffers["param_flat"],
            native_buffers["grad_flat"],
            native_buffers["exp_avg"],
            native_buffers["exp_avg_sq"],
            json.dumps(bench_config),
        )
        assert native["ok"] is True, native
        assert native["training_dispatch"] is False, native
        assert native["training_path_enabled"] is False, native
        assert native["performance_test_ready"] is False, native
    finally:
        destroyed_runtime = lulynx_native.destroy_adamw_cuda_kernel_runtime_session_py(runtime_id)
        assert destroyed_runtime["ok"] is True, destroyed_runtime

    diffs = {
        "param_flat": _max_abs_diff(native_buffers["param_flat"], torch_baseline["param_flat"]),
        "exp_avg": _max_abs_diff(native_buffers["exp_avg"], torch_baseline["exp_avg"]),
        "exp_avg_sq": _max_abs_diff(native_buffers["exp_avg_sq"], torch_baseline["exp_avg_sq"]),
    }
    rel_diffs = {
        "param_flat": _max_rel_diff(native_buffers["param_flat"], torch_baseline["param_flat"]),
        "exp_avg": _max_rel_diff(native_buffers["exp_avg"], torch_baseline["exp_avg"]),
        "exp_avg_sq": _max_rel_diff(native_buffers["exp_avg_sq"], torch_baseline["exp_avg_sq"]),
    }
    max_diff = max(diffs.values())
    max_rel_diff = max(rel_diffs.values())
    parity_ok = (
        (diffs["param_flat"] <= 3e-4 or rel_diffs["param_flat"] <= 5e-6)
        and (diffs["exp_avg"] <= 5e-6 or rel_diffs["exp_avg"] <= 5e-6)
        and (diffs["exp_avg_sq"] <= 5e-5 or rel_diffs["exp_avg_sq"] <= 2e-5)
    )
    assert parity_ok, {"numel": numel, "diffs": diffs, "rel_diffs": rel_diffs, "native": native}
    native_avg = float(native["avg_ms_native_loop"])
    torch_ms = float(torch_baseline["elapsed_ms"])
    torch_avg = torch_ms / max(iterations, 1)
    return {
        "numel": numel,
        "label": _case_label(numel),
        "iterations": iterations,
        "warmup": warmup,
        "native_avg_ms": native_avg,
        "native_avg_ms_host_wall": float(native.get("avg_ms_host_wall", 0.0) or 0.0),
        "native_avg_ms_cuda_event": float(native.get("avg_ms_cuda_event", 0.0) or 0.0),
        "timing_source": str(native.get("timing_source", "")),
        "torch_adamw_avg_ms": torch_avg,
        "torch_adamw_provider": str(torch_baseline["provider"]),
        "speedup_vs_torch_adamw_loop": torch_avg / native_avg if native_avg > 0 else 0.0,
        "native_elapsed_ms": float(native["elapsed_ms_native_loop"]),
        "native_elapsed_ms_host_wall": float(native.get("elapsed_ms_host_wall", 0.0) or 0.0),
        "native_elapsed_ms_cuda_event": float(native.get("elapsed_ms_cuda_event", 0.0) or 0.0),
        "torch_adamw_elapsed_ms": torch_ms,
        "max_abs_diff": max_diff,
        "max_rel_diff": max_rel_diff,
        "diffs": diffs,
        "rel_diffs": rel_diffs,
        "native_scope": str(native.get("benchmark_scope", "")),
    }


def _benchmark_case(
    lulynx_native: Any,
    *,
    numel: int,
    iterations: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    runs = [
        _benchmark_once(lulynx_native, numel=numel, iterations=iterations, warmup=warmup)
        for _ in range(max(int(repeats), 1))
    ]
    native_event_values = [float(run["native_avg_ms_cuda_event"]) for run in runs if float(run["native_avg_ms_cuda_event"]) > 0]
    native_primary_values = [float(run["native_avg_ms"]) for run in runs]
    torch_values = [float(run["torch_adamw_avg_ms"]) for run in runs]
    speedups = [float(run["speedup_vs_torch_adamw_loop"]) for run in runs]
    best_parity = max(runs, key=lambda run: float(run["max_abs_diff"]))
    return {
        "numel": numel,
        "label": _case_label(numel),
        "iterations": iterations,
        "warmup": warmup,
        "repeats": max(int(repeats), 1),
        "native_avg_ms_stats": _stats(native_primary_values),
        "native_cuda_event_avg_ms_stats": _stats(native_event_values),
        "torch_adamw_avg_ms_stats": _stats(torch_values),
        "speedup_vs_torch_adamw_loop_stats": _stats(speedups),
        "torch_adamw_provider": str(runs[-1]["torch_adamw_provider"]),
        "timing_source": str(runs[-1]["timing_source"]),
        "max_abs_diff": float(best_parity["max_abs_diff"]),
        "max_rel_diff": float(best_parity["max_rel_diff"]),
        "diffs": best_parity["diffs"],
        "rel_diffs": best_parity["rel_diffs"],
        "runs": runs,
    }


def run_smoke(
    shapes: tuple[int, ...] = (4096, 32768, 262144, 1048576),
    iterations: int = 24,
    warmup: int = 4,
    repeats: int = 3,
) -> dict[str, Any]:
    _inject_native_artifact_dir_from_env()
    spec = importlib.util.find_spec("lulynx_native")
    if spec is None:
        return {"schema_version": 1, "probe": "turbocore_cuda_runtime_benchmark_smoke", "ok": True, "skipped": True, "reason": "lulynx_native_not_importable"}
    if not torch.cuda.is_available():
        return {"schema_version": 1, "probe": "turbocore_cuda_runtime_benchmark_smoke", "ok": True, "skipped": True, "reason": "torch_cuda_unavailable"}

    import lulynx_native  # type: ignore

    required = [
        "create_adamw_cuda_kernel_runtime_session_py",
        "destroy_adamw_cuda_kernel_runtime_session_py",
        "benchmark_adamw_cuda_kernel_runtime_session_py",
    ]
    missing = [name for name in required if not hasattr(lulynx_native, name)]
    assert not missing, missing

    cases = [
        _benchmark_case(
            lulynx_native,
            numel=int(numel),
            iterations=int(iterations),
            warmup=int(warmup),
            repeats=int(repeats),
        )
        for numel in shapes
    ]
    return {
        "schema_version": 1,
        "probe": "turbocore_cuda_runtime_benchmark_smoke",
        "ok": True,
        "skipped": False,
        "origin": str(getattr(spec, "origin", "")),
        "device": torch.cuda.get_device_name(0),
        "cases": cases,
        "evidence_level": "short_smoke_not_promotion_gate",
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_kernel_present": False,
        "performance_test_ready": False,
    }


if __name__ == "__main__":
    result = run_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not bool(result.get("ok", False)):
        raise SystemExit(1)
