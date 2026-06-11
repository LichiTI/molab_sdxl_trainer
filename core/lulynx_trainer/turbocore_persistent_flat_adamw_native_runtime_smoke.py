"""Owner-backed smoke for the dev-only CUDA AdamW runtime benchmark.

This probe uses ``PersistentFlatAdamW`` as the flat-buffer owner, validates the
current-process handle/request boundary, then launches the cached native AdamW
kernel directly against the owner's buffers. It is not a training dispatcher.
"""

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

from core.turbocore_flat_adamw_state import FlatAdamWConfig, PersistentFlatAdamW  # noqa: E402
from core.turbocore_native_tensor_binding import build_flat_adamw_native_binding_request  # noqa: E402
from core.turbocore_tensor_handle_registry import (  # noqa: E402
    build_tensor_object_map_for_handles,
    register_persistent_flat_adamw_buffers,
)
from core.services.native_module_loader import ensure_lulynx_native_artifact_path  # noqa: E402


SHAPE_PRESETS: dict[int, list[tuple[int, ...]]] = {
    4096: [(64, 32), (32, 32), (1024,)],
    32768: [(128, 128), (64, 128), (8192,)],
    262144: [(256, 256), (256, 512), (256, 256)],
    1048576: [(512, 512), (512, 1024), (512, 512)],
}


def _inject_native_artifact_dir_from_env() -> None:
    ensure_lulynx_native_artifact_path()
    raw = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    if not raw:
        return
    path = Path(raw).expanduser()
    if path.is_dir():
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    return {"min": ordered[0], "median": float(statistics.median(ordered)), "max": ordered[-1]}


def _make_owner_inputs(numel: int, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    shapes = SHAPE_PRESETS[int(numel)]
    total = sum(_shape_numel(shape) for shape in shapes)
    flat = (torch.arange(total, dtype=torch.float32, device=device) * 0.0005 + 0.5).contiguous()
    grad = (torch.arange(total, dtype=torch.float32, device=device) * 0.00001 + 0.01).contiguous()
    values: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    offset = 0
    for shape in shapes:
        count = _shape_numel(shape)
        values.append(flat[offset : offset + count].view(shape).clone().contiguous())
        grads.append(grad[offset : offset + count].view(shape).clone().contiguous())
        offset += count
    return values, grads


def _shape_numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _make_owner(values: list[torch.Tensor], grads: list[torch.Tensor], cfg: FlatAdamWConfig) -> PersistentFlatAdamW:
    owner = PersistentFlatAdamW([tensor.detach().clone() for tensor in values], cfg)
    owner.set_grads([tensor.detach().clone() for tensor in grads])
    return owner


def _reference_step(owner: PersistentFlatAdamW, step_index: int) -> None:
    beta1, beta2 = owner.config.betas
    exp_avg = owner.exp_avg
    exp_avg_sq = owner.exp_avg_sq
    grad = owner.grad_flat
    param = owner.param_flat
    step_number = int(step_index) + 1
    exp_avg.mul_(float(beta1)).add_(grad, alpha=1.0 - float(beta1))
    exp_avg_sq.mul_(float(beta2)).addcmul_(grad, grad, value=1.0 - float(beta2))
    if float(owner.config.weight_decay):
        param.mul_(1.0 - float(owner.config.lr) * float(owner.config.weight_decay))
    step_size = float(owner.config.lr) / (1.0 - float(beta1) ** step_number)
    denom = exp_avg_sq.sqrt().div_((1.0 - float(beta2) ** step_number) ** 0.5).add_(float(owner.config.eps))
    param.addcdiv_(exp_avg, denom, value=-step_size)


def _time_reference_owner(owner: PersistentFlatAdamW, *, iterations: int, warmup: int) -> dict[str, float]:
    for step in range(max(int(warmup), 0)):
        _reference_step(owner, step)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    host_started = time.perf_counter()
    for offset in range(max(int(iterations), 1)):
        _reference_step(owner, max(int(warmup), 0) + offset)
    end_event.record()
    end_event.synchronize()
    host_ms = (time.perf_counter() - host_started) * 1000.0
    event_ms = float(start_event.elapsed_time(end_event))
    return {
        "elapsed_ms_cuda_event": event_ms,
        "avg_ms_cuda_event": event_ms / max(int(iterations), 1),
        "elapsed_ms_host_wall": host_ms,
        "avg_ms_host_wall": host_ms / max(int(iterations), 1),
    }


def _validate_owner_binding(lulynx_native: Any, owner: PersistentFlatAdamW) -> dict[str, Any]:
    registry, handles, descriptor = register_persistent_flat_adamw_buffers(owner)
    request = build_flat_adamw_native_binding_request(registry, handles)
    tensor_map = build_tensor_object_map_for_handles(registry, handles)
    session = lulynx_native.create_flat_adamw_tensor_binding_session(json.dumps(request), tensor_map)
    assert session["ok"] is True, session
    session_id = int(session["session_id"])
    try:
        launch_plan = lulynx_native.tensor_binding_session_launch_plan(session_id)
        assert launch_plan["ok"] is True, launch_plan
        assert int(launch_plan["numel"]) == int(owner.param_flat.numel()), launch_plan
        for role, handle_id in handles.items():
            assert int(tensor_map[handle_id].data_ptr()) == int(getattr(owner, role).data_ptr())
    finally:
        destroyed = lulynx_native.destroy_tensor_binding_session(session_id)
        assert destroyed["ok"] is True, destroyed
    return {
        "descriptor_layout": descriptor.get("layout"),
        "handle_count": len(handles),
        "launch_plan_numel": int(launch_plan["numel"]),
        "training_path_enabled": False,
    }


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _max_rel_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    diff = (left.detach() - right.detach()).abs()
    denom = right.detach().abs().clamp_min(1e-12)
    return float((diff / denom).max().item())


def _assert_owner_parity(native_owner: PersistentFlatAdamW, ref_owner: PersistentFlatAdamW) -> dict[str, Any]:
    diffs = {
        "param_flat": _max_abs_diff(native_owner.param_flat, ref_owner.param_flat),
        "exp_avg": _max_abs_diff(native_owner.exp_avg, ref_owner.exp_avg),
        "exp_avg_sq": _max_abs_diff(native_owner.exp_avg_sq, ref_owner.exp_avg_sq),
    }
    rel_diffs = {
        "param_flat": _max_rel_diff(native_owner.param_flat, ref_owner.param_flat),
        "exp_avg": _max_rel_diff(native_owner.exp_avg, ref_owner.exp_avg),
        "exp_avg_sq": _max_rel_diff(native_owner.exp_avg_sq, ref_owner.exp_avg_sq),
    }
    parity_ok = (
        (diffs["param_flat"] <= 3e-4 or rel_diffs["param_flat"] <= 5e-6)
        and (diffs["exp_avg"] <= 5e-6 or rel_diffs["exp_avg"] <= 5e-6)
        and (diffs["exp_avg_sq"] <= 5e-5 or rel_diffs["exp_avg_sq"] <= 2e-5)
    )
    assert parity_ok, {"diffs": diffs, "rel_diffs": rel_diffs}
    return {"max_abs_diff": max(diffs.values()), "max_rel_diff": max(rel_diffs.values()), "diffs": diffs, "rel_diffs": rel_diffs}


def _run_once(lulynx_native: Any, *, numel: int, iterations: int, warmup: int) -> dict[str, Any]:
    device = torch.device("cuda")
    cfg = FlatAdamWConfig(lr=1e-3, weight_decay=0.01, max_grad_norm=0.0, finite_check=False, block_size=256)
    values, grads = _make_owner_inputs(numel, device)
    native_owner = _make_owner(values, grads, cfg)
    ref_owner = _make_owner(values, grads, cfg)
    binding = _validate_owner_binding(lulynx_native, native_owner)
    ptrs_before = {role: int(getattr(native_owner, role).data_ptr()) for role in ("param_flat", "grad_flat", "exp_avg", "exp_avg_sq")}
    ref_timing = _time_reference_owner(ref_owner, iterations=iterations, warmup=warmup)

    runtime = lulynx_native.create_adamw_cuda_kernel_runtime_session_py(str(PROJECT_ROOT), "compute_89")
    assert runtime["ok"] is True, runtime
    runtime_id = int(runtime["runtime_session_id"])
    try:
        native = lulynx_native.benchmark_adamw_cuda_kernel_runtime_session_py(
            runtime_id,
            native_owner.param_flat,
            native_owner.grad_flat,
            native_owner.exp_avg,
            native_owner.exp_avg_sq,
            json.dumps({
                "lr": cfg.lr,
                "betas": list(cfg.betas),
                "eps": cfg.eps,
                "weight_decay": cfg.weight_decay,
                "step_index": 0,
                "block_size": cfg.block_size,
                "iterations": int(iterations),
                "warmup": int(warmup),
                "max_numel": max(int(numel), 4096),
                "use_cuda_events": True,
            }),
        )
        assert native["ok"] is True, native
    finally:
        destroyed = lulynx_native.destroy_adamw_cuda_kernel_runtime_session_py(runtime_id)
        assert destroyed["ok"] is True, destroyed
    ptrs_after = {role: int(getattr(native_owner, role).data_ptr()) for role in ptrs_before}
    assert ptrs_before == ptrs_after, {"before": ptrs_before, "after": ptrs_after}
    parity = _assert_owner_parity(native_owner, ref_owner)
    native_avg = float(native["avg_ms_cuda_event"])
    ref_avg = float(ref_timing["avg_ms_cuda_event"])
    return {
        "numel": int(numel),
        "parameter_tensors": len(SHAPE_PRESETS[int(numel)]),
        "iterations": int(iterations),
        "warmup": int(warmup),
        "binding": binding,
        "native_avg_ms_cuda_event": native_avg,
        "native_avg_ms_host_wall": float(native["avg_ms_host_wall"]),
        "owner_torch_avg_ms_cuda_event": ref_avg,
        "owner_torch_avg_ms_host_wall": float(ref_timing["avg_ms_host_wall"]),
        "speedup_vs_owner_torch_math": ref_avg / native_avg if native_avg > 0 else 0.0,
        "pointers_stable": True,
        "owner_step_index_managed_by_probe": False,
        **parity,
    }


def _run_case(lulynx_native: Any, *, numel: int, iterations: int, warmup: int, repeats: int) -> dict[str, Any]:
    runs = [_run_once(lulynx_native, numel=numel, iterations=iterations, warmup=warmup) for _ in range(max(int(repeats), 1))]
    return {
        "numel": int(numel),
        "repeats": max(int(repeats), 1),
        "native_avg_ms_stats": _stats([float(run["native_avg_ms_cuda_event"]) for run in runs]),
        "owner_torch_avg_ms_stats": _stats([float(run["owner_torch_avg_ms_cuda_event"]) for run in runs]),
        "speedup_vs_owner_torch_math_stats": _stats([float(run["speedup_vs_owner_torch_math"]) for run in runs]),
        "max_abs_diff": max(float(run["max_abs_diff"]) for run in runs),
        "max_rel_diff": max(float(run["max_rel_diff"]) for run in runs),
        "runs": runs,
    }


def run_smoke(shapes: tuple[int, ...] = (4096, 262144, 1048576), iterations: int = 24, warmup: int = 4, repeats: int = 2) -> dict[str, Any]:
    _inject_native_artifact_dir_from_env()
    spec = importlib.util.find_spec("lulynx_native")
    if spec is None:
        return {"schema_version": 1, "probe": "turbocore_persistent_flat_adamw_native_runtime_smoke", "ok": True, "skipped": True, "reason": "lulynx_native_not_importable"}
    if not torch.cuda.is_available():
        return {"schema_version": 1, "probe": "turbocore_persistent_flat_adamw_native_runtime_smoke", "ok": True, "skipped": True, "reason": "torch_cuda_unavailable"}

    import lulynx_native  # type: ignore

    required = [
        "create_flat_adamw_tensor_binding_session",
        "tensor_binding_session_launch_plan",
        "destroy_tensor_binding_session",
        "create_adamw_cuda_kernel_runtime_session_py",
        "destroy_adamw_cuda_kernel_runtime_session_py",
        "benchmark_adamw_cuda_kernel_runtime_session_py",
    ]
    missing = [name for name in required if not hasattr(lulynx_native, name)]
    assert not missing, missing

    cases = [_run_case(lulynx_native, numel=int(numel), iterations=int(iterations), warmup=int(warmup), repeats=int(repeats)) for numel in shapes]
    return {
        "schema_version": 1,
        "probe": "turbocore_persistent_flat_adamw_native_runtime_smoke",
        "ok": True,
        "skipped": False,
        "origin": str(getattr(spec, "origin", "")),
        "device": torch.cuda.get_device_name(0),
        "cases": cases,
        "evidence_level": "owner_backed_short_smoke_not_training_dispatch",
        "persistent_flat_owner": True,
        "flatten_scatter_in_timed_loop": False,
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
