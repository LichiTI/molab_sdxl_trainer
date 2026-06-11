"""Owner-to-training-parameter copyback dispatch validation for TurboCore."""

from __future__ import annotations

import time
from typing import Any, Iterable, Mapping

import torch


def build_copyback_dispatch_probe(
    owner: Any,
    params: Iterable[torch.nn.Parameter],
    *,
    scratch_probe: Mapping[str, Any] | None = None,
    restore_after_validation: bool = True,
) -> dict[str, Any]:
    """Validate owner.copy_params_to_ against real parameters behind an explicit flag."""

    param_list = [param for param in params if isinstance(param, torch.Tensor)]
    started = time.perf_counter()
    scratch = dict(scratch_probe or {})
    scratch_ok = bool(scratch.get("scratch_copyback_validated", False))
    payload = _base_payload(param_list, scratch_ok=scratch_ok)
    if not scratch_ok:
        payload.update(
            {
                "copyback_dispatch_validated": False,
                "real_parameters_mutated": False,
                "parameters_mutated": False,
                "restore_after_validation": bool(restore_after_validation),
                "real_parameters_restored": True,
                "reason": "scratch_copyback_not_validated",
                "elapsed_ms": _elapsed_ms(started),
            }
        )
        return payload

    before_tensors = [param.detach().clone() for param in param_list]
    before_flat = _flatten_params(before_tensors)
    before_strides = [tuple(param.stride()) for param in param_list]
    try:
        expected = _flatten_owner_like_params(owner.param_flat.detach(), param_list)
        owner.copy_params_to_(param_list)
        after_flat = _flatten_params(param_list)
        dispatch_diff = (after_flat.float() - expected.float()).abs()
        before_diff = (after_flat.float() - before_flat.float()).abs()
        dispatch_max = _max(dispatch_diff)
        dispatch_mean = _mean(dispatch_diff)
        before_max = _max(before_diff)
        shape_ok = all(tuple(param.shape) == tuple(src.shape) for param, src in zip(param_list, before_tensors))
        stride_ok = all(tuple(param.stride()) == stride for param, stride in zip(param_list, before_strides))
        dtype_ok = dispatch_max <= _dtype_tolerance(param_list)
        mutated = bool(before_max > 0.0)
        restored = False
        restore_max = None
        if restore_after_validation:
            with torch.no_grad():
                for param, before in zip(param_list, before_tensors):
                    param.copy_(before)
            restored_flat = _flatten_params(param_list)
            restore_diff = (restored_flat.float() - before_flat.float()).abs()
            restore_max = _max(restore_diff)
            restored = bool(restore_max <= _dtype_tolerance(param_list))
        validated = bool(shape_ok and stride_ok and dtype_ok and (restored or not restore_after_validation))
        payload.update(
            {
                "copyback_dispatch_validated": validated,
                "shape_validated": shape_ok,
                "stride_preserved": stride_ok,
                "dtype_cast_validated": dtype_ok,
                "expected_flat_numel": int(expected.numel()),
                "dispatched_flat_numel": int(after_flat.numel()),
                "dispatch_max_abs_diff": dispatch_max,
                "dispatch_mean_abs_diff": dispatch_mean,
                "real_parameter_max_abs_diff_from_before": before_max,
                "real_parameters_mutated": mutated,
                "parameters_mutated": mutated,
                "restore_after_validation": bool(restore_after_validation),
                "real_parameters_restored": restored,
                "restore_max_abs_diff": restore_max,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive report for research probes
        _restore_params(param_list, before_tensors)
        payload.update(
            {
                "copyback_dispatch_validated": False,
                "shape_validated": False,
                "stride_preserved": False,
                "dtype_cast_validated": False,
                "real_parameters_mutated": False,
                "parameters_mutated": False,
                "restore_after_validation": bool(restore_after_validation),
                "real_parameters_restored": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    payload["elapsed_ms"] = _elapsed_ms(started)
    return payload


def _base_payload(params: list[torch.Tensor], *, scratch_ok: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "probe": "turbocore_copyback_dispatch_probe_v0",
        "enabled": True,
        "copyback_dispatch_enabled": True,
        "copyback_target": "training_parameters",
        "copyback_dispatch_target": "training_parameters",
        "requires_scratch_copyback_validated": True,
        "scratch_copyback_validated": bool(scratch_ok),
        "parameter_tensors": len(params),
        "parameter_numel": int(sum(param.numel() for param in params)),
        "target_dtypes": sorted({str(param.dtype).replace("torch.", "") for param in params}),
        "target_devices": sorted({str(param.device) for param in params}),
        "training_path_enabled": False,
    }


def _restore_params(params: list[torch.Tensor], before_tensors: list[torch.Tensor]) -> None:
    if not before_tensors:
        return
    with torch.no_grad():
        for param, before in zip(params, before_tensors):
            param.copy_(before)


def _flatten_params(params: Iterable[torch.Tensor]) -> torch.Tensor:
    tensors = [param.detach().float().reshape(-1) for param in params if isinstance(param, torch.Tensor)]
    if not tensors:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(tensors).contiguous()


def _flatten_owner_like_params(flat: torch.Tensor, params: Iterable[torch.Tensor]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    offset = 0
    for param in params:
        count = int(param.numel())
        parts.append(flat.narrow(0, offset, count).view_as(param).detach().float().reshape(-1))
        offset += count
    if not parts:
        return torch.empty(0, dtype=torch.float32, device=flat.device)
    return torch.cat(parts).contiguous()


def _dtype_tolerance(params: list[torch.Tensor]) -> float:
    dtypes = {param.dtype for param in params}
    if torch.float16 in dtypes or torch.bfloat16 in dtypes:
        return 5e-3
    return 1e-6


def _max(tensor: torch.Tensor) -> float:
    return float(tensor.max().detach().cpu().item()) if tensor.numel() else 0.0


def _mean(tensor: torch.Tensor) -> float:
    return float(tensor.mean().detach().cpu().item()) if tensor.numel() else 0.0


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 4)


__all__ = ["build_copyback_dispatch_probe"]
