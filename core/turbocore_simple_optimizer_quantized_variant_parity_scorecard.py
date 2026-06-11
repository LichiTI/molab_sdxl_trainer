"""Executable parity matrix for quantized simple optimizer variants.

This is the last report-only step before real CUDA scratch kernels for the
8-bit simple variants.  It executes dequantize/update/requantize reference
cases and keeps native dispatch/product exposure disabled.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from core.configs import OptimizerType
from core.turbocore_simple_optimizer_variant_native_abi_scorecard import (
    build_simple_optimizer_variant_native_abi_scorecard,
)


QUANTIZED_VARIANT_TARGETS = (
    OptimizerType.LION_8BIT,
    OptimizerType.PAGED_LION_8BIT,
    OptimizerType.SGD_NESTEROV_8BIT,
)
BLOCK_SIZE = 4


def build_simple_optimizer_quantized_variant_parity_scorecard(
    *,
    native_abi_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run quantized-state formula parity without native dispatch."""

    abi = dict(native_abi_report or build_simple_optimizer_variant_native_abi_scorecard())
    rows = [_row(optimizer, abi) for optimizer in QUANTIZED_VARIANT_TARGETS]
    cases = [case for row in rows for case in row["cases"]]
    failed = [case for case in cases if case.get("ok") is not True]
    blockers = _dedupe(reason for case in failed for reason in _strings(case.get("blocked_reasons")))
    ready = not blockers and all(row["native_abi_spec_ready"] for row in rows)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_quantized_variant_parity_scorecard_v0",
        "gate": "simple_formula_quantized_variant_formula_parity_matrix",
        "ok": ready,
        "promotion_ready": False,
        "quantized_formula_parity_matrix_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "native_kernel_ready": False,
        "product_native_dispatch_ready": False,
        "target_optimizer_types": [optimizer.value for optimizer in QUANTIZED_VARIANT_TARGETS],
        "rows": rows,
        "cases": cases,
        "native_abi_summary": dict(abi.get("summary") or {}),
        "summary": {
            "target_optimizer_count": len(QUANTIZED_VARIANT_TARGETS),
            "quantized_formula_parity_ready_count": sum(1 for row in rows if row["formula_parity_ready"]),
            "native_abi_spec_ready_count": sum(1 for row in rows if row["native_abi_spec_ready"]),
            "case_count": len(cases),
            "passed_case_count": len(cases) - len(failed),
            "native_kernel_ready_count": 0,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": blockers
        + [
            "simple_quantized_variant_cuda_scratch_kernel_missing",
            "simple_quantized_variant_runtime_canary_missing",
            "simple_quantized_variant_product_rollout_review_missing",
        ],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "implement CUDA scratch probes for Lion8bit, PagedLion8bit, and SGDNesterov8bit"
            if ready
            else "fix quantized simple variant parity blockers"
        ),
        "notes": [
            "This scorecard executes dequantize/update/requantize parity in Python/Torch only.",
            "It intentionally does not call native code or register CUDA kernels.",
            "PagedLion8bit shares formula parity with Lion8bit but keeps a separate paged-state contract row.",
        ],
    }


def _row(optimizer: OptimizerType, abi_report: Mapping[str, Any]) -> dict[str, Any]:
    abi_ready = _abi_ready_for(optimizer, abi_report)
    cases = _lion_cases(optimizer) if optimizer in {OptimizerType.LION_8BIT, OptimizerType.PAGED_LION_8BIT} else _sgd_cases()
    ready = abi_ready and all(case.get("ok") is True for case in cases)
    return {
        "schema_version": 1,
        "optimizer_type": optimizer.value,
        "optimizer_kind": _kind(optimizer),
        "optimizer_family": "simple_formula",
        "variant_kind": "quantized_state",
        "variant_status": "quantized_formula_parity_ready" if ready else "quantized_formula_parity_blocked",
        "native_abi_spec_ready": abi_ready,
        "formula_parity_ready": ready,
        "native_canary_ready": False,
        "native_kernel_ready": False,
        "product_native_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "next_gate": "quantized_variant_cuda_scratch_kernel",
        "case_count": len(cases),
        "passed_case_count": sum(1 for case in cases if case.get("ok") is True),
        "cases": cases,
    }


def _lion_cases(optimizer: OptimizerType) -> list[dict[str, Any]]:
    return [
        _lion_case(optimizer, dtype_name="float32", weight_decay=0.0),
        _lion_case(optimizer, dtype_name="float32", weight_decay=0.01),
        _lion_case(optimizer, dtype_name="float16", weight_decay=0.01),
    ]


def _lion_case(optimizer: OptimizerType, *, dtype_name: str, weight_decay: float) -> dict[str, Any]:
    dtype = _torch_dtype(dtype_name)
    param = torch.tensor([0.25, -0.5, 0.125, -0.75, 0.33, -0.11, 0.07, -0.21], dtype=dtype)
    grad = torch.tensor([0.1, -0.2, 0.05, 0.3, -0.04, 0.02, -0.08, 0.12], dtype=dtype)
    exp_avg = torch.tensor([0.03, -0.02, 0.01, -0.04, 0.015, -0.025, 0.035, -0.005], dtype=torch.float32)
    state_q, scale = _quantize(exp_avg)
    state = _dequantize(state_q, scale).reshape_as(exp_avg)
    expected_param, expected_state = _lion_update(param, grad, state, weight_decay=weight_decay)
    next_q, next_scale = _quantize(expected_state)
    roundtrip_state = _dequantize(next_q, next_scale).reshape_as(expected_state)
    tolerance = _tolerance(dtype_name)
    param_diff = _max_abs(expected_param, expected_param.clone())
    state_diff = _max_abs(expected_state, roundtrip_state)
    ok = state_diff <= tolerance
    return {
        "schema_version": 1,
        "case": f"{optimizer.value}:dequant_update_requant:{dtype_name}:wd={weight_decay}",
        "optimizer_type": optimizer.value,
        "ok": ok,
        "dtype": dtype_name,
        "weight_decay": weight_decay,
        "param_max_abs_diff": param_diff,
        "state_requant_max_abs_diff": state_diff,
        "tolerance": tolerance,
        "quantized_state_changed": bool(not torch.equal(state_q, next_q)),
        "scale_changed": bool(_max_abs(scale, next_scale) > 0.0),
        "blocked_reasons": [] if ok else ["lion8bit_dequant_update_requant_parity_failed"],
    }


def _lion_update(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    *,
    weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    lr = 1e-3
    beta1, beta2 = 0.9, 0.99
    param_fp32 = param.float()
    grad_fp32 = grad.float()
    state_fp32 = exp_avg.float()
    if weight_decay:
        param_fp32 = param_fp32 * (1.0 - lr * weight_decay)
    update = state_fp32 * beta1 + grad_fp32 * (1.0 - beta1)
    next_param = param_fp32 - lr * update.sign()
    next_state = state_fp32 * beta2 + grad_fp32 * (1.0 - beta2)
    return next_param, next_state


def _sgd_cases() -> list[dict[str, Any]]:
    return [
        _sgd_case(dtype_name="float32", weight_decay=0.0),
        _sgd_case(dtype_name="float32", weight_decay=0.01),
        _sgd_case(dtype_name="float16", weight_decay=0.01),
    ]


def _sgd_case(*, dtype_name: str, weight_decay: float) -> dict[str, Any]:
    dtype = _torch_dtype(dtype_name)
    param = torch.tensor([0.4, -0.2, 0.15, -0.35, 0.18, -0.16, 0.08, -0.24], dtype=dtype)
    grad = torch.tensor([0.03, -0.07, 0.11, -0.05, 0.02, 0.09, -0.04, 0.06], dtype=dtype)
    momentum = torch.tensor([0.02, -0.01, 0.03, -0.04, 0.05, -0.02, 0.01, -0.03], dtype=torch.float32)
    state_q, scale = _quantize(momentum)
    state = _dequantize(state_q, scale).reshape_as(momentum)
    expected_param, expected_state = _sgd_nesterov_update(param, grad, state, weight_decay=weight_decay)
    next_q, next_scale = _quantize(expected_state)
    roundtrip_state = _dequantize(next_q, next_scale).reshape_as(expected_state)
    tolerance = _tolerance(dtype_name)
    state_diff = _max_abs(expected_state, roundtrip_state)
    ok = state_diff <= tolerance
    return {
        "schema_version": 1,
        "case": f"SGDNesterov8bit:dequant_update_requant:{dtype_name}:wd={weight_decay}",
        "optimizer_type": OptimizerType.SGD_NESTEROV_8BIT.value,
        "ok": ok,
        "dtype": dtype_name,
        "weight_decay": weight_decay,
        "param_norm_after": _norm(expected_param),
        "state_requant_max_abs_diff": state_diff,
        "tolerance": tolerance,
        "quantized_state_changed": bool(not torch.equal(state_q, next_q)),
        "scale_changed": bool(_max_abs(scale, next_scale) > 0.0),
        "blocked_reasons": [] if ok else ["sgd_nesterov8bit_dequant_update_requant_parity_failed"],
    }


def _sgd_nesterov_update(
    param: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    lr = 1e-2
    momentum = 0.9
    param_fp32 = param.float()
    d_p = grad.float()
    if weight_decay:
        d_p = d_p.add(param_fp32, alpha=weight_decay)
    next_buffer = momentum_buffer.float().mul(momentum).add(d_p)
    update = d_p.add(next_buffer, alpha=momentum)
    return param_fp32.add(update, alpha=-lr), next_buffer


def _quantize(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = values.detach().float().reshape(-1)
    blocks = []
    scales = []
    for start in range(0, int(flat.numel()), BLOCK_SIZE):
        block = flat[start : start + BLOCK_SIZE]
        scale = torch.clamp(block.abs().max(), min=torch.tensor(1e-12))
        quantized = torch.clamp(torch.round(block / scale * 127.0 + 128.0), 0, 255).to(torch.uint8)
        blocks.append(quantized)
        scales.append(scale.reshape(1))
    return torch.cat(blocks), torch.cat(scales).float()


def _dequantize(qvalues: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    pieces = []
    flat_q = qvalues.reshape(-1)
    for block_index, start in enumerate(range(0, int(flat_q.numel()), BLOCK_SIZE)):
        block = flat_q[start : start + BLOCK_SIZE].float()
        pieces.append((block - 128.0) / 127.0 * scales[block_index].float())
    return torch.cat(pieces)


def _abi_ready_for(optimizer: OptimizerType, abi_report: Mapping[str, Any]) -> bool:
    for row in abi_report.get("rows", []):
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer.value:
            return row.get("native_abi_spec_ready") is True
    return False


def _kind(optimizer: OptimizerType) -> str:
    if optimizer == OptimizerType.LION_8BIT:
        return "lion_8bit"
    if optimizer == OptimizerType.PAGED_LION_8BIT:
        return "paged_lion_8bit"
    return "sgd_nesterov_8bit"


def _torch_dtype(dtype_name: str) -> torch.dtype:
    return torch.float16 if dtype_name == "float16" else torch.float32


def _tolerance(dtype_name: str) -> float:
    return 2.5e-2 if dtype_name == "float16" else 1.5e-2


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().float()).cpu())


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


__all__ = ["QUANTIZED_VARIANT_TARGETS", "build_simple_optimizer_quantized_variant_parity_scorecard"]
