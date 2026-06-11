"""Report-only scorecard for TurboCore simple-formula optimizer expansion.

V2-P7 starts by proving optimizer math and state schemas before adding any
native kernel.  This module deliberately does not dispatch training updates.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from core.configs import OptimizerType


SIMPLE_FORMULA_OPTIMIZERS = (
    OptimizerType.LION,
    OptimizerType.LION_8BIT,
    OptimizerType.PAGED_LION_8BIT,
    OptimizerType.SGD_NESTEROV,
    OptimizerType.SGD_NESTEROV_8BIT,
    OptimizerType.RADAM_SCHEDULE_FREE,
    OptimizerType.SGD_SCHEDULE_FREE,
)

DEFAULT_DTYPES = ("float32", "float16", "bfloat16")


def build_simple_optimizer_reference_scorecard(
    *,
    dtype_cases: Sequence[str] = DEFAULT_DTYPES,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Build the first V2-P7 scorecard without enabling native dispatch."""

    target_device = device or torch.device("cpu")
    normalized_dtypes = [_normalize_dtype(dtype) for dtype in dtype_cases]
    parity = {
        "lion": _run_lion_reference_matrix(normalized_dtypes, target_device),
        "sgd_nesterov": _run_sgd_nesterov_reference_matrix(normalized_dtypes, target_device),
        "schedule_free": _schedule_free_reference_placeholder(),
    }
    rows = [_optimizer_row(optimizer, parity) for optimizer in SIMPLE_FORMULA_OPTIMIZERS]
    missing = [row["optimizer_type"] for row in rows if row["reference_status"] == "unclassified"]
    parity_cases = _flatten_cases(parity)
    failed_parity = [case for case in parity_cases if not bool(case.get("ok", False))]
    seed_ready = not missing and not failed_parity and _family_ready(rows, "lion") and _family_ready(rows, "sgd_nesterov")
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_reference_scorecard_v0",
        "gate": "simple_formula_optimizer_reference",
        "ok": not missing and not failed_parity,
        "promotion_ready": False,
        "first_stage_ready": seed_ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "runtime_canary_ready": False,
        "e2e_no_regression_ready": False,
        "target_optimizer_types": [optimizer.value for optimizer in SIMPLE_FORMULA_OPTIMIZERS],
        "rows": rows,
        "parity": parity,
        "summary": {
            "target_optimizer_count": len(rows),
            "classified_optimizer_count": len(rows) - len(missing),
            "formula_reference_ready_count": sum(
                1 for row in rows if str(row.get("reference_status", "")).startswith("formula_reference_ready")
            ),
            "state_layout_pending_count": sum(1 for row in rows if "layout_pending" in row["reference_status"]),
            "state_machine_pending_count": sum(1 for row in rows if "state_machine_pending" in row["reference_status"]),
            "parity_case_count": len(parity_cases),
            "passed_parity_case_count": len(parity_cases) - len(failed_parity),
            "native_kernel_ready_count": 0,
        },
        "promotion_blockers": [
            "lion_native_kernel_parity_missing",
            "sgd_nesterov_native_kernel_parity_missing",
            "runtime_canary_hit_missing",
            "e2e_no_regression_missing",
        ],
        "blocked_reasons": _dedupe(
            [f"unclassified_optimizer:{name}" for name in missing]
            + [f"parity_case_failed:{case.get('case')}" for case in failed_parity]
        ),
        "recommended_next_step": "add Rust/CUDA optimizer_kind ABI for Lion and SGDNesterov after reference seed",
        "notes": [
            "This scorecard is report-only and never enables optimizer dispatch.",
            "Lion and SGD Nesterov formula seeds are ready before native kernels.",
            "8-bit, paged, and schedule-free variants still need state layout or state-machine work.",
        ],
    }


def _optimizer_row(optimizer: OptimizerType, parity: Mapping[str, Any]) -> dict[str, Any]:
    if optimizer in {OptimizerType.LION, OptimizerType.LION_8BIT, OptimizerType.PAGED_LION_8BIT}:
        layout_pending = optimizer != OptimizerType.LION
        return {
            "optimizer_type": optimizer.value,
            "family": "lion",
            "formula": "decoupled_weight_decay_sign_momentum",
            "state_schema": {
                "required": ["exp_avg"],
                "state_dtype": "float32_preferred",
                "param_dtype": "float32|float16|bfloat16",
                "quantized_state": optimizer in {OptimizerType.LION_8BIT, OptimizerType.PAGED_LION_8BIT},
                "paged_state": optimizer == OptimizerType.PAGED_LION_8BIT,
            },
            "reference_status": "formula_reference_ready_layout_pending" if layout_pending else "formula_reference_ready",
            "parity_ready": _parity_group_ok(parity, "lion"),
            "native_kernel_status": "not_started",
            "runtime_gate_status": "not_started",
            "training_path_enabled": False,
            "default_behavior_changed": False,
        }
    if optimizer in {OptimizerType.SGD_NESTEROV, OptimizerType.SGD_NESTEROV_8BIT}:
        layout_pending = optimizer == OptimizerType.SGD_NESTEROV_8BIT
        return {
            "optimizer_type": optimizer.value,
            "family": "sgd_nesterov",
            "formula": "coupled_weight_decay_momentum_nesterov",
            "state_schema": {
                "required": ["momentum_buffer"],
                "state_dtype": "param_dtype_or_float32_master",
                "param_dtype": "float32|float16|bfloat16",
                "quantized_state": layout_pending,
                "paged_state": False,
            },
            "reference_status": "formula_reference_ready_layout_pending" if layout_pending else "formula_reference_ready",
            "parity_ready": _parity_group_ok(parity, "sgd_nesterov"),
            "native_kernel_status": "not_started",
            "runtime_gate_status": "not_started",
            "training_path_enabled": False,
            "default_behavior_changed": False,
        }
    if optimizer in {OptimizerType.RADAM_SCHEDULE_FREE, OptimizerType.SGD_SCHEDULE_FREE}:
        return {
            "optimizer_type": optimizer.value,
            "family": "schedule_free_simple",
            "formula": "schedule_free_state_machine_pending",
            "state_schema": {
                "required": ["step", "z", "train_mode", "schedule_weight"],
                "state_dtype": "upstream_reference_required",
                "param_dtype": "float32|float16|bfloat16",
                "scheduler_coupled": True,
            },
            "reference_status": "state_machine_pending",
            "parity_ready": False,
            "native_kernel_status": "not_started",
            "runtime_gate_status": "not_started",
            "training_path_enabled": False,
            "default_behavior_changed": False,
        }
    return {
        "optimizer_type": optimizer.value,
        "family": "unknown",
        "formula": "unknown",
        "state_schema": {},
        "reference_status": "unclassified",
        "parity_ready": False,
        "native_kernel_status": "not_started",
        "runtime_gate_status": "not_started",
        "training_path_enabled": False,
        "default_behavior_changed": False,
    }


def _run_lion_reference_matrix(dtype_cases: Sequence[str], device: torch.device) -> dict[str, Any]:
    cases = []
    for dtype_name in dtype_cases:
        for weight_decay in (0.0, 0.01):
            cases.append(_lion_case(dtype_name, device, weight_decay))
    return {"ok": all(case["ok"] for case in cases), "cases": cases}


def _lion_case(dtype_name: str, device: torch.device, weight_decay: float) -> dict[str, Any]:
    torch_dtype = _torch_dtype(dtype_name)
    if torch_dtype is None:
        return _blocked_case("lion", dtype_name, weight_decay, "unsupported_dtype")
    try:
        param = torch.tensor([0.25, -0.5, 0.125, -0.75], dtype=torch_dtype, device=device)
        grad = torch.tensor([0.1, -0.2, 0.05, 0.3], dtype=torch_dtype, device=device)
        exp_avg = torch.zeros_like(param, dtype=torch.float32)
        vector_param, vector_state = _lion_vector_step(param, grad, exp_avg, lr=1e-3, betas=(0.9, 0.99), weight_decay=weight_decay)
        loop_param, loop_state = _lion_loop_step(param, grad, exp_avg, lr=1e-3, betas=(0.9, 0.99), weight_decay=weight_decay)
        param_diff = _max_abs(vector_param, loop_param)
        state_diff = _max_abs(vector_state, loop_state)
        tolerance = _tolerance(dtype_name)
        return {
            "case": f"lion:{dtype_name}:wd={weight_decay}",
            "optimizer_family": "lion",
            "ok": param_diff <= tolerance and state_diff <= tolerance,
            "dtype": dtype_name,
            "device": str(device),
            "weight_decay": weight_decay,
            "max_param_diff": param_diff,
            "max_state_diff": state_diff,
            "tolerance": tolerance,
        }
    except Exception as exc:
        return _blocked_case("lion", dtype_name, weight_decay, f"{type(exc).__name__}: {exc}")


def _lion_vector_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    *,
    lr: float,
    betas: tuple[float, float],
    weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    beta1, beta2 = betas
    param_fp32 = param.float()
    grad_fp32 = grad.float()
    state_fp32 = exp_avg.float()
    if weight_decay:
        param_fp32 = param_fp32 * (1.0 - lr * weight_decay)
    update = state_fp32 * beta1 + grad_fp32 * (1.0 - beta1)
    next_param = param_fp32 - lr * update.sign()
    next_state = state_fp32 * beta2 + grad_fp32 * (1.0 - beta2)
    return next_param, next_state


def _lion_loop_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    *,
    lr: float,
    betas: tuple[float, float],
    weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    beta1, beta2 = betas
    param_values = param.float().flatten().tolist()
    grad_values = grad.float().flatten().tolist()
    state_values = exp_avg.float().flatten().tolist()
    out_param = []
    out_state = []
    for p_value, g_value, m_value in zip(param_values, grad_values, state_values):
        decayed = p_value * (1.0 - lr * weight_decay) if weight_decay else p_value
        update = m_value * beta1 + g_value * (1.0 - beta1)
        sign = 1.0 if update > 0.0 else -1.0 if update < 0.0 else 0.0
        out_param.append(decayed - lr * sign)
        out_state.append(m_value * beta2 + g_value * (1.0 - beta2))
    return (
        torch.tensor(out_param, dtype=torch.float32, device=param.device).reshape_as(param),
        torch.tensor(out_state, dtype=torch.float32, device=param.device).reshape_as(exp_avg),
    )


def _run_sgd_nesterov_reference_matrix(dtype_cases: Sequence[str], device: torch.device) -> dict[str, Any]:
    cases = []
    for dtype_name in dtype_cases:
        for weight_decay in (0.0, 0.01):
            cases.append(_sgd_nesterov_case(dtype_name, device, weight_decay))
    return {"ok": all(case["ok"] for case in cases), "cases": cases}


def _sgd_nesterov_case(dtype_name: str, device: torch.device, weight_decay: float) -> dict[str, Any]:
    torch_dtype = _torch_dtype(dtype_name)
    if torch_dtype is None:
        return _blocked_case("sgd_nesterov", dtype_name, weight_decay, "unsupported_dtype")
    try:
        param = torch.tensor([0.4, -0.2, 0.15, -0.35], dtype=torch_dtype, device=device)
        grad = torch.tensor([0.03, -0.07, 0.11, -0.05], dtype=torch_dtype, device=device)
        momentum = 0.9
        ours_param, ours_buf = _sgd_nesterov_vector_step(param, grad, torch.zeros_like(param), lr=1e-2, momentum=momentum, weight_decay=weight_decay)
        torch_param = torch.nn.Parameter(param.detach().clone())
        torch_param.grad = grad.detach().clone()
        optimizer = torch.optim.SGD([torch_param], lr=1e-2, momentum=momentum, dampening=0.0, weight_decay=weight_decay, nesterov=True)
        optimizer.step()
        expected_buf = optimizer.state[torch_param]["momentum_buffer"]
        param_diff = _max_abs(ours_param, torch_param.detach())
        state_diff = _max_abs(ours_buf, expected_buf)
        tolerance = _tolerance(dtype_name)
        return {
            "case": f"sgd_nesterov:{dtype_name}:wd={weight_decay}",
            "optimizer_family": "sgd_nesterov",
            "ok": param_diff <= tolerance and state_diff <= tolerance,
            "dtype": dtype_name,
            "device": str(device),
            "weight_decay": weight_decay,
            "max_param_diff": param_diff,
            "max_state_diff": state_diff,
            "tolerance": tolerance,
        }
    except Exception as exc:
        return _blocked_case("sgd_nesterov", dtype_name, weight_decay, f"{type(exc).__name__}: {exc}")


def _sgd_nesterov_vector_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_p = grad
    if weight_decay:
        d_p = d_p.add(param, alpha=weight_decay)
    next_buffer = momentum_buffer.mul(momentum).add(d_p)
    update = d_p.add(next_buffer, alpha=momentum)
    return param.add(update, alpha=-lr), next_buffer


def _schedule_free_reference_placeholder() -> dict[str, Any]:
    return {
        "ok": True,
        "cases": [],
        "status": "state_machine_reference_pending",
        "blocked_reasons": ["schedule_free_state_machine_reference_pending"],
    }


def _flatten_cases(parity: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for payload in parity.values():
        if isinstance(payload, Mapping):
            cases.extend([dict(case) for case in payload.get("cases", []) if isinstance(case, Mapping)])
    return cases


def _family_ready(rows: Sequence[Mapping[str, Any]], family: str) -> bool:
    family_rows = [row for row in rows if row.get("family") == family]
    return bool(family_rows) and all(bool(row.get("parity_ready", False)) for row in family_rows)


def _parity_group_ok(parity: Mapping[str, Any], group: str) -> bool:
    payload = parity.get(group, {})
    return bool(isinstance(payload, Mapping) and payload.get("ok", False))


def _normalize_dtype(value: str) -> str:
    normalized = str(value or "").replace("torch.", "").strip().lower()
    return {"fp16": "float16", "half": "float16", "fp32": "float32", "bf16": "bfloat16"}.get(
        normalized,
        normalized,
    )


def _torch_dtype(dtype_name: str) -> torch.dtype | None:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return None


def _tolerance(dtype_name: str) -> float:
    if dtype_name == "float16":
        return 2e-3
    if dtype_name == "bfloat16":
        return 2e-2
    return 1e-6


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _blocked_case(family: str, dtype_name: str, weight_decay: float, reason: str) -> dict[str, Any]:
    return {
        "case": f"{family}:{dtype_name}:wd={weight_decay}",
        "optimizer_family": family,
        "ok": False,
        "dtype": dtype_name,
        "weight_decay": weight_decay,
        "blocked_reasons": [reason],
    }


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["SIMPLE_FORMULA_OPTIMIZERS", "build_simple_optimizer_reference_scorecard"]
