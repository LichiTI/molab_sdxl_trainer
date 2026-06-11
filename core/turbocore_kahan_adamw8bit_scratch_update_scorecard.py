"""Report-only scratch update parity for KahanAdamW8bit."""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from core.lulynx_trainer.kahan_adamw8bit import KahanAdamW8bit


SCRATCH_UPDATE_KIND = "kahan_adamw8bit_python_scratch_update_v0"
MATRIX_CASES = (
    {"case": "kahan_cpu_fp32_4096", "numel": 4096, "dtype": torch.float32},
    {"case": "kahan_cpu_bf16_4096", "numel": 4096, "dtype": torch.bfloat16},
    {"case": "kahan_cpu_bf16_8192", "numel": 8192, "dtype": torch.bfloat16},
)


def build_kahan_adamw8bit_scratch_update_scorecard() -> dict[str, Any]:
    """Compare direct scratch formula with the local KahanAdamW8bit optimizer."""

    cases = [_safe_case(case, lambda item=case: _run_case(item)) for case in MATRIX_CASES]
    matrix_ready = all(str(case.get("status")) == "passed" for case in cases)
    validations = _validations(cases, matrix_ready)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_kahan_adamw8bit_scratch_update_scorecard_v0",
        "gate": "kahan_adamw8bit_scratch_update_parity",
        "ok": ready,
        "promotion_ready": False,
        "scratch_update_parity_ready": ready,
        "native_kernel_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "optimizer_kind": "kahan_adamw8bit",
        "optimizer_family": "adamw_quantized_kahan",
        "scratch_update_kind": SCRATCH_UPDATE_KIND,
        "matrix_cases": cases,
        "validations": validations,
        "summary": {
            "case_count": len(cases),
            "passed_case_count": sum(1 for case in cases if str(case.get("status")) == "passed"),
            "failed_case_count": sum(1 for case in cases if str(case.get("status")) == "failed"),
            "max_param_diff": _max_case_value(cases, "max_param_diff"),
            "max_kahan_comp_diff": _max_case_value(cases, "max_kahan_comp_diff"),
            "max_absmax_diff": _max_case_value(cases, "max_absmax_diff"),
            "quantized_state_mismatch_count": sum(
                int(case.get("quantized_state_mismatch_count") or 0) for case in cases
            ),
            "native_kernel_ready": False,
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "kahan_adamw8bit_native_scratch_kernel_missing",
                "kahan_adamw8bit_runtime_canary_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "implement KahanAdamW8bit native scratch kernel behind report-only canary"
            if ready
            else "fix KahanAdamW8bit scratch update parity blockers"
        ),
        "notes": [
            "This scorecard is CPU/Python reference work only.",
            "It proves the scratch formula can reproduce the local optimizer step.",
            "It does not enable native dispatch or touch user training runs.",
        ],
    }


def _run_case(case: Mapping[str, Any]) -> dict[str, Any]:
    numel = int(case["numel"])
    dtype = case["dtype"]
    base = torch.linspace(-1.0, 1.0, steps=numel, dtype=torch.float32).to(dtype)
    grad1 = torch.linspace(-0.05, 0.05, steps=numel, dtype=torch.float32).to(dtype)
    grad2 = torch.linspace(0.04, -0.03, steps=numel, dtype=torch.float32).to(dtype)

    prime_param = torch.nn.Parameter(base.clone())
    prime = KahanAdamW8bit([prime_param], lr=1e-3, weight_decay=0.01)
    _optimizer_step(prime_param, prime, grad1)
    checkpoint = copy.deepcopy(prime.state_dict())
    saved_param = prime_param.detach().clone()

    reference_param = torch.nn.Parameter(saved_param.clone())
    reference = KahanAdamW8bit([reference_param], lr=1e-3, weight_decay=0.01)
    reference.load_state_dict(copy.deepcopy(checkpoint))
    _optimizer_step(reference_param, reference, grad2)

    scratch_param = torch.nn.Parameter(saved_param.clone())
    scratch = KahanAdamW8bit([scratch_param], lr=1e-3, weight_decay=0.01)
    scratch.load_state_dict(copy.deepcopy(checkpoint))
    _scratch_step(scratch_param, scratch, grad2)

    reference_state = reference.state[reference_param]
    scratch_state = scratch.state[scratch_param]
    param_compare = _compare_tensor(reference_param.detach(), scratch_param.detach())
    state_compare = _compare_kahan_state(reference_state, scratch_state)
    step_match = int(reference_state.get("step", 0)) == int(scratch_state.get("step", 0))
    ok = bool(param_compare["ok"]) and bool(state_compare["ok"]) and step_match
    return {
        "schema_version": 1,
        "case": str(case["case"]),
        "numel": numel,
        "dtype": str(dtype).replace("torch.", ""),
        "status": "passed" if ok else "failed",
        "ok": ok,
        "step_match": step_match,
        "reference_step": int(reference_state.get("step", 0)),
        "scratch_step": int(scratch_state.get("step", 0)),
        "param_compare": param_compare,
        "state_compare": state_compare,
        "max_param_diff": param_compare.get("max_diff"),
        "max_kahan_comp_diff": state_compare.get("max_kahan_comp_diff"),
        "max_absmax_diff": state_compare.get("max_absmax_diff"),
        "quantized_state_mismatch_count": state_compare.get("quantized_state_mismatch_count"),
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [] if ok else _case_blockers(param_compare, state_compare, step_match),
    }


def _optimizer_step(param: torch.nn.Parameter, optimizer: KahanAdamW8bit, grad: torch.Tensor) -> None:
    param.grad = grad.detach().clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _scratch_step(param: torch.nn.Parameter, optimizer: KahanAdamW8bit, grad: torch.Tensor) -> None:
    group = optimizer.param_groups[0]
    state = optimizer.state[param]
    beta1, beta2 = group["betas"]
    lr = float(group["lr"])
    eps = float(group["eps"])
    weight_decay = float(group["weight_decay"])

    state["step"] = int(state.get("step", 0)) + 1
    step = int(state["step"])
    exp_avg = state["exp_avg_q"].dequantize(torch.float32).to(param.device)
    exp_avg_sq = state["exp_avg_sq_q"].dequantize(torch.float32).to(param.device)
    kahan_comp = state["kahan_comp"]
    grad_fp32 = grad.float()

    exp_avg.mul_(float(beta1)).add_(grad_fp32, alpha=1.0 - float(beta1))
    exp_avg_sq.mul_(float(beta2)).addcmul_(grad_fp32, grad_fp32, value=1.0 - float(beta2))
    state["exp_avg_q"].update(exp_avg)
    state["exp_avg_sq_q"].update(exp_avg_sq)

    bias_correction1 = 1.0 - float(beta1) ** step
    bias_correction2 = 1.0 - float(beta2) ** step
    step_size = lr / bias_correction1
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
    update = exp_avg / denom * (-step_size)
    if weight_decay != 0.0:
        update.add_(param.data.float(), alpha=-lr * weight_decay)

    kahan_y = update - kahan_comp
    param_fp32 = param.data.float()
    kahan_t = param_fp32 + kahan_y
    kahan_comp.copy_((kahan_t - param_fp32) - kahan_y)
    param.data.copy_(kahan_t.to(param.dtype))


def _compare_tensor(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    same_shape = left.shape == right.shape
    same_dtype = left.dtype == right.dtype
    max_diff = _max_abs(left, right) if same_shape else float("inf")
    return {
        "schema_version": 1,
        "ok": same_shape and same_dtype and max_diff == 0.0,
        "shape_match": same_shape,
        "dtype_match": same_dtype,
        "left_dtype": str(left.dtype).replace("torch.", ""),
        "right_dtype": str(right.dtype).replace("torch.", ""),
        "max_diff": max_diff,
        "tolerance": 0.0,
    }


def _compare_kahan_state(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    quant_mismatch = 0
    max_absmax_diff = 0.0
    for key in ("exp_avg_q", "exp_avg_sq_q"):
        left_q = left[key]
        right_q = right[key]
        quant_mismatch += int((left_q.data != right_q.data).sum().item())
        max_absmax_diff = max(max_absmax_diff, _max_abs(left_q.absmax, right_q.absmax))
    max_kahan = _max_abs(left["kahan_comp"], right["kahan_comp"])
    ok = quant_mismatch == 0 and max_absmax_diff == 0.0 and max_kahan == 0.0
    return {
        "schema_version": 1,
        "ok": ok,
        "quantized_state_mismatch_count": quant_mismatch,
        "max_absmax_diff": max_absmax_diff,
        "max_kahan_comp_diff": max_kahan,
    }


def _validations(cases: Sequence[Mapping[str, Any]], matrix_ready: bool) -> list[dict[str, Any]]:
    return [
        _validation(
            "scratch_update_matrix_ready",
            matrix_ready,
            "kahan_adamw8bit_scratch_update_matrix_failed",
        ),
        _validation(
            "quantized_state_exact",
            sum(int(case.get("quantized_state_mismatch_count") or 0) for case in cases) == 0,
            "kahan_adamw8bit_scratch_update_quantized_state_mismatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not any(bool(case.get("training_path_enabled", True)) for case in cases),
            "kahan_adamw8bit_scratch_update_changed_default_behavior",
        ),
    ]


def _safe_case(case: Mapping[str, Any], fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = fn()
        payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        return payload
    except Exception as exc:
        return {
            "schema_version": 1,
            "case": str(case.get("case") or "unknown"),
            "numel": int(case.get("numel") or 0),
            "status": "failed",
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": [f"kahan_adamw8bit_scratch_update_case_failed:{type(exc).__name__}"],
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _case_blockers(
    param_compare: Mapping[str, Any],
    state_compare: Mapping[str, Any],
    step_match: bool,
) -> list[str]:
    blockers = []
    if not bool(param_compare.get("ok", False)):
        blockers.append("kahan_adamw8bit_scratch_update_param_parity_failed")
    if not bool(state_compare.get("ok", False)):
        blockers.append("kahan_adamw8bit_scratch_update_state_parity_failed")
    if not step_match:
        blockers.append("kahan_adamw8bit_scratch_update_step_mismatch")
    return blockers


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 and right.numel() == 0:
        return 0.0
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def _max_case_value(cases: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(case[key]) for case in cases if case.get(key) is not None]
    return max(values) if values else None


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["SCRATCH_UPDATE_KIND", "build_kahan_adamw8bit_scratch_update_scorecard"]
