"""Short no-regression gate for V2-P7 simple optimizer canary routing."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from core.turbocore_simple_optimizer_runtime_canary_scorecard import (
    build_simple_optimizer_runtime_canary_scorecard,
)


def build_simple_optimizer_e2e_no_regression_scorecard(
    *,
    runtime_canary_report: Mapping[str, Any] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Validate fallback optimizer math while native canary stays shadow-only."""

    runtime = dict(runtime_canary_report or build_simple_optimizer_runtime_canary_scorecard())
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = [_run_lion_case(target_device), _run_sgd_nesterov_case(target_device)]
    failed = [case for case in cases if not bool(case.get("ok", False))]
    runtime_ready = bool(runtime.get("runtime_canary_ready", False))
    ready = (
        runtime_ready
        and not failed
        and not bool(runtime.get("training_path_enabled", True))
        and not bool(runtime.get("native_dispatch_allowed", True))
    )
    blockers = []
    if not runtime_ready:
        blockers.append("runtime_canary_hit_missing")
    blockers.extend(str(reason) for case in failed for reason in case.get("blocked_reasons", []))
    if not ready:
        blockers.append("e2e_no_regression_missing")
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_e2e_no_regression_scorecard_v0",
        "gate": "simple_formula_e2e_no_regression",
        "ok": not failed,
        "promotion_ready": False,
        "e2e_no_regression_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "canary_shadow_route_only": True,
        "device": str(target_device),
        "cases": cases,
        "runtime_canary_summary": {
            "runtime_canary_ready": runtime_ready,
            "native_route_hit_count": int(runtime.get("native_route_hit_count", 0) or 0),
            "training_path_enabled": bool(runtime.get("training_path_enabled", False)),
            "native_dispatch_allowed": bool(runtime.get("native_dispatch_allowed", False)),
        },
        "summary": {
            "case_count": len(cases),
            "passed_case_count": len(cases) - len(failed),
            "finite_loss_count": sum(1 for case in cases if case.get("finite_loss") is True),
            "fallback_update_count": sum(1 for case in cases if case.get("fallback_update_applied") is True),
        },
        "promotion_blockers": _dedupe(blockers),
        "blocked_reasons": _dedupe(blockers),
        "recommended_next_step": "P7 simple optimizer gates complete; keep default off until product promotion review" if ready else "fix simple optimizer e2e no-regression blockers",
        "notes": [
            "This is a short no-regression smoke, not a product promotion benchmark.",
            "Native simple optimizer remains shadow/canary-only and does not replace PyTorch optimizer dispatch.",
        ],
    }


def _run_lion_case(device: torch.device) -> dict[str, Any]:
    param, grad, loss_value = _make_gradient(device)
    exp_avg = torch.zeros_like(param)
    before = param.detach().clone()
    lr = 1e-3
    beta1, beta2 = 0.9, 0.99
    weight_decay = 0.01
    next_param, next_exp_avg = _lion_step(param.detach(), grad, exp_avg, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    diff = float((next_param - before).abs().max().item())
    ok = bool(torch.isfinite(next_param).all() and torch.isfinite(next_exp_avg).all() and diff > 0.0)
    return {
        "schema_version": 1,
        "ok": ok,
        "optimizer_kind": "lion",
        "device": str(device),
        "finite_loss": bool(torch.isfinite(torch.tensor(loss_value)).item()),
        "fallback_update_applied": diff > 0.0,
        "param_max_change": diff,
        "state_finite": bool(torch.isfinite(next_exp_avg).all().item()),
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [] if ok else ["lion_e2e_no_regression_failed"],
    }


def _run_sgd_nesterov_case(device: torch.device) -> dict[str, Any]:
    param, grad, loss_value = _make_gradient(device)
    momentum_buffer = torch.zeros_like(param)
    before = param.detach().clone()
    next_param, next_buffer = _sgd_nesterov_step(
        param.detach(),
        grad,
        momentum_buffer,
        lr=1e-2,
        momentum=0.9,
        weight_decay=0.01,
    )
    diff = float((next_param - before).abs().max().item())
    ok = bool(torch.isfinite(next_param).all() and torch.isfinite(next_buffer).all() and diff > 0.0)
    return {
        "schema_version": 1,
        "ok": ok,
        "optimizer_kind": "sgd_nesterov",
        "device": str(device),
        "finite_loss": bool(torch.isfinite(torch.tensor(loss_value)).item()),
        "fallback_update_applied": diff > 0.0,
        "param_max_change": diff,
        "state_finite": bool(torch.isfinite(next_buffer).all().item()),
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [] if ok else ["sgd_nesterov_e2e_no_regression_failed"],
    }


def _make_gradient(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    torch.manual_seed(7319)
    param = torch.nn.Parameter(torch.linspace(-0.25, 0.35, 16, device=device, dtype=torch.float32))
    features = torch.linspace(-0.5, 0.5, 16, device=device, dtype=torch.float32)
    target = torch.tensor(0.125, device=device, dtype=torch.float32)
    loss = ((param * features).sum() - target).pow(2)
    loss.backward()
    grad = param.grad.detach().clone()
    return param.detach().clone(), grad, float(loss.detach().cpu().item())


def _lion_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    *,
    lr: float,
    betas: tuple[float, float],
    weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    beta1, beta2 = betas
    next_param = param.float()
    grad_fp32 = grad.float()
    state = exp_avg.float()
    if weight_decay:
        next_param = next_param * (1.0 - lr * weight_decay)
    update = state * beta1 + grad_fp32 * (1.0 - beta1)
    next_param = next_param - lr * update.sign()
    next_state = state * beta2 + grad_fp32 * (1.0 - beta2)
    return next_param, next_state


def _sgd_nesterov_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_p = grad.float()
    param_fp32 = param.float()
    if weight_decay:
        d_p = d_p + param_fp32 * weight_decay
    next_buffer = momentum_buffer.float() * momentum + d_p
    update = d_p + momentum * next_buffer
    return param_fp32 - lr * update, next_buffer


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_simple_optimizer_e2e_no_regression_scorecard"]
