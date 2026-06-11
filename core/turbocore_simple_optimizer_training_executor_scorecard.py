"""Scorecard for simple optimizer native training executor parity."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from core.turbocore_simple_optimizer_training_executor import (
    build_simple_optimizer_training_executor,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("lion", "sgd_nesterov")


def build_simple_optimizer_training_executor_scorecard(
    *,
    workspace_root: str | Path | None = None,
    steps: int = 4,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run real backward gradients through the default-off native executor."""

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if target_device.type != "cuda":
        return _blocked("cuda_required_for_simple_optimizer_training_executor")
    root = Path(workspace_root or REPO_ROOT)
    cases = [_run_case(kind, root, max(int(steps), 1), target_device) for kind in TARGETS]
    ready = all(bool(case.get("training_executor_ready", False)) for case in cases)
    blockers = [reason for case in cases for reason in case.get("blocked_reasons", [])]
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_training_executor_scorecard_v0",
        "gate": "simple_formula_training_executor_native_steps",
        "ok": all(bool(case.get("ok", False)) for case in cases),
        "promotion_ready": False,
        "training_executor_stage_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "workspace_root": str(root.resolve()),
        "device": str(target_device),
        "steps": max(int(steps), 1),
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "native_step_count": sum(int(case.get("native_step_count", 0) or 0) for case in cases),
            "native_kernel_launch_count": sum(int(case.get("native_kernel_launch_count", 0) or 0) for case in cases),
            "max_param_abs_diff": max(float(case.get("max_param_abs_diff", 0.0) or 0.0) for case in cases),
            "max_state_abs_diff": max(float(case.get("max_state_abs_diff", 0.0) or 0.0) for case in cases),
        },
        "promotion_blockers": _dedupe(blockers + ["product_training_route_not_bound"]),
        "blocked_reasons": _dedupe(blockers),
        "recommended_next_step": (
            "wire simple formula optimizer executor through native dispatch runtime"
            if ready
            else "complete simple formula optimizer training executor parity blockers"
        ),
    }


def _run_case(optimizer_kind: str, workspace_root: Path, steps: int, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(20260603)
    param = torch.nn.Parameter(torch.linspace(-0.25, 0.35, 32, device=device, dtype=torch.float32))
    ref_param = param.detach().clone()
    ref_state = torch.zeros_like(ref_param)
    config = _config(optimizer_kind)
    executor = build_simple_optimizer_training_executor(
        params=[param],
        config=config,
        workspace_root=workspace_root,
    )
    reports: list[dict[str, Any]] = []
    try:
        for step in range(steps):
            if param.grad is not None:
                param.grad = None
            features = torch.linspace(-0.5, 0.5, 32, device=device, dtype=torch.float32)
            target = torch.tensor(0.05 * (step + 1), device=device, dtype=torch.float32)
            loss = ((param * features).sum() - target).pow(2)
            loss.backward()
            grad = param.grad.detach().clone()
            ref_param, ref_state = _reference_step(optimizer_kind, ref_param, grad, ref_state, config)
            reports.append(executor({"training_dispatch": True, "training_path_enabled": True}))
        torch.cuda.synchronize()
    finally:
        executor.close()
    param_diff = float((param.detach() - ref_param).abs().max().item())
    state_diff = float((executor.state_flat.detach() - ref_state).abs().max().item())
    ok_reports = all(bool(report.get("ok", False)) for report in reports)
    parity_ok = param_diff <= 5e-6 and state_diff <= 5e-6
    ready = ok_reports and parity_ok and len(reports) == steps
    return {
        "schema_version": 1,
        "ok": ready,
        "optimizer_kind": optimizer_kind,
        "training_executor_ready": ready,
        "native_step_count": sum(1 for report in reports if report.get("native_step_executed") is True),
        "native_kernel_launch_count": sum(1 for report in reports if report.get("native_kernel_launched") is True),
        "max_param_abs_diff": param_diff,
        "max_state_abs_diff": state_diff,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "blocked_reasons": [] if ready else [f"{optimizer_kind}_training_executor_parity_failed"],
    }


def _config(optimizer_kind: str) -> dict[str, Any]:
    if optimizer_kind == "lion":
        return {"optimizer_kind": "lion", "lr": 1e-3, "betas": [0.9, 0.99], "weight_decay": 0.01}
    return {"optimizer_kind": "sgd_nesterov", "lr": 1e-2, "momentum": 0.9, "weight_decay": 0.01}


def _reference_step(
    optimizer_kind: str,
    param: torch.Tensor,
    grad: torch.Tensor,
    state: torch.Tensor,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    param_fp32 = param.detach().clone().float()
    grad_fp32 = grad.detach().clone().float()
    state_fp32 = state.detach().clone().float()
    lr = float(config["lr"])
    weight_decay = float(config.get("weight_decay", 0.0))
    if optimizer_kind == "lion":
        beta1, beta2 = [float(item) for item in config["betas"]]
        if weight_decay:
            param_fp32 = param_fp32 * (1.0 - lr * weight_decay)
        update = state_fp32 * beta1 + grad_fp32 * (1.0 - beta1)
        return param_fp32 - lr * torch.sign(update), state_fp32 * beta2 + grad_fp32 * (1.0 - beta2)
    momentum = float(config["momentum"])
    d_p = grad_fp32 + param_fp32 * weight_decay if weight_decay else grad_fp32
    next_state = state_fp32 * momentum + d_p
    return param_fp32 - lr * (d_p + momentum * next_state), next_state


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_training_executor_scorecard_v0",
        "gate": "simple_formula_training_executor_native_steps",
        "ok": False,
        "promotion_ready": False,
        "training_executor_stage_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "cases": [],
        "summary": {
            "case_count": 0,
            "native_step_count": 0,
            "native_kernel_launch_count": 0,
            "max_param_abs_diff": 0.0,
            "max_state_abs_diff": 0.0,
        },
        "promotion_blockers": [reason, "product_training_route_not_bound"],
        "blocked_reasons": [reason],
        "recommended_next_step": "run simple optimizer training executor on CUDA",
    }


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_simple_optimizer_training_executor_scorecard"]
