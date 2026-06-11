"""Backward-hook native canary for LOMO-style fused-backward updates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pytorch_optimizer import LOMO

from core.services.native_module_loader import native_with_entrypoints


ENTRYPOINT = "probe_lomo_fused_backward_training_hook_canary_py"
REPO_ROOT = Path(__file__).resolve().parents[2]


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.linspace(-0.11, 0.17, steps=64, device="cuda", dtype=torch.float32).view(8, 8).contiguous()
        )

    def forward(self) -> torch.Tensor:
        return ((self.weight * 0.29) ** 2).mean() + self.weight.mean() * 0.006


def build_lomo_fused_backward_hook_canary_scorecard() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _blocked("cuda_required_for_lomo_fused_backward_hook_canary")
    case = _run_case()
    ok = case.get("ok") is True
    blockers = list(case.get("blocked_reasons", []) or [])
    return {
        "schema_version": 1,
        "scorecard": "turbocore_lomo_fused_backward_hook_canary_scorecard_v0",
        "gate": "lomo_fused_backward_native_hook_canary",
        "ok": ok,
        "promotion_ready": False,
        "backward_hook_canary_ready": ok,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready": False,
        "optimizer_family": "plugin_fused_backward",
        "case": case,
        "summary": {
            "optimizer_count": 1,
            "backward_hook_canary_ready_count": 1 if ok else 0,
            "native_step_count": 1 if case.get("native_step_executed") is True else 0,
            "native_kernel_launch_count": 1 if case.get("native_kernel_launched") is True else 0,
            "backward_hook_native_launch_count": int(case.get("backward_hook_native_launch_count", 0) or 0),
            "optimizer_step_called_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": blockers
        + [
            "lomo_e2e_shadow_matrix_missing",
            "lomo_owner_release_approval_missing",
            "fused_backward_family_runtime_launch_incomplete",
        ],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "expand LOMO fused-backward hook canary into selected fused-backward shadow matrix"
            if ok
            else "fix LOMO fused-backward hook canary blockers"
        ),
    }


def _run_case() -> dict[str, Any]:
    model = _ToyModel()
    optimizer = LOMO(model, lr=1.0e-3, clip_grad_norm=None, clip_grad_value=None)
    native = native_with_entrypoints(ENTRYPOINT)
    if native is None:
        return _case_blocked("lomo_fused_backward_native_entrypoint_missing")
    launches: list[dict[str, Any]] = []
    step_called = False
    native_lr = 1.0e-3

    def _native_hook(grad: torch.Tensor) -> torch.Tensor:
        launch = dict(
            getattr(native, ENTRYPOINT)(
                model.weight,
                grad.detach().float().contiguous(),
                int(model.weight.numel()),
                float(native_lr),
                0.0,
                1.0,
                str(REPO_ROOT.resolve()),
                _cuda_arch(model.weight.device),
            )
        )
        launches.append(launch)
        return grad

    def _step_guard(*_args: Any, **_kwargs: Any) -> None:
        nonlocal step_called
        step_called = True
        raise AssertionError("lomo_canary_forbids_public_optimizer_step")

    hook_handle = model.weight.register_hook(_native_hook)
    optimizer.step = _step_guard  # type: ignore[method-assign]
    before = model.weight.detach().clone()
    try:
        loss = model.forward()
        optimizer.fused_backward(loss, lr=0.0)
    finally:
        hook_handle.remove()
    after = model.weight.detach().clone()
    native_ok = bool(launches and launches[0].get("ok") is True)
    mutated = _max_abs_diff(before, after) > 0.0
    grad_cleared = model.weight.grad is None
    ok = bool(
        native_ok
        and launches[0].get("kernel_executed") is True
        and launches[0].get("backward_hook_gradient_owner") is True
        and mutated
        and grad_cleared
        and not step_called
        and int(getattr(optimizer, "lr", 0.0) or 0.0) == 0
    )
    return {
        "schema_version": 1,
        "ok": ok,
        "probe": "lomo_fused_backward_native_hook_canary_v0",
        "optimizer_kind": "lomo",
        "loss": float(loss.detach().cpu().item()),
        "native_step_executed": native_ok,
        "native_kernel_launched": launches[0].get("kernel_executed") is True if launches else False,
        "backward_hook_gradient_owner": launches[0].get("backward_hook_gradient_owner") is True if launches else False,
        "backward_hook_native_launch_count": len(launches),
        "training_parameters_mutated": mutated,
        "optimizer_step_called": step_called,
        "public_optimizer_step_forbidden": True,
        "grad_cleared_after_fused_backward": grad_cleared,
        "python_lomo_lr_during_backward": float(getattr(optimizer, "lr", 0.0) or 0.0),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "launch": launches[0] if launches else {},
        "blocked_reasons": [] if ok else _case_blockers(launches, mutated, grad_cleared, step_called),
    }


def _case_blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "probe": "lomo_fused_backward_native_hook_canary_v0",
        "blocked_reasons": [reason],
    }


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_lomo_fused_backward_hook_canary_scorecard_v0",
        "gate": "lomo_fused_backward_native_hook_canary",
        "ok": False,
        "promotion_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "case": {},
        "summary": {"native_step_count": 0, "native_kernel_launch_count": 0},
        "promotion_blockers": [reason],
        "blocked_reasons": [reason],
        "recommended_next_step": "run LOMO fused-backward hook canary on CUDA",
    }


def _case_blockers(launches: list[dict[str, Any]], mutated: bool, grad_cleared: bool, step_called: bool) -> list[str]:
    blockers: list[str] = []
    if not launches:
        blockers.append("lomo_backward_hook_native_launch_missing")
    else:
        launch = launches[0]
        blockers.extend(str(item) for item in launch.get("blocked_reasons", []) or [])
        if launch.get("kernel_executed") is not True:
            blockers.append("lomo_backward_hook_kernel_not_executed")
        if launch.get("backward_hook_gradient_owner") is not True:
            blockers.append("lomo_backward_hook_gradient_owner_missing")
    if not mutated:
        blockers.append("lomo_backward_hook_parameters_not_mutated")
    if not grad_cleared:
        blockers.append("lomo_fused_backward_grad_not_cleared")
    if step_called:
        blockers.append("lomo_public_optimizer_step_called")
    return _dedupe(blockers or ["lomo_fused_backward_hook_canary_failed"])


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().detach().cpu().item())


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


__all__ = ["build_lomo_fused_backward_hook_canary_scorecard"]
