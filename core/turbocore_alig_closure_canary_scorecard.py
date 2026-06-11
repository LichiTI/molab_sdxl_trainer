"""Closure-native canary for AliG representative runtime rehearsal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pytorch_optimizer import AliG

from core.services.native_module_loader import native_with_entrypoints


ENTRYPOINT = "probe_alig_closure_training_canary_py"
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_alig_closure_canary_scorecard() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _blocked("cuda_required_for_alig_closure_canary")
    case = _run_case()
    ok = case.get("ok") is True
    blockers = list(case.get("blocked_reasons", []) or [])
    return {
        "schema_version": 1,
        "scorecard": "turbocore_alig_closure_canary_scorecard_v0",
        "gate": "alig_closure_native_canary",
        "ok": ok,
        "promotion_ready": False,
        "closure_canary_ready": ok,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready": False,
        "optimizer_family": "plugin_closure_or_second_order",
        "case": case,
        "summary": {
            "optimizer_count": 1,
            "closure_canary_ready_count": 1 if ok else 0,
            "closure_call_count": int(case.get("closure_call_count", 0) or 0),
            "native_step_count": 1 if case.get("native_step_executed") is True else 0,
            "native_kernel_launch_count": 1 if case.get("native_kernel_launched") is True else 0,
            "optimizer_step_called_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": blockers
        + [
            "alig_e2e_shadow_matrix_missing",
            "alig_owner_release_approval_missing",
            "closure_second_order_family_runtime_launch_incomplete",
        ],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "expand AliG closure canary into selected closure/second-order shadow matrix"
            if ok
            else "fix AliG closure canary blockers"
        ),
    }


def _run_case() -> dict[str, Any]:
    param = torch.nn.Parameter(
        torch.linspace(-0.13, 0.19, steps=64, device="cuda", dtype=torch.float32).view(8, 8).contiguous()
    )
    optimizer = AliG([param], max_lr=1.0e-2, momentum=0.0)
    native = native_with_entrypoints(ENTRYPOINT)
    if native is None:
        return _case_blocked("alig_closure_native_entrypoint_missing")
    closure_calls = 0
    step_called = False

    def closure() -> float:
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad(set_to_none=True)
        loss_tensor = ((param.float() * 0.23) ** 2).mean() + param.float().mean() * 0.005
        loss_tensor.backward()
        return float(loss_tensor.detach().cpu().item())

    def _step_guard(*_args: Any, **_kwargs: Any) -> None:
        nonlocal step_called
        step_called = True
        raise AssertionError("alig_canary_forbids_public_optimizer_step")

    before = param.detach().clone()
    loss = closure()
    step_size = float(optimizer.compute_step_size(loss))
    max_lr = optimizer.param_groups[0].get("max_lr")
    if max_lr is not None:
        step_size = min(step_size, float(max_lr))
    grad = param.grad.detach().float().contiguous() if param.grad is not None else torch.zeros_like(param)
    optimizer.step = _step_guard  # type: ignore[method-assign]
    try:
        launch = dict(
            getattr(native, ENTRYPOINT)(
                param,
                grad,
                int(param.numel()),
                float(step_size),
                str(REPO_ROOT.resolve()),
                _cuda_arch(param.device),
            )
        )
    except Exception as exc:  # pragma: no cover - native/CUDA dependent
        return _case_blocked(f"alig_native_step_call_failed:{type(exc).__name__}:{exc}")
    optimizer.param_groups[0]["step"] = int(optimizer.param_groups[0].get("step", 0) or 0) + 1
    optimizer.zero_grad(set_to_none=True)
    after = param.detach().clone()
    mutated = _max_abs_diff(before, after) > 0.0
    ok = bool(
        launch.get("ok") is True
        and launch.get("kernel_executed") is True
        and launch.get("closure_replay") is True
        and closure_calls == 1
        and step_size > 0.0
        and mutated
        and not step_called
        and param.grad is None
    )
    return {
        "schema_version": 1,
        "ok": ok,
        "probe": "alig_closure_native_canary_v0",
        "optimizer_kind": "alig",
        "loss": loss,
        "step_size": step_size,
        "closure_call_count": closure_calls,
        "native_step_executed": launch.get("ok") is True,
        "native_kernel_launched": launch.get("kernel_executed") is True,
        "closure_replay": launch.get("closure_replay") is True,
        "training_parameters_mutated": mutated,
        "optimizer_step_called": step_called,
        "public_optimizer_step_forbidden": True,
        "grad_cleared_after_native_step": param.grad is None,
        "step_after_native": int(optimizer.param_groups[0].get("step", 0) or 0),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "launch": launch,
        "blocked_reasons": [] if ok else _case_blockers(launch, closure_calls, step_size, mutated, step_called, param.grad is None),
    }


def _case_blocked(reason: str) -> dict[str, Any]:
    return {"schema_version": 1, "ok": False, "probe": "alig_closure_native_canary_v0", "blocked_reasons": [reason]}


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_alig_closure_canary_scorecard_v0",
        "gate": "alig_closure_native_canary",
        "ok": False,
        "promotion_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "case": {},
        "summary": {"native_step_count": 0, "native_kernel_launch_count": 0},
        "promotion_blockers": [reason],
        "blocked_reasons": [reason],
        "recommended_next_step": "run AliG closure canary on CUDA",
    }


def _case_blockers(
    launch: dict[str, Any],
    closure_calls: int,
    step_size: float,
    mutated: bool,
    step_called: bool,
    grad_cleared: bool,
) -> list[str]:
    blockers = [str(item) for item in launch.get("blocked_reasons", []) or []]
    if launch.get("kernel_executed") is not True:
        blockers.append("alig_closure_kernel_not_executed")
    if launch.get("closure_replay") is not True:
        blockers.append("alig_closure_replay_not_recorded")
    if closure_calls != 1:
        blockers.append("alig_closure_call_count_mismatch")
    if step_size <= 0.0:
        blockers.append("alig_closure_step_size_invalid")
    if not mutated:
        blockers.append("alig_closure_parameters_not_mutated")
    if step_called:
        blockers.append("alig_public_optimizer_step_called")
    if not grad_cleared:
        blockers.append("alig_grad_not_cleared")
    return _dedupe(blockers or ["alig_closure_canary_failed"])


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


__all__ = ["build_alig_closure_canary_scorecard"]
