"""Runtime tensor-launch scorecard for simple formula optimizer kernels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from core.services.native_module_loader import native_with_entrypoints


ENTRYPOINTS = (
    "create_simple_optimizer_cuda_kernel_runtime_session_py",
    "step_simple_optimizer_cuda_kernel_runtime_session_py",
    "destroy_simple_optimizer_cuda_kernel_runtime_session_py",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("lion", "sgd_nesterov")
TOLERANCE = 5e-6


def build_simple_optimizer_runtime_launch_scorecard(
    *,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Launch Lion/SGD Nesterov kernels on real CUDA tensors without training dispatch."""

    native = native_with_entrypoints(*ENTRYPOINTS)
    if native is None:
        return _blocked("simple_optimizer_runtime_entrypoints_missing")
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if target_device.type != "cuda":
        return _blocked("cuda_required_for_simple_optimizer_runtime_launch")
    root = str(Path(workspace_root or REPO_ROOT).resolve())
    cuda_arch = str(arch or _cuda_arch(target_device))
    cases = [_run_case(native, kind, root, cuda_arch, target_device) for kind in TARGETS]
    ready_cases = [case for case in cases if bool(case.get("runtime_launch_ready", False))]
    blockers = [reason for case in cases for reason in case.get("blocked_reasons", [])]
    ready = len(ready_cases) == len(TARGETS)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_runtime_launch_scorecard_v0",
        "gate": "simple_formula_runtime_tensor_launch",
        "ok": all(bool(case.get("ok", False)) for case in cases),
        "promotion_ready": False,
        "runtime_launch_stage_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "workspace_root": root,
        "arch": cuda_arch,
        "device": str(target_device),
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "runtime_launch_ready_count": len(ready_cases),
            "kernel_executed_count": sum(1 for case in cases if case.get("kernel_executed") is True),
            "parity_ready_count": sum(1 for case in cases if case.get("parity_ok") is True),
        },
        "promotion_blockers": _dedupe(blockers + ["training_dispatch_executor_missing"]),
        "blocked_reasons": _dedupe(blockers),
        "recommended_next_step": (
            "wire simple formula optimizer runtime launch into a default-off training executor"
            if ready
            else "complete simple formula runtime tensor launch blockers"
        ),
    }


def _run_case(native: Any, optimizer_kind: str, workspace_root: str, arch: str, device: torch.device) -> dict[str, Any]:
    runtime_id: int | None = None
    try:
        created = native.create_simple_optimizer_cuda_kernel_runtime_session_py(
            optimizer_kind,
            workspace_root,
            arch,
        )
        create_report = dict(created) if isinstance(created, Mapping) else {}
        if not bool(create_report.get("ok", False)):
            return _case_blocked(
                optimizer_kind,
                "simple_optimizer_runtime_create_failed",
                create_report=create_report,
            )
        runtime_id = int(create_report.get("runtime_session_id", 0) or 0)
        param, grad, state, config = _case_inputs(optimizer_kind, device)
        expected_param, expected_state = _reference(optimizer_kind, param, grad, state, config)
        step = native.step_simple_optimizer_cuda_kernel_runtime_session_py(
            runtime_id,
            param,
            grad,
            state,
            json.dumps({**config, "max_numel": int(param.numel()), "training_dispatch": False}),
        )
        step_report = dict(step) if isinstance(step, Mapping) else {}
        param_diff = float((param.detach() - expected_param).abs().max().item())
        state_diff = float((state.detach() - expected_state).abs().max().item())
        parity_ok = max(param_diff, state_diff) <= TOLERANCE
        ok = bool(step_report.get("ok", False)) and parity_ok
        return {
            "schema_version": 1,
            "ok": ok,
            "optimizer_kind": optimizer_kind,
            "runtime_session_created": True,
            "kernel_executed": bool(step_report.get("kernel_executed", False)),
            "parameters_mutated": bool(step_report.get("parameters_mutated", False)),
            "training_tensor_binding": bool(step_report.get("training_tensor_binding", False)),
            "training_path_enabled": bool(step_report.get("training_path_enabled", False)),
            "native_dispatch_allowed": False,
            "runtime_launch_ready": ok,
            "parity_ok": parity_ok,
            "param_max_abs_diff": param_diff,
            "state_max_abs_diff": state_diff,
            "max_abs_diff": max(param_diff, state_diff),
            "tolerance": TOLERANCE,
            "create_report": create_report,
            "step_report": step_report,
            "blocked_reasons": [] if ok else [f"{optimizer_kind}_runtime_launch_parity_missing"],
        }
    except Exception as exc:  # pragma: no cover - CUDA/native dependent
        return _case_blocked(
            optimizer_kind,
            f"simple_optimizer_runtime_launch_failed:{type(exc).__name__}",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if runtime_id is not None:
            try:
                native.destroy_simple_optimizer_cuda_kernel_runtime_session_py(runtime_id)
            except Exception:
                pass


def _case_inputs(optimizer_kind: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    numel = 64
    index = torch.arange(numel, device=device, dtype=torch.float32)
    sign2 = torch.where((index.remainder(2) == 0), 1.0, -1.0)
    sign3 = torch.where((index.remainder(3) == 0), -1.0, 1.0)
    param = (sign2 * (0.125 + index * 0.003)).contiguous()
    grad = (sign3 * (0.01 + index * 0.0002)).contiguous()
    state = (torch.where((index.remainder(5) == 0), -1.0, 1.0) * index * 0.0001).contiguous()
    if optimizer_kind == "lion":
        return param, grad, state, {"lr": 1e-3, "betas": [0.9, 0.99], "weight_decay": 0.01, "block_size": 128}
    return param, grad, state, {"lr": 1e-2, "momentum": 0.9, "weight_decay": 0.01, "block_size": 128}


def _reference(
    optimizer_kind: str,
    param: torch.Tensor,
    grad: torch.Tensor,
    state: torch.Tensor,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    param_fp32 = param.detach().clone().float()
    grad_fp32 = grad.detach().clone().float()
    state_fp32 = state.detach().clone().float()
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))
    if optimizer_kind == "lion":
        beta1, beta2 = [float(item) for item in config.get("betas", [0.9, 0.99])]
        if weight_decay:
            param_fp32 = param_fp32 * (1.0 - lr * weight_decay)
        update = state_fp32 * beta1 + grad_fp32 * (1.0 - beta1)
        return param_fp32 - lr * torch.sign(update), state_fp32 * beta2 + grad_fp32 * (1.0 - beta2)
    momentum = float(config.get("momentum", 0.9))
    d_p = grad_fp32 + param_fp32 * weight_decay if weight_decay else grad_fp32
    next_state = state_fp32 * momentum + d_p
    return param_fp32 - lr * (d_p + momentum * next_state), next_state


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


def _case_blocked(optimizer_kind: str, reason: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "ok": False,
        "optimizer_kind": optimizer_kind,
        "runtime_launch_ready": False,
        "kernel_executed": False,
        "parameters_mutated": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [reason],
    }
    payload.update(extra)
    return payload


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_runtime_launch_scorecard_v0",
        "gate": "simple_formula_runtime_tensor_launch",
        "ok": False,
        "promotion_ready": False,
        "runtime_launch_stage_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "cases": [],
        "summary": {
            "case_count": 0,
            "runtime_launch_ready_count": 0,
            "kernel_executed_count": 0,
            "parity_ready_count": 0,
        },
        "promotion_blockers": [reason, "training_dispatch_executor_missing"],
        "blocked_reasons": [reason],
        "recommended_next_step": "build or load lulynx_native with simple optimizer runtime entrypoints",
    }


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_simple_optimizer_runtime_launch_scorecard"]
