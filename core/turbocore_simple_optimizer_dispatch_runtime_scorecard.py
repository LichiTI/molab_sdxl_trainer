"""Scorecard for simple optimizer execution through native dispatch runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from core.turbocore_native_update_dispatch_runtime import TurboCoreNativeUpdateDispatchRuntime
from core.turbocore_simple_optimizer_training_executor import build_simple_optimizer_training_executor


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("lion", "sgd_nesterov")


def build_simple_optimizer_dispatch_runtime_scorecard(
    *,
    workspace_root: str | Path | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Use the shared dispatch runtime wrapper to call simple native executors."""

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if target_device.type != "cuda":
        return _blocked("cuda_required_for_simple_optimizer_dispatch_runtime")
    root = Path(workspace_root or REPO_ROOT)
    cases = [_run_case(kind, root, target_device) for kind in TARGETS]
    ready = all(bool(case.get("dispatch_runtime_ready", False)) for case in cases)
    blockers = [reason for case in cases for reason in case.get("blocked_reasons", [])]
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_dispatch_runtime_scorecard_v0",
        "gate": "simple_formula_dispatch_runtime_native_steps",
        "ok": all(bool(case.get("ok", False)) for case in cases),
        "promotion_ready": False,
        "dispatch_runtime_stage_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "workspace_root": str(root.resolve()),
        "device": str(target_device),
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "native_step_count": sum(1 for case in cases if case.get("native_step_executed") is True),
            "skip_pytorch_count": sum(1 for case in cases if case.get("should_call_pytorch_optimizer_step") is False),
        },
        "promotion_blockers": _dedupe(blockers + ["product_training_route_not_bound"]),
        "blocked_reasons": _dedupe(blockers),
        "recommended_next_step": (
            "bind simple formula optimizer dispatch to representative product training canary"
            if ready
            else "complete simple formula dispatch runtime blockers"
        ),
    }


def _run_case(optimizer_kind: str, workspace_root: Path, device: torch.device) -> dict[str, Any]:
    param = torch.nn.Parameter(torch.linspace(-0.2, 0.3, 16, device=device, dtype=torch.float32))
    features = torch.linspace(-0.5, 0.5, 16, device=device, dtype=torch.float32)
    loss = ((param * features).sum() - torch.tensor(0.125, device=device)).pow(2)
    loss.backward()
    executor = build_simple_optimizer_training_executor(
        params=[param],
        config=_config(optimizer_kind),
        workspace_root=workspace_root,
    )
    runtime = TurboCoreNativeUpdateDispatchRuntime()
    try:
        report = runtime.prepare_step(
            step=1,
            arming_report={
                "previous_request_requested": True,
                "armed_for_native_dispatch": True,
                "execute_native_step": True,
            },
            kernel_launch_plan={"launch_allowed": True, "launch_attempted": False},
            runtime_context={
                "native_update_executor_present": True,
                "native_update_runtime_execution_guard_enabled": True,
                "native_update_training_mutation_guard_enabled": True,
                "native_update_training_dispatch_enabled": True,
                "native_update_runtime_dispatch_available": True,
                "training_path_enabled": True,
            },
            native_executor=executor,
        )
    finally:
        executor.close()
    ready = (
        bool(report.get("native_step_executed", False))
        and bool(report.get("native_kernel_launched", False))
        and bool(report.get("training_executor", {}).get("ok", False))
        and report.get("should_call_pytorch_optimizer_step") is False
        and not report.get("blocked_reasons")
    )
    return {
        "schema_version": 1,
        "ok": ready,
        "optimizer_kind": optimizer_kind,
        "dispatch_runtime_ready": ready,
        "native_step_executed": bool(report.get("native_step_executed", False)),
        "native_kernel_launched": bool(report.get("native_kernel_launched", False)),
        "should_call_pytorch_optimizer_step": bool(report.get("should_call_pytorch_optimizer_step", True)),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "blocked_reasons": list(report.get("blocked_reasons", []) or []),
    }


def _config(optimizer_kind: str) -> dict[str, Any]:
    if optimizer_kind == "lion":
        return {"optimizer_kind": "lion", "lr": 1e-3, "betas": [0.9, 0.99], "weight_decay": 0.01}
    return {"optimizer_kind": "sgd_nesterov", "lr": 1e-2, "momentum": 0.9, "weight_decay": 0.01}


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_dispatch_runtime_scorecard_v0",
        "gate": "simple_formula_dispatch_runtime_native_steps",
        "ok": False,
        "promotion_ready": False,
        "dispatch_runtime_stage_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "cases": [],
        "summary": {"case_count": 0, "native_step_count": 0, "skip_pytorch_count": 0},
        "promotion_blockers": [reason, "product_training_route_not_bound"],
        "blocked_reasons": [reason],
        "recommended_next_step": "run simple optimizer dispatch runtime on CUDA",
    }


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_simple_optimizer_dispatch_runtime_scorecard"]
