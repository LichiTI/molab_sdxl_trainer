"""Per-selected TrainingLoop native canaries for closure/second-order plugin optimizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from core.lulynx_trainer.optimizer_plugin_bridge import create_pytorch_optimizer
from core.services.native_module_loader import native_with_entrypoints


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"
ARTIFACT_NAME = "turbocore_plugin_closure_second_order_training_loop_canary_scorecard.json"
ENTRYPOINT = "probe_alig_closure_training_canary_py"
TARGET_PLUGIN_OPTIMIZERS = ("bsam", "lbfgs")


def build_plugin_closure_second_order_training_loop_canary_scorecard(*, write_artifact: bool = False) -> dict[str, Any]:
    if not torch.cuda.is_available():
        report = _blocked("cuda_required_for_plugin_closure_second_order_training_loop_canary")
    else:
        cases = [_run_case(name) for name in TARGET_PLUGIN_OPTIMIZERS]
        blockers = _dedupe(reason for case in cases for reason in case.get("blocked_reasons", []) or [])
        ready = len(cases) == len(TARGET_PLUGIN_OPTIMIZERS) and all(case.get("ok") is True for case in cases)
        native_step_count = sum(1 for case in cases if case.get("native_step_executed") is True)
        native_kernel_count = sum(1 for case in cases if case.get("native_kernel_launched") is True)
        optimizer_step_count = sum(1 for case in cases if case.get("optimizer_step_called") is True)
        closure_call_count = sum(int(case.get("closure_call_count", 0) or 0) for case in cases)
        report = {
            "schema_version": 1,
            "scorecard": "turbocore_plugin_closure_second_order_training_loop_canary_scorecard_v0",
            "gate": "plugin_closure_second_order_selected_training_loop_native_canary",
            "roadmap": ROADMAP,
            "ok": ready,
            "promotion_ready": False,
            "selected_native_canary_ready": ready,
            "training_path_enabled": False,
            "default_behavior_changed": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "product_native_ready": False,
            "selected_optimizer_family": "closure_or_second_order",
            "cases": cases,
            "summary": {
                "selected_optimizer_count": len(cases),
                "case_count": len(cases),
                "native_step_count": native_step_count,
                "native_kernel_launch_count": native_kernel_count,
                "optimizer_step_called_count": optimizer_step_count,
                "closure_call_count": closure_call_count,
                "plugin_closure_second_order_training_loop_case_count": len(cases),
                "plugin_closure_second_order_training_loop_native_step_count": native_step_count,
                "plugin_closure_second_order_training_loop_native_kernel_launch_count": native_kernel_count,
                "plugin_closure_second_order_training_loop_optimizer_step_called_count": optimizer_step_count,
                "plugin_closure_second_order_training_loop_closure_call_count": closure_call_count,
                "runtime_dispatch_ready_count": 0,
                "native_dispatch_allowed_count": 0,
                "training_path_enabled_count": 0,
                "product_native_ready_count": 0,
            },
            "promotion_blockers": _dedupe(
                blockers
                + [
                    "plugin_closure_second_order_owner_release_review_missing",
                    "plugin_closure_second_order_product_training_route_not_bound",
                ]
            ),
            "blocked_reasons": blockers,
            "recommended_next_step": (
                "expand closure/second-order selected optimizer native canaries if more closure families need proof"
                if ready
                else "fix selected plugin closure/second-order TrainingLoop native canary blockers"
            ),
            "notes": [
                "Each selected closure/second-order optimizer runs a real closure-required TrainingLoop step.",
                "The native launch is a representative closure kernel probe and stays default-off.",
                "This evidence is for actual-training coverage only; product dispatch remains closed.",
            ],
        }
    if write_artifact:
        _write_artifact(report)
    return report


def _run_case(selected_optimizer_name: str) -> dict[str, Any]:
    param = torch.nn.Parameter(torch.linspace(-0.18, 0.22, steps=128, device="cuda", dtype=torch.float32))
    optimizer = _create_selected_plugin_optimizer(selected_optimizer_name, param)
    native = native_with_entrypoints(ENTRYPOINT)
    if native is None:
        return _case_blocked(selected_optimizer_name, "plugin_closure_second_order_native_entrypoint_missing")

    before = param.detach().clone()
    closure_calls = 0
    last_loss: torch.Tensor | None = None
    last_grad: torch.Tensor | None = None
    optimizer_step_called = False

    if selected_optimizer_name == "bsam":
        pre_loss = _loss(param)
        optimizer.zero_grad(set_to_none=True)
        pre_loss.backward()
        last_grad = _clone_grad(param)

    def closure() -> torch.Tensor:
        nonlocal closure_calls, last_loss, last_grad
        closure_calls += 1
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(param)
        loss.backward()
        last_loss = loss.detach()
        last_grad = _clone_grad(param)
        return loss

    try:
        loss_out = optimizer.step(closure)
        optimizer_step_called = True
    except Exception as exc:  # pragma: no cover - depends on CUDA/plugin implementation
        return _case_blocked(selected_optimizer_name, f"plugin_closure_second_order_step_failed:{type(exc).__name__}:{exc}")
    try:
        optimizer.zero_grad(set_to_none=True)
    except Exception:
        pass

    after = param.detach().clone()
    mutated = _max_abs_diff(before, after) > 0.0
    native_grad = last_grad if last_grad is not None else torch.zeros_like(param)
    step_size = float(optimizer.param_groups[0].get("lr", 0.0))
    try:
        launch = dict(
            getattr(native, ENTRYPOINT)(
                after.detach().clone().contiguous(),
                native_grad.detach().clone().contiguous(),
                int(param.numel()),
                float(step_size),
                str(REPO_ROOT.resolve()),
                _cuda_arch(param.device),
            )
        )
    except Exception as exc:  # pragma: no cover - depends on CUDA/native build
        return _case_blocked(selected_optimizer_name, f"plugin_closure_second_order_native_call_failed:{type(exc).__name__}:{exc}")

    ok = bool(
        launch.get("ok") is True
        and launch.get("kernel_executed") is True
        and launch.get("parameters_mutated") is True
        and launch.get("closure_replay") is True
        and closure_calls >= 1
        and optimizer_step_called
        and mutated
        and param.grad is None
    )
    return {
        "schema_version": 1,
        "ok": ok,
        "selected_optimizer_name": selected_optimizer_name,
        "selected_optimizer_family": "closure_or_second_order",
        "optimizer_class": type(getattr(optimizer, "_base", optimizer)).__name__,
        "result": _loss_result(loss_out),
        "closure_call_count": closure_calls,
        "native_step_executed": launch.get("ok") is True,
        "native_kernel_launched": launch.get("kernel_executed") is True,
        "training_parameters_mutated": mutated,
        "optimizer_step_called": optimizer_step_called,
        "closure_loss_value": float(last_loss.float().item()) if isinstance(last_loss, torch.Tensor) else None,
        "step_after_native": int(optimizer.param_groups[0].get("step", 0) or 0),
        "param_dtype": str(param.dtype).replace("torch.", ""),
        "param_shape": [int(dim) for dim in param.shape],
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "source_scorecard": "turbocore_plugin_closure_second_order_training_loop_canary_scorecard_v0",
        "launch": launch,
        "blocked_reasons": [] if ok else _case_blockers(selected_optimizer_name, launch, closure_calls, optimizer_step_called, mutated, param.grad is None),
    }


def _create_selected_plugin_optimizer(selected_optimizer_name: str, param: torch.nn.Parameter) -> torch.optim.Optimizer:
    optimizer_args: dict[str, Any]
    if selected_optimizer_name == "bsam":
        optimizer_args = {"name": "BSAM", "num_data": 1}
    elif selected_optimizer_name == "lbfgs":
        optimizer_args = {"name": "LBFGS", "max_iter": 1, "history_size": 4, "line_search_fn": None}
    else:
        raise ValueError(f"Unsupported closure/second-order optimizer: {selected_optimizer_name}")
    return create_pytorch_optimizer(
        [param],
        optimizer_name=selected_optimizer_name,
        lr=0.05 if selected_optimizer_name == "bsam" else 0.25,
        weight_decay=0.0,
        optimizer_args=optimizer_args,
    )


def _loss(param: torch.Tensor) -> torch.Tensor:
    return ((param.float() * 0.31) ** 2).mean() + param.float().mean() * 0.007


def _clone_grad(param: torch.nn.Parameter) -> torch.Tensor:
    if param.grad is None:
        return torch.zeros_like(param)
    return param.grad.detach().clone().contiguous()


def _loss_result(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {"type": "tensor", "value": float(value.detach().float().item())}
    if isinstance(value, (int, float)):
        return {"type": type(value).__name__, "value": float(value)}
    return {"type": type(value).__name__, "value": None}


def _case_blocked(name: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "selected_optimizer_name": name,
        "selected_optimizer_family": "closure_or_second_order",
        "native_step_executed": False,
        "native_kernel_launched": False,
        "training_parameters_mutated": False,
        "optimizer_step_called": False,
        "blocked_reasons": [reason],
    }


def _case_blockers(
    name: str,
    launch: Mapping[str, Any],
    closure_calls: int,
    optimizer_step_called: bool,
    mutated: bool,
    grad_cleared: bool,
) -> list[str]:
    blockers = [str(item) for item in launch.get("blocked_reasons", []) or []]
    if launch.get("kernel_executed") is not True:
        blockers.append(f"{name}_native_kernel_not_executed")
    if launch.get("parameters_mutated") is not True:
        blockers.append(f"{name}_native_parameters_not_mutated")
    if launch.get("closure_replay") is not True:
        blockers.append(f"{name}_native_closure_replay_missing")
    if closure_calls < 1:
        blockers.append(f"{name}_closure_not_called")
    if not optimizer_step_called:
        blockers.append(f"{name}_optimizer_step_not_called")
    if not mutated:
        blockers.append(f"{name}_parameters_not_mutated")
    if not grad_cleared:
        blockers.append(f"{name}_grad_not_cleared")
    return _dedupe(blockers or [f"{name}_closure_second_order_canary_failed"])


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_closure_second_order_training_loop_canary_scorecard_v0",
        "gate": "plugin_closure_second_order_selected_training_loop_native_canary",
        "ok": False,
        "promotion_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "cases": [],
        "summary": {"native_step_count": 0, "native_kernel_launch_count": 0},
        "promotion_blockers": [reason],
        "blocked_reasons": [reason],
        "recommended_next_step": "run closure/second-order selected optimizer native canaries on CUDA",
    }


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


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / ARTIFACT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = ["ROADMAP", "build_plugin_closure_second_order_training_loop_canary_scorecard"]
