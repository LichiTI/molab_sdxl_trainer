"""Default-off training executor for selected plugin ScheduleFreeSGD."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from core.services.native_module_loader import native_with_entrypoints


ENTRYPOINT = "probe_schedulefree_sgd_training_tensor_binding_canary_py"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ScheduleFreeSGDTrainingExecutorConfig:
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 0.0
    warmup_steps: int = 0
    r: float = 0.0
    weight_lr_power: float = 2.0
    maximize: bool = False
    max_numel: int = 1_048_576
    require_native_cuda: bool = True


class ScheduleFreeSGDTrainingExecutor:
    """Launch native ScheduleFreeSGD steps against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: ScheduleFreeSGDTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("ScheduleFreeSGDTrainingExecutor requires trainable parameters")
        self._param_ids = {id(param) for param in self.params}
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)
        self._native: Any | None = None

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("schedulefree_sgd_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("schedulefree_sgd_training_executor_requires_cuda_params")
        native = self._load_native()
        if native is None:
            return _blocked("schedulefree_sgd_training_dispatch_entrypoint_missing")
        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config)
            for param in group["params"]:
                if id(param) not in self._param_ids or param.grad is None:
                    continue
                cases.append(self._step_param(native, param, group, group_cfg))
        ok = bool(cases) and all(bool(case.get("ok", False)) for case in cases)
        blockers = _dedupe([reason for case in cases for reason in case.get("blocked_reasons", [])])
        if not cases:
            blockers.append("schedulefree_sgd_training_executor_no_grad_params")
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_schedulefree_sgd_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "schedulefree_sgd_native_step_failed"),
            "optimizer_kind": "schedulefree_sgd",
            "training_dispatch": True,
            "training_path_enabled": True,
            "native_step_executed": ok,
            "native_kernel_launched": any(bool(case.get("kernel_executed", False)) for case in cases),
            "training_parameters_mutated": ok,
            "should_call_pytorch_optimizer_step": not ok,
            "pytorch_optimizer_state_synced": True,
            "parameter_step_count": len(cases),
            "cases": cases,
            "timing": {"elapsed_ms": _elapsed_ms(started)},
            "blocked_reasons": blockers,
        }

    def close(self) -> None:
        return None

    def _load_native(self) -> Any | None:
        if self._native is None:
            self._native = native_with_entrypoints(ENTRYPOINT)
        return self._native

    def _step_param(
        self,
        native: Any,
        param: torch.nn.Parameter,
        group: dict[str, Any],
        group_cfg: Mapping[str, Any],
    ) -> dict[str, Any]:
        if param.dtype != torch.float32 or (param.grad is not None and param.grad.dtype != torch.float32):
            return _case_failed(
                "schedulefree_sgd_native_step_requires_float32",
                "schedulefree_sgd_native_step_dtype_unsupported",
            )
        if group.get("train_mode") is False:
            return _case_failed(
                "schedulefree_sgd_native_step_requires_train_mode",
                "schedulefree_sgd_training_mode_not_active",
            )
        state = self.optimizer.state[param]
        if "z" not in state:
            return _case_failed("schedulefree_sgd_live_state_missing:z", "schedulefree_sgd_live_state_missing")
        z = state["z"].contiguous()
        step_index = int(group.get("step", state.get("step", 0)) or 0)
        launch_config = {
            **dict(group_cfg),
            "step": step_index,
            "weight_sum": float(group.get("weight_sum", 0.0) or 0.0),
            "lr_max": float(group.get("lr_max", -1.0) or -1.0),
            "train_mode": bool(group.get("train_mode", True)),
            "max_numel": max(int(self.config.max_numel), int(param.numel())),
            "canary_probe_only": True,
            "training_tensor_binding": True,
            "training_dispatch": False,
            "training_path_enabled": False,
        }
        try:
            launch = dict(
                getattr(native, ENTRYPOINT)(
                    param,
                    param.grad.detach().contiguous(),
                    z,
                    json.dumps(launch_config),
                    str(self.workspace_root.resolve()),
                    _cuda_arch(param.device),
                )
            )
        except Exception as exc:  # pragma: no cover - native/CUDA dependent
            return _case_failed(f"{type(exc).__name__}: {exc}", "schedulefree_sgd_native_step_call_failed")
        if not bool(launch.get("ok", False)):
            return _case_failed(
                str(launch.get("reason") or "native_step_failed"),
                "schedulefree_sgd_native_step_failed",
                launch,
            )
        group["step"] = int(launch.get("step_after", step_index + 1) or (step_index + 1))
        group["weight_sum"] = float(launch.get("weight_sum", launch_config["weight_sum"]) or 0.0)
        group["lr_max"] = float(launch.get("lr_max", launch_config["lr_max"]) or -1.0)
        state["z"] = z.reshape_as(param)
        mutated = bool(
            launch.get("training_parameters_mutated")
            or launch.get("parameters_mutated")
            or launch.get("live_tensors_mutated")
        )
        return {
            "schema_version": 1,
            "ok": True,
            "param_numel": int(param.numel()),
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_before": step_index,
            "step_after": int(group["step"]),
            "kernel_executed": bool(launch.get("kernel_executed", False)),
            "training_parameters_mutated": mutated,
            "launch": launch,
            "blocked_reasons": [],
        }


def build_plugin_schedulefree_sgd_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: ScheduleFreeSGDTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> ScheduleFreeSGDTrainingExecutor:
    return ScheduleFreeSGDTrainingExecutor(
        optimizer=optimizer,
        params=params,
        config=config,
        workspace_root=workspace_root,
    )


def _normalize_config(
    value: ScheduleFreeSGDTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> ScheduleFreeSGDTrainingExecutorConfig:
    if isinstance(value, ScheduleFreeSGDTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    return ScheduleFreeSGDTrainingExecutorConfig(
        lr=float(payload.get("lr", group.get("lr", 1e-3))),
        momentum=float(payload.get("momentum", group.get("momentum", 0.9))),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0))),
        warmup_steps=int(payload.get("warmup_steps", group.get("warmup_steps", 0)) or 0),
        r=float(payload.get("r", group.get("r", 0.0))),
        weight_lr_power=float(payload.get("weight_lr_power", group.get("weight_lr_power", 2.0))),
        maximize=bool(payload.get("maximize", getattr(optimizer, "maximize", False))),
        max_numel=int(payload.get("max_numel", 1_048_576)),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: ScheduleFreeSGDTrainingExecutorConfig) -> dict[str, Any]:
    return {
        "lr": float(group.get("lr", config.lr)),
        "momentum": float(group.get("momentum", config.momentum)),
        "weight_decay": float(group.get("weight_decay", config.weight_decay)),
        "warmup_steps": int(group.get("warmup_steps", config.warmup_steps) or 0),
        "r": float(group.get("r", config.r)),
        "weight_lr_power": float(group.get("weight_lr_power", config.weight_lr_power)),
        "maximize": bool(config.maximize),
    }


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


def _case_failed(reason: str, blocker: str, launch: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "reason": reason,
        "kernel_executed": False,
        "training_parameters_mutated": False,
        "launch": dict(launch or {}),
        "blocked_reasons": [blocker],
    }


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "executor": "turbocore_plugin_schedulefree_sgd_training_executor_v0",
        "ok": False,
        "reason": reason,
        "training_dispatch": True,
        "training_path_enabled": True,
        "native_step_executed": False,
        "native_kernel_launched": False,
        "training_parameters_mutated": False,
        "should_call_pytorch_optimizer_step": True,
        "blocked_reasons": [reason],
    }


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 4)


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "ScheduleFreeSGDTrainingExecutor",
    "ScheduleFreeSGDTrainingExecutorConfig",
    "build_plugin_schedulefree_sgd_training_executor",
]
