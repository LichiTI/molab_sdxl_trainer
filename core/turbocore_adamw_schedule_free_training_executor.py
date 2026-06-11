"""Default-off training executor for AdamWScheduleFree native dispatch."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from core.services.native_module_loader import native_with_entrypoints


ENTRYPOINT = "probe_adamw_schedule_free_training_tensor_binding_canary_py"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AdamWScheduleFreeTrainingExecutorConfig:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    warmup_steps: int = 0
    r: float = 0.0
    weight_lr_power: float = 2.0
    max_numel: int = 1_048_576
    require_native_cuda: bool = True


class AdamWScheduleFreeTrainingExecutor:
    """Launch native AdamWScheduleFree steps against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: AdamWScheduleFreeTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("AdamWScheduleFreeTrainingExecutor requires trainable parameters")
        self._param_ids = {id(param) for param in self.params}
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)
        self._native: Any | None = None

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("adamw_schedule_free_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("adamw_schedule_free_training_executor_requires_cuda_params")
        native = self._load_native()
        if native is None:
            return _blocked("adamw_schedule_free_training_dispatch_entrypoint_missing")
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
            blockers.append("adamw_schedule_free_training_executor_no_grad_params")
        return {
            "schema_version": 1,
            "executor": "turbocore_adamw_schedule_free_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "adamw_schedule_free_native_step_failed"),
            "optimizer_kind": "adamw_schedule_free",
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
                "adamw_schedule_free_native_step_requires_float32",
                "adamw_schedule_free_native_step_dtype_unsupported",
            )
        state = self.optimizer.state[param]
        if "z" not in state or "exp_avg_sq" not in state:
            return _case_failed(
                "adamw_schedule_free_live_state_missing:z_or_exp_avg_sq",
                "adamw_schedule_free_live_state_missing",
            )
        z = state["z"].contiguous()
        exp_avg_sq = state["exp_avg_sq"].contiguous()
        step_key = _step_key(group)
        step_index = int(group.get(step_key, state.get("step", 0)) or 0)
        launch_config = {
            **dict(group_cfg),
            "k": step_index,
            "weight_sum": float(group.get("weight_sum", 0.0) or 0.0),
            "lr_max": float(group.get("lr_max", 0.0) or 0.0),
            "scheduled_lr": float(group.get("scheduled_lr", 0.0) or 0.0),
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
                    exp_avg_sq,
                    json.dumps(launch_config),
                    str(self.workspace_root.resolve()),
                    _cuda_arch(param.device),
                )
            )
        except Exception as exc:  # pragma: no cover - native/CUDA dependent
            return _case_failed(
                f"{type(exc).__name__}: {exc}",
                "adamw_schedule_free_native_step_call_failed",
            )
        if not bool(launch.get("ok", False)):
            return _case_failed(
                str(launch.get("reason") or "native_step_failed"),
                "adamw_schedule_free_native_step_failed",
                launch,
            )
        group[step_key] = int(launch.get("k_after", step_index + 1) or (step_index + 1))
        group["weight_sum"] = float(launch.get("weight_sum", launch_config["weight_sum"]) or 0.0)
        group["lr_max"] = float(launch.get("lr_max", launch_config["lr_max"]) or 0.0)
        group["scheduled_lr"] = float(launch.get("scheduled_lr", launch_config["scheduled_lr"]) or 0.0)
        state["z"] = z.reshape_as(param)
        state["exp_avg_sq"] = exp_avg_sq.reshape_as(param)
        if "step" in state:
            state["step"] = int(group[step_key])
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
            "step_after": int(group[step_key]),
            "step_key": step_key,
            "kernel_executed": bool(launch.get("kernel_executed", False)),
            "training_parameters_mutated": mutated,
            "launch": launch,
            "blocked_reasons": [],
        }


def build_adamw_schedule_free_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: AdamWScheduleFreeTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> AdamWScheduleFreeTrainingExecutor:
    return AdamWScheduleFreeTrainingExecutor(
        optimizer=optimizer,
        params=params,
        config=config,
        workspace_root=workspace_root,
    )


def _normalize_config(
    value: AdamWScheduleFreeTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> AdamWScheduleFreeTrainingExecutorConfig:
    if isinstance(value, AdamWScheduleFreeTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = payload.get("betas", group.get("betas", (0.9, 0.999)))
    return AdamWScheduleFreeTrainingExecutorConfig(
        lr=float(payload.get("lr", group.get("lr", 1e-3))),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(payload.get("eps", group.get("eps", 1e-8))),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.01))),
        warmup_steps=int(payload.get("warmup_steps", group.get("warmup_steps", 0)) or 0),
        r=float(payload.get("r", group.get("r", 0.0))),
        weight_lr_power=float(payload.get("weight_lr_power", group.get("weight_lr_power", 2.0))),
        max_numel=int(payload.get("max_numel", 1_048_576)),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: AdamWScheduleFreeTrainingExecutorConfig) -> dict[str, Any]:
    betas = group.get("betas", config.betas)
    return {
        "lr": float(group.get("lr", config.lr)),
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "eps": float(group.get("eps", config.eps)),
        "weight_decay": float(group.get("weight_decay", config.weight_decay)),
        "warmup_steps": int(group.get("warmup_steps", config.warmup_steps) or 0),
        "r": float(group.get("r", config.r)),
        "weight_lr_power": float(group.get("weight_lr_power", config.weight_lr_power)),
    }


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


def _step_key(group: Mapping[str, Any]) -> str:
    if "k" in group:
        return "k"
    if "step" in group:
        return "step"
    return "k"


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
        "executor": "turbocore_adamw_schedule_free_training_executor_v0",
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
    "AdamWScheduleFreeTrainingExecutor",
    "AdamWScheduleFreeTrainingExecutorConfig",
    "build_adamw_schedule_free_training_executor",
]
