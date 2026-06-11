"""Default-off training executor for selected plugin PAdam native dispatch."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from core.services.native_module_loader import native_with_entrypoints


ENTRYPOINT = "probe_padam_training_tensor_binding_canary_py"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PluginPAdamTrainingExecutorConfig:
    lr: float = 1e-1
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    partial: float = 0.25
    weight_decay: float = 0.0
    weight_decouple: bool = False
    fixed_decay: bool = False
    maximize: bool = False
    max_numel: int = 1_048_576
    require_native_cuda: bool = True


class PluginPAdamTrainingExecutor:
    """Launch selected PAdam native steps against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: PluginPAdamTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("PluginPAdamTrainingExecutor requires trainable parameters")
        self._param_ids = {id(param) for param in self.params}
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)
        self._native: Any | None = None

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("plugin_padam_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("plugin_padam_training_executor_requires_cuda_params")
        native = self._load_native()
        if native is None:
            return _blocked("plugin_padam_training_dispatch_entrypoint_missing")
        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        stepped_groups: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config)
            group_step_before = _group_step_to_int(group)
            group_cases: list[dict[str, Any]] = []
            for param in group["params"]:
                if id(param) not in self._param_ids or param.grad is None:
                    continue
                group_cases.append(self._step_param(native, param, group_cfg, group_step_before))
            cases.extend(group_cases)
            if group_cases and all(bool(case.get("ok", False)) for case in group_cases):
                _set_group_step(group, group_step_before + 1)
                stepped_groups.append(
                    {
                        "schema_version": 1,
                        "step_before": group_step_before,
                        "step_after": _group_step_to_int(group),
                        "parameter_count": len(group_cases),
                    }
                )
        ok = bool(cases) and all(bool(case.get("ok", False)) for case in cases)
        blockers = _dedupe([reason for case in cases for reason in case.get("blocked_reasons", [])])
        if not cases:
            blockers.append("plugin_padam_training_executor_no_grad_params")
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_padam_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "plugin_padam_native_step_failed"),
            "optimizer_kind": "padam",
            "training_dispatch": True,
            "training_path_enabled": True,
            "native_step_executed": ok,
            "native_kernel_launched": any(bool(case.get("kernel_executed", False)) for case in cases),
            "training_parameters_mutated": ok,
            "should_call_pytorch_optimizer_step": not ok,
            "pytorch_optimizer_state_synced": True,
            "parameter_step_count": len(cases),
            "cases": cases,
            "stepped_groups": stepped_groups,
            "timing": {"elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4)},
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
        group_cfg: Mapping[str, Any],
        group_step_before: int,
    ) -> dict[str, Any]:
        if type(self.optimizer).__name__.lower() != "padam":
            return _case_failed("plugin_padam_training_executor_optimizer_not_padam", "plugin_padam_optimizer_class_unsupported")
        if param.dtype != torch.float32 or param.grad is None or param.grad.dtype != torch.float32:
            return _case_failed("plugin_padam_native_step_requires_float32", "plugin_padam_native_step_dtype_unsupported")
        if not param.is_contiguous() or not param.grad.is_contiguous():
            return _case_failed("plugin_padam_native_step_requires_contiguous_param_grad", "plugin_padam_native_step_layout_unsupported")
        state = self.optimizer.state[param]
        if "exp_avg" not in state or "exp_avg_sq" not in state:
            return _case_failed("plugin_padam_live_state_missing", "plugin_padam_live_state_missing")
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        if not (torch.is_tensor(exp_avg) and torch.is_tensor(exp_avg_sq)):
            return _case_failed("plugin_padam_live_state_tensors_missing", "plugin_padam_live_state_missing")
        if exp_avg.dtype != torch.float32 or exp_avg_sq.dtype != torch.float32:
            return _case_failed("plugin_padam_native_state_requires_float32", "plugin_padam_native_state_dtype_unsupported")
        if not exp_avg.is_contiguous() or not exp_avg_sq.is_contiguous():
            return _case_failed("plugin_padam_native_state_requires_contiguous", "plugin_padam_native_state_layout_unsupported")
        launch_config = {
            **dict(group_cfg),
            "step": group_step_before,
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
                    exp_avg,
                    exp_avg_sq,
                    json.dumps(launch_config),
                    str(self.workspace_root.resolve()),
                    _cuda_arch(param.device),
                )
            )
        except Exception as exc:  # pragma: no cover - native/CUDA dependent
            return _case_failed(f"{type(exc).__name__}: {exc}", "plugin_padam_native_step_call_failed")
        if not bool(launch.get("ok", False)):
            return _case_failed(
                str(launch.get("reason") or "plugin_padam_native_step_failed"),
                "plugin_padam_native_step_failed",
                launch=launch,
            )
        return {
            "schema_version": 1,
            "ok": True,
            "param_numel": int(param.numel()),
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_before": group_step_before,
            "step_after": int(launch.get("step_after", group_step_before + 1) or (group_step_before + 1)),
            "kernel_executed": bool(launch.get("kernel_executed", False)),
            "training_parameters_mutated": bool(
                launch.get("training_parameters_mutated")
                or launch.get("parameters_mutated")
                or launch.get("live_tensors_mutated")
            ),
            "launch": launch,
            "blocked_reasons": [],
        }


def build_plugin_padam_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: PluginPAdamTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> PluginPAdamTrainingExecutor:
    return PluginPAdamTrainingExecutor(
        optimizer=optimizer,
        params=params,
        config=config,
        workspace_root=workspace_root,
    )


def _normalize_config(
    value: PluginPAdamTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> PluginPAdamTrainingExecutorConfig:
    if isinstance(value, PluginPAdamTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = payload.get("betas", group.get("betas", (0.9, 0.999)))
    return PluginPAdamTrainingExecutorConfig(
        lr=float(payload.get("lr", group.get("lr", 1e-1))),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(payload.get("eps", group.get("eps", 1e-8))),
        partial=float(payload.get("partial", group.get("partial", 0.25))),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0))),
        weight_decouple=bool(payload.get("weight_decouple", group.get("weight_decouple", False))),
        fixed_decay=bool(payload.get("fixed_decay", group.get("fixed_decay", False))),
        maximize=bool(payload.get("maximize", getattr(optimizer, "maximize", False))),
        max_numel=int(payload.get("max_numel", 1_048_576)),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: PluginPAdamTrainingExecutorConfig) -> dict[str, Any]:
    betas = group.get("betas", config.betas)
    return {
        "lr": float(group.get("lr", config.lr)),
        "betas": [float(betas[0]), float(betas[1])],
        "eps": float(group.get("eps", config.eps)),
        "partial": float(group.get("partial", config.partial)),
        "weight_decay": float(group.get("weight_decay", config.weight_decay)),
        "weight_decouple": bool(group.get("weight_decouple", config.weight_decouple)),
        "fixed_decay": bool(group.get("fixed_decay", config.fixed_decay)),
        "maximize": bool(config.maximize),
    }


def _group_step_to_int(group: Mapping[str, Any]) -> int:
    value = group.get("step", 0)
    if torch.is_tensor(value) and value.numel() > 0:
        return int(value.detach().reshape(-1)[0].cpu().item())
    return int(value or 0)


def _set_group_step(group: dict[str, Any], value: int) -> None:
    current = group.get("step")
    if torch.is_tensor(current):
        current.fill_(int(value))
    else:
        group["step"] = int(value)


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


def _case_failed(reason: str, blocker: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "reason": reason,
        "kernel_executed": False,
        "training_parameters_mutated": False,
        "blocked_reasons": [blocker],
        **extra,
    }


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "executor": "turbocore_plugin_padam_training_executor_v0",
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


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "PluginPAdamTrainingExecutor",
    "PluginPAdamTrainingExecutorConfig",
    "build_plugin_padam_training_executor",
]
