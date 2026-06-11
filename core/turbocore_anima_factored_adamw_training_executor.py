"""Default-off training executor for AnimaFactoredAdamW native dispatch."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from core.services.native_module_loader import native_with_entrypoints


ENTRYPOINT = "probe_anima_factored_adamw_training_tensor_binding_canary_py"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AnimaFactoredAdamWTrainingExecutorConfig:
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    factored_eps: float = 1e-30
    max_numel: int = 1_048_576
    require_native_cuda: bool = True


class AnimaFactoredAdamWTrainingExecutor:
    """Launch native AnimaFactoredAdamW steps against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: AnimaFactoredAdamWTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("AnimaFactoredAdamWTrainingExecutor requires trainable parameters")
        self._param_ids = {id(param) for param in self.params}
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)
        self._native: Any | None = None

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("anima_factored_adamw_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("anima_factored_adamw_training_executor_requires_cuda_params")
        native = self._load_native()
        if native is None:
            return _blocked("anima_factored_adamw_training_dispatch_entrypoint_missing")
        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config)
            for param in group["params"]:
                if id(param) not in self._param_ids or param.grad is None:
                    continue
                cases.append(self._step_param(native, param, group_cfg))
        ok = bool(cases) and all(bool(case.get("ok", False)) for case in cases)
        blockers = _dedupe([reason for case in cases for reason in case.get("blocked_reasons", [])])
        if not cases:
            blockers.append("anima_factored_adamw_training_executor_no_grad_params")
        return {
            "schema_version": 1,
            "executor": "turbocore_anima_factored_adamw_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "anima_factored_adamw_native_step_failed"),
            "optimizer_kind": "anima_factored_adamw",
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

    def _step_param(self, native: Any, param: torch.nn.Parameter, group_cfg: Mapping[str, Any]) -> dict[str, Any]:
        if param.dtype != torch.float32 or (param.grad is not None and param.grad.dtype != torch.float32):
            return _case_failed("anima_factored_adamw_native_step_requires_float32", "anima_factored_adamw_native_step_dtype_unsupported")
        if param.dim() == 2:
            rows, cols = int(param.shape[0]), int(param.shape[1])
        elif param.dim() == 1:
            rows, cols = 1, int(param.numel())
        else:
            return _case_failed("anima_factored_adamw_native_step_requires_1d_or_2d", "anima_factored_adamw_native_step_shape_unsupported")
        state = self.optimizer.state[param]
        missing = [key for key in ("step", "exp_avg", "factored") if key not in state]
        if missing:
            return _case_failed(f"anima_factored_adamw_live_state_missing:{','.join(missing)}", "anima_factored_adamw_live_state_missing")
        factored = bool(state.get("factored", False))
        if factored and ("exp_avg_sq_row" not in state or "exp_avg_sq_col" not in state):
            return _case_failed("anima_factored_adamw_factored_state_missing", "anima_factored_adamw_live_state_missing")
        if not factored and "exp_avg_sq" not in state:
            return _case_failed("anima_factored_adamw_unfactored_state_missing", "anima_factored_adamw_live_state_missing")
        exp_avg = state["exp_avg"].contiguous()
        row_tensor = state["exp_avg_sq_row"].reshape(rows).contiguous() if factored else torch.zeros(rows, device=param.device, dtype=torch.float32)
        col_tensor = state["exp_avg_sq_col"].reshape(cols).contiguous() if factored else torch.zeros(cols, device=param.device, dtype=torch.float32)
        exp_avg_sq = torch.zeros_like(param.detach()).contiguous() if factored else state["exp_avg_sq"].contiguous()
        step_after = int(state["step"].detach().cpu().item() if torch.is_tensor(state["step"]) else state["step"]) + 1
        launch_config = {
            **dict(group_cfg),
            "factored": factored,
            "rows": rows,
            "cols": cols,
            "step": step_after,
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
                    row_tensor,
                    col_tensor,
                    exp_avg_sq,
                    json.dumps(launch_config),
                    str(self.workspace_root.resolve()),
                    _cuda_arch(param.device),
                )
            )
        except Exception as exc:  # pragma: no cover - native/CUDA dependent
            return _case_failed(f"{type(exc).__name__}: {exc}", "anima_factored_adamw_native_step_call_failed")
        if not bool(launch.get("ok", False)):
            return _case_failed(str(launch.get("reason") or "native_step_failed"), "anima_factored_adamw_native_step_failed", launch)
        state["step"] = torch.tensor(step_after, dtype=torch.int64, device=param.device)
        state["exp_avg"] = exp_avg.reshape_as(param)
        if factored:
            state["exp_avg_sq_row"] = row_tensor.reshape(rows, 1)
            state["exp_avg_sq_col"] = col_tensor.reshape(1, cols)
        else:
            state["exp_avg_sq"] = exp_avg_sq.reshape_as(param)
        return {
            "schema_version": 1,
            "ok": True,
            "param_numel": int(param.numel()),
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "factored": factored,
            "step_after": step_after,
            "kernel_executed": bool(launch.get("kernel_executed", False)),
            "training_parameters_mutated": bool(launch.get("parameters_mutated", False)),
            "launch": launch,
            "blocked_reasons": [],
        }


def build_anima_factored_adamw_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: AnimaFactoredAdamWTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> AnimaFactoredAdamWTrainingExecutor:
    return AnimaFactoredAdamWTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: AnimaFactoredAdamWTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> AnimaFactoredAdamWTrainingExecutorConfig:
    if isinstance(value, AnimaFactoredAdamWTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = group.get("betas", (0.9, 0.999))
    return AnimaFactoredAdamWTrainingExecutorConfig(
        lr=float(payload.get("lr", group.get("lr", 1e-4))),
        beta1=float(payload.get("beta1", betas[0])),
        beta2=float(payload.get("beta2", betas[1])),
        eps=float(payload.get("eps", group.get("eps", 1e-8))),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0))),
        factored_eps=float(payload.get("factored_eps", group.get("factored_eps", 1e-30))),
        max_numel=int(payload.get("max_numel", 1_048_576)),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: AnimaFactoredAdamWTrainingExecutorConfig) -> dict[str, Any]:
    betas = group.get("betas", (config.beta1, config.beta2))
    return {
        "lr": float(group.get("lr", config.lr)),
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "eps": float(group.get("eps", config.eps)),
        "weight_decay": float(group.get("weight_decay", config.weight_decay)),
        "factored_eps": float(group.get("factored_eps", config.factored_eps)),
    }


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 4)


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
        "executor": "turbocore_anima_factored_adamw_training_executor_v0",
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
    "AnimaFactoredAdamWTrainingExecutor",
    "AnimaFactoredAdamWTrainingExecutorConfig",
    "build_anima_factored_adamw_training_executor",
]
