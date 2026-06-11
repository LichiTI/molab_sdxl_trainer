"""Default-off training executor for KahanAdamW8bit native dispatch."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from core.lulynx_trainer.kahan_adamw8bit import KahanAdamW8bit, _QuantizedState
from core.services.native_module_loader import native_with_entrypoints


ENTRYPOINT = "step_kahan_adamw8bit_training_dispatch_py"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class KahanAdamW8bitTrainingExecutorConfig:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    max_numel: int = 1_048_576
    require_native_cuda: bool = True


class KahanAdamW8bitTrainingExecutor:
    """Launch native KahanAdamW8bit steps against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: KahanAdamW8bit,
        params: Iterable[torch.nn.Parameter],
        config: KahanAdamW8bitTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        if not isinstance(optimizer, KahanAdamW8bit):
            raise ValueError("KahanAdamW8bitTrainingExecutor requires KahanAdamW8bit optimizer")
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("KahanAdamW8bitTrainingExecutor requires trainable parameters")
        self._param_ids = {id(param) for param in self.params}
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)
        self._native: Any | None = None

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("kahan_adamw8bit_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("kahan_adamw8bit_training_executor_requires_cuda_params")
        native = self._load_native()
        if native is None:
            return _blocked("kahan_adamw8bit_training_dispatch_entrypoint_missing")
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
            blockers.append("kahan_adamw8bit_training_executor_no_grad_params")
        return {
            "schema_version": 1,
            "executor": "turbocore_kahan_adamw8bit_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "kahan_adamw8bit_native_step_failed"),
            "optimizer_kind": "kahan_adamw8bit",
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
        state = self.optimizer.state[param]
        _ensure_state(state, param)
        step_index = int(state.get("step") or 0)
        exp_avg_q = state["exp_avg_q"]
        exp_avg_sq_q = state["exp_avg_sq_q"]
        launch_config = {
            **dict(group_cfg),
            "step_index": step_index,
            "max_numel": max(int(self.config.max_numel), int(param.numel())),
            "canary_probe_only": False,
            "training_tensor_binding": True,
            "training_dispatch": True,
            "training_path_enabled": True,
        }
        try:
            launch = dict(
                getattr(native, ENTRYPOINT)(
                    param,
                    param.grad.detach().contiguous(),
                    exp_avg_q.data,
                    exp_avg_sq_q.data,
                    exp_avg_q.absmax,
                    exp_avg_sq_q.absmax,
                    state["kahan_comp"],
                    json.dumps(launch_config),
                    str(self.workspace_root.resolve()),
                    _cuda_arch(param.device),
                )
            )
        except Exception as exc:  # pragma: no cover - native/CUDA dependent
            return _case_failed(f"{type(exc).__name__}: {exc}", "kahan_adamw8bit_native_step_call_failed")
        if not bool(launch.get("ok", False)):
            return _case_failed(str(launch.get("reason") or "native_step_failed"), "kahan_adamw8bit_native_step_failed", launch)
        state["step"] = step_index + 1
        return {
            "schema_version": 1,
            "ok": True,
            "param_numel": int(param.numel()),
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_before": step_index,
            "step_after": int(state["step"]),
            "kernel_executed": bool(launch.get("kernel_executed", False)),
            "training_parameters_mutated": bool(launch.get("training_parameters_mutated", False)),
            "launch": launch,
            "blocked_reasons": [],
        }


def build_kahan_adamw8bit_training_executor(
    *,
    optimizer: KahanAdamW8bit,
    params: Iterable[torch.nn.Parameter],
    config: KahanAdamW8bitTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> KahanAdamW8bitTrainingExecutor:
    return KahanAdamW8bitTrainingExecutor(
        optimizer=optimizer,
        params=params,
        config=config,
        workspace_root=workspace_root,
    )


def _normalize_config(
    value: KahanAdamW8bitTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: KahanAdamW8bit,
) -> KahanAdamW8bitTrainingExecutorConfig:
    if isinstance(value, KahanAdamW8bitTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = payload.get("betas", group.get("betas", (0.9, 0.999)))
    return KahanAdamW8bitTrainingExecutorConfig(
        lr=float(payload.get("lr", group.get("lr", 1e-3))),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(payload.get("eps", group.get("eps", 1e-8))),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.01))),
        max_numel=int(payload.get("max_numel", 1_048_576)),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: KahanAdamW8bitTrainingExecutorConfig) -> dict[str, Any]:
    betas = group.get("betas", config.betas)
    return {
        "lr": float(group.get("lr", config.lr)),
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "eps": float(group.get("eps", config.eps)),
        "weight_decay": float(group.get("weight_decay", config.weight_decay)),
    }


def _ensure_state(state: dict[str, Any], param: torch.nn.Parameter) -> None:
    if not state:
        state["step"] = 0
        state["exp_avg_q"] = _QuantizedState(torch.zeros_like(param, dtype=torch.float32))
        state["exp_avg_sq_q"] = _QuantizedState(torch.zeros_like(param, dtype=torch.float32))
        state["kahan_comp"] = torch.zeros_like(param, dtype=torch.float32)
    comp = state.get("kahan_comp")
    if isinstance(comp, torch.Tensor) and comp.dtype != torch.float32:
        state["kahan_comp"] = comp.float()


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


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "executor": "turbocore_kahan_adamw8bit_training_executor_v0",
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
    "KahanAdamW8bitTrainingExecutor",
    "KahanAdamW8bitTrainingExecutorConfig",
    "build_kahan_adamw8bit_training_executor",
]
