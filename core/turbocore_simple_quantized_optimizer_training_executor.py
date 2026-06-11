"""Default-off training executor for quantized simple native optimizers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from core.services.native_module_loader import native_with_entrypoints


ENTRYPOINT = "step_simple_quantized_optimizer_training_dispatch_py"
REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_KINDS = {"lion8bit", "paged_lion8bit", "sgd_nesterov8bit"}
BLOCK_SIZE = 256


@dataclass(frozen=True)
class SimpleQuantizedOptimizerTrainingExecutorConfig:
    optimizer_kind: str = "lion8bit"
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.99)
    momentum: float = 0.9
    weight_decay: float = 0.0
    max_numel: int = 1_048_576
    require_native_cuda: bool = True


class SimpleQuantizedOptimizerTrainingExecutor:
    """Launch native dequantize/update/requantize steps against live params."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        params: Iterable[torch.nn.Parameter],
        config: SimpleQuantizedOptimizerTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("SimpleQuantizedOptimizerTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config)
        self.workspace_root = Path(workspace_root or REPO_ROOT)
        self._native: Any | None = None
        self._state: dict[int, dict[str, torch.Tensor | int]] = {}
        self._pytorch_optimizer_state_dirty = False
        self._last_optimizer_state_sync: dict[str, Any] = {}

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("simple_quantized_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("simple_quantized_training_executor_requires_cuda_params")
        native = self._load_native()
        if native is None:
            return _blocked("simple_quantized_training_dispatch_entrypoint_missing")
        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        for param in self.params:
            if param.grad is None:
                continue
            cases.append(self._step_param(native, param))
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        if ok:
            self._pytorch_optimizer_state_dirty = True
        optimizer_sync = (
            self.sync_optimizer_state_to_pytorch(reason="simple_quantized_native_step")
            if ok
            else _optimizer_sync_blocked("simple_quantized_native_step_not_executed", len(self.params))
        )
        blockers = _dedupe([reason for case in cases for reason in case.get("blocked_reasons", [])])
        if not cases:
            blockers.append("simple_quantized_training_executor_no_grad_params")
        return {
            "schema_version": 1,
            "executor": "turbocore_simple_quantized_optimizer_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "simple_quantized_native_step_failed"),
            "optimizer_kind": self.config.optimizer_kind,
            "training_dispatch": True,
            "training_path_enabled": True,
            "native_step_executed": ok,
            "native_kernel_launched": any(case.get("kernel_executed") is True for case in cases),
            "training_parameters_mutated": ok,
            "should_call_pytorch_optimizer_step": not ok,
            "pytorch_optimizer_state_synced": bool(optimizer_sync.get("synced", False)),
            "optimizer_state_sync": optimizer_sync,
            "parameter_step_count": len(cases),
            "cases": cases,
            "timing": {"elapsed_ms": _elapsed_ms(started)},
            "blocked_reasons": blockers,
        }

    def sync_optimizer_state_to_pytorch(self, *, reason: str = "manual_sync") -> dict[str, Any]:
        if self.optimizer is None:
            return _optimizer_sync_blocked("simple_quantized_optimizer_missing", len(self.params), reason)
        if not self._state:
            return _optimizer_sync_blocked("simple_quantized_state_empty", len(self.params), reason)
        if not self._pytorch_optimizer_state_dirty:
            return {
                "schema_version": 1,
                "synced": False,
                "direction": "simple_quantized_executor_to_pytorch_optimizer",
                "state_tensors": 0,
                "parameter_tensors": len(self.params),
                "optimizer_kind": self.config.optimizer_kind,
                "training_path_enabled": True,
                "reason": "pytorch_optimizer_state_already_current",
                "requested_reason": str(reason or "manual_sync"),
            }
        synced = 0
        state_tensors = 0
        for param in self.params:
            local = self._state.get(id(param))
            if local is None:
                continue
            target = self.optimizer.state[param]
            target["turbocore_quantized_optimizer_kind"] = self.config.optimizer_kind
            target["turbocore_quantized_optimizer_step"] = int(local["step"])
            target["turbocore_quantized_state_q"] = local["state_q"].detach().clone()
            target["turbocore_quantized_scale"] = local["scale"].detach().clone()
            synced += 1
            state_tensors += 2
        self._pytorch_optimizer_state_dirty = False
        self._last_optimizer_state_sync = {
            "schema_version": 1,
            "synced": synced > 0,
            "direction": "simple_quantized_executor_to_pytorch_optimizer",
            "state_tensors": state_tensors,
            "parameter_tensors": synced,
            "optimizer_kind": self.config.optimizer_kind,
            "training_path_enabled": True,
            "reason": "synced" if synced > 0 else "simple_quantized_no_parameter_state_synced",
            "requested_reason": str(reason or "manual_sync"),
        }
        return dict(self._last_optimizer_state_sync)

    def restore_optimizer_state_from_pytorch(self, *, reason: str = "manual_restore") -> dict[str, Any]:
        if self.optimizer is None:
            return _optimizer_restore_blocked("simple_quantized_optimizer_missing", len(self.params), reason)
        restored = 0
        state_tensors = 0
        for param in self.params:
            source = self.optimizer.state.get(param, {})
            if not isinstance(source, Mapping):
                continue
            state_q = source.get("turbocore_quantized_state_q")
            scale = source.get("turbocore_quantized_scale")
            if not isinstance(state_q, torch.Tensor) or not isinstance(scale, torch.Tensor):
                continue
            local = self._state_for(param)
            local["step"] = int(source.get("turbocore_quantized_optimizer_step", 0) or 0)
            local["state_q"] = state_q.detach().clone().to(device=param.device, dtype=torch.uint8).reshape(-1)
            local["scale"] = scale.detach().clone().to(device=param.device, dtype=torch.float32).reshape(-1)
            restored += 1
            state_tensors += 2
        self._pytorch_optimizer_state_dirty = False
        return {
            "schema_version": 1,
            "restored": restored > 0,
            "direction": "pytorch_optimizer_to_simple_quantized_executor",
            "state_tensors": state_tensors,
            "parameter_tensors": restored,
            "optimizer_kind": self.config.optimizer_kind,
            "training_path_enabled": True,
            "reason": "restored" if restored > 0 else "simple_quantized_no_parameter_state_restored",
            "requested_reason": str(reason or "manual_restore"),
        }

    def close(self) -> None:
        self.sync_optimizer_state_to_pytorch(reason="simple_quantized_training_executor_close")
        self._native = None
        self._state.clear()

    def _load_native(self) -> Any | None:
        if self._native is None:
            self._native = native_with_entrypoints(ENTRYPOINT)
        return self._native

    def _step_param(self, native: Any, param: torch.nn.Parameter) -> dict[str, Any]:
        if param.dtype != torch.float32 or not param.is_contiguous():
            return _case_failed("simple_quantized_training_executor_requires_contiguous_fp32_param")
        grad = param.grad.detach().contiguous().float()
        state = self._state_for(param)
        step_before = int(state["step"])
        launch_config = {
            **_launch_config(self.config),
            "step_index": step_before,
            "max_numel": max(int(self.config.max_numel), int(param.numel())),
            "canary_probe_only": False,
            "training_tensor_binding": True,
            "training_dispatch": True,
            "training_path_enabled": True,
        }
        try:
            launch = dict(
                getattr(native, ENTRYPOINT)(
                    self.config.optimizer_kind,
                    param,
                    grad,
                    state["state_q"],
                    state["scale"],
                    json.dumps(launch_config),
                    str(self.workspace_root.resolve()),
                    _cuda_arch(param.device),
                )
            )
        except Exception as exc:  # pragma: no cover - native/CUDA dependent
            return _case_failed(f"{type(exc).__name__}: {exc}", "simple_quantized_native_step_call_failed")
        if not bool(launch.get("ok", False)):
            return _case_failed(
                str(launch.get("reason") or "simple_quantized_native_step_failed"),
                "simple_quantized_native_step_failed",
                launch,
            )
        state["step"] = step_before + 1
        return {
            "schema_version": 1,
            "ok": True,
            "param_numel": int(param.numel()),
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_before": step_before,
            "step_after": int(state["step"]),
            "kernel_executed": bool(launch.get("kernel_executed", False)),
            "training_parameters_mutated": bool(launch.get("training_parameters_mutated", False)),
            "launch": launch,
            "blocked_reasons": [],
        }

    def _state_for(self, param: torch.nn.Parameter) -> dict[str, torch.Tensor | int]:
        key = id(param)
        state = self._state.get(key)
        if state is None or int(state["state_q"].numel()) != int(param.numel()):  # type: ignore[index]
            state = {
                "step": 0,
                "state_q": torch.full_like(param.detach().reshape(-1), 128, dtype=torch.uint8),
                "scale": torch.full(
                    (_blocks_for(int(param.numel())),),
                    1.0e-20,
                    device=param.device,
                    dtype=torch.float32,
                ),
            }
            self._state[key] = state
        return state


def build_simple_quantized_optimizer_training_executor(
    *,
    optimizer: torch.optim.Optimizer | None = None,
    params: Iterable[torch.nn.Parameter],
    config: SimpleQuantizedOptimizerTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> SimpleQuantizedOptimizerTrainingExecutor:
    return SimpleQuantizedOptimizerTrainingExecutor(
        optimizer=optimizer,
        params=params,
        config=config,
        workspace_root=workspace_root,
    )


def _normalize_config(
    value: SimpleQuantizedOptimizerTrainingExecutorConfig | Mapping[str, Any] | None,
) -> SimpleQuantizedOptimizerTrainingExecutorConfig:
    if isinstance(value, SimpleQuantizedOptimizerTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    kind = str(payload.get("optimizer_kind", "lion8bit") or "lion8bit").strip().lower()
    if kind in {"lion_8bit", "lion8"}:
        kind = "lion8bit"
    elif kind in {"pagedlion8bit", "paged_lion_8bit"}:
        kind = "paged_lion8bit"
    elif kind in {"sgdnesterov8bit", "sgd_nesterov_8bit", "sgd8bit"}:
        kind = "sgd_nesterov8bit"
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"Unsupported simple quantized optimizer kind: {kind}")
    betas = payload.get("betas", (0.9, 0.99))
    return SimpleQuantizedOptimizerTrainingExecutorConfig(
        optimizer_kind=kind,
        lr=float(payload.get("lr", 1e-3 if kind != "sgd_nesterov8bit" else 1e-2)),
        betas=(float(betas[0]), float(betas[1])),
        momentum=float(payload.get("momentum", 0.9)),
        weight_decay=float(payload.get("weight_decay", 0.0)),
        max_numel=int(payload.get("max_numel", 1_048_576)),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _launch_config(config: SimpleQuantizedOptimizerTrainingExecutorConfig) -> dict[str, Any]:
    return {
        "lr": float(config.lr),
        "betas": [float(config.betas[0]), float(config.betas[1])],
        "momentum": float(config.momentum),
        "weight_decay": float(config.weight_decay),
    }


def _case_failed(reason: str, blocker: str | None = None, launch: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "reason": reason,
        "kernel_executed": False,
        "training_parameters_mutated": False,
        "launch": dict(launch or {}),
        "blocked_reasons": [blocker or reason],
    }


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "executor": "turbocore_simple_quantized_optimizer_training_executor_v0",
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


def _optimizer_sync_blocked(reason: str, parameter_tensors: int, requested_reason: str = "manual_sync") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "synced": False,
        "direction": "simple_quantized_executor_to_pytorch_optimizer",
        "state_tensors": 0,
        "parameter_tensors": int(parameter_tensors),
        "training_path_enabled": True,
        "reason": str(reason or "simple_quantized_optimizer_state_sync_blocked"),
        "requested_reason": str(requested_reason or "manual_sync"),
    }


def _optimizer_restore_blocked(
    reason: str,
    parameter_tensors: int,
    requested_reason: str = "manual_restore",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "restored": False,
        "direction": "pytorch_optimizer_to_simple_quantized_executor",
        "state_tensors": 0,
        "parameter_tensors": int(parameter_tensors),
        "training_path_enabled": True,
        "reason": str(reason or "simple_quantized_optimizer_state_restore_blocked"),
        "requested_reason": str(requested_reason or "manual_restore"),
    }


def _blocks_for(numel: int) -> int:
    return max(1, (int(numel) + BLOCK_SIZE - 1) // BLOCK_SIZE)


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


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
    "SimpleQuantizedOptimizerTrainingExecutor",
    "SimpleQuantizedOptimizerTrainingExecutorConfig",
    "build_simple_quantized_optimizer_training_executor",
]
