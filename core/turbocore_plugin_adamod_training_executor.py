"""Default-off training executor for selected plugin adamod native dispatch."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from core.services.native_module_loader import native_with_entrypoints
from core.turbocore_native_tensor_binding import build_flat_adamw_native_binding_request
from core.turbocore_tensor_handle_registry import (
    TurboCoreTensorHandleRegistry,
    build_tensor_object_map_for_handles,
)


ENTRYPOINTS = (
    "create_flat_adamw_tensor_binding_session",
    "tensor_binding_session_cuda_adamod_tensor_probe",
    "destroy_tensor_binding_session",
)
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PluginAdaModTrainingExecutorConfig:
    lr: float = 1e-3
    betas: tuple[float, float, float] = (0.9, 0.99, 0.9999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    block_size: int = 128
    require_native_cuda: bool = True


class PluginAdaModTrainingExecutor:
    """Launch selected AdaMod native steps against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: PluginAdaModTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("PluginAdaModTrainingExecutor requires trainable parameters")
        self._param_ids = {id(param) for param in self.params}
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)
        self._native: Any | None = None

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("plugin_adamod_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("plugin_adamod_training_executor_requires_cuda_params")
        native = self._load_native()
        if native is None:
            return _blocked("plugin_adamod_training_dispatch_entrypoints_missing")
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
            blockers.append("plugin_adamod_training_executor_no_grad_params")
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_adamod_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "plugin_adamod_native_step_failed"),
            "optimizer_kind": "adamod",
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
            self._native = native_with_entrypoints(*ENTRYPOINTS)
        return self._native

    def _step_param(
        self,
        native: Any,
        param: torch.nn.Parameter,
        group_cfg: Mapping[str, Any],
        group_step_before: int,
    ) -> dict[str, Any]:
        if type(self.optimizer).__name__.lower() != "adamod":
            return _case_failed("plugin_adamod_training_executor_optimizer_not_adamod", "plugin_adamod_optimizer_class_unsupported")
        if bool(group_cfg.get("maximize", False)):
            return _case_failed("plugin_adamod_native_step_requires_plain_gradient", "plugin_adamod_gradient_mode_unsupported")
        if param.dtype != torch.float32 or param.grad is None or param.grad.dtype != torch.float32:
            return _case_failed("plugin_adamod_native_step_requires_float32", "plugin_adamod_native_step_dtype_unsupported")
        if not param.is_contiguous() or not param.grad.is_contiguous():
            return _case_failed("plugin_adamod_native_step_requires_contiguous_param_grad", "plugin_adamod_native_step_layout_unsupported")
        state = self.optimizer.state[param]
        if "exp_avg" not in state or "exp_avg_sq" not in state or "exp_avg_lr" not in state:
            return _case_failed("plugin_adamod_live_state_missing", "plugin_adamod_live_state_missing")
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg_lr = state["exp_avg_lr"]
        if not (torch.is_tensor(exp_avg) and torch.is_tensor(exp_avg_sq) and torch.is_tensor(exp_avg_lr)):
            return _case_failed("plugin_adamod_live_state_tensors_missing", "plugin_adamod_live_state_missing")
        if exp_avg.dtype != torch.float32 or exp_avg_sq.dtype != torch.float32 or exp_avg_lr.dtype != torch.float32:
            return _case_failed("plugin_adamod_native_state_requires_float32", "plugin_adamod_native_state_dtype_unsupported")
        if not exp_avg.is_contiguous() or not exp_avg_sq.is_contiguous() or not exp_avg_lr.is_contiguous():
            return _case_failed("plugin_adamod_native_state_requires_contiguous", "plugin_adamod_native_state_layout_unsupported")
        registry = TurboCoreTensorHandleRegistry(namespace="plugin_adamod_training_executor")
        param_flat = param.detach().reshape(-1)
        grad_flat = param.grad.detach().reshape(-1)
        exp_avg_flat = exp_avg.reshape(-1)
        exp_avg_sq_flat = exp_avg_sq.reshape(-1)
        exp_avg_lr_flat = exp_avg_lr.reshape(-1)
        try:
            handles = registry.register_flat_adamw_buffers(
                param_flat=param_flat,
                grad_flat=grad_flat,
                exp_avg=exp_avg_flat,
                exp_avg_sq=exp_avg_sq_flat,
            )
            exp_avg_lr_record = registry.register(
                exp_avg_lr_flat,
                role="exp_avg_lr",
                expected_numel=int(param_flat.numel()),
            )
            request = build_flat_adamw_native_binding_request(registry, handles)
            request["optimizer"] = "AdaMod"
            request["bindings"] = [
                *list(request["bindings"]),
                {
                    "role": exp_avg_lr_record.role,
                    "handle_id": exp_avg_lr_record.handle_id,
                    "handle_kind": exp_avg_lr_record.handle_kind,
                    "numel": exp_avg_lr_record.numel,
                    "dtype": exp_avg_lr_record.dtype,
                    "device_type": exp_avg_lr_record.device_type,
                    "device_index": exp_avg_lr_record.device_index,
                    "layout": exp_avg_lr_record.layout,
                    "contiguous": exp_avg_lr_record.contiguous,
                    "alignment_bytes": exp_avg_lr_record.alignment_bytes,
                    "pointer_exported": False,
                },
            ]
            tensor_map = build_tensor_object_map_for_handles(registry, handles)
            tensor_map[exp_avg_lr_record.handle_id] = registry.resolve(exp_avg_lr_record.handle_id)
            session = dict(native.create_flat_adamw_tensor_binding_session(json.dumps(request), tensor_map))
            if not bool(session.get("ok", False)):
                return _case_failed(
                    "plugin_adamod_tensor_binding_session_create_failed",
                    "plugin_adamod_tensor_binding_session_create_failed",
                    session=session,
                )
            session_id = int(session["session_id"])
            try:
                launch = dict(
                    native.tensor_binding_session_cuda_adamod_tensor_probe(
                        session_id,
                        json.dumps(
                            {
                                **dict(group_cfg),
                                "step_index": group_step_before,
                                "block_size": int(self.config.block_size),
                                "max_numel": int(param_flat.numel()),
                                "training_tensor_binding": True,
                                "training_dispatch": False,
                                "training_path_enabled": False,
                            }
                        ),
                    )
                )
            finally:
                try:
                    native.destroy_tensor_binding_session(session_id)
                except Exception:
                    pass
        except Exception as exc:  # pragma: no cover - native/CUDA dependent
            return _case_failed(f"{type(exc).__name__}: {exc}", "plugin_adamod_native_step_call_failed")
        if not bool(launch.get("ok", False)):
            return _case_failed(
                str(launch.get("reason") or "plugin_adamod_native_step_failed"),
                "plugin_adamod_native_step_failed",
                launch=launch,
            )
        return {
            "schema_version": 1,
            "ok": True,
            "param_numel": int(param.numel()),
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_before": group_step_before,
            "step_after": group_step_before + 1,
            "kernel_executed": bool(launch.get("kernel_executed", False)),
            "training_parameters_mutated": bool(launch.get("parameters_mutated", False)),
            "launch": launch,
            "blocked_reasons": [],
        }


def build_plugin_adamod_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: PluginAdaModTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> PluginAdaModTrainingExecutor:
    return PluginAdaModTrainingExecutor(
        optimizer=optimizer,
        params=params,
        config=config,
        workspace_root=workspace_root,
    )


def _normalize_config(
    value: PluginAdaModTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> PluginAdaModTrainingExecutorConfig:
    if isinstance(value, PluginAdaModTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = _adamod_beta_triplet(payload.get("betas", group.get("betas", (0.9, 0.99, 0.9999))), group)
    return PluginAdaModTrainingExecutorConfig(
        lr=float(payload.get("lr", group.get("lr", 1e-3))),
        betas=betas,
        eps=float(payload.get("eps", group.get("eps", 1e-8))),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0))),
        block_size=int(payload.get("block_size", 128)),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: PluginAdaModTrainingExecutorConfig) -> dict[str, Any]:
    betas = _adamod_beta_triplet(group.get("betas", config.betas), group)
    return {
        "lr": float(group.get("lr", config.lr)),
        "betas": [float(betas[0]), float(betas[1]), float(betas[2])],
        "eps": float(group.get("eps", config.eps)),
        "weight_decay": float(group.get("weight_decay", config.weight_decay)),
        "weight_decouple": bool(group.get("weight_decouple", True)),
        "fixed_decay": bool(group.get("fixed_decay", False)),
        "maximize": bool(group.get("maximize", False)),
    }


def _adamod_beta_triplet(value: Any, group: Mapping[str, Any] | None = None) -> tuple[float, float, float]:
    values = list(value) if isinstance(value, (list, tuple)) else []
    while len(values) < 2:
        values.append((0.9, 0.99)[len(values)])
    if len(values) < 3:
        group_beta3 = dict(group or {}).get("beta3", 0.9999)
        values.append(group_beta3)
    return (float(values[0]), float(values[1]), float(values[2]))


def _group_step_to_int(group: Mapping[str, Any]) -> int:
    value = group.get("step", 0)
    if torch.is_tensor(value) and value.numel() > 0:
        return int(value.detach().reshape(-1)[0].cpu().item())
    return int(value or 0)


def _set_group_step(group: dict[Any, Any], value: int) -> None:
    current = group.get("step")
    if torch.is_tensor(current):
        current.fill_(int(value))
    else:
        group["step"] = int(value)


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
        "executor": "turbocore_plugin_adamod_training_executor_v0",
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
    "PluginAdaModTrainingExecutor",
    "PluginAdaModTrainingExecutorConfig",
    "build_plugin_adamod_training_executor",
]

