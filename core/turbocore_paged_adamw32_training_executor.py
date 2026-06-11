"""Default-off training executor for fp32 PagedAdamW native dispatch."""

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
    "tensor_binding_session_cuda_adamw_tensor_probe",
    "destroy_tensor_binding_session",
)
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PagedAdamW32TrainingExecutorConfig:
    optimizer_kind: str = "paged_adamw"
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    block_size: int = 128
    require_native_cuda: bool = True


class PagedAdamW32TrainingExecutor:
    """Launch AdamW native steps against live bitsandbytes fp32 paged state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: PagedAdamW32TrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("PagedAdamW32TrainingExecutor requires trainable parameters")
        self._param_ids = {id(param) for param in self.params}
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)
        self._native: Any | None = None

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("paged_adamw32_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("paged_adamw32_training_executor_requires_cuda_params")
        native = self._load_native()
        if native is None:
            return _blocked("paged_adamw32_training_dispatch_entrypoints_missing")
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
            blockers.append("paged_adamw32_training_executor_no_grad_params")
        return {
            "schema_version": 1,
            "executor": "turbocore_paged_adamw32_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "paged_adamw32_native_step_failed"),
            "optimizer_kind": self.config.optimizer_kind,
            "training_dispatch": True,
            "training_path_enabled": True,
            "native_step_executed": ok,
            "native_kernel_launched": any(bool(case.get("kernel_executed", False)) for case in cases),
            "training_parameters_mutated": ok,
            "should_call_pytorch_optimizer_step": not ok,
            "pytorch_optimizer_state_synced": True,
            "parameter_step_count": len(cases),
            "cases": cases,
            "timing": {"elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4)},
            "blocked_reasons": blockers,
        }

    def close(self) -> None:
        return None

    def _load_native(self) -> Any | None:
        if self._native is None:
            self._native = native_with_entrypoints(*ENTRYPOINTS)
        return self._native

    def _step_param(self, native: Any, param: torch.nn.Parameter, group_cfg: Mapping[str, Any]) -> dict[str, Any]:
        if type(self.optimizer).__name__ not in {"PagedAdamW", "PagedAdamW32bit"}:
            return _case_failed("paged_adamw32_optimizer_class_unsupported", "paged_adamw32_optimizer_class_unsupported")
        if param.dtype != torch.float32 or param.grad is None or param.grad.dtype != torch.float32:
            return _case_failed("paged_adamw32_native_step_requires_float32", "paged_adamw32_native_step_dtype_unsupported")
        state = self.optimizer.state[param]
        if "step" not in state or "state1" not in state or "state2" not in state:
            return _case_failed("paged_adamw32_live_state_missing", "paged_adamw32_live_state_missing")
        state1 = state["state1"]
        state2 = state["state2"]
        if not (torch.is_tensor(state1) and torch.is_tensor(state2)):
            return _case_failed("paged_adamw32_live_state_tensors_missing", "paged_adamw32_live_state_missing")
        if state1.dtype != torch.float32 or state2.dtype != torch.float32:
            return _case_failed("paged_adamw32_state_requires_float32", "paged_adamw32_state_dtype_unsupported")
        step_before = _step_to_int(state.get("step"))
        registry = TurboCoreTensorHandleRegistry(namespace="paged_adamw32_training_executor")
        param_flat = param.detach().reshape(-1).contiguous()
        grad_flat = param.grad.detach().reshape(-1).contiguous()
        state1_flat = state1.reshape(-1).contiguous()
        state2_flat = state2.reshape(-1).contiguous()
        try:
            handles = registry.register_flat_adamw_buffers(
                param_flat=param_flat,
                grad_flat=grad_flat,
                exp_avg=state1_flat,
                exp_avg_sq=state2_flat,
            )
            request = build_flat_adamw_native_binding_request(registry, handles)
            tensor_map = build_tensor_object_map_for_handles(registry, handles)
            session = dict(native.create_flat_adamw_tensor_binding_session(json.dumps(request), tensor_map))
            if not bool(session.get("ok", False)):
                return _case_failed("paged_adamw32_tensor_binding_session_create_failed", "paged_adamw32_tensor_binding_session_create_failed", session=session)
            session_id = int(session["session_id"])
            try:
                launch = dict(
                    native.tensor_binding_session_cuda_adamw_tensor_probe(
                        session_id,
                        json.dumps(
                            {
                                **dict(group_cfg),
                                "step_index": step_before,
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
            return _case_failed(f"{type(exc).__name__}: {exc}", "paged_adamw32_native_step_call_failed")
        if not bool(launch.get("ok", False)):
            return _case_failed(str(launch.get("reason") or "paged_adamw32_native_step_failed"), "paged_adamw32_native_step_failed", launch=launch)
        _set_step(state, step_before + 1)
        return {
            "schema_version": 1,
            "ok": True,
            "param_numel": int(param.numel()),
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_before": step_before,
            "step_after": _step_to_int(state.get("step")),
            "kernel_executed": bool(launch.get("kernel_executed", False)),
            "training_parameters_mutated": bool(launch.get("parameters_mutated", False)),
            "launch": launch,
            "blocked_reasons": [],
        }


def build_paged_adamw32_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: PagedAdamW32TrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> PagedAdamW32TrainingExecutor:
    return PagedAdamW32TrainingExecutor(
        optimizer=optimizer,
        params=params,
        config=config,
        workspace_root=workspace_root,
    )


def _normalize_config(
    value: PagedAdamW32TrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> PagedAdamW32TrainingExecutorConfig:
    if isinstance(value, PagedAdamW32TrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = payload.get("betas", group.get("betas", (0.9, 0.999)))
    return PagedAdamW32TrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind", "paged_adamw") or "paged_adamw"),
        lr=float(payload.get("lr", group.get("lr", 1e-3))),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(payload.get("eps", group.get("eps", 1e-8))),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.01))),
        block_size=int(payload.get("block_size", 128)),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: PagedAdamW32TrainingExecutorConfig) -> dict[str, Any]:
    betas = group.get("betas", config.betas)
    return {
        "lr": float(group.get("lr", config.lr)),
        "betas": [float(betas[0]), float(betas[1])],
        "eps": float(group.get("eps", config.eps)),
        "weight_decay": float(group.get("weight_decay", config.weight_decay)),
    }


def _step_to_int(value: Any) -> int:
    if torch.is_tensor(value) and value.numel() > 0:
        return int(value.detach().reshape(-1)[0].cpu().item())
    return int(value or 0)


def _set_step(state: dict[Any, Any], value: int) -> None:
    current = state.get("step")
    if torch.is_tensor(current):
        current.fill_(int(value))
    else:
        state["step"] = int(value)


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
        "executor": "turbocore_paged_adamw32_training_executor_v0",
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
    "PagedAdamW32TrainingExecutor",
    "PagedAdamW32TrainingExecutorConfig",
    "build_paged_adamw32_training_executor",
]
