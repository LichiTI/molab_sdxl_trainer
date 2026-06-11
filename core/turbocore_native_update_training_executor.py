"""Training executor adapter for TurboCore native update dispatch.

The adapter is intentionally narrow: it replaces exactly one AdamW optimizer
step for an already-selected parameter group, mirrors state back to the PyTorch
optimizer for fallback/checkpoint safety, and reports enough evidence for the
dispatch runtime to decide whether the Python optimizer step must still run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

from core.turbocore_update_checkpoint_contract import (
    sync_flat_owner_state_from_optimizer,
    sync_optimizer_state_from_flat_owner,
)
from core.turbocore_update_executor import TurboCoreUpdateExecutor, TurboCoreUpdateExecutorConfig


@dataclass(frozen=True)
class NativeUpdateTrainingExecutorConfig:
    optimizer: str = "AdamW"
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 0.0
    finite_check: bool = True
    direct_grad: bool = False
    prefer_native_cuda: bool = True
    require_native_cuda: bool = False
    prefer_triton: bool = True
    block_size: int = 1024
    sync_optimizer_state_each_step: bool = True
    sync_params_from_optimizer_each_step: bool = True
    sync_pytorch_optimizer_state_each_step: bool = True
    native_runtime_synchronization_policy: str = "context_synchronize"
    native_runtime_stream_guard_descriptor: dict[str, Any] | None = None
    native_runtime_stream_lifetime_lease_evidence: dict[str, Any] | None = None

    def executor_config(self) -> TurboCoreUpdateExecutorConfig:
        return TurboCoreUpdateExecutorConfig(
            optimizer=self.optimizer,
            lr=float(self.lr),
            betas=(float(self.betas[0]), float(self.betas[1])),
            eps=float(self.eps),
            weight_decay=float(self.weight_decay),
            max_grad_norm=float(self.max_grad_norm),
            finite_check=bool(self.finite_check),
            direct_grad=bool(self.direct_grad),
            copy_params_back=True,
            zero_owner_grad=True,
            prefer_native_cuda=bool(self.prefer_native_cuda),
            native_training_dispatch=True,
            require_native_cuda=bool(self.require_native_cuda),
            prefer_triton=bool(self.prefer_triton),
            block_size=int(self.block_size),
            native_runtime_synchronization_policy=str(
                self.native_runtime_synchronization_policy or "context_synchronize"
            ),
            native_runtime_stream_guard_descriptor=dict(self.native_runtime_stream_guard_descriptor or {}),
            native_runtime_stream_lifetime_lease_evidence=dict(
                self.native_runtime_stream_lifetime_lease_evidence or {}
            ),
        )


class NativeUpdateTrainingExecutor:
    """Callable native-update executor used by dispatch runtime."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: NativeUpdateTrainingExecutorConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("NativeUpdateTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.executor = TurboCoreUpdateExecutor(self.params, self.config.executor_config())
        self._call_index = 0
        self._pytorch_optimizer_state_dirty = False
        self._last_optimizer_state_sync: dict[str, Any] = {}

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        request_payload = dict(request or {})
        if not bool(request_payload.get("training_dispatch", False)) or not bool(request_payload.get("training_path_enabled", False)):
            return _blocked("native_update_training_executor_requires_training_dispatch")
        if _has_non_adamw_optimizer(self.optimizer):
            return _blocked("native_update_training_executor_optimizer_not_adamw")

        initial_sync = self._call_index == 0
        started = time.perf_counter()
        if bool(self.config.sync_optimizer_state_each_step) or initial_sync:
            state_sync = sync_flat_owner_state_from_optimizer(self.executor.owner, self.optimizer, self.params)
        else:
            state_sync = _deferred_state_sync_report(self.executor.owner)
        state_sync_ms = _elapsed_ms(started)
        param_sync_started = time.perf_counter()
        if bool(self.config.sync_params_from_optimizer_each_step) or initial_sync:
            self.executor.owner.param_flat.copy_(_flatten_params(self.params, device=self.executor.owner.param_flat.device))
        param_sync_ms = _elapsed_ms(param_sync_started)
        step_started = time.perf_counter()
        report = self.executor.step()
        step_ms = _elapsed_ms(step_started)
        optimizer_sync_started = time.perf_counter()
        if bool(self.config.sync_pytorch_optimizer_state_each_step):
            optimizer_sync = sync_optimizer_state_from_flat_owner(self.executor.owner, self.optimizer, self.params)
            self._pytorch_optimizer_state_dirty = False
        else:
            optimizer_sync = _deferred_optimizer_sync_report(self.executor.owner, len(self.params))
            if not bool(report.owner_step.get("skipped", False)):
                self._pytorch_optimizer_state_dirty = True
        self._last_optimizer_state_sync = dict(optimizer_sync)
        optimizer_sync_ms = _elapsed_ms(optimizer_sync_started)
        owner_step = dict(report.owner_step)
        skipped = bool(owner_step.get("skipped", False))
        self._call_index += 1
        elapsed_ms = _elapsed_ms(started)
        return {
            "schema_version": 1,
            "executor": "turbocore_native_update_training_executor_v0",
            "ok": not skipped,
            "reason": str(owner_step.get("reason", "stepped") or "stepped"),
            "training_dispatch": True,
            "training_path_enabled": True,
            "native_step_executed": not skipped,
            "native_kernel_launched": bool(report.native_kernel_present),
            "training_parameters_mutated": bool(report.copied_params_back and not skipped),
            "pytorch_optimizer_state_synced": bool(optimizer_sync.get("synced", False)),
            "should_call_pytorch_optimizer_step": skipped,
            "state_sync": state_sync,
            "optimizer_state_sync": optimizer_sync,
            "update_report": report.as_dict(),
            "timing": {
                "elapsed_ms": elapsed_ms,
                "state_sync_ms": state_sync_ms,
                "param_sync_ms": param_sync_ms,
                "executor_step_ms": step_ms,
                "optimizer_state_sync_ms": optimizer_sync_ms,
            },
            "blocked_reasons": [] if not skipped else [str(owner_step.get("reason", "native_update_step_skipped") or "native_update_step_skipped")],
        }

    def sync_optimizer_state_to_pytorch(self, *, reason: str = "manual_sync") -> dict[str, Any]:
        if not self._pytorch_optimizer_state_dirty:
            return {
                "schema_version": 1,
                "synced": False,
                "direction": "flat_owner_to_pytorch_optimizer",
                "state_tensors": 0,
                "parameter_tensors": len(self.params),
                "step_index": int(getattr(self.executor.owner, "step_index", 0) or 0),
                "training_path_enabled": True,
                "reason": "pytorch_optimizer_state_already_current",
                "requested_reason": str(reason or "manual_sync"),
            }
        report = sync_optimizer_state_from_flat_owner(self.executor.owner, self.optimizer, self.params)
        report["requested_reason"] = str(reason or "manual_sync")
        self._pytorch_optimizer_state_dirty = False
        self._last_optimizer_state_sync = dict(report)
        return report

    def close(self) -> None:
        self.sync_optimizer_state_to_pytorch(reason="native_training_executor_close")
        self.executor.close()


def build_native_update_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: NativeUpdateTrainingExecutorConfig | Mapping[str, Any] | None = None,
) -> NativeUpdateTrainingExecutor:
    return NativeUpdateTrainingExecutor(optimizer=optimizer, params=params, config=config)


def _normalize_config(
    value: NativeUpdateTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> NativeUpdateTrainingExecutorConfig:
    if isinstance(value, NativeUpdateTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = payload.get("betas", group.get("betas", (0.9, 0.999)))
    return NativeUpdateTrainingExecutorConfig(
        optimizer=str(payload.get("optimizer", "AdamW") or "AdamW"),
        lr=float(payload.get("lr", group.get("lr", 1e-4))),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(payload.get("eps", group.get("eps", 1e-8))),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.01))),
        max_grad_norm=float(payload.get("max_grad_norm", 0.0)),
        finite_check=bool(payload.get("finite_check", True)),
        direct_grad=bool(payload.get("direct_grad", False)),
        prefer_native_cuda=bool(payload.get("prefer_native_cuda", True)),
        require_native_cuda=bool(payload.get("require_native_cuda", False)),
        prefer_triton=bool(payload.get("prefer_triton", True)),
        block_size=int(payload.get("block_size", 1024)),
        sync_optimizer_state_each_step=bool(payload.get("sync_optimizer_state_each_step", True)),
        sync_params_from_optimizer_each_step=bool(payload.get("sync_params_from_optimizer_each_step", True)),
        sync_pytorch_optimizer_state_each_step=bool(payload.get("sync_pytorch_optimizer_state_each_step", True)),
        native_runtime_synchronization_policy=str(
            payload.get("native_runtime_synchronization_policy", "context_synchronize")
            or "context_synchronize"
        ),
        native_runtime_stream_guard_descriptor=dict(payload.get("native_runtime_stream_guard_descriptor", {}) or {}),
        native_runtime_stream_lifetime_lease_evidence=dict(
            payload.get("native_runtime_stream_lifetime_lease_evidence", {}) or {}
        ),
    )


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "executor": "turbocore_native_update_training_executor_v0",
        "ok": False,
        "reason": str(reason),
        "training_dispatch": True,
        "training_path_enabled": True,
        "native_step_executed": False,
        "native_kernel_launched": False,
        "training_parameters_mutated": False,
        "should_call_pytorch_optimizer_step": True,
        "blocked_reasons": [str(reason)],
    }


def _flatten_params(params: Iterable[torch.Tensor], *, device: torch.device) -> torch.Tensor:
    tensors = [param.detach().float().reshape(-1).to(device=device) for param in params if isinstance(param, torch.Tensor)]
    if not tensors:
        return torch.empty(0, device=device, dtype=torch.float32)
    return torch.cat(tensors).contiguous()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 4)


def _deferred_state_sync_report(owner: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "synced": False,
        "state_tensors": 0,
        "missing_state_tensors": 0,
        "step_index": int(getattr(owner, "step_index", 0) or 0),
        "training_path_enabled": False,
        "reason": "owner_authoritative_state_reused",
    }


def _deferred_optimizer_sync_report(owner: Any, parameter_tensors: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "synced": False,
        "direction": "flat_owner_to_pytorch_optimizer",
        "state_tensors": 0,
        "parameter_tensors": int(parameter_tensors),
        "step_index": int(getattr(owner, "step_index", 0) or 0),
        "training_path_enabled": True,
        "reason": "pytorch_optimizer_state_sync_deferred",
    }


def _has_non_adamw_optimizer(optimizer: torch.optim.Optimizer) -> bool:
    return "adamw" not in type(optimizer).__name__.lower()


__all__ = [
    "NativeUpdateTrainingExecutor",
    "NativeUpdateTrainingExecutorConfig",
    "build_native_update_training_executor",
]
