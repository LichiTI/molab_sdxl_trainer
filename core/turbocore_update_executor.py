"""TurboCore update-path executor prototype.

This module composes persistent flat AdamW ownership with an optional direct
gradient binding. It is deliberately not connected to the product trainer yet;
the goal is to validate the lifecycle shape for a future native update path.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch

from core.turbocore_direct_grad import TurboCoreDirectGradBinding
from core.turbocore_flat_adamw_state import FlatAdamWConfig, PersistentFlatAdamW


@dataclass(frozen=True)
class TurboCoreUpdateExecutorConfig:
    optimizer: str = "AdamW"
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    finite_check: bool = True
    direct_grad: bool = False
    copy_params_back: bool = True
    zero_owner_grad: bool = True
    prefer_native_cuda: bool = False
    native_training_dispatch: bool = False
    require_native_cuda: bool = False
    prefer_triton: bool = False
    block_size: int = 1024
    native_runtime_synchronization_policy: str = "context_synchronize"
    native_runtime_stream_guard_descriptor: dict[str, Any] | None = None
    native_runtime_stream_lifetime_lease_evidence: dict[str, Any] | None = None

    def flat_adamw_config(self) -> FlatAdamWConfig:
        return FlatAdamWConfig(
            lr=float(self.lr),
            betas=(float(self.betas[0]), float(self.betas[1])),
            eps=float(self.eps),
            weight_decay=float(self.weight_decay),
            max_grad_norm=float(self.max_grad_norm),
            finite_check=bool(self.finite_check),
            prefer_native_cuda=bool(self.prefer_native_cuda),
            native_training_dispatch=bool(self.native_training_dispatch),
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

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["betas"] = [float(self.betas[0]), float(self.betas[1])]
        return payload


@dataclass(frozen=True)
class TurboCoreUpdateReport:
    schema_version: int
    executor: str
    optimizer: str
    step_index: int
    used_direct_grad: bool
    copied_params_back: bool
    zeroed_owner_grad: bool
    owner_backend: str
    native_kernel_present: bool
    training_path_enabled: bool
    owner_step: dict[str, Any]
    direct_grad_snapshot: dict[str, Any]
    timing: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TurboCoreUpdateExecutor:
    """Owns an experimental flat update path for one parameter group."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        config: TurboCoreUpdateExecutorConfig | dict[str, Any] | None = None,
    ) -> None:
        self.params = _as_params(params)
        self.config = _normalize_config(config)
        if self.config.optimizer.strip().lower() != "adamw":
            raise ValueError("TurboCoreUpdateExecutor prototype currently supports AdamW only")
        self.owner = PersistentFlatAdamW(self.params, self.config.flat_adamw_config())
        self.direct_grad_binding: TurboCoreDirectGradBinding | None = None
        if self.config.direct_grad:
            self.direct_grad_binding = TurboCoreDirectGradBinding(self.owner, self.params).install()

    def step(self) -> TurboCoreUpdateReport:
        started = time.perf_counter()
        grad_sync_ms = 0.0
        if self.direct_grad_binding is None:
            grad_sync_started = time.perf_counter()
            self.owner.set_grads([param.grad for param in self.params])
            grad_sync_ms = _elapsed_ms(grad_sync_started)
        owner_step_started = time.perf_counter()
        owner_report = self.owner.step()
        owner_step_ms = _elapsed_ms(owner_step_started)
        copied = False
        copyback_ms = 0.0
        if bool(self.config.copy_params_back) and not bool(owner_report.skipped):
            copyback_started = time.perf_counter()
            self.owner.copy_params_to_(self.params)
            copyback_ms = _elapsed_ms(copyback_started)
            copied = True
        zeroed = False
        zero_grad_ms = 0.0
        if bool(self.config.zero_owner_grad):
            zero_grad_started = time.perf_counter()
            self.owner.zero_grad()
            zero_grad_ms = _elapsed_ms(zero_grad_started)
            zeroed = True
        direct_snapshot = self.direct_grad_binding.snapshot() if self.direct_grad_binding is not None else {}
        return TurboCoreUpdateReport(
            schema_version=1,
            executor="turbocore_update_executor_v0",
            optimizer="AdamW",
            step_index=int(self.owner.step_index),
            used_direct_grad=self.direct_grad_binding is not None,
            copied_params_back=copied,
            zeroed_owner_grad=zeroed,
            owner_backend=str(owner_report.backend),
            native_kernel_present=bool(owner_report.native_kernel_present),
            training_path_enabled=False,
            owner_step=owner_report.as_dict(),
            direct_grad_snapshot=direct_snapshot,
            timing={
                "elapsed_ms": _elapsed_ms(started),
                "grad_sync_ms": grad_sync_ms,
                "owner_step_ms": owner_step_ms,
                "copyback_ms": copyback_ms,
                "zero_grad_ms": zero_grad_ms,
            },
        )

    def shadow_step_from_grads(
        self,
        grads: Iterable[torch.Tensor | None] | None = None,
        *,
        sync_grads: bool = True,
    ) -> TurboCoreUpdateReport:
        started = time.perf_counter()
        grad_sync_ms = 0.0
        if bool(sync_grads):
            grad_sync_started = time.perf_counter()
            self.owner.set_grads(list(grads) if grads is not None else [param.grad for param in self.params])
            grad_sync_ms = _elapsed_ms(grad_sync_started)
        owner_step_started = time.perf_counter()
        owner_report = self.owner.step()
        owner_step_ms = _elapsed_ms(owner_step_started)
        direct_snapshot = self.direct_grad_binding.snapshot() if self.direct_grad_binding is not None else {}
        return TurboCoreUpdateReport(
            schema_version=1,
            executor="turbocore_update_executor_v0",
            optimizer="AdamW",
            step_index=int(self.owner.step_index),
            used_direct_grad=self.direct_grad_binding is not None,
            copied_params_back=False,
            zeroed_owner_grad=False,
            owner_backend=str(owner_report.backend),
            native_kernel_present=bool(owner_report.native_kernel_present),
            training_path_enabled=False,
            owner_step=owner_report.as_dict(),
            direct_grad_snapshot=direct_snapshot,
            timing={
                "elapsed_ms": _elapsed_ms(started),
                "grad_sync_ms": grad_sync_ms,
                "owner_step_ms": owner_step_ms,
                "copyback_ms": 0.0,
                "zero_grad_ms": 0.0,
            },
        )

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for param in self.params:
            if bool(set_to_none):
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()
        self.owner.zero_grad()
        if self.direct_grad_binding is not None:
            self.direct_grad_binding.zero_owner_grad()

    def close(self) -> None:
        if self.direct_grad_binding is not None:
            self.direct_grad_binding.remove()
            self.direct_grad_binding = None
        close_owner = getattr(self.owner, "close", None)
        if callable(close_owner):
            close_owner()

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "executor": "turbocore_update_executor_v0",
            "optimizer": "AdamW",
            "training_path_enabled": False,
            "native_kernel_present": False,
            "config": self.config.as_dict(),
            "owner": self.owner.snapshot(),
            "direct_grad": self.direct_grad_binding.snapshot() if self.direct_grad_binding is not None else None,
        }


def _as_params(params: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    result = [param for param in params if isinstance(param, torch.nn.Parameter)]
    if not result:
        raise ValueError("TurboCoreUpdateExecutor requires at least one parameter")
    devices = {str(param.device) for param in result}
    if len(devices) != 1:
        raise ValueError("TurboCoreUpdateExecutor requires all parameters on the same device")
    return result


def _normalize_config(
    config: TurboCoreUpdateExecutorConfig | dict[str, Any] | None,
) -> TurboCoreUpdateExecutorConfig:
    if config is None:
        return TurboCoreUpdateExecutorConfig()
    if isinstance(config, TurboCoreUpdateExecutorConfig):
        return config
    betas = config.get("betas", (0.9, 0.999))
    return TurboCoreUpdateExecutorConfig(
        optimizer=str(config.get("optimizer", "AdamW") or "AdamW"),
        lr=float(config.get("lr", 1e-4)),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(config.get("eps", 1e-8)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        finite_check=bool(config.get("finite_check", True)),
        direct_grad=bool(config.get("direct_grad", False)),
        copy_params_back=bool(config.get("copy_params_back", True)),
        zero_owner_grad=bool(config.get("zero_owner_grad", True)),
        prefer_native_cuda=bool(config.get("prefer_native_cuda", False)),
        native_training_dispatch=bool(config.get("native_training_dispatch", False)),
        require_native_cuda=bool(config.get("require_native_cuda", False)),
        prefer_triton=bool(config.get("prefer_triton", False)),
        block_size=int(config.get("block_size", 1024)),
        native_runtime_synchronization_policy=str(
            config.get("native_runtime_synchronization_policy", "context_synchronize")
            or "context_synchronize"
        ),
        native_runtime_stream_guard_descriptor=dict(config.get("native_runtime_stream_guard_descriptor", {}) or {}),
        native_runtime_stream_lifetime_lease_evidence=dict(
            config.get("native_runtime_stream_lifetime_lease_evidence", {}) or {}
        ),
    )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 4)


__all__ = [
    "TurboCoreUpdateExecutor",
    "TurboCoreUpdateExecutorConfig",
    "TurboCoreUpdateReport",
]
