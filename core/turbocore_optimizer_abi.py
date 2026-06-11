"""Stateful optimizer ABI prototype for future TurboCore native backends.

This module is Python-only and developer-facing. It models the lifecycle a
Rust/CUDA optimizer must preserve before it can be considered for training:
persistent state, finite-gradient handling, clipping, zero-grad, state save,
state restore, and introspection.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch


NATIVE_OPTIMIZER_STATEFUL_ENTRYPOINTS = [
    "create_stateful_adamw_optimizer",
    "optimizer_step",
    "optimizer_zero_grad",
    "optimizer_state_dict",
    "optimizer_load_state_dict",
    "optimizer_snapshot",
    "destroy_optimizer",
]

NATIVE_FLAT_ADAMW_OWNER_ENTRYPOINTS = [
    "create_flat_adamw_owner",
    "flat_adamw_set_grad_buffer",
    "flat_adamw_step",
    "flat_adamw_zero_grad",
    "flat_adamw_state_dict",
    "flat_adamw_load_state_dict",
    "flat_adamw_snapshot",
    "destroy_flat_adamw_owner",
]


@dataclass(frozen=True)
class AdamWStatefulOptimizerConfig:
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    finite_check: bool = True
    set_to_none: bool = True

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["betas"] = [float(self.betas[0]), float(self.betas[1])]
        return payload


@dataclass(frozen=True)
class OptimizerStateSummary:
    parameter_tensors: int
    parameter_count: int
    state_tensors: int
    state_bytes: int
    state_mb: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptimizerStepReport:
    optimizer: str
    step_index: int
    finite: bool
    skipped: bool
    grad_norm_before_clip: float
    grad_norm_after_clip: float
    clipped: bool
    reason: str
    state: OptimizerStateSummary
    native_kernel_present: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.as_dict()
        return payload


def _as_params(params: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    return [param for param in params if isinstance(param, torch.nn.Parameter)]


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _grad_is_finite(params: Iterable[torch.nn.Parameter]) -> bool:
    for param in params:
        grad = param.grad
        if grad is not None and not bool(torch.isfinite(grad.detach()).all().item()):
            return False
    return True


def _grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    norms: list[torch.Tensor] = []
    for param in params:
        grad = param.grad
        if grad is not None:
            norms.append(grad.detach().float().norm(2))
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), ord=2).detach().cpu().item())


def summarize_optimizer_state(
    params: Iterable[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
) -> OptimizerStateSummary:
    param_list = _as_params(params)
    state_tensors = 0
    state_bytes = 0
    seen: set[int] = set()
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for value in state.values():
            if isinstance(value, torch.Tensor) and id(value) not in seen:
                seen.add(id(value))
                state_tensors += 1
                state_bytes += _tensor_bytes(value)
    return OptimizerStateSummary(
        parameter_tensors=len(param_list),
        parameter_count=sum(int(param.numel()) for param in param_list),
        state_tensors=state_tensors,
        state_bytes=state_bytes,
        state_mb=round(state_bytes / 1024.0 / 1024.0, 3),
    )


class PyTorchStatefulAdamWBackend:
    """Reference stateful AdamW backend matching the proposed native lifecycle."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        config: AdamWStatefulOptimizerConfig | None = None,
        *,
        backend_name: str = "pytorch_stateful_adamw",
    ) -> None:
        self.params = _as_params(params)
        self.config = config or AdamWStatefulOptimizerConfig()
        self.backend_name = backend_name
        self.step_index = 0
        self.optimizer = torch.optim.AdamW(
            self.params,
            lr=float(self.config.lr),
            betas=(float(self.config.betas[0]), float(self.config.betas[1])),
            eps=float(self.config.eps),
            weight_decay=float(self.config.weight_decay),
        )

    def step(self) -> OptimizerStepReport:
        finite = _grad_is_finite(self.params)
        grad_norm_before = _grad_norm(self.params)
        if self.config.finite_check and not finite:
            return OptimizerStepReport(
                optimizer=self.backend_name,
                step_index=self.step_index,
                finite=False,
                skipped=True,
                grad_norm_before_clip=grad_norm_before,
                grad_norm_after_clip=grad_norm_before,
                clipped=False,
                reason="non_finite_gradient",
                state=summarize_optimizer_state(self.params, self.optimizer),
            )

        clipped = False
        if float(self.config.max_grad_norm or 0.0) > 0:
            torch.nn.utils.clip_grad_norm_(self.params, float(self.config.max_grad_norm))
            clipped = bool(grad_norm_before > float(self.config.max_grad_norm))
        grad_norm_after = _grad_norm(self.params)
        self.optimizer.step()
        self.step_index += 1
        return OptimizerStepReport(
            optimizer=self.backend_name,
            step_index=self.step_index,
            finite=finite,
            skipped=False,
            grad_norm_before_clip=grad_norm_before,
            grad_norm_after_clip=grad_norm_after,
            clipped=clipped,
            reason="stepped",
            state=summarize_optimizer_state(self.params, self.optimizer),
        )

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=bool(self.config.set_to_none))

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "backend": self.backend_name,
            "step_index": int(self.step_index),
            "config": self.config.as_dict(),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(dict(state_dict.get("optimizer") or {}))
        self.step_index = int(state_dict.get("step_index", self.step_index) or 0)

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "backend": self.backend_name,
            "optimizer": "AdamW",
            "step_index": int(self.step_index),
            "training_path_enabled": False,
            "native_kernel_present": False,
            "config": self.config.as_dict(),
            "state": summarize_optimizer_state(self.params, self.optimizer).as_dict(),
        }


def build_native_optimizer_stateful_capability_stub() -> dict[str, Any]:
    flat_owner = build_flat_adamw_owner_capability_stub()
    return {
        "available": False,
        "status": "python_abi_prototype",
        "reason": "native_stateful_optimizer_not_implemented",
        "entrypoints": list(NATIVE_OPTIMIZER_STATEFUL_ENTRYPOINTS),
        "stateful": True,
        "supported_optimizers": ["AdamW"],
        "flat_owner": flat_owner,
        "training_path_enabled": False,
    }


def build_flat_adamw_owner_capability_stub() -> dict[str, Any]:
    """Return the expected native schema for persistent flat AdamW ownership."""

    return {
        "available": False,
        "status": "python_abi_prototype",
        "reason": "native_flat_adamw_owner_not_implemented",
        "entrypoints": list(NATIVE_FLAT_ADAMW_OWNER_ENTRYPOINTS),
        "optimizer": "AdamW",
        "stateful": True,
        "layout": "flat_contiguous_fp32_buffers",
        "required_buffers": ["param_flat", "grad_flat", "exp_avg", "exp_avg_sq"],
        "owns_parameter_buffer": True,
        "owns_gradient_buffer": True,
        "supports_direct_gradient_write": False,
        "supports_external_tensor_handles": False,
        "supports_stream_descriptor": True,
        "descriptor_schema": "flat_adamw_owner_descriptor_v1",
        "descriptor_required_fields": ["layout", "buffers", "stream"],
        "binding_request": {
            "available": False,
            "status": "python_abi_prototype",
            "reason": "native_tensor_binding_not_implemented",
            "entrypoints": [
                "validate_flat_adamw_tensor_binding_request",
                "probe_flat_adamw_tensor_object_binding",
                "create_flat_adamw_tensor_binding_session",
                "tensor_binding_session_snapshot",
                "tensor_binding_session_validate",
                "tensor_binding_session_launch_plan",
                "tensor_binding_session_stream_guard_probe",
                "tensor_binding_session_noop_launch",
                "tensor_binding_session_cpu_reference_guard",
                "tensor_binding_session_cuda_stub_launch",
                "tensor_binding_session_cuda_adamw_tensor_probe",
                "tensor_binding_session_cuda_adamw_runtime_probe",
                "get_adamw_cuda_kernel_contract",
                "destroy_tensor_binding_session",
            ],
            "binding_request_schema": "turbocore_native_tensor_binding_request_v1",
            "supports_external_tensor_handles": False,
            "supports_tensor_object_sessions": True,
            "supports_launch_plan": True,
            "kernel_registry": {
                "available": False,
                "status": "dry_run_registry",
                "reason": "cuda_adamw_kernel_not_registered",
                "entrypoints": [
                    "tensor_binding_session_noop_launch",
                    "tensor_binding_session_cpu_reference_guard",
                    "tensor_binding_session_cuda_stub_launch",
                ],
                "supported_plans": ["adamw_flat_fp32_launch_plan_v0"],
                "kernel_contract": {
                    "contract": "adamw_flat_fp32_cuda_kernel_v0",
                    "available": False,
                    "native_kernel_present": False,
                    "training_path_enabled": False,
                    "launch_plan": "adamw_flat_fp32_launch_plan_v0",
                },
                "dry_run_launch_supported": True,
                "cpu_reference_guard_supported": True,
                "cuda_stub_launch_supported": True,
                "native_kernel_present": False,
                "training_path_enabled": False,
            },
            "pointer_export_supported": False,
            "native_kernel_present": False,
            "training_path_enabled": False,
        },
        "native_kernel_present": False,
        "reference_owner": "core.turbocore_flat_adamw_state.PersistentFlatAdamW",
        "training_path_enabled": False,
        "notes": [
            "developer_only_contract",
            "no_training_dispatch",
            "persistent_flat_buffers_required",
        ],
    }


__all__ = [
    "AdamWStatefulOptimizerConfig",
    "NATIVE_FLAT_ADAMW_OWNER_ENTRYPOINTS",
    "NATIVE_OPTIMIZER_STATEFUL_ENTRYPOINTS",
    "OptimizerStateSummary",
    "OptimizerStepReport",
    "PyTorchStatefulAdamWBackend",
    "build_flat_adamw_owner_capability_stub",
    "build_native_optimizer_stateful_capability_stub",
    "summarize_optimizer_state",
]
