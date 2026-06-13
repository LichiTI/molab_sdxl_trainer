"""Default-off SCIONLight Triton TrainingLoop executor for selected plugin canaries."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from core.turbocore_triton_optimizer import triton_adamw_flat_available, triton_adamw_flat_unavailable_reason


try:  # pragma: no cover - host-specific import availability
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ScionLightTrainingExecutorConfig:
    optimizer_kind: str = "scionlight"
    lr: float = 1.0e-3
    momentum: float = 0.1
    constraint: bool = False
    norm_type: int = 0
    scale: float = 1.0
    weight_decay: float = 0.0
    weight_decouple: bool = True
    foreach: bool = False
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _scionlight_first_step_none_kernel(
        param_ptr,
        grad_ptr,
        n_elements,
        lr,
        momentum,
        scale,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        param_new = param - lr * scale * grad
        grad_new = grad * (1.0 - momentum)

        tl.store(param_ptr + offsets, param_new, mask=mask)
        tl.store(grad_ptr + offsets, grad_new, mask=mask)


class ScionLightTrainingExecutor:
    """Launch a real SCIONLight norm_type=NONE first-step fp32 Triton kernel."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: ScionLightTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("ScionLightTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("scionlight_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("scionlight_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("scionlight_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"scionlight_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("scionlight_maximize_not_supported")
        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config)
            blockers = _unsupported_group_blockers(group_cfg)
            if blockers:
                cases.append(_case_failed(blockers[0]))
                continue
            group["step"] = int(group.get("step", 0) or 0) + 1  # type: ignore[index]
            for param in group["params"]:
                if param.grad is None:
                    continue
                cases.append(self._step_param(param, group, group_cfg))
        blockers = _dedupe(reason for case in cases for reason in case.get("blocked_reasons", []) or [])
        if not cases:
            blockers.append("scionlight_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_scionlight_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "scionlight_native_step_failed"),
            "optimizer_kind": self.config.optimizer_kind,
            "training_dispatch": True,
            "training_path_enabled": True,
            "native_step_executed": ok,
            "native_kernel_launched": any(case.get("kernel_executed") is True for case in cases),
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

    def _step_param(
        self,
        param: torch.nn.Parameter,
        group: Mapping[str, Any],
        group_cfg: Mapping[str, Any],
    ) -> dict[str, Any]:
        if param.dtype != torch.float32 or (param.grad is not None and param.grad.dtype != torch.float32):
            return _case_failed("scionlight_training_executor_requires_float32")
        if not param.is_contiguous() or param.grad is None or not param.grad.is_contiguous():
            return _case_failed("scionlight_training_executor_requires_contiguous_tensors")
        step = int(group.get("step", 0) or 0)
        if step != 1:
            return _case_failed("scionlight_training_executor_only_first_step_supported_for_canary")
        before = param.detach().clone()
        grid = (triton.cdiv(int(param.numel()), int(group_cfg["block_size"])),)
        _scionlight_first_step_none_kernel[grid](
            param,
            param.grad.detach(),
            int(param.numel()),
            float(group_cfg["lr"]),
            float(group_cfg["momentum"]),
            float(group_cfg["scale"]),
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        mutated = _max_abs_diff(before, param.detach()) > 0.0
        return {
            "schema_version": 1,
            "ok": mutated and torch.isfinite(param).all().item(),
            "param_shape": [int(dim) for dim in param.shape],
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_after": step,
            "kernel_executed": True,
            "training_parameters_mutated": bool(mutated),
            "state_keys": sorted(str(key) for key in self.optimizer.state[param].keys()),
            "blocked_reasons": [] if mutated else ["scionlight_training_executor_parameters_not_mutated"],
        }


def build_plugin_scionlight_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: ScionLightTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> ScionLightTrainingExecutor:
    return ScionLightTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: ScionLightTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> ScionLightTrainingExecutorConfig:
    if isinstance(value, ScionLightTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    return ScionLightTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "scionlight"),
        lr=float(payload.get("lr", group.get("lr", 1.0e-3)) or 1.0e-3),
        momentum=float(payload.get("momentum", group.get("momentum", 0.1)) or 0.1),
        constraint=bool(payload.get("constraint", group.get("constraint", False))),
        norm_type=int(payload.get("norm_type", group.get("norm_type", 0)) or 0),
        scale=float(payload.get("scale", group.get("scale", 1.0)) or 1.0),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0)) or 0.0),
        weight_decouple=bool(payload.get("weight_decouple", group.get("weight_decouple", True))),
        foreach=bool(payload.get("foreach", group.get("foreach", False))),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: ScionLightTrainingExecutorConfig) -> dict[str, Any]:
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "momentum": float(group.get("momentum", config.momentum) or config.momentum),
        "constraint": bool(group.get("constraint", config.constraint)),
        "norm_type": int(group.get("norm_type", config.norm_type) or config.norm_type),
        "norm_kwargs": group.get("norm_kwargs", {}),
        "scale": float(group.get("scale", config.scale) or config.scale),
        "weight_decay": float(group.get("weight_decay", config.weight_decay) or config.weight_decay),
        "weight_decouple": bool(group.get("weight_decouple", config.weight_decouple)),
        "foreach": bool(group.get("foreach", config.foreach)),
        "block_size": int(config.block_size),
    }


def _unsupported_group_blockers(group_cfg: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if bool(group_cfg.get("constraint", False)):
        blockers.append("scionlight_constraint_not_supported_for_canary")
    if int(group_cfg.get("norm_type", 0) or 0) != 0:
        blockers.append("scionlight_only_norm_type_none_supported_for_canary")
    if dict(group_cfg.get("norm_kwargs") or {}):
        blockers.append("scionlight_norm_kwargs_not_supported_for_canary")
    if abs(float(group_cfg.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("scionlight_weight_decay_not_supported_for_canary")
    if float(group_cfg.get("scale", 0.0) or 0.0) <= 0.0:
        blockers.append("scionlight_scale_must_be_positive")
    momentum = float(group_cfg.get("momentum", 0.0) or 0.0)
    if momentum <= 0.0 or momentum > 1.0:
        blockers.append("scionlight_momentum_out_of_range")
    return blockers


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().detach().cpu().item())


def _case_failed(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "kernel_executed": False,
        "training_parameters_mutated": False,
        "blocked_reasons": [reason],
    }


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "executor": "turbocore_plugin_scionlight_training_executor_v0",
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


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "ScionLightTrainingExecutor",
    "ScionLightTrainingExecutorConfig",
    "build_plugin_scionlight_training_executor",
]
