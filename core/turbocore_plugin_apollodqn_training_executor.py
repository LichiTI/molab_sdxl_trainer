"""Default-off ApolloDQN Triton TrainingLoop executor for selected plugin canaries."""

from __future__ import annotations

import math
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
class ApolloDqnTrainingExecutorConfig:
    optimizer_kind: str = "apollodqn"
    lr: float = 1.0e-2
    init_lr: float = 1.0e-5
    beta: float = 0.9
    rebound: str = "constant"
    weight_decay: float = 0.0
    weight_decay_type: str = "l2"
    warmup_steps: int = 500
    eps: float = 1.0e-4
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _apollodqn_first_step_constant_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_grad_ptr,
        approx_hessian_ptr,
        update_ptr,
        n_elements,
        current_lr,
        alpha,
        eps_scaled,
        rebound,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        exp_avg_grad = tl.load(exp_avg_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        approx_hessian = tl.load(approx_hessian_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        delta_grad = grad - exp_avg_grad
        exp_avg_grad_new = exp_avg_grad + delta_grad * alpha

        update_norm4 = tl.sqrt(tl.sqrt(tl.sum(tl.where(mask, update * update * update * update, 0.0), axis=0)))
        de_nom0 = update_norm4 + eps_scaled
        normalized_update = update / de_nom0
        v_sq = normalized_update * normalized_update
        delta = tl.sum(tl.where(mask, (delta_grad / de_nom0) * normalized_update, 0.0), axis=0) * (-alpha)
        delta -= tl.sum(tl.where(mask, approx_hessian * v_sq, 0.0), axis=0)

        approx_hessian_new = approx_hessian + v_sq * delta
        denom = tl.maximum(tl.abs(approx_hessian_new), rebound)
        update_new = exp_avg_grad_new / denom
        param_new = param - current_lr * update_new

        tl.store(param_ptr + offsets, param_new, mask=mask)
        tl.store(exp_avg_grad_ptr + offsets, exp_avg_grad_new, mask=mask)
        tl.store(approx_hessian_ptr + offsets, approx_hessian_new, mask=mask)
        tl.store(update_ptr + offsets, update_new, mask=mask)


class ApolloDqnTrainingExecutor:
    """Launch a real ApolloDQN first-step constant-rebound fp32 Triton kernel."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: ApolloDqnTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("ApolloDqnTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("apollodqn_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("apollodqn_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("apollodqn_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"apollodqn_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("apollodqn_maximize_not_supported")
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
            blockers.append("apollodqn_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_apollodqn_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "apollodqn_native_step_failed"),
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
            return _case_failed("apollodqn_training_executor_requires_float32")
        if not param.is_contiguous() or param.grad is None or not param.grad.is_contiguous():
            return _case_failed("apollodqn_training_executor_requires_contiguous_tensors")
        step = int(group.get("step", 0) or 0)
        if step != 1:
            return _case_failed("apollodqn_training_executor_only_first_step_supported_for_canary")
        state = self.optimizer.state[param]
        exp_avg_grad = _state_tensor(state, "exp_avg_grad", param)
        approx_hessian = _state_tensor(state, "approx_hessian", param)
        update = _state_tensor(state, "update", param)
        before = param.detach().clone()
        beta = float(group_cfg["beta"])
        alpha = (1.0 - beta) / (1.0 - math.pow(beta, step))
        current_lr = _current_lr(group_cfg, step)
        grid = (triton.cdiv(int(param.numel()), int(group_cfg["block_size"])),)
        _apollodqn_first_step_constant_kernel[grid](
            param,
            param.grad.detach(),
            exp_avg_grad,
            approx_hessian,
            update,
            int(param.numel()),
            float(current_lr),
            float(alpha),
            float(group_cfg["eps"]) / 1.0e-2,
            1.0e-2,
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        state["exp_avg_grad"] = exp_avg_grad
        state["approx_hessian"] = approx_hessian
        state["update"] = update
        mutated = _max_abs_diff(before, param.detach()) > 0.0
        return {
            "schema_version": 1,
            "ok": mutated and torch.isfinite(param).all().item(),
            "param_shape": [int(dim) for dim in param.shape],
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_after": step,
            "kernel_executed": True,
            "training_parameters_mutated": bool(mutated),
            "state_keys": sorted(str(key) for key in state.keys()),
            "blocked_reasons": [] if mutated else ["apollodqn_training_executor_parameters_not_mutated"],
        }


def build_plugin_apollodqn_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: ApolloDqnTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> ApolloDqnTrainingExecutor:
    return ApolloDqnTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: ApolloDqnTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> ApolloDqnTrainingExecutorConfig:
    if isinstance(value, ApolloDqnTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    return ApolloDqnTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "apollodqn"),
        lr=float(payload.get("lr", group.get("lr", 1.0e-2)) or 1.0e-2),
        init_lr=float(payload.get("init_lr", group.get("init_lr", 1.0e-5)) or 1.0e-5),
        beta=float(payload.get("beta", group.get("beta", 0.9)) or 0.9),
        rebound=str(payload.get("rebound", group.get("rebound", "constant")) or "constant"),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0)) or 0.0),
        weight_decay_type=str(payload.get("weight_decay_type", group.get("weight_decay_type", "l2")) or "l2"),
        warmup_steps=int(payload.get("warmup_steps", group.get("warmup_steps", 500)) or 500),
        eps=float(payload.get("eps", group.get("eps", 1.0e-4)) or 1.0e-4),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: ApolloDqnTrainingExecutorConfig) -> dict[str, Any]:
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "init_lr": float(group.get("init_lr", config.init_lr) or config.init_lr),
        "beta": float(group.get("beta", config.beta) or config.beta),
        "rebound": str(group.get("rebound", config.rebound) or config.rebound),
        "weight_decay": float(group.get("weight_decay", config.weight_decay) or config.weight_decay),
        "weight_decay_type": str(group.get("weight_decay_type", config.weight_decay_type) or config.weight_decay_type),
        "warmup_steps": int(group.get("warmup_steps", config.warmup_steps) or config.warmup_steps),
        "eps": float(group.get("eps", config.eps) or config.eps),
        "block_size": int(config.block_size),
    }


def _unsupported_group_blockers(group_cfg: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if str(group_cfg.get("rebound", "constant")) != "constant":
        blockers.append("apollodqn_rebound_belief_not_supported_for_canary")
    if abs(float(group_cfg.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("apollodqn_weight_decay_not_supported_for_canary")
    if str(group_cfg.get("weight_decay_type", "l2")) != "l2":
        blockers.append("apollodqn_non_l2_weight_decay_type_not_supported_for_canary")
    if int(group_cfg.get("warmup_steps", 0) or 0) <= 0:
        blockers.append("apollodqn_warmup_steps_must_be_positive")
    if float(group_cfg.get("eps", 0.0) or 0.0) <= 0.0:
        blockers.append("apollodqn_eps_must_be_positive")
    return blockers


def _current_lr(group_cfg: Mapping[str, Any], step: int) -> float:
    lr = float(group_cfg["lr"])
    init_lr = float(group_cfg["init_lr"])
    warmup_steps = int(group_cfg["warmup_steps"])
    if step >= warmup_steps:
        return lr
    return (lr - init_lr) * step / warmup_steps + init_lr


def _state_tensor(state: dict[str, Any], key: str, param: torch.nn.Parameter) -> torch.Tensor:
    existing = state.get(key)
    if torch.is_tensor(existing) and tuple(existing.shape) == tuple(param.shape):
        if existing.device == param.device and existing.dtype == torch.float32 and existing.is_contiguous():
            return existing
        converted = existing.detach().to(device=param.device, dtype=torch.float32).contiguous()
        state[key] = converted
        return converted
    state[key] = torch.zeros_like(param.detach(), dtype=torch.float32).contiguous()
    return state[key]


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
        "executor": "turbocore_plugin_apollodqn_training_executor_v0",
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
    "ApolloDqnTrainingExecutor",
    "ApolloDqnTrainingExecutorConfig",
    "build_plugin_apollodqn_training_executor",
]
