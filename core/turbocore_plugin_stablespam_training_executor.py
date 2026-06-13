"""Default-off StableSPAM Triton TrainingLoop executor for selected plugin canaries."""

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
class StableSPAMTrainingExecutorConfig:
    optimizer_kind: str = "stablespam"
    lr: float = 1.0e-3
    beta1: float = 0.9
    beta2: float = 0.999
    gamma1: float = 0.7
    gamma2: float = 0.9
    theta: float = 0.999
    t_max: int | None = None
    weight_decay: float = 0.0
    update_proj_gap: int = 1000
    eps: float = 1.0e-8
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _stablespam_first_step_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        grad_norm,
        c_norm,
        bias_correction2_sqrt,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        scaled_grad = grad / grad_norm * c_norm
        exp_avg = scaled_grad * (1.0 - beta1)
        exp_avg_sq = scaled_grad * scaled_grad * (1.0 - beta2)
        denom = tl.sqrt(exp_avg_sq) / bias_correction2_sqrt + eps
        param_new = param - (lr / (1.0 - beta1)) * exp_avg / denom

        tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
        tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)
        tl.store(param_ptr + offsets, param_new, mask=mask)


class StableSPAMTrainingExecutor:
    """Launch a real StableSPAM no-warmup first-step fp32 Triton kernel."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: StableSPAMTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("StableSPAMTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("stablespam_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("stablespam_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("stablespam_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"stablespam_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("stablespam_maximize_not_supported")
        if self.config.t_max is not None:
            return _blocked("stablespam_warmup_not_supported_for_canary")
        if int(getattr(self.optimizer, "total_step", 0) or 0) != 0:
            return _blocked("stablespam_training_executor_only_cold_start_supported")
        started = time.perf_counter()
        self.optimizer.total_step = int(getattr(self.optimizer, "total_step", 0) or 0) + 1  # type: ignore[attr-defined]
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
            blockers.append("stablespam_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_stablespam_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "stablespam_native_step_failed"),
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
            return _case_failed("stablespam_training_executor_requires_float32")
        if not param.is_contiguous() or param.grad is None or not param.grad.is_contiguous():
            return _case_failed("stablespam_training_executor_requires_contiguous_tensors")
        step = int(group.get("step", 0) or 0)
        if step != 1:
            return _case_failed("stablespam_training_executor_only_first_step_supported_for_canary")
        state = self.optimizer.state[param]
        exp_avg = _state_tensor(state, "exp_avg", param)
        exp_avg_sq = _state_tensor(state, "exp_avg_sq", param)
        m_norm_t = _state_scalar(state, "m_norm_t", param)
        v_norm_t = _state_scalar(state, "v_norm_t", param)
        m_max_t = _state_scalar(state, "m_max_t", param)
        if exp_avg is None or exp_avg_sq is None or m_norm_t is None or v_norm_t is None or m_max_t is None:
            return _case_failed("stablespam_training_executor_requires_fp32_state")
        if not exp_avg.is_contiguous() or not exp_avg_sq.is_contiguous():
            return _case_failed("stablespam_training_executor_requires_contiguous_state")

        grad = param.grad.detach()
        max_grad = torch.max(grad.abs())
        if float(max_grad.detach().cpu().item()) <= 0.0:
            return _case_failed("stablespam_training_executor_zero_grad_not_supported")
        theta = float(group_cfg["theta"])
        gamma1 = float(group_cfg["gamma1"])
        gamma2 = float(group_cfg["gamma2"])
        m_max_t.lerp_(max_grad.reshape(1), weight=1.0 - theta)
        m_max_hat = m_max_t / (1.0 - theta**step)
        if bool((grad.abs() > m_max_hat).any().detach().cpu().item()):
            return _case_failed("stablespam_training_executor_clip_path_not_supported")

        grad_norm = torch.linalg.norm(grad)
        grad_norm_value = float(grad_norm.detach().cpu().item())
        if grad_norm_value <= 0.0:
            return _case_failed("stablespam_training_executor_zero_norm_not_supported")
        m_norm_t.lerp_(grad_norm.reshape(1), weight=1.0 - gamma1)
        v_norm_t.lerp_(grad_norm.pow(2).reshape(1), weight=1.0 - gamma2)
        m_norm_hat = m_norm_t / (1.0 - gamma1**step)
        v_norm_hat = v_norm_t / (1.0 - gamma2**step)
        c_norm = m_norm_hat / (v_norm_hat.sqrt() + float(group_cfg["eps"]))

        before = param.detach().clone()
        beta1 = float(group_cfg["beta1"])
        beta2 = float(group_cfg["beta2"])
        bias_correction2_sqrt = math.sqrt(1.0 - beta2**step)
        grid = (triton.cdiv(int(param.numel()), int(group_cfg["block_size"])),)
        _stablespam_first_step_kernel[grid](
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            int(param.numel()),
            float(group_cfg["lr"]),
            beta1,
            beta2,
            float(group_cfg["eps"]),
            grad_norm_value,
            float(c_norm.detach().cpu().item()),
            bias_correction2_sqrt,
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        mutated = _max_abs_diff(before, param.detach()) > 0.0
        return {
            "schema_version": 1,
            "ok": mutated
            and torch.isfinite(param).all().item()
            and torch.isfinite(exp_avg).all().item()
            and torch.isfinite(exp_avg_sq).all().item(),
            "param_shape": [int(dim) for dim in param.shape],
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_after": step,
            "total_step_after": int(getattr(self.optimizer, "total_step", 0) or 0),
            "kernel_executed": True,
            "training_parameters_mutated": bool(mutated),
            "state_keys": sorted(str(key) for key in state.keys()),
            "blocked_reasons": [] if mutated else ["stablespam_training_executor_parameters_not_mutated"],
        }


def build_plugin_stablespam_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: StableSPAMTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> StableSPAMTrainingExecutor:
    return StableSPAMTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: StableSPAMTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> StableSPAMTrainingExecutorConfig:
    if isinstance(value, StableSPAMTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = payload.get("betas", group.get("betas", (0.9, 0.999)))
    if not isinstance(betas, (tuple, list)):
        betas = (0.9, 0.999)
    t_max = payload.get("t_max", getattr(optimizer, "t_max", None))
    return StableSPAMTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "stablespam"),
        lr=float(payload.get("lr", group.get("lr", 1.0e-3)) or 1.0e-3),
        beta1=float(betas[0] if len(betas) > 0 else 0.9),
        beta2=float(betas[1] if len(betas) > 1 else 0.999),
        gamma1=float(payload.get("gamma1", getattr(optimizer, "gamma1", 0.7))),
        gamma2=float(payload.get("gamma2", getattr(optimizer, "gamma2", 0.9))),
        theta=float(payload.get("theta", getattr(optimizer, "theta", 0.999))),
        t_max=None if t_max is None else int(t_max),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0)) or 0.0),
        update_proj_gap=int(payload.get("update_proj_gap", getattr(optimizer, "update_proj_gap", 1000)) or 1000),
        eps=float(payload.get("eps", group.get("eps", 1.0e-8)) or 1.0e-8),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: StableSPAMTrainingExecutorConfig) -> dict[str, Any]:
    betas = group.get("betas", (config.beta1, config.beta2))
    if not isinstance(betas, (tuple, list)):
        betas = (config.beta1, config.beta2)
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "beta1": float(betas[0] if len(betas) > 0 else config.beta1),
        "beta2": float(betas[1] if len(betas) > 1 else config.beta2),
        "gamma1": float(config.gamma1),
        "gamma2": float(config.gamma2),
        "theta": float(config.theta),
        "t_max": config.t_max,
        "weight_decay": float(group.get("weight_decay", config.weight_decay) or config.weight_decay),
        "update_proj_gap": int(config.update_proj_gap),
        "eps": float(group.get("eps", config.eps) or config.eps),
        "block_size": int(config.block_size),
    }


def _unsupported_group_blockers(group_cfg: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if abs(float(group_cfg.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("stablespam_weight_decay_not_supported_for_canary")
    if int(group_cfg.get("update_proj_gap", 0) or 0) <= 1:
        blockers.append("stablespam_projection_refresh_not_supported_for_canary")
    if float(group_cfg.get("lr", 0.0) or 0.0) <= 0.0:
        blockers.append("stablespam_lr_must_be_positive")
    beta1 = float(group_cfg.get("beta1", 0.0) or 0.0)
    beta2 = float(group_cfg.get("beta2", 0.0) or 0.0)
    if beta1 < 0.0 or beta1 >= 1.0 or beta2 < 0.0 or beta2 >= 1.0:
        blockers.append("stablespam_betas_out_of_range")
    return blockers


def _state_tensor(state: dict[str, Any], key: str, param: torch.nn.Parameter) -> torch.Tensor | None:
    value = state.get(key)
    if value is None:
        value = torch.zeros_like(param)
        state[key] = value
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape) or value.dtype != torch.float32:
        return None
    return value


def _state_scalar(state: dict[str, Any], key: str, param: torch.nn.Parameter) -> torch.Tensor | None:
    value = state.get(key)
    if value is None:
        value = torch.zeros(1, device=param.device, dtype=torch.float32)
        state[key] = value
    if not torch.is_tensor(value) or tuple(value.shape) != (1,) or value.dtype != torch.float32:
        return None
    return value


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
        "executor": "turbocore_plugin_stablespam_training_executor_v0",
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
    "StableSPAMTrainingExecutor",
    "StableSPAMTrainingExecutorConfig",
    "build_plugin_stablespam_training_executor",
]
