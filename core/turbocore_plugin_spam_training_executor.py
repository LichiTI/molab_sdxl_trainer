"""Default-off SPAM Triton TrainingLoop executor for selected plugin canaries."""

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
class SPAMTrainingExecutorConfig:
    optimizer_kind: str = "spam"
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-6
    density: float = 1.0
    threshold: int = 0
    grad_accu_steps: int = 20
    update_proj_gap: int = 500
    warmup_epoch: int = 1
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _spam_dense_mask_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        n_elements,
        step_size,
        beta1,
        beta2,
        eps,
        scale_factor,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        m_new = m * beta1 + g * (1.0 - beta1)
        v_new = v * beta2 + g * g * (1.0 - beta2)
        update = m_new / (tl.sqrt(v_new) + eps)
        p_new = p - step_size * scale_factor * update

        tl.store(exp_avg_ptr + offsets, m_new, mask=mask)
        tl.store(exp_avg_sq_ptr + offsets, v_new, mask=mask)
        tl.store(param_ptr + offsets, p_new, mask=mask)


class SPAMTrainingExecutor:
    """Launch a real SPAM dense-mask Triton kernel against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: SPAMTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("SPAMTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("spam_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("spam_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("spam_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"spam_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("spam_maximize_not_supported")
        if abs(float(self.config.density) - 1.0) > 1.0e-12:
            return _blocked("spam_training_executor_requires_density_1")
        if int(self.config.threshold) != 0:
            return _blocked("spam_training_executor_threshold_mask_not_supported")
        if int(self.config.update_proj_gap) <= 1:
            return _blocked("spam_training_executor_projection_refresh_not_supported")
        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config)
            if float(group_cfg["weight_decay"]) != 0.0:
                cases.append(_case_failed("spam_weight_decay_not_supported_for_dense_mask_canary"))
                continue
            group["step"] = int(group.get("step", 0) or 0) + 1  # type: ignore[index]
            for param in group["params"]:
                if param.grad is None:
                    continue
                cases.append(self._step_param(param, group, group_cfg))
        blockers = _dedupe(reason for case in cases for reason in case.get("blocked_reasons", []) or [])
        if not cases:
            blockers.append("spam_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_spam_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "spam_native_step_failed"),
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
        if param.ndim != 2:
            return _case_failed("spam_training_executor_requires_2d_dense_param")
        if param.dtype != torch.float32 or (param.grad is not None and param.grad.dtype != torch.float32):
            return _case_failed("spam_training_executor_requires_float32")
        if not param.is_contiguous() or param.grad is None or not param.grad.is_contiguous():
            return _case_failed("spam_training_executor_requires_contiguous_tensors")
        state = self.optimizer.state[param]
        mask = _state_dense_mask(state, param)
        if mask is None:
            return _case_failed("spam_training_executor_requires_all_true_bool_mask")
        if _projection_reset_due(self.optimizer, self.config):
            return _case_failed("spam_projection_refresh_not_supported")
        exp_avg = _state_flat_tensor(state, "exp_avg", param)
        exp_avg_sq = _state_flat_tensor(state, "exp_avg_sq", param)
        before = param.detach().clone()
        step = int(group.get("step", 0) or 0)
        beta1 = float(group_cfg["beta1"])
        beta2 = float(group_cfg["beta2"])
        step_size = float(group_cfg["lr"]) * math.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)
        grid = (triton.cdiv(int(param.numel()), int(group_cfg["block_size"])),)
        _spam_dense_mask_kernel[grid](
            param,
            param.grad.detach(),
            exp_avg,
            exp_avg_sq,
            int(param.numel()),
            float(step_size),
            beta1,
            beta2,
            float(group_cfg["eps"]),
            _scale_factor(self.optimizer, self.config),
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        state["mask"] = mask
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq
        self.optimizer.state["total_step"] = int(self.optimizer.state.get("total_step", 0) or 0) + 1
        self.optimizer.state["current_step"] = int(self.optimizer.state.get("current_step", 0) or 0) + 1
        mutated = not torch.allclose(before, param.detach())
        return {
            "schema_version": 1,
            "ok": mutated and torch.isfinite(param).all().item(),
            "param_shape": [int(dim) for dim in param.shape],
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_after": step,
            "total_step_after": int(self.optimizer.state.get("total_step", 0) or 0),
            "current_step_after": int(self.optimizer.state.get("current_step", 0) or 0),
            "kernel_executed": True,
            "training_parameters_mutated": bool(mutated),
            "mask_density": 1.0,
            "state_keys": sorted(str(key) for key in state.keys()),
            "blocked_reasons": [] if mutated else ["spam_training_executor_parameters_not_mutated"],
        }


def build_plugin_spam_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: SPAMTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> SPAMTrainingExecutor:
    return SPAMTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: SPAMTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> SPAMTrainingExecutorConfig:
    if isinstance(value, SPAMTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = payload.get("betas", group.get("betas", (0.9, 0.999)))
    return SPAMTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "spam"),
        lr=float(payload.get("lr", group.get("lr", 1e-3)) or 1e-3),
        beta1=float(betas[0]),
        beta2=float(betas[1]),
        eps=float(payload.get("eps", group.get("eps", 1e-6)) or 1e-6),
        density=float(payload.get("density", getattr(optimizer, "density", 1.0)) or 1.0),
        threshold=int(payload.get("threshold", getattr(optimizer, "threshold", 0)) or 0),
        grad_accu_steps=int(payload.get("grad_accu_steps", getattr(optimizer, "grad_accu_steps", 20)) or 20),
        update_proj_gap=int(payload.get("update_proj_gap", getattr(optimizer, "update_proj_gap", 500)) or 500),
        warmup_epoch=int(payload.get("warmup_epoch", getattr(optimizer, "warmup_epoch", 1)) or 1),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: SPAMTrainingExecutorConfig) -> dict[str, Any]:
    betas = group.get("betas", (config.beta1, config.beta2))
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "eps": float(group.get("eps", config.eps) or config.eps),
        "weight_decay": float(group.get("weight_decay", 0.0) or 0.0),
        "block_size": int(config.block_size),
    }


def _state_dense_mask(state: dict[str, Any], param: torch.nn.Parameter) -> torch.Tensor | None:
    existing = state.get("mask")
    if existing is None:
        existing = torch.ones_like(param.detach(), dtype=torch.bool)
        state["mask"] = existing
    if not torch.is_tensor(existing) or existing.dtype != torch.bool or tuple(existing.shape) != tuple(param.shape):
        return None
    mask = existing.to(device=param.device, dtype=torch.bool).contiguous()
    if not bool(mask.all().detach().cpu().item()):
        return None
    state["mask"] = mask
    return mask


def _state_flat_tensor(state: dict[str, Any], key: str, param: torch.nn.Parameter) -> torch.Tensor:
    existing = state.get(key)
    shape = (int(param.numel()),)
    if torch.is_tensor(existing) and tuple(existing.shape) == shape:
        if existing.device == param.device and existing.dtype == torch.float32 and existing.is_contiguous():
            return existing
        converted = existing.detach().to(device=param.device, dtype=torch.float32).contiguous()
        state[key] = converted
        return converted
    state[key] = torch.zeros(shape, device=param.device, dtype=torch.float32)
    return state[key]


def _projection_reset_due(optimizer: torch.optim.Optimizer, config: SPAMTrainingExecutorConfig) -> bool:
    total_step = int(optimizer.state.get("total_step", 0) or 0)
    return total_step != 0 and (total_step + 1) % int(config.update_proj_gap) == 0


def _scale_factor(optimizer: torch.optim.Optimizer, config: SPAMTrainingExecutorConfig) -> float:
    current_step = int(optimizer.state.get("current_step", int(config.warmup_epoch) + 1) or 0)
    if current_step >= int(config.warmup_epoch):
        return 1.0
    return 1.0


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
        "executor": "turbocore_plugin_spam_training_executor_v0",
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


__all__ = ["SPAMTrainingExecutor", "SPAMTrainingExecutorConfig", "build_plugin_spam_training_executor"]
