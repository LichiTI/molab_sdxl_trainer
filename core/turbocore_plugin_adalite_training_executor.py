"""Default-off Adalite Triton TrainingLoop executor for selected plugin canaries."""

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
class AdaliteTrainingExecutorConfig:
    optimizer_kind: str = "adalite"
    lr: float = 1.0e-3
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    weight_decouple: bool = False
    fixed_decay: bool = False
    g_norm_min: float = 1.0e-10
    ratio_min: float = 1.0e-4
    tau: float = 1.0
    eps1: float = 1.0e-6
    eps2: float = 1.0e-10
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _adalite_first_step_1d_kernel(
        param_ptr,
        grad_ptr,
        m_avg_ptr,
        v_avg_ptr,
        n_elements,
        lr,
        beta1,
        beta2,
        g_norm_min,
        ratio_min,
        eps1,
        bias_correction2,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m_avg = tl.load(m_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v_avg = tl.load(v_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        param_norm = tl.sqrt(tl.sum(tl.where(mask, param * param, 0.0), axis=0))
        grad_norm = tl.sqrt(tl.sum(tl.where(mask, grad * grad, 0.0), axis=0))
        trust_ratio = tl.maximum(param_norm / tl.maximum(grad_norm, g_norm_min), ratio_min)
        scaled_grad = grad * trust_ratio

        m_new = m_avg * beta1 + scaled_grad * (1.0 - beta1)
        residual = scaled_grad - m_new
        v_new = v_avg * beta2 + residual * residual * (1.0 - beta2)
        denom = tl.sqrt(v_new / bias_correction2 + eps1)
        update = m_new + (scaled_grad - m_new) * (1.0 - beta1)
        param_new = param - lr * (update / denom)

        tl.store(param_ptr + offsets, param_new, mask=mask)
        tl.store(grad_ptr + offsets, scaled_grad, mask=mask)
        tl.store(m_avg_ptr + offsets, m_new, mask=mask)
        tl.store(v_avg_ptr + offsets, v_new, mask=mask)


class AdaliteTrainingExecutor:
    """Launch a real Adalite first-step 1D fp32 Triton kernel against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: AdaliteTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("AdaliteTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("adalite_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("adalite_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("adalite_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"adalite_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("adalite_maximize_not_supported")
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
            blockers.append("adalite_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_adalite_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "adalite_native_step_failed"),
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
        if param.ndim != 1 or int(param.numel()) <= 1:
            return _case_failed("adalite_training_executor_requires_1d_dense_param")
        if int(param.numel()) > int(group_cfg["block_size"]):
            return _case_failed("adalite_training_executor_requires_single_block_canary")
        if param.dtype != torch.float32 or (param.grad is not None and param.grad.dtype != torch.float32):
            return _case_failed("adalite_training_executor_requires_float32")
        if not param.is_contiguous() or param.grad is None or not param.grad.is_contiguous():
            return _case_failed("adalite_training_executor_requires_contiguous_tensors")
        step = int(group.get("step", 0) or 0)
        if step != 1:
            return _case_failed("adalite_training_executor_only_first_step_supported_for_canary")
        state = self.optimizer.state[param]
        m_avg = _state_tensor(state, "m_avg", param)
        v_avg = _state_tensor(state, "v_avg", param)
        before = param.detach().clone()
        _adalite_first_step_1d_kernel[(1,)](
            param,
            param.grad.detach(),
            m_avg,
            v_avg,
            int(param.numel()),
            float(group_cfg["lr"]),
            float(group_cfg["beta1"]),
            float(group_cfg["beta2"]),
            float(group_cfg["g_norm_min"]),
            float(group_cfg["ratio_min"]),
            float(group_cfg["eps1"]),
            1.0 - math.pow(float(group_cfg["beta2"]), step),
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        state["m_avg"] = m_avg
        state["v_avg"] = v_avg
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
            "blocked_reasons": [] if mutated else ["adalite_training_executor_parameters_not_mutated"],
        }


def build_plugin_adalite_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: AdaliteTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> AdaliteTrainingExecutor:
    return AdaliteTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: AdaliteTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> AdaliteTrainingExecutorConfig:
    if isinstance(value, AdaliteTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = payload.get("betas", group.get("betas", (0.9, 0.999)))
    beta1, beta2 = _two_betas(betas)
    return AdaliteTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "adalite"),
        lr=float(payload.get("lr", group.get("lr", 1.0e-3)) or 1.0e-3),
        beta1=beta1,
        beta2=beta2,
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0)) or 0.0),
        weight_decouple=bool(payload.get("weight_decouple", group.get("weight_decouple", False))),
        fixed_decay=bool(payload.get("fixed_decay", group.get("fixed_decay", False))),
        g_norm_min=float(payload.get("g_norm_min", group.get("g_norm_min", 1.0e-10)) or 1.0e-10),
        ratio_min=float(payload.get("ratio_min", group.get("ratio_min", 1.0e-4)) or 1.0e-4),
        tau=float(payload.get("tau", group.get("tau", 1.0)) or 1.0),
        eps1=float(payload.get("eps1", group.get("eps1", 1.0e-6)) or 1.0e-6),
        eps2=float(payload.get("eps2", group.get("eps2", 1.0e-10)) or 1.0e-10),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: AdaliteTrainingExecutorConfig) -> dict[str, Any]:
    beta1, beta2 = _two_betas(group.get("betas", (config.beta1, config.beta2)))
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": float(group.get("weight_decay", config.weight_decay) or config.weight_decay),
        "weight_decouple": bool(group.get("weight_decouple", config.weight_decouple)),
        "fixed_decay": bool(group.get("fixed_decay", config.fixed_decay)),
        "g_norm_min": float(group.get("g_norm_min", config.g_norm_min) or config.g_norm_min),
        "ratio_min": float(group.get("ratio_min", config.ratio_min) or config.ratio_min),
        "tau": float(group.get("tau", config.tau) or config.tau),
        "eps1": float(group.get("eps1", config.eps1) or config.eps1),
        "eps2": float(group.get("eps2", config.eps2) or config.eps2),
        "block_size": int(config.block_size),
    }


def _two_betas(value: Any) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)):
        value = (0.9, 0.999)
    return (
        float(value[0] if len(value) > 0 else 0.9),
        float(value[1] if len(value) > 1 else 0.999),
    )


def _unsupported_group_blockers(group_cfg: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if abs(float(group_cfg.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("adalite_weight_decay_not_supported_for_canary")
    if float(group_cfg.get("g_norm_min", 0.0) or 0.0) <= 0.0:
        blockers.append("adalite_g_norm_min_must_be_positive")
    if float(group_cfg.get("ratio_min", 0.0) or 0.0) <= 0.0:
        blockers.append("adalite_ratio_min_must_be_positive")
    return blockers


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
        "executor": "turbocore_plugin_adalite_training_executor_v0",
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
    "AdaliteTrainingExecutor",
    "AdaliteTrainingExecutorConfig",
    "build_plugin_adalite_training_executor",
]
