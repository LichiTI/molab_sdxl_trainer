"""Default-off MARS Triton TrainingLoop executor for selected plugin canaries."""

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
class MarsTrainingExecutorConfig:
    optimizer_kind: str = "mars"
    lr: float = 3.0e-3
    lr_1d: float = 3.0e-3
    beta1_1d: float = 0.9
    beta2_1d: float = 0.95
    weight_decay: float = 0.0
    weight_decouple: bool = True
    fixed_decay: bool = False
    ams_bound: bool = False
    cautious: bool = False
    optimize_1d: bool = False
    mars_type: str = "adamw"
    eps: float = 1.0e-8
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _mars_first_step_1d_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        last_grad_ptr,
        n_elements,
        step_size,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2_sqrt,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        exp_avg_new = exp_avg * beta1 + grad * (1.0 - beta1)
        exp_avg_sq_new = exp_avg_sq * beta2 + grad * grad * (1.0 - beta2)
        denom = tl.sqrt(exp_avg_sq_new + 1.0e-15) + eps
        denom = (denom / bias_correction2_sqrt) * bias_correction1
        update = exp_avg_new / denom
        param_new = param - step_size * update

        tl.store(param_ptr + offsets, param_new, mask=mask)
        tl.store(exp_avg_ptr + offsets, exp_avg_new, mask=mask)
        tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq_new, mask=mask)
        tl.store(last_grad_ptr + offsets, grad, mask=mask)


class MarsTrainingExecutor:
    """Launch a real MARS 1D fallback first-step fp32 Triton kernel."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: MarsTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("MarsTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("mars_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("mars_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("mars_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"mars_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("mars_maximize_not_supported")
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
            blockers.append("mars_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_mars_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "mars_native_step_failed"),
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
            return _case_failed("mars_training_executor_requires_float32")
        if not param.is_contiguous() or param.grad is None or not param.grad.is_contiguous():
            return _case_failed("mars_training_executor_requires_contiguous_tensors")
        if param.ndim != 1:
            return _case_failed("mars_training_executor_1d_fallback_canary_only")
        step = int(group.get("step", 0) or 0)
        if step != 1:
            return _case_failed("mars_training_executor_only_first_step_supported_for_canary")
        state = self.optimizer.state[param]
        exp_avg = _state_tensor(state, "exp_avg", param)
        exp_avg_sq = _state_tensor(state, "exp_avg_sq", param)
        last_grad = _state_tensor(state, "last_grad", param)
        before = param.detach().clone()
        beta1 = float(group_cfg["beta1_1d"])
        beta2 = float(group_cfg["beta2_1d"])
        bias_correction1 = 1.0 - math.pow(beta1, step)
        bias_correction2_sqrt = math.sqrt(1.0 - math.pow(beta2, step))
        grid = (triton.cdiv(int(param.numel()), int(group_cfg["block_size"])),)
        _mars_first_step_1d_kernel[grid](
            param,
            param.grad.detach(),
            exp_avg,
            exp_avg_sq,
            last_grad,
            int(param.numel()),
            float(group_cfg["step_size"]),
            beta1,
            beta2,
            bias_correction1,
            bias_correction2_sqrt,
            float(group_cfg["eps"]),
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq
        state["last_grad"] = last_grad
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
            "blocked_reasons": [] if mutated else ["mars_training_executor_parameters_not_mutated"],
        }


def build_plugin_mars_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: MarsTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> MarsTrainingExecutor:
    return MarsTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: MarsTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> MarsTrainingExecutorConfig:
    if isinstance(value, MarsTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas_1d = payload.get("betas_1d", group.get("betas_1d", (0.9, 0.95)))
    return MarsTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "mars"),
        lr=float(payload.get("lr", group.get("lr", 3.0e-3)) or 3.0e-3),
        lr_1d=float(payload.get("lr_1d", group.get("lr_1d", 3.0e-3)) or 3.0e-3),
        beta1_1d=float(betas_1d[0]),
        beta2_1d=float(betas_1d[1]),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0)) or 0.0),
        weight_decouple=bool(payload.get("weight_decouple", group.get("weight_decouple", True))),
        fixed_decay=bool(payload.get("fixed_decay", group.get("fixed_decay", False))),
        ams_bound=bool(payload.get("ams_bound", group.get("ams_bound", False))),
        cautious=bool(payload.get("cautious", group.get("cautious", False))),
        optimize_1d=bool(payload.get("optimize_1d", group.get("optimize_1d", False))),
        mars_type=str(payload.get("mars_type", group.get("mars_type", "adamw")) or "adamw"),
        eps=float(payload.get("eps", group.get("eps", 1.0e-8)) or 1.0e-8),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: MarsTrainingExecutorConfig) -> dict[str, Any]:
    betas_1d = group.get("betas_1d", (config.beta1_1d, config.beta2_1d))
    lr = float(group.get("lr", config.lr) or config.lr)
    lr_1d = float(group.get("lr_1d", config.lr_1d) or config.lr_1d)
    lr_1d_factor = float(group.get("lr_1d_factor", lr_1d / lr) or (lr_1d / lr))
    return {
        "lr": lr,
        "lr_1d": lr_1d,
        "lr_1d_factor": lr_1d_factor,
        "step_size": lr * lr_1d_factor,
        "beta1_1d": float(betas_1d[0]),
        "beta2_1d": float(betas_1d[1]),
        "weight_decay": float(group.get("weight_decay", config.weight_decay) or config.weight_decay),
        "weight_decouple": bool(group.get("weight_decouple", config.weight_decouple)),
        "fixed_decay": bool(group.get("fixed_decay", config.fixed_decay)),
        "ams_bound": bool(group.get("ams_bound", config.ams_bound)),
        "cautious": bool(group.get("cautious", config.cautious)),
        "optimize_1d": bool(group.get("optimize_1d", config.optimize_1d)),
        "mars_type": str(group.get("mars_type", config.mars_type) or config.mars_type),
        "eps": float(group.get("eps", config.eps) or config.eps),
        "block_size": int(config.block_size),
    }


def _unsupported_group_blockers(group_cfg: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if bool(group_cfg.get("optimize_1d", False)):
        blockers.append("mars_mixed_optimize_1d_path_not_supported_for_canary")
    if str(group_cfg.get("mars_type", "adamw")) != "adamw":
        blockers.append("mars_non_adamw_type_not_supported_for_canary")
    if bool(group_cfg.get("ams_bound", False)):
        blockers.append("mars_ams_bound_not_supported_for_canary")
    if bool(group_cfg.get("cautious", False)):
        blockers.append("mars_cautious_not_supported_for_canary")
    if abs(float(group_cfg.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("mars_weight_decay_not_supported_for_canary")
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
        "executor": "turbocore_plugin_mars_training_executor_v0",
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
    "MarsTrainingExecutor",
    "MarsTrainingExecutorConfig",
    "build_plugin_mars_training_executor",
]
