"""Default-off AdaShift Triton TrainingLoop executor for selected plugin canaries."""

from __future__ import annotations

import math
import time
from collections import deque
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
class AdaShiftTrainingExecutorConfig:
    optimizer_kind: str = "adashift"
    lr: float = 1.0e-3
    beta1: float = 0.9
    beta2: float = 0.999
    keep_num: int = 1
    weight_decay: float = 0.0
    eps: float = 1.0e-10
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _adashift_dense_kernel(
        param_ptr,
        grad_ptr,
        offset_grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        n_elements,
        lr,
        beta1,
        beta2,
        first_grad_weight,
        last_grad_weight,
        bias_correction,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        offset_grad = tl.load(offset_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        exp_avg_new = (exp_avg - offset_grad * first_grad_weight) * beta1 + grad * last_grad_weight
        reduced_grad_sq = offset_grad * offset_grad
        exp_avg_sq_new = exp_avg_sq * beta2 + reduced_grad_sq * (1.0 - beta2)
        denom = tl.sqrt(exp_avg_sq_new / bias_correction) + eps
        param_new = param - lr * (exp_avg_new / denom)

        tl.store(param_ptr + offsets, param_new, mask=mask)
        tl.store(exp_avg_ptr + offsets, exp_avg_new, mask=mask)
        tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq_new, mask=mask)


class AdaShiftTrainingExecutor:
    """Launch a real AdaShift warmed-step fp32 Triton kernel against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: AdaShiftTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("AdaShiftTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("adashift_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("adashift_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("adashift_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"adashift_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("adashift_maximize_not_supported")
        if not _identity_reduce_func(getattr(self.optimizer, "reduce_func", None)):
            return _blocked("adashift_reduce_func_identity_only_supported_for_canary")
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
            blockers.append("adashift_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_adashift_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "adashift_native_step_failed"),
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
            return _case_failed("adashift_training_executor_requires_float32")
        if not param.is_contiguous() or param.grad is None or not param.grad.is_contiguous():
            return _case_failed("adashift_training_executor_requires_contiguous_tensors")
        step = int(group.get("step", 0) or 0)
        keep_num = int(group_cfg["keep_num"])
        if step <= keep_num:
            return _case_failed("adashift_training_executor_requires_warmed_queue_step")
        state = self.optimizer.state[param]
        grad_queue = state.get("grad_queue")
        if not isinstance(grad_queue, deque) or len(grad_queue) != keep_num:
            return _case_failed("adashift_training_executor_requires_full_grad_queue")
        grad_queue.append(param.grad.detach().clone())
        offset_grad = grad_queue[0]
        if not torch.is_tensor(offset_grad) or tuple(offset_grad.shape) != tuple(param.shape):
            return _case_failed("adashift_training_executor_requires_tensor_offset_grad")
        if offset_grad.device != param.device or offset_grad.dtype != torch.float32 or not offset_grad.is_contiguous():
            return _case_failed("adashift_training_executor_requires_contiguous_fp32_offset_grad")
        offset_grad_for_sq = offset_grad.detach().clone()
        exp_avg = _state_tensor(state, "exp_avg", param)
        exp_avg_sq = _state_tensor(state, "exp_avg_sq", param)
        if exp_avg is None or exp_avg_sq is None:
            return _case_failed("adashift_training_executor_requires_fp32_state")
        weights = _shift_weights(group_cfg)
        bias_correction = 1.0 - float(group_cfg["beta2"]) ** (step - keep_num)
        if not math.isfinite(bias_correction) or bias_correction <= 0.0:
            return _case_failed("adashift_training_executor_requires_positive_bias_correction")
        before = param.detach().clone()
        grid = (triton.cdiv(int(param.numel()), int(group_cfg["block_size"])),)
        _adashift_dense_kernel[grid](
            param,
            param.grad.detach(),
            offset_grad,
            exp_avg,
            exp_avg_sq,
            int(param.numel()),
            float(group_cfg["lr"]),
            float(group_cfg["beta1"]),
            float(group_cfg["beta2"]),
            weights["first_grad_weight"],
            weights["last_grad_weight"],
            bias_correction,
            float(group_cfg["eps"]),
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        offset_grad.copy_(offset_grad_for_sq.square())
        state["grad_queue"] = grad_queue
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq
        mutated = _max_abs_diff(before, param.detach()) > 0.0
        finite = torch.isfinite(param).all().item() and torch.isfinite(exp_avg).all().item() and torch.isfinite(exp_avg_sq).all().item()
        return {
            "schema_version": 1,
            "ok": mutated and finite,
            "param_shape": [int(dim) for dim in param.shape],
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_after": step,
            "keep_num": keep_num,
            "kernel_executed": True,
            "training_parameters_mutated": bool(mutated),
            "state_keys": sorted(str(key) for key in state.keys()),
            "grad_queue_len": len(grad_queue),
            "blocked_reasons": [] if mutated else ["adashift_training_executor_parameters_not_mutated"],
        }


def build_plugin_adashift_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: AdaShiftTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> AdaShiftTrainingExecutor:
    return AdaShiftTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: AdaShiftTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> AdaShiftTrainingExecutorConfig:
    if isinstance(value, AdaShiftTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = payload.get("betas", group.get("betas", (0.9, 0.999)))
    if not isinstance(betas, (tuple, list)):
        betas = (0.9, 0.999)
    return AdaShiftTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "adashift"),
        lr=float(payload.get("lr", group.get("lr", 1.0e-3)) or 1.0e-3),
        beta1=float(betas[0] if len(betas) > 0 else 0.9),
        beta2=float(betas[1] if len(betas) > 1 else 0.999),
        keep_num=int(payload.get("keep_num", group.get("keep_num", 1)) or 1),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0)) or 0.0),
        eps=float(payload.get("eps", group.get("eps", 1.0e-10)) or 1.0e-10),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: AdaShiftTrainingExecutorConfig) -> dict[str, Any]:
    betas = group.get("betas", (config.beta1, config.beta2))
    if not isinstance(betas, (tuple, list)):
        betas = (config.beta1, config.beta2)
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "beta1": float(betas[0] if len(betas) > 0 else config.beta1),
        "beta2": float(betas[1] if len(betas) > 1 else config.beta2),
        "keep_num": int(group.get("keep_num", config.keep_num) or config.keep_num),
        "weight_decay": float(group.get("weight_decay", config.weight_decay) or config.weight_decay),
        "eps": float(group.get("eps", config.eps) or config.eps),
        "block_size": int(config.block_size),
    }


def _unsupported_group_blockers(group_cfg: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if int(group_cfg.get("keep_num", 0) or 0) != 1:
        blockers.append("adashift_keep_num_one_only_supported_for_canary")
    if abs(float(group_cfg.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("adashift_weight_decay_not_supported_for_canary")
    beta1 = float(group_cfg.get("beta1", 0.0) or 0.0)
    beta2 = float(group_cfg.get("beta2", 0.0) or 0.0)
    if beta1 < 0.0 or beta1 >= 1.0 or beta2 < 0.0 or beta2 >= 1.0:
        blockers.append("adashift_betas_out_of_range")
    if float(group_cfg.get("lr", 0.0) or 0.0) <= 0.0:
        blockers.append("adashift_lr_must_be_positive")
    return blockers


def _shift_weights(group_cfg: Mapping[str, Any]) -> dict[str, float]:
    beta1 = float(group_cfg["beta1"])
    keep_num = int(group_cfg["keep_num"])
    exp_weight_sum = sum(beta1**i for i in range(keep_num))
    return {
        "first_grad_weight": beta1 ** (keep_num - 1) / exp_weight_sum,
        "last_grad_weight": 1.0 / exp_weight_sum,
    }


def _identity_reduce_func(value: Any) -> bool:
    if value is None:
        return True
    name = str(getattr(value, "__name__", ""))
    return name == "<lambda>"


def _state_tensor(state: dict[str, Any], key: str, param: torch.nn.Parameter) -> torch.Tensor | None:
    value = state.get(key)
    if value is None:
        value = torch.zeros_like(param.detach(), dtype=torch.float32).contiguous()
        state[key] = value
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
        return None
    if value.device != param.device or value.dtype != torch.float32 or not value.is_contiguous():
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
        "executor": "turbocore_plugin_adashift_training_executor_v0",
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
    "AdaShiftTrainingExecutor",
    "AdaShiftTrainingExecutorConfig",
    "build_plugin_adashift_training_executor",
]
