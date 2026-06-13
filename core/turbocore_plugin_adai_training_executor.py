"""Default-off Adai Triton TrainingLoop executor for selected plugin canaries."""

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
class AdaiTrainingExecutorConfig:
    optimizer_kind: str = "adai"
    lr: float = 1.0e-3
    beta0: float = 0.1
    beta2: float = 0.99
    eps: float = 1.0e-3
    weight_decay: float = 0.0
    weight_decouple: bool = False
    fixed_decay: bool = False
    stable_weight_decay: bool = False
    dampening: float = 1.0
    use_gc: bool = False
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _adai_dense_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        beta1_prod_ptr,
        n_elements,
        exp_avg_sq_hat_mean,
        bias_correction2,
        beta0,
        beta2,
        eps,
        lr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b1_prod = tl.load(beta1_prod_ptr + offsets, mask=mask, other=1.0).to(tl.float32)

        v_new = v * beta2 + g * g * (1.0 - beta2)
        v_hat = v_new / bias_correction2
        beta1_raw = 1.0 - (v_hat / exp_avg_sq_hat_mean) * beta0
        beta1 = tl.minimum(tl.maximum(beta1_raw, 0.0), 1.0 - eps)
        beta3 = 1.0 - beta1
        b1_prod_new = b1_prod * beta1
        m_new = m * beta1 + beta3 * g
        update = m_new / (1.0 - b1_prod_new)
        p_new = p - lr * update

        tl.store(exp_avg_ptr + offsets, m_new, mask=mask)
        tl.store(exp_avg_sq_ptr + offsets, v_new, mask=mask)
        tl.store(beta1_prod_ptr + offsets, b1_prod_new, mask=mask)
        tl.store(param_ptr + offsets, p_new, mask=mask)


class AdaiTrainingExecutor:
    """Launch a real Adai fp32 Triton kernel against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: AdaiTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("AdaiTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("adai_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("adai_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("adai_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"adai_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("adai_maximize_not_supported")
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
            blockers.append("adai_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_adai_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "adai_native_step_failed"),
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
            return _case_failed("adai_training_executor_requires_float32")
        if not param.is_contiguous() or param.grad is None or not param.grad.is_contiguous():
            return _case_failed("adai_training_executor_requires_contiguous_tensors")
        state = self.optimizer.state[param]
        exp_avg = _state_tensor(state, "exp_avg", param, fill=0.0)
        exp_avg_sq = _state_tensor(state, "exp_avg_sq", param, fill=0.0)
        beta1_prod = _state_tensor(state, "beta1_prod", param, fill=1.0)
        before = param.detach().clone()
        step = int(group.get("step", 0) or 0)
        beta2 = float(group_cfg["beta2"])
        bias_correction2 = 1.0 - math.pow(beta2, step)
        exp_avg_sq_next = exp_avg_sq.detach() * beta2 + param.grad.detach() * param.grad.detach() * (1.0 - beta2)
        exp_avg_sq_hat_mean = float((exp_avg_sq_next / bias_correction2).mean().detach().cpu().item())
        if not math.isfinite(exp_avg_sq_hat_mean) or exp_avg_sq_hat_mean <= 0.0:
            return _case_failed("adai_exp_avg_sq_hat_mean_invalid_for_canary")
        grid = (triton.cdiv(int(param.numel()), int(group_cfg["block_size"])),)
        _adai_dense_kernel[grid](
            param,
            param.grad.detach(),
            exp_avg,
            exp_avg_sq,
            beta1_prod,
            int(param.numel()),
            float(exp_avg_sq_hat_mean),
            float(bias_correction2),
            float(group_cfg["beta0"]),
            beta2,
            float(group_cfg["eps"]),
            float(group_cfg["lr"]),
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq
        state["beta1_prod"] = beta1_prod
        mutated = not torch.allclose(before, param.detach())
        return {
            "schema_version": 1,
            "ok": mutated and torch.isfinite(param).all().item(),
            "param_shape": [int(dim) for dim in param.shape],
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_after": step,
            "kernel_executed": True,
            "training_parameters_mutated": bool(mutated),
            "state_keys": sorted(str(key) for key in state.keys()),
            "blocked_reasons": [] if mutated else ["adai_training_executor_parameters_not_mutated"],
        }


def build_plugin_adai_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: AdaiTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> AdaiTrainingExecutor:
    return AdaiTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: AdaiTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> AdaiTrainingExecutorConfig:
    if isinstance(value, AdaiTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = payload.get("betas", group.get("betas", (0.1, 0.99)))
    return AdaiTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "adai"),
        lr=float(payload.get("lr", group.get("lr", 1.0e-3)) or 1.0e-3),
        beta0=float(betas[0]),
        beta2=float(betas[1]),
        eps=float(payload.get("eps", group.get("eps", 1.0e-3)) or 1.0e-3),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0)) or 0.0),
        weight_decouple=bool(payload.get("weight_decouple", group.get("weight_decouple", False))),
        fixed_decay=bool(payload.get("fixed_decay", group.get("fixed_decay", False))),
        stable_weight_decay=bool(payload.get("stable_weight_decay", group.get("stable_weight_decay", False))),
        dampening=float(payload.get("dampening", group.get("dampening", 1.0)) or 1.0),
        use_gc=bool(payload.get("use_gc", group.get("use_gc", False))),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: AdaiTrainingExecutorConfig) -> dict[str, Any]:
    betas = group.get("betas", (config.beta0, config.beta2))
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "beta0": float(betas[0]),
        "beta2": float(betas[1]),
        "eps": float(group.get("eps", config.eps) or config.eps),
        "weight_decay": float(group.get("weight_decay", config.weight_decay) or config.weight_decay),
        "weight_decouple": bool(group.get("weight_decouple", config.weight_decouple)),
        "fixed_decay": bool(group.get("fixed_decay", config.fixed_decay)),
        "stable_weight_decay": bool(group.get("stable_weight_decay", config.stable_weight_decay)),
        "dampening": float(group.get("dampening", config.dampening) or config.dampening),
        "use_gc": bool(group.get("use_gc", config.use_gc)),
        "block_size": int(config.block_size),
    }


def _unsupported_group_blockers(group_cfg: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if abs(float(group_cfg.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("adai_weight_decay_not_supported_for_canary")
    if bool(group_cfg.get("stable_weight_decay", False)):
        blockers.append("adai_stable_weight_decay_not_supported_for_canary")
    if bool(group_cfg.get("use_gc", False)):
        blockers.append("adai_gradient_centralization_not_supported_for_canary")
    if abs(float(group_cfg.get("dampening", 1.0) or 1.0) - 1.0) > 1.0e-12:
        blockers.append("adai_non_default_dampening_not_supported_for_canary")
    return blockers


def _state_tensor(state: dict[str, Any], key: str, param: torch.nn.Parameter, *, fill: float) -> torch.Tensor:
    existing = state.get(key)
    if torch.is_tensor(existing) and tuple(existing.shape) == tuple(param.shape):
        if existing.device == param.device and existing.dtype == torch.float32 and existing.is_contiguous():
            return existing
        converted = existing.detach().to(device=param.device, dtype=torch.float32).contiguous()
        state[key] = converted
        return converted
    state[key] = torch.full_like(param.detach(), float(fill), dtype=torch.float32).contiguous()
    return state[key]


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
        "executor": "turbocore_plugin_adai_training_executor_v0",
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


__all__ = ["AdaiTrainingExecutor", "AdaiTrainingExecutorConfig", "build_plugin_adai_training_executor"]
