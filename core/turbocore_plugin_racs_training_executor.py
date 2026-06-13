"""Default-off RACS Triton TrainingLoop executor for selected plugin canaries."""

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
class RacsTrainingExecutorConfig:
    optimizer_kind: str = "racs"
    lr: float = 1.0e-3
    beta: float = 0.9
    alpha: float = 0.05
    gamma: float = 1.01
    weight_decay: float = 0.0
    weight_decouple: bool = True
    fixed_decay: bool = False
    eps: float = 1.0e-8
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _racs_first_step_1d_kernel(
        param_ptr,
        grad_ptr,
        s_ptr,
        q_ptr,
        theta_ptr,
        n_elements,
        lr,
        beta,
        alpha,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(s_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        q = tl.load(q_ptr).to(tl.float32)

        grad_sq = grad * grad
        mean_grad_sq = tl.sum(tl.where(mask, grad_sq, 0.0), axis=0) / n_elements
        s_raw = s * beta + grad_sq * (1.0 - beta)
        q_raw = q * beta + mean_grad_sq * (1.0 - beta)
        s_sqrt = tl.sqrt(s_raw + eps)
        q_sqrt = tl.sqrt(q_raw + eps)
        grad_hat = grad / (s_sqrt * q_sqrt)
        param_new = param - lr * alpha * grad_hat
        theta = tl.sqrt(tl.sum(tl.where(mask, grad_hat * grad_hat, 0.0), axis=0))

        tl.store(param_ptr + offsets, param_new, mask=mask)
        tl.store(s_ptr + offsets, s_raw, mask=mask)
        tl.store(q_ptr, q_raw)
        tl.store(theta_ptr, theta)


class RacsTrainingExecutor:
    """Launch a real RACS first-step 1D fp32 Triton kernel against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: RacsTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("RacsTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("racs_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("racs_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("racs_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"racs_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("racs_maximize_not_supported")
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
            blockers.append("racs_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_racs_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "racs_native_step_failed"),
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
            return _case_failed("racs_training_executor_requires_float32")
        if not param.is_contiguous() or param.grad is None or not param.grad.is_contiguous():
            return _case_failed("racs_training_executor_requires_contiguous_tensors")
        if param.ndim != 1:
            return _case_failed("racs_training_executor_1d_canary_only")
        if int(param.numel()) > int(group_cfg["block_size"]):
            return _case_failed("racs_training_executor_single_block_canary_only")
        step = int(group.get("step", 0) or 0)
        if step != 1:
            return _case_failed("racs_training_executor_only_first_step_supported_for_canary")
        state = self.optimizer.state[param]
        s = _state_tensor(state, "s", (int(param.numel()),), param)
        q = _state_tensor(state, "q", (1,), param, fill=1.0)
        theta = _state_tensor(state, "theta", (1,), param)
        before = param.detach().clone()
        _racs_first_step_1d_kernel[(1,)](
            param,
            param.grad.detach(),
            s,
            q,
            theta,
            int(param.numel()),
            float(group_cfg["lr"]),
            float(group_cfg["beta"]),
            float(group_cfg["alpha"]),
            float(group_cfg["eps"]),
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        state["s"] = s
        state["q"] = q
        state["theta"] = theta
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
            "blocked_reasons": [] if mutated else ["racs_training_executor_parameters_not_mutated"],
        }


def build_plugin_racs_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: RacsTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> RacsTrainingExecutor:
    return RacsTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: RacsTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> RacsTrainingExecutorConfig:
    if isinstance(value, RacsTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    return RacsTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "racs"),
        lr=float(payload.get("lr", group.get("lr", 1.0e-3)) or 1.0e-3),
        beta=float(payload.get("beta", group.get("beta", 0.9)) or 0.9),
        alpha=float(payload.get("alpha", group.get("alpha", 0.05)) or 0.05),
        gamma=float(payload.get("gamma", group.get("gamma", 1.01)) or 1.01),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0)) or 0.0),
        weight_decouple=bool(payload.get("weight_decouple", group.get("weight_decouple", True))),
        fixed_decay=bool(payload.get("fixed_decay", group.get("fixed_decay", False))),
        eps=float(payload.get("eps", group.get("eps", 1.0e-8)) or 1.0e-8),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: RacsTrainingExecutorConfig) -> dict[str, Any]:
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "beta": float(group.get("beta", config.beta) or config.beta),
        "alpha": float(group.get("alpha", config.alpha) or config.alpha),
        "gamma": float(group.get("gamma", config.gamma) or config.gamma),
        "weight_decay": float(group.get("weight_decay", config.weight_decay) or config.weight_decay),
        "weight_decouple": bool(group.get("weight_decouple", config.weight_decouple)),
        "fixed_decay": bool(group.get("fixed_decay", config.fixed_decay)),
        "eps": float(group.get("eps", config.eps) or config.eps),
        "block_size": int(config.block_size),
    }


def _unsupported_group_blockers(group_cfg: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if abs(float(group_cfg.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("racs_weight_decay_not_supported_for_canary")
    return blockers


def _state_tensor(
    state: dict[str, Any],
    key: str,
    shape: tuple[int, ...],
    param: torch.nn.Parameter,
    *,
    fill: float = 0.0,
) -> torch.Tensor:
    existing = state.get(key)
    if torch.is_tensor(existing) and tuple(existing.shape) == shape:
        if existing.device == param.device and existing.dtype == torch.float32 and existing.is_contiguous():
            return existing
        converted = existing.detach().to(device=param.device, dtype=torch.float32).contiguous()
        state[key] = converted
        return converted
    state[key] = torch.full(shape, fill, device=param.device, dtype=torch.float32)
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
        "executor": "turbocore_plugin_racs_training_executor_v0",
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
    "RacsTrainingExecutor",
    "RacsTrainingExecutorConfig",
    "build_plugin_racs_training_executor",
]
