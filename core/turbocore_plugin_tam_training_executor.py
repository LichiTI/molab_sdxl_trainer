"""Default-off TAM Triton TrainingLoop executor for selected plugin canaries."""

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
class TamTrainingExecutorConfig:
    optimizer_kind: str = "tam"
    lr: float = 1.0e-3
    momentum: float = 0.9
    decay_rate: float = 0.9
    weight_decay: float = 0.0
    eps: float = 1.0e-8
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _tam_first_step_scalar_kernel(
        param_ptr,
        grad_ptr,
        s_ptr,
        momentum_buffer_ptr,
        lr,
        momentum,
        decay_rate,
        eps,
    ):
        p = tl.load(param_ptr).to(tl.float32)
        g = tl.load(grad_ptr).to(tl.float32)
        s = tl.load(s_ptr).to(tl.float32)
        buf = tl.load(momentum_buffer_ptr).to(tl.float32)

        grad_abs = tl.abs(g)
        buf_abs = tl.abs(buf)
        grad_norm = tl.where(grad_abs > 0.0, g / grad_abs, 0.0)
        buf_norm = tl.where(buf_abs > 0.0, buf / buf_abs, 0.0)
        corr = buf_norm * grad_norm
        s_new = s * decay_rate + corr * (1.0 - decay_rate)
        d = (((1.0 + s_new) / 2.0) + eps) * g
        buf_new = buf * momentum + d
        p_new = p - lr * buf_new

        tl.store(s_ptr, s_new)
        tl.store(momentum_buffer_ptr, buf_new)
        tl.store(param_ptr, p_new)


class TamTrainingExecutor:
    """Launch a real TAM first-step scalar fp32 Triton kernel against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: TamTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("TamTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("tam_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("tam_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("tam_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"tam_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("tam_maximize_not_supported")
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
            blockers.append("tam_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_tam_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "tam_native_step_failed"),
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
            return _case_failed("tam_training_executor_requires_float32")
        if not param.is_contiguous() or param.grad is None or not param.grad.is_contiguous():
            return _case_failed("tam_training_executor_requires_contiguous_tensors")
        if int(param.numel()) != 1:
            return _case_failed("tam_training_executor_scalar_canary_only")
        step = int(group.get("step", 0) or 0)
        if step != 1:
            return _case_failed("tam_training_executor_only_first_step_supported_for_canary")
        state = self.optimizer.state[param]
        s = _state_tensor(state, "s", param, init="zero")
        momentum_buffer = _state_tensor(state, "momentum_buffer", param, init="grad")
        before = param.detach().clone()
        _tam_first_step_scalar_kernel[(1,)](
            param,
            param.grad.detach(),
            s,
            momentum_buffer,
            float(group_cfg["lr"]),
            float(group_cfg["momentum"]),
            float(group_cfg["decay_rate"]),
            float(group_cfg["eps"]),
            num_warps=1,
        )
        state["s"] = s
        state["momentum_buffer"] = momentum_buffer
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
            "blocked_reasons": [] if mutated else ["tam_training_executor_parameters_not_mutated"],
        }


def build_plugin_tam_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: TamTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> TamTrainingExecutor:
    return TamTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: TamTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> TamTrainingExecutorConfig:
    if isinstance(value, TamTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    return TamTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "tam"),
        lr=float(payload.get("lr", group.get("lr", 1.0e-3)) or 1.0e-3),
        momentum=float(payload.get("momentum", group.get("momentum", 0.9)) or 0.9),
        decay_rate=float(payload.get("decay_rate", group.get("decay_rate", 0.9)) or 0.9),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0)) or 0.0),
        eps=float(payload.get("eps", group.get("eps", 1.0e-8)) or 1.0e-8),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: TamTrainingExecutorConfig) -> dict[str, Any]:
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "momentum": float(group.get("momentum", config.momentum) or config.momentum),
        "decay_rate": float(group.get("decay_rate", config.decay_rate) or config.decay_rate),
        "weight_decay": float(group.get("weight_decay", config.weight_decay) or config.weight_decay),
        "eps": float(group.get("eps", config.eps) or config.eps),
    }


def _unsupported_group_blockers(group_cfg: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if abs(float(group_cfg.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("tam_weight_decay_not_supported_for_canary")
    return blockers


def _state_tensor(state: dict[str, Any], key: str, param: torch.nn.Parameter, *, init: str) -> torch.Tensor:
    existing = state.get(key)
    if torch.is_tensor(existing) and tuple(existing.shape) == tuple(param.shape):
        if existing.device == param.device and existing.dtype == torch.float32 and existing.is_contiguous():
            return existing
        converted = existing.detach().to(device=param.device, dtype=torch.float32).contiguous()
        state[key] = converted
        return converted
    if init == "grad" and param.grad is not None:
        state[key] = param.grad.detach().to(dtype=torch.float32).contiguous().clone()
    else:
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
        "executor": "turbocore_plugin_tam_training_executor_v0",
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


__all__ = ["TamTrainingExecutor", "TamTrainingExecutorConfig", "build_plugin_tam_training_executor"]
