"""Default-off FTRL Triton TrainingLoop executor for selected plugin canaries."""

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
class FTRLTrainingExecutorConfig:
    optimizer_kind: str = "ftrl"
    lr: float = 1.0e-3
    lr_power: float = -0.5
    beta: float = 0.0
    lambda_1: float = 0.0
    lambda_2: float = 0.0
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _ftrl_dense_kernel(
        param_ptr,
        grad_ptr,
        z_ptr,
        n_ptr,
        n_elements,
        lr,
        beta,
        lambda_1,
        lambda_2,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        n = tl.load(n_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        g2 = g * g
        sigma = (tl.sqrt(n + g2) - tl.sqrt(n)) / lr
        z_new = z + g - sigma * p
        n_new = n + g2
        denom = (beta + tl.sqrt(n_new)) / lr + lambda_2
        z_sign = tl.where(z_new > 0.0, 1.0, tl.where(z_new < 0.0, -1.0, 0.0))
        update = (z_sign * lambda_1 - z_new) / denom
        p_new = tl.where(tl.abs(z_new) < lambda_1, 0.0, update)

        tl.store(z_ptr + offsets, z_new, mask=mask)
        tl.store(n_ptr + offsets, n_new, mask=mask)
        tl.store(param_ptr + offsets, p_new, mask=mask)


class FTRLTrainingExecutor:
    """Launch a real FTRL fp32 Triton kernel against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: FTRLTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("FTRLTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("ftrl_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("ftrl_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("ftrl_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"ftrl_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("ftrl_maximize_not_supported")
        if float(self.config.lr) <= 0.0:
            return _blocked("ftrl_training_executor_requires_positive_lr")
        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config)
            if float(group_cfg["lr"]) <= 0.0:
                cases.append(_case_failed("ftrl_training_executor_requires_positive_group_lr"))
                continue
            if abs(float(group_cfg["lr_power"]) + 0.5) > 1.0e-12:
                cases.append(_case_failed("ftrl_training_executor_requires_lr_power_minus_half"))
                continue
            group["step"] = int(group.get("step", 0) or 0) + 1  # type: ignore[index]
            for param in group["params"]:
                if param.grad is None:
                    continue
                cases.append(self._step_param(param, group, group_cfg))
        blockers = _dedupe(reason for case in cases for reason in case.get("blocked_reasons", []) or [])
        if not cases:
            blockers.append("ftrl_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_ftrl_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "ftrl_native_step_failed"),
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
            return _case_failed("ftrl_training_executor_requires_float32")
        if not param.is_contiguous() or param.grad is None or not param.grad.is_contiguous():
            return _case_failed("ftrl_training_executor_requires_contiguous_tensors")
        state = self.optimizer.state[param]
        z = _state_tensor(state, "z", param)
        n = _state_tensor(state, "n", param)
        before = param.detach().clone()
        grid = (triton.cdiv(int(param.numel()), int(group_cfg["block_size"])),)
        _ftrl_dense_kernel[grid](
            param,
            param.grad.detach(),
            z,
            n,
            int(param.numel()),
            float(group_cfg["lr"]),
            float(group_cfg["beta"]),
            float(group_cfg["lambda_1"]),
            float(group_cfg["lambda_2"]),
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        state["z"] = z
        state["n"] = n
        mutated = not torch.allclose(before, param.detach())
        return {
            "schema_version": 1,
            "ok": mutated and torch.isfinite(param).all().item(),
            "param_shape": [int(dim) for dim in param.shape],
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_after": int(group.get("step", 0) or 0),
            "kernel_executed": True,
            "training_parameters_mutated": bool(mutated),
            "state_keys": sorted(str(key) for key in state.keys()),
            "blocked_reasons": [] if mutated else ["ftrl_training_executor_parameters_not_mutated"],
        }


def build_plugin_ftrl_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: FTRLTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> FTRLTrainingExecutor:
    return FTRLTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: FTRLTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> FTRLTrainingExecutorConfig:
    if isinstance(value, FTRLTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    return FTRLTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "ftrl"),
        lr=float(payload.get("lr", group.get("lr", 1.0e-3)) or 1.0e-3),
        lr_power=float(payload.get("lr_power", group.get("lr_power", -0.5))),
        beta=float(payload.get("beta", group.get("beta", 0.0)) or 0.0),
        lambda_1=float(payload.get("lambda_1", group.get("lambda_1", 0.0)) or 0.0),
        lambda_2=float(payload.get("lambda_2", group.get("lambda_2", 0.0)) or 0.0),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: FTRLTrainingExecutorConfig) -> dict[str, Any]:
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "lr_power": float(group.get("lr_power", config.lr_power)),
        "beta": float(group.get("beta", config.beta) or config.beta),
        "lambda_1": float(group.get("lambda_1", config.lambda_1) or config.lambda_1),
        "lambda_2": float(group.get("lambda_2", config.lambda_2) or config.lambda_2),
        "block_size": int(config.block_size),
    }


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
        "executor": "turbocore_plugin_ftrl_training_executor_v0",
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


__all__ = ["FTRLTrainingExecutor", "FTRLTrainingExecutorConfig", "build_plugin_ftrl_training_executor"]
