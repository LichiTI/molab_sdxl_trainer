"""Default-off A2Grad Triton TrainingLoop executor for selected plugin canaries."""

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
class A2GradTrainingExecutorConfig:
    optimizer_kind: str = "a2grad"
    beta: float = 10.0
    lips: float = 10.0
    rho: float = 0.5
    variant: str = "uni"
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _a2grad_uni_dense_kernel(
        param_ptr,
        grad_ptr,
        avg_grad_ptr,
        x_k_ptr,
        n_elements,
        step_plus_one,
        alpha_k_old,
        alpha_k_next,
        coefficient,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        avg = tl.load(avg_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_k_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        avg_new = avg + (g - avg) * step_plus_one
        x_new = x + g * coefficient
        p_new = p * (1.0 - alpha_k_next) + x_new * alpha_k_next
        p_new = p_new + g * ((1.0 - alpha_k_next) * alpha_k_old * coefficient)

        tl.store(avg_grad_ptr + offsets, avg_new, mask=mask)
        tl.store(x_k_ptr + offsets, x_new, mask=mask)
        tl.store(param_ptr + offsets, p_new, mask=mask)


class A2GradTrainingExecutor:
    """Launch a real A2Grad uni-variant fp32 Triton kernel against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: A2GradTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("A2GradTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("a2grad_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("a2grad_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("a2grad_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"a2grad_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("a2grad_maximize_not_supported")
        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config, self.optimizer)
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
            blockers.append("a2grad_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_a2grad_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "a2grad_native_step_failed"),
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
            return _case_failed("a2grad_training_executor_requires_float32")
        if not param.is_contiguous() or param.grad is None or not param.grad.is_contiguous():
            return _case_failed("a2grad_training_executor_requires_contiguous_tensors")
        state = self.optimizer.state[param]
        avg_grad = _state_tensor(state, "avg_grad", param, fill_from=param.grad.detach())
        x_k = _state_tensor(state, "x_k", param, fill_from=param.detach())
        v_k = _state_scalar_tensor(state, "v_k", param)
        alpha_k_old = float(state.get("alpha_k", 1.0) or 1.0)
        before = param.detach().clone()
        step = int(group.get("step", 0) or 0)
        step_plus_one = float(step + 1)
        avg_new = avg_grad + (param.grad.detach() - avg_grad) * step_plus_one
        delta_k_sq = (param.grad.detach() - avg_new).pow(2).sum()
        v_k.add_(delta_k_sq)
        h_k = float(v_k.sqrt().detach().cpu().item())
        gamma_k = 2.0 * float(group_cfg["lips"]) / float(step + 1)
        alpha_k_next = 2.0 / float(step + 3)
        coefficient = -1.0 / (gamma_k + float(group_cfg["beta"]) * h_k)
        grid = (triton.cdiv(int(param.numel()), int(group_cfg["block_size"])),)
        _a2grad_uni_dense_kernel[grid](
            param,
            param.grad.detach(),
            avg_grad,
            x_k,
            int(param.numel()),
            step_plus_one,
            float(alpha_k_old),
            float(alpha_k_next),
            float(coefficient),
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        state["avg_grad"] = avg_grad
        state["x_k"] = x_k
        state["v_k"] = v_k
        state["alpha_k"] = alpha_k_next
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
            "blocked_reasons": [] if mutated else ["a2grad_training_executor_parameters_not_mutated"],
        }


def build_plugin_a2grad_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: A2GradTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> A2GradTrainingExecutor:
    return A2GradTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: A2GradTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> A2GradTrainingExecutorConfig:
    if isinstance(value, A2GradTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    return A2GradTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "a2grad"),
        beta=float(payload.get("beta", group.get("beta", 10.0)) or 10.0),
        lips=float(payload.get("lips", group.get("lips", 10.0)) or 10.0),
        rho=float(payload.get("rho", group.get("rho", 0.5)) or 0.5),
        variant=str(payload.get("variant", getattr(optimizer, "variant", "uni")) or "uni"),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(
    group: Mapping[str, Any],
    config: A2GradTrainingExecutorConfig,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return {
        "beta": float(group.get("beta", config.beta) or config.beta),
        "lips": float(group.get("lips", config.lips) or config.lips),
        "rho": float(group.get("rho", config.rho) or config.rho),
        "variant": str(getattr(optimizer, "variant", config.variant) or config.variant),
        "block_size": int(config.block_size),
    }


def _unsupported_group_blockers(group_cfg: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if str(group_cfg.get("variant", "uni")) != "uni":
        blockers.append("a2grad_only_uni_variant_supported_for_canary")
    return blockers


def _state_tensor(
    state: dict[str, Any],
    key: str,
    param: torch.nn.Parameter,
    *,
    fill_from: torch.Tensor,
) -> torch.Tensor:
    existing = state.get(key)
    if torch.is_tensor(existing) and tuple(existing.shape) == tuple(param.shape):
        if existing.device == param.device and existing.dtype == torch.float32 and existing.is_contiguous():
            return existing
        converted = existing.detach().to(device=param.device, dtype=torch.float32).contiguous()
        state[key] = converted
        return converted
    state[key] = fill_from.detach().to(device=param.device, dtype=torch.float32).clone().contiguous()
    return state[key]


def _state_scalar_tensor(state: dict[str, Any], key: str, param: torch.nn.Parameter) -> torch.Tensor:
    existing = state.get(key)
    if torch.is_tensor(existing) and existing.numel() == 1:
        if existing.device == param.device and existing.dtype == torch.float32 and existing.is_contiguous():
            return existing
        converted = existing.detach().to(device=param.device, dtype=torch.float32).reshape(1).contiguous()
        state[key] = converted
        return converted
    state[key] = torch.zeros((1,), dtype=torch.float32, device=param.device)
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
        "executor": "turbocore_plugin_a2grad_training_executor_v0",
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


__all__ = ["A2GradTrainingExecutor", "A2GradTrainingExecutorConfig", "build_plugin_a2grad_training_executor"]
