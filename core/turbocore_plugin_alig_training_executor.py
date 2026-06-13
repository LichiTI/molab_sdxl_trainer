"""Default-off AliG Triton TrainingLoop executor for selected plugin canaries."""

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
class AliGTrainingExecutorConfig:
    optimizer_kind: str = "alig"
    max_lr: float | None = None
    momentum: float = 0.0
    adjusted_momentum: bool = False
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _alig_dense_step_kernel(
        param_ptr,
        grad_ptr,
        n_elements,
        step_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(param_ptr + offsets, p - step_size * g, mask=mask)


class AliGTrainingExecutor:
    """Launch a real AliG no-momentum fp32 Triton kernel against live params."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: AliGTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("AliGTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("alig_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("alig_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("alig_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"alig_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)) or bool(getattr(getattr(self.optimizer, "_base", None), "maximize", False)):
            return _blocked("alig_maximize_not_supported")

        loss_value, loss_blocker = _loss_value_from_request(payload, self.optimizer)
        if loss_blocker:
            return _blocked(loss_blocker)

        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config)
            blockers = _unsupported_group_blockers(group_cfg, self.optimizer)
            if blockers:
                cases.append(_case_failed(blockers[0]))
                continue
            group["step"] = int(group.get("step", 0) or 0) + 1  # type: ignore[index]
            group["step_size"] = _compute_step_size(self.optimizer.param_groups, float(loss_value), group_cfg)
            for param in group["params"]:
                if param.grad is None:
                    continue
                cases.append(self._step_param(param, group, group_cfg))

        _clear_bound_loss_value(self.optimizer)
        blockers = _dedupe(reason for case in cases for reason in case.get("blocked_reasons", []) or [])
        if not cases:
            blockers.append("alig_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_alig_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "alig_native_step_failed"),
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
            return _case_failed("alig_training_executor_requires_float32")
        if not param.is_contiguous() or param.grad is None or not param.grad.is_contiguous():
            return _case_failed("alig_training_executor_requires_contiguous_tensors")
        if param.grad.is_sparse:
            return _case_failed("alig_training_executor_sparse_grad_not_supported")
        step = int(group.get("step", 0) or 0)
        if step != 1:
            return _case_failed("alig_training_executor_only_first_step_supported_for_canary")
        before = param.detach().clone()
        grid = (triton.cdiv(int(param.numel()), int(group_cfg["block_size"])),)
        _alig_dense_step_kernel[grid](
            param,
            param.grad.detach(),
            int(param.numel()),
            float(group.get("step_size", 0.0) or 0.0),
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        mutated = _max_abs_diff(before, param.detach()) > 0.0
        return {
            "schema_version": 1,
            "ok": mutated and torch.isfinite(param).all().item(),
            "param_shape": [int(dim) for dim in param.shape],
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_after": step,
            "step_size": float(group.get("step_size", 0.0) or 0.0),
            "kernel_executed": True,
            "training_parameters_mutated": bool(mutated),
            "state_keys": sorted(str(key) for key in self.optimizer.state[param].keys()),
            "blocked_reasons": [] if mutated else ["alig_training_executor_parameters_not_mutated"],
        }


def build_plugin_alig_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: AliGTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> AliGTrainingExecutor:
    return AliGTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: AliGTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> AliGTrainingExecutorConfig:
    if isinstance(value, AliGTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    max_lr = payload.get("max_lr", group.get("max_lr"))
    return AliGTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "alig"),
        max_lr=None if max_lr is None else float(max_lr),
        momentum=float(payload.get("momentum", group.get("momentum", 0.0)) or 0.0),
        adjusted_momentum=bool(payload.get("adjusted_momentum", group.get("adjusted_momentum", False))),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: AliGTrainingExecutorConfig) -> dict[str, Any]:
    max_lr = group.get("max_lr", config.max_lr)
    return {
        "max_lr": None if max_lr is None else float(max_lr),
        "momentum": float(group.get("momentum", config.momentum) or 0.0),
        "adjusted_momentum": bool(group.get("adjusted_momentum", config.adjusted_momentum)),
        "block_size": int(config.block_size),
    }


def _unsupported_group_blockers(group_cfg: Mapping[str, Any], optimizer: torch.optim.Optimizer) -> list[str]:
    blockers: list[str] = []
    if abs(float(group_cfg.get("momentum", 0.0) or 0.0)) > 0.0:
        blockers.append("alig_momentum_not_supported_for_canary")
    if bool(group_cfg.get("adjusted_momentum", False)):
        blockers.append("alig_adjusted_momentum_not_supported_for_canary")
    base = getattr(optimizer, "_base", optimizer)
    if getattr(base, "projection_fn", None) is not None:
        blockers.append("alig_projection_fn_not_supported_for_canary")
    return blockers


def _loss_value_from_request(payload: Mapping[str, Any], optimizer: torch.optim.Optimizer) -> tuple[float, str]:
    context = payload.get("runtime_context")
    if isinstance(context, Mapping) and "optimizer_loss_value_for_step" in context:
        return float(context["optimizer_loss_value_for_step"]), ""
    value = getattr(optimizer, "_lulynx_loss_value_for_step", None)
    if value is None:
        return 0.0, "alig_loss_value_missing_for_canary"
    if isinstance(value, torch.Tensor):
        value = float(value.detach().float().item())
    return float(value), ""


def _compute_step_size(param_groups: list[dict[str, Any]], loss_value: float, group_cfg: Mapping[str, Any]) -> float:
    grad_norm_sq = torch.zeros(1, dtype=torch.float32, device=param_groups[0]["params"][0].device)
    for group in param_groups:
        for param in group["params"]:
            grad = getattr(param, "grad", None)
            if grad is not None:
                grad_norm_sq.add_(grad.norm().pow(2))
    step_size = float(loss_value) / float(torch.sqrt(grad_norm_sq).add_(1.0e-6).detach().cpu().item())
    max_lr = group_cfg.get("max_lr")
    return min(step_size, float(max_lr)) if max_lr is not None else step_size


def _clear_bound_loss_value(optimizer: torch.optim.Optimizer) -> None:
    if hasattr(optimizer, "_lulynx_loss_value_for_step"):
        setattr(optimizer, "_lulynx_loss_value_for_step", None)


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
        "executor": "turbocore_plugin_alig_training_executor_v0",
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


__all__ = ["AliGTrainingExecutor", "AliGTrainingExecutorConfig", "build_plugin_alig_training_executor"]
