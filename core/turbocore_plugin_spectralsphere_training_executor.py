"""Default-off SpectralSphere TrainingLoop executor for selected plugin canaries."""

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

try:  # pragma: no cover - plugin import availability is environment-specific
    from pytorch_optimizer.optimizer.sso import compute_spectral_ball_update
except Exception:  # pragma: no cover
    compute_spectral_ball_update = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SpectralSphereTrainingExecutorConfig:
    optimizer_kind: str = "spectralsphere"
    lr: float = 3.0e-4
    momentum: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0
    weight_decouple: bool = True
    power_iteration_steps: int = 1
    msign_steps: int = 1
    solver_tolerance_f: float = 1.0e-8
    solver_max_iterations: int = 10
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _spectralsphere_apply_update_kernel(
        param_ptr,
        update_ptr,
        n_elements,
        lr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(param_ptr + offsets, param - lr * update, mask=mask)


class SpectralSphereTrainingExecutor:
    """Launch SpectralSphere first-step CUDA canary against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: SpectralSphereTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("SpectralSphereTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("spectralsphere_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("spectralsphere_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("spectralsphere_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"spectralsphere_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if compute_spectral_ball_update is None:
            return _blocked("spectralsphere_plugin_update_helper_unavailable")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("spectralsphere_maximize_not_supported")
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
            blockers.append("spectralsphere_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_spectralsphere_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "spectralsphere_native_step_failed"),
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
        if type(getattr(self.optimizer, "_base", self.optimizer)).__name__.lower() != "spectralsphere":
            return _case_failed("spectralsphere_training_executor_optimizer_kind_unsupported")
        if param.ndim != 2:
            return _case_failed("spectralsphere_training_executor_requires_2d_param")
        if param.dtype != torch.float32 or param.grad is None or param.grad.dtype != torch.float32:
            return _case_failed("spectralsphere_training_executor_requires_float32")
        if not param.is_contiguous() or not param.grad.is_contiguous():
            return _case_failed("spectralsphere_training_executor_requires_contiguous_tensors")
        step = int(group.get("step", 0) or 0)
        if step != 1:
            return _case_failed("spectralsphere_training_executor_only_first_step_supported_for_canary")
        state = self.optimizer.state[param]
        momentum_buffer = state.get("momentum_buffer")
        if momentum_buffer is None:
            momentum_buffer = torch.zeros_like(param)
            state["momentum_buffer"] = momentum_buffer
        if not torch.is_tensor(momentum_buffer) or tuple(momentum_buffer.shape) != tuple(param.shape):
            return _case_failed("spectralsphere_training_executor_momentum_state_missing")
        if momentum_buffer.dtype != torch.float32 or not momentum_buffer.is_contiguous():
            return _case_failed("spectralsphere_training_executor_requires_contiguous_float32_momentum")

        before = param.detach().clone()
        momentum_buffer.lerp_(param.grad.detach(), weight=1.0 - float(group_cfg["momentum"]))
        update = compute_spectral_ball_update(  # type: ignore[misc]
            param.detach(),
            momentum=momentum_buffer,
            power_iteration_steps=int(group_cfg["power_iteration_steps"]),
            msign_steps=int(group_cfg["msign_steps"]),
            solver_tolerance_f=float(group_cfg["solver_tolerance_f"]),
            solver_max_iterations=int(group_cfg["solver_max_iterations"]),
        ).contiguous()
        if not torch.isfinite(update.float()).all().item():
            return _case_failed("spectralsphere_training_executor_update_nonfinite")
        grid = (triton.cdiv(int(param.numel()), int(group_cfg["block_size"])),)
        _spectralsphere_apply_update_kernel[grid](
            param,
            update,
            int(param.numel()),
            float(group_cfg["lr"]),
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        mutated = _max_abs_diff(before, param.detach()) > 0.0
        return {
            "schema_version": 1,
            "ok": mutated and torch.isfinite(param).all().item() and torch.isfinite(momentum_buffer).all().item(),
            "param_shape": [int(dim) for dim in param.shape],
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_after": step,
            "kernel_executed": True,
            "training_parameters_mutated": bool(mutated),
            "state_keys": sorted(str(key) for key in state.keys()),
            "update_dtype": str(update.dtype).replace("torch.", ""),
            "blocked_reasons": [] if mutated else ["spectralsphere_training_executor_parameters_not_mutated"],
        }


def build_plugin_spectralsphere_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: SpectralSphereTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> SpectralSphereTrainingExecutor:
    return SpectralSphereTrainingExecutor(
        optimizer=optimizer,
        params=params,
        config=config,
        workspace_root=workspace_root,
    )


def _normalize_config(
    value: SpectralSphereTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> SpectralSphereTrainingExecutorConfig:
    if isinstance(value, SpectralSphereTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    return SpectralSphereTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "spectralsphere"),
        lr=float(payload.get("lr", group.get("lr", 3.0e-4)) or 3.0e-4),
        momentum=float(payload.get("momentum", group.get("momentum", 0.0)) or 0.0),
        nesterov=bool(payload.get("nesterov", group.get("nesterov", False))),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0)) or 0.0),
        weight_decouple=bool(payload.get("weight_decouple", group.get("weight_decouple", True))),
        power_iteration_steps=int(payload.get("power_iteration_steps", 1) or 1),
        msign_steps=int(payload.get("msign_steps", 1) or 1),
        solver_tolerance_f=float(payload.get("solver_tolerance_f", 1.0e-8) or 1.0e-8),
        solver_max_iterations=int(payload.get("solver_max_iterations", 10) or 10),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: SpectralSphereTrainingExecutorConfig) -> dict[str, Any]:
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "momentum": float(group.get("momentum", config.momentum) or config.momentum),
        "nesterov": bool(group.get("nesterov", config.nesterov)),
        "weight_decay": float(group.get("weight_decay", config.weight_decay) or config.weight_decay),
        "weight_decouple": bool(group.get("weight_decouple", config.weight_decouple)),
        "power_iteration_steps": int(config.power_iteration_steps),
        "msign_steps": int(config.msign_steps),
        "solver_tolerance_f": float(config.solver_tolerance_f),
        "solver_max_iterations": int(config.solver_max_iterations),
        "block_size": int(config.block_size),
    }


def _unsupported_group_blockers(group_cfg: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if bool(group_cfg.get("nesterov", False)):
        blockers.append("spectralsphere_nesterov_not_supported_for_canary")
    if abs(float(group_cfg.get("momentum", 0.0) or 0.0)) > 0.0:
        blockers.append("spectralsphere_momentum_not_supported_for_canary")
    if abs(float(group_cfg.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("spectralsphere_weight_decay_not_supported_for_canary")
    if int(group_cfg.get("power_iteration_steps", 0) or 0) != 1:
        blockers.append("spectralsphere_power_iteration_steps_must_be_1_for_canary")
    if int(group_cfg.get("msign_steps", 0) or 0) != 1:
        blockers.append("spectralsphere_msign_steps_must_be_1_for_canary")
    return blockers


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
        "executor": "turbocore_plugin_spectralsphere_training_executor_v0",
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
    "SpectralSphereTrainingExecutor",
    "SpectralSphereTrainingExecutorConfig",
    "build_plugin_spectralsphere_training_executor",
]
