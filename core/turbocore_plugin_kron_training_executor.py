"""Default-off Kron Triton TrainingLoop executor for plugin canaries."""

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
class KronTrainingExecutorConfig:
    optimizer_kind: str = "kron"
    lr: float = 1.0e-3
    momentum: float = 0.0
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _kron_identity_step_kernel(
        param_ptr,
        grad_ptr,
        momentum_ptr,
        n_elements,
        lr,
        momentum,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(momentum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m_new = m * momentum + g * (1.0 - momentum)
        tl.store(momentum_ptr + offsets, m_new, mask=mask)
        tl.store(param_ptr + offsets, p - lr * m_new, mask=mask)


class KronTrainingExecutor:
    """Launch a no-RNG identity-Q Kron fp32 Triton canary."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: KronTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("KronTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("kron_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("kron_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("kron_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"kron_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("kron_maximize_not_supported")

        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        update_counter = int(getattr(self.optimizer, "update_counter", 0))
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config)
            blockers = _unsupported_group_blockers(group, group_cfg)
            if blockers:
                cases.append(_case_failed(blockers[0]))
                continue
            group["step"] = int(group.get("step", 0) or 0) + 1  # type: ignore[index]
            setattr(self.optimizer, "prob_step", int(getattr(self.optimizer, "prob_step", 0)) + 1)
            setattr(self.optimizer, "update_counter", update_counter + 1)
            for param in group["params"]:
                if param.grad is None:
                    continue
                cases.append(self._step_param(param, group, group_cfg))

        blockers = _dedupe(reason for case in cases for reason in case.get("blocked_reasons", []) or [])
        if not cases:
            blockers.append("kron_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_kron_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "kron_native_step_failed"),
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
            return _case_failed("kron_training_executor_requires_float32")
        if param.grad is None or param.grad.is_sparse:
            return _case_failed("kron_training_executor_sparse_or_missing_grad")
        if not param.is_contiguous() or not param.grad.is_contiguous():
            return _case_failed("kron_training_executor_requires_contiguous_tensors")
        if param.dim() != 1:
            return _case_failed("kron_training_executor_requires_1d_canary")
        step = int(group.get("step", 0) or 0)
        if step != 1:
            return _case_failed("kron_training_executor_only_first_step_supported_for_canary")

        state = self.optimizer.state[param]
        momentum_buffer = _state_tensor(state, "momentum_buffer", param)
        _ensure_identity_q_state(state, param)
        before = param.detach().clone()
        grid = (triton.cdiv(int(param.numel()), int(group_cfg["block_size"])),)
        _kron_identity_step_kernel[grid](
            param,
            param.grad.detach(),
            momentum_buffer,
            int(param.numel()),
            float(group_cfg["lr"]),
            float(group_cfg["momentum"]),
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        mutated = _max_abs_diff(before, param.detach()) > 0.0
        return _case("kron", mutated, param, state, step)


def build_plugin_kron_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: KronTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> KronTrainingExecutor:
    return KronTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: KronTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> KronTrainingExecutorConfig:
    if isinstance(value, KronTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    return KronTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "kron"),
        lr=float(payload.get("lr", group.get("lr", 1.0e-3)) or 1.0e-3),
        momentum=float(payload.get("momentum", group.get("momentum", 0.0)) or 0.0),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: KronTrainingExecutorConfig) -> dict[str, Any]:
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "momentum": float(group.get("momentum", config.momentum) or config.momentum),
        "block_size": int(config.block_size),
    }


def _unsupported_group_blockers(group: Mapping[str, Any], group_cfg: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if abs(float(group.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("kron_weight_decay_not_supported_for_canary")
    if abs(float(group_cfg["momentum"])) > 0.0:
        blockers.append("kron_momentum_not_supported_for_canary")
    if not bool(group.get("weight_decouple", True)):
        blockers.append("kron_coupled_weight_decay_not_supported_for_canary")
    if str(group.get("memory_save_mode", "all_diag") or "all_diag") != "all_diag":
        blockers.append("kron_requires_all_diag_memory_save_mode_canary")
    if float(getattr(group.get("pre_conditioner_update_probability"), "__float__", lambda: 0.0)()) > 1.0e-6:
        blockers.append("kron_precondition_update_probability_not_disabled_for_canary")
    return blockers


def _state_tensor(state: dict[str, Any], key: str, param: torch.nn.Parameter) -> torch.Tensor:
    value = state.get(key)
    if torch.is_tensor(value) and tuple(value.shape) == tuple(param.shape):
        return value
    state[key] = torch.zeros_like(param.detach(), dtype=torch.float32)
    return state[key]


def _ensure_identity_q_state(state: dict[str, Any], param: torch.nn.Parameter) -> None:
    q = state.get("Q")
    if not isinstance(q, list) or len(q) != 1 or not torch.is_tensor(q[0]) or tuple(q[0].shape) != tuple(param.shape):
        state["Q"] = [torch.ones_like(param.detach(), dtype=torch.float32)]
    state["expressions"] = ("a,a->a", ["a,a->a"], "a,a,a->a")


def _case(kind: str, ok: bool, param: torch.nn.Parameter, state: dict[str, Any], step: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": bool(ok) and torch.isfinite(param).all().item(),
        "param_shape": [int(dim) for dim in param.shape],
        "param_dtype": str(param.dtype).replace("torch.", ""),
        "step_after": int(step),
        "kernel_executed": True,
        "training_parameters_mutated": bool(ok),
        "state_keys": sorted(str(key) for key in state.keys()),
        "blocked_reasons": [] if ok else [f"{kind}_training_executor_parameters_not_mutated"],
    }


def _case_failed(reason: str) -> dict[str, Any]:
    return {"schema_version": 1, "ok": False, "kernel_executed": False, "training_parameters_mutated": False, "blocked_reasons": [reason]}


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "executor": "turbocore_plugin_kron_training_executor_v0",
        "ok": False,
        "reason": reason,
        "optimizer_kind": "kron",
        "training_dispatch": True,
        "training_path_enabled": True,
        "native_step_executed": False,
        "native_kernel_launched": False,
        "training_parameters_mutated": False,
        "should_call_pytorch_optimizer_step": True,
        "blocked_reasons": [reason],
    }


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().detach().cpu().item())


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["KronTrainingExecutor", "KronTrainingExecutorConfig", "build_plugin_kron_training_executor"]
