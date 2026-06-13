"""Default-off Shampoo-family Triton TrainingLoop executors for plugin canaries."""

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
class ShampooFamilyTrainingExecutorConfig:
    optimizer_kind: str
    lr: float = 1.0e-3
    beta1: float = 0.0
    beta2: float = 0.999
    eps: float = 1.0e-8
    matrix_eps: float = 1.0e-6
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _apply_update_kernel(param_ptr, update_ptr, n_elements, lr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(param_ptr + offsets, p - lr * u, mask=mask)

    @triton.jit
    def _soap_second_step_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        n_elements,
        beta1,
        beta2,
        step_size,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m_new = m * beta1 + g * (1.0 - beta1)
        v_new = v * beta2 + g * g * (1.0 - beta2)
        p_new = p - step_size * (m_new / (tl.sqrt(v_new) + eps))
        tl.store(exp_avg_ptr + offsets, m_new, mask=mask)
        tl.store(exp_avg_sq_ptr + offsets, v_new, mask=mask)
        tl.store(param_ptr + offsets, p_new, mask=mask)


class ShampooFamilyTrainingExecutor:
    """Launch narrow Shampoo-family CUDA canaries against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: ShampooFamilyTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("ShampooFamilyTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked(self.config.optimizer_kind, f"{self.config.optimizer_kind}_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked(self.config.optimizer_kind, f"{self.config.optimizer_kind}_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked(self.config.optimizer_kind, f"{self.config.optimizer_kind}_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(self.config.optimizer_kind, f"{self.config.optimizer_kind}_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked(self.config.optimizer_kind, f"{self.config.optimizer_kind}_maximize_not_supported")

        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config)
            blockers = _unsupported_group_blockers(self.config.optimizer_kind, group, group_cfg)
            if blockers:
                cases.append(_case_failed(blockers[0]))
                continue
            _ensure_plugin_state(self.optimizer, group)
            group["step"] = int(group.get("step", 0) or 0) + 1  # type: ignore[index]
            for param in group["params"]:
                if param.grad is None:
                    continue
                cases.append(self._step_param(param, group, group_cfg))

        blockers = _dedupe(reason for case in cases for reason in case.get("blocked_reasons", []) or [])
        if not cases:
            blockers.append(f"{self.config.optimizer_kind}_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": f"turbocore_plugin_{self.config.optimizer_kind}_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else f"{self.config.optimizer_kind}_native_step_failed"),
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
            return _case_failed(f"{self.config.optimizer_kind}_training_executor_requires_float32")
        if not param.is_contiguous() or param.grad is None or not param.grad.is_contiguous():
            return _case_failed(f"{self.config.optimizer_kind}_training_executor_requires_contiguous_tensors")
        if param.grad.is_sparse:
            return _case_failed(f"{self.config.optimizer_kind}_training_executor_sparse_grad_not_supported")
        kind = self.config.optimizer_kind
        if kind == "shampoo":
            return self._step_shampoo(param, group, group_cfg)
        if kind == "scalableshampoo":
            return self._step_scalable_shampoo(param, group, group_cfg)
        if kind == "soap":
            return self._step_soap(param, group, group_cfg)
        return _case_failed(f"{kind}_training_executor_optimizer_kind_unsupported")

    def _step_shampoo(self, param: torch.nn.Parameter, group: Mapping[str, Any], group_cfg: Mapping[str, Any]) -> dict[str, Any]:
        step = int(group.get("step", 0) or 0)
        if step != 1 or param.dim() != 1:
            return _case_failed("shampoo_training_executor_requires_first_step_1d_canary")
        state = self.optimizer.state[param]
        grad = param.grad.detach()
        pre_cond = state.setdefault("pre_cond_0", float(group_cfg["matrix_eps"]) * torch.eye(param.numel(), device=param.device, dtype=param.dtype))
        inv_pre_cond = state.setdefault("inv_pre_cond_0", torch.zeros_like(pre_cond))
        pre_cond.add_(grad.view(-1, 1) @ grad.view(1, -1))
        inv_pre_cond.copy_(_matrix_inverse_power_svd(pre_cond, float(param.dim())))
        update = grad.view(1, -1).matmul(inv_pre_cond).view_as(param).contiguous()
        state["momentum_buffer"] = update
        return _apply_update(kind="shampoo", param=param, update=update, lr=float(group_cfg["lr"]), block_size=int(group_cfg["block_size"]), state=state, step=step)

    def _step_scalable_shampoo(self, param: torch.nn.Parameter, group: Mapping[str, Any], group_cfg: Mapping[str, Any]) -> dict[str, Any]:
        step = int(group.get("step", 0) or 0)
        if step != 1:
            return _case_failed("scalableshampoo_training_executor_only_first_step_supported_for_canary")
        state = self.optimizer.state[param]
        grad = param.grad.detach()
        pre_conditioner = state.get("pre_conditioner")
        graft = state.get("graft")
        if pre_conditioner is not None:
            pre_conditioner.add_statistics(grad)
        if graft is not None:
            graft.add_statistics(grad, float(group_cfg["beta2"]))
            update = graft.update_momentum(grad, float(group_cfg["beta1"]))
        else:
            update = grad
        update = update.contiguous()
        state["momentum"].mul_(float(group_cfg["beta1"])).add_(grad)
        return _apply_update(kind="scalableshampoo", param=param, update=update, lr=float(group_cfg["lr"]), block_size=int(group_cfg["block_size"]), state=state, step=step)

    def _step_soap(self, param: torch.nn.Parameter, group: Mapping[str, Any], group_cfg: Mapping[str, Any]) -> dict[str, Any]:
        step = int(group.get("step", 0) or 0)
        if step != 2:
            return _case_failed("soap_training_executor_requires_warmed_second_step_canary")
        state = self.optimizer.state[param]
        q_values = state.get("Q")
        if not isinstance(q_values, list) or any(torch.is_tensor(item) for item in q_values):
            return _case_failed("soap_training_executor_requires_unprojected_1d_canary")
        exp_avg = _state_tensor(state, "exp_avg", param)
        exp_avg_sq = _state_tensor(state, "exp_avg_sq", param)
        beta1 = float(group_cfg["beta1"])
        beta2 = float(group_cfg["beta2"])
        step_size = float(group_cfg["lr"])
        if bool(group_cfg["correct_bias"]):
            step_size *= math.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)
        before = param.detach().clone()
        grid = (triton.cdiv(int(param.numel()), int(group_cfg["block_size"])),)
        _soap_second_step_kernel[grid](
            param,
            param.grad.detach(),
            exp_avg,
            exp_avg_sq,
            int(param.numel()),
            beta1,
            beta2,
            step_size,
            float(group_cfg["eps"]),
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        mutated = _max_abs_diff(before, param.detach()) > 0.0
        return _case(kind="soap", ok=mutated, param=param, state=state, step=step, extra={"step_size": step_size})


def build_plugin_shampoo_training_executor(**kwargs: Any) -> ShampooFamilyTrainingExecutor:
    return ShampooFamilyTrainingExecutor(**kwargs)


def build_plugin_scalableshampoo_training_executor(**kwargs: Any) -> ShampooFamilyTrainingExecutor:
    return ShampooFamilyTrainingExecutor(**kwargs)


def build_plugin_soap_training_executor(**kwargs: Any) -> ShampooFamilyTrainingExecutor:
    return ShampooFamilyTrainingExecutor(**kwargs)


def _normalize_config(value: ShampooFamilyTrainingExecutorConfig | Mapping[str, Any] | None, optimizer: torch.optim.Optimizer) -> ShampooFamilyTrainingExecutorConfig:
    if isinstance(value, ShampooFamilyTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = payload.get("betas", group.get("betas", (0.0, 0.999)))
    return ShampooFamilyTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or ""),
        lr=float(payload.get("lr", group.get("lr", 1.0e-3)) or 1.0e-3),
        beta1=float(betas[0]),
        beta2=float(betas[1]),
        eps=float(payload.get("eps", group.get("eps", 1.0e-8)) or 1.0e-8),
        matrix_eps=float(payload.get("matrix_eps", group.get("matrix_eps", 1.0e-6)) or 1.0e-6),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: ShampooFamilyTrainingExecutorConfig) -> dict[str, Any]:
    betas = group.get("betas", (config.beta1, config.beta2))
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "eps": float(group.get("eps", config.eps) or config.eps),
        "matrix_eps": float(group.get("matrix_eps", config.matrix_eps) or config.matrix_eps),
        "correct_bias": bool(group.get("correct_bias", True)),
        "block_size": int(config.block_size),
    }


def _unsupported_group_blockers(kind: str, group: Mapping[str, Any], group_cfg: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if abs(float(group.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append(f"{kind}_weight_decay_not_supported_for_canary")
    if kind == "shampoo" and abs(float(group.get("momentum", 0.0) or 0.0)) > 0.0:
        blockers.append("shampoo_momentum_not_supported_for_canary")
    if kind == "scalableshampoo":
        if bool(group.get("nesterov", False)):
            blockers.append("scalableshampoo_nesterov_not_supported_for_canary")
        if int(group.get("graft_type", 0) or 0) != 0:
            blockers.append("scalableshampoo_requires_no_graft_canary")
    if kind == "soap":
        if bool(group.get("precondition_1d", False)):
            blockers.append("soap_precondition_1d_not_supported_for_canary")
        if bool(group.get("merge_dims", False)):
            blockers.append("soap_merge_dims_not_supported_for_canary")
        if bool(group.get("normalize_gradient", False)):
            blockers.append("soap_normalize_gradient_not_supported_for_canary")
    return blockers


def _ensure_plugin_state(optimizer: torch.optim.Optimizer, group: Mapping[str, Any]) -> None:
    init_group = getattr(optimizer, "init_group", None)
    if callable(init_group):
        init_group(group)


def _matrix_inverse_power_svd(matrix: torch.Tensor, power: float) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(matrix.to(torch.float32), full_matrices=False)
    s.pow_(-1.0 / power)
    return (u @ s.diag() @ vh).to(matrix.dtype)


def _apply_update(kind: str, param: torch.nn.Parameter, update: torch.Tensor, lr: float, block_size: int, state: dict[str, Any], step: int) -> dict[str, Any]:
    before = param.detach().clone()
    grid = (triton.cdiv(int(param.numel()), int(block_size)),)
    _apply_update_kernel[grid](param, update, int(param.numel()), float(lr), BLOCK_SIZE=int(block_size), num_warps=4)
    mutated = _max_abs_diff(before, param.detach()) > 0.0
    return _case(kind=kind, ok=mutated, param=param, state=state, step=step)


def _state_tensor(state: dict[str, Any], key: str, param: torch.nn.Parameter) -> torch.Tensor:
    value = state.get(key)
    if torch.is_tensor(value) and tuple(value.shape) == tuple(param.shape):
        return value
    state[key] = torch.zeros_like(param.detach(), dtype=torch.float32)
    return state[key]


def _case(kind: str, ok: bool, param: torch.nn.Parameter, state: dict[str, Any], step: int, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
        **dict(extra or {}),
    }


def _case_failed(reason: str) -> dict[str, Any]:
    return {"schema_version": 1, "ok": False, "kernel_executed": False, "training_parameters_mutated": False, "blocked_reasons": [reason]}


def _blocked(kind: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "executor": f"turbocore_plugin_{kind}_training_executor_v0",
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


__all__ = [
    "ShampooFamilyTrainingExecutor",
    "ShampooFamilyTrainingExecutorConfig",
    "build_plugin_scalableshampoo_training_executor",
    "build_plugin_shampoo_training_executor",
    "build_plugin_soap_training_executor",
]
