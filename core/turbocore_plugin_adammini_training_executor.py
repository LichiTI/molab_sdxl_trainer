"""Default-off AdamMini Triton TrainingLoop executor for selected plugin canaries."""

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
class AdamMiniTrainingExecutorConfig:
    optimizer_kind: str = "adammini"
    lr: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    eps: float = 1.0e-8
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _adammini_lefts_kernel(
        param_ptr,
        grad_ptr,
        m_ptr,
        n_elements,
        lr,
        beta1,
        bias_correction1,
        h,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        m_new = m * beta1 + grad * (1.0 - beta1)
        update = m_new * ((1.0 / bias_correction1) / h)
        param_new = param - lr * update

        tl.store(m_ptr + offsets, m_new, mask=mask)
        tl.store(param_ptr + offsets, param_new, mask=mask)


class AdamMiniTrainingExecutor:
    """Launch a real AdamMini step_lefts fp32 Triton kernel against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: AdamMiniTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("AdamMiniTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("adammini_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("adammini_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("adammini_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"adammini_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("adammini_maximize_not_supported")
        if bool(getattr(self.optimizer, "model_sharding", False)):
            return _blocked("adammini_model_sharding_not_supported_for_canary")
        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config)
            blockers = _unsupported_group_blockers(group, group_cfg)
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
            blockers.append("adammini_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_adammini_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "adammini_native_step_failed"),
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
            return _case_failed("adammini_training_executor_requires_float32")
        if not param.is_contiguous() or param.grad is None or not param.grad.is_contiguous():
            return _case_failed("adammini_training_executor_requires_contiguous_tensors")
        step = int(group.get("step", 0) or 0)
        if step != 1:
            return _case_failed("adammini_training_executor_only_first_step_supported_for_canary")
        state = self.optimizer.state[param]
        m = _state_tensor(state, "m", param)
        v_mean = _state_scalar(state, "v_mean", param)
        dimension = _dimension_scalar(state, param)
        if m is None or v_mean is None or dimension is None:
            return _case_failed("adammini_training_executor_requires_lefts_state")
        if bool(state.get("reduced", False)):
            return _case_failed("adammini_reduced_state_not_supported_for_canary")
        beta1 = float(group_cfg["beta1"])
        beta2 = float(group_cfg["beta2"])
        bias_correction1 = 1.0 - beta1**step
        bias_correction2_sq = math.sqrt(1.0 - beta2**step)
        if bias_correction1 <= 0.0 or bias_correction2_sq <= 0.0:
            return _case_failed("adammini_training_executor_invalid_bias_correction")
        tmp_lr = torch.sum(param.grad.detach().float() * param.grad.detach().float()) / dimension
        v_new = v_mean.detach().float() * beta2 + tmp_lr * (1.0 - beta2)
        h = float((torch.sqrt(v_new) / bias_correction2_sq + float(group_cfg["eps"])).detach().cpu().item())
        if not math.isfinite(h) or h <= 0.0:
            return _case_failed("adammini_training_executor_invalid_h")
        before = param.detach().clone()
        grid = (triton.cdiv(int(param.numel()), int(group_cfg["block_size"])),)
        _adammini_lefts_kernel[grid](
            param,
            param.grad.detach(),
            m,
            int(param.numel()),
            float(group_cfg["lr"]),
            beta1,
            bias_correction1,
            h,
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        v_mean.copy_(v_new.reshape_as(v_mean))
        state["m"] = m
        state["v_mean"] = v_mean
        state["dimension"] = dimension
        state["reduced"] = False
        mutated = _max_abs_diff(before, param.detach()) > 0.0
        finite = torch.isfinite(param).all().item() and torch.isfinite(m).all().item() and torch.isfinite(v_mean).all().item()
        return {
            "schema_version": 1,
            "ok": mutated and finite,
            "param_shape": [int(dim) for dim in param.shape],
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_after": step,
            "kernel_executed": True,
            "training_parameters_mutated": bool(mutated),
            "state_keys": sorted(str(key) for key in state.keys()),
            "v_mean": float(v_mean.detach().cpu().item()),
            "blocked_reasons": [] if mutated else ["adammini_training_executor_parameters_not_mutated"],
        }


def build_plugin_adammini_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: AdamMiniTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> AdamMiniTrainingExecutor:
    return AdamMiniTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: AdamMiniTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> AdamMiniTrainingExecutorConfig:
    if isinstance(value, AdamMiniTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = payload.get("betas", group.get("betas", (0.9, 0.999)))
    if not isinstance(betas, (tuple, list)):
        betas = (0.9, 0.999)
    return AdamMiniTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "adammini"),
        lr=float(payload.get("lr", group.get("lr", 1.0)) or 1.0),
        beta1=float(betas[0] if len(betas) > 0 else 0.9),
        beta2=float(betas[1] if len(betas) > 1 else 0.999),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0)) or 0.0),
        eps=float(payload.get("eps", group.get("eps", 1.0e-8)) or 1.0e-8),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: AdamMiniTrainingExecutorConfig) -> dict[str, Any]:
    betas = group.get("betas", (config.beta1, config.beta2))
    if not isinstance(betas, (tuple, list)):
        betas = (config.beta1, config.beta2)
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "beta1": float(betas[0] if len(betas) > 0 else config.beta1),
        "beta2": float(betas[1] if len(betas) > 1 else config.beta2),
        "weight_decay": float(group.get("weight_decay", config.weight_decay) or config.weight_decay),
        "eps": float(group.get("eps", config.eps) or config.eps),
        "block_size": int(config.block_size),
    }


def _unsupported_group_blockers(group: Mapping[str, Any], group_cfg: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    name = str(group.get("name") or "")
    if any(block in name for block in ("embed", "embd", "wte", "lm_head.weight", "output.weight")):
        blockers.append("adammini_embed_branch_not_supported_for_canary")
    if any(block in name for block in ("k_proj.weight", "q_proj.weight", "wq.weight", "wk.weight")):
        blockers.append("adammini_attn_proj_branch_not_supported_for_canary")
    if "attn.attn.weight" in name or "attn.qkv.weight" in name:
        blockers.append("adammini_attn_branch_not_supported_for_canary")
    if abs(float(group_cfg.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("adammini_weight_decay_not_supported_for_canary")
    beta1 = float(group_cfg.get("beta1", 0.0) or 0.0)
    beta2 = float(group_cfg.get("beta2", 0.0) or 0.0)
    if beta1 < 0.0 or beta1 >= 1.0 or beta2 < 0.0 or beta2 >= 1.0:
        blockers.append("adammini_betas_out_of_range")
    return blockers


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


def _state_scalar(state: dict[str, Any], key: str, param: torch.nn.Parameter) -> torch.Tensor | None:
    value = state.get(key)
    if value is None:
        value = torch.zeros((), device=param.device, dtype=torch.float32)
        state[key] = value
    if not torch.is_tensor(value) or tuple(value.shape) != () or value.device != param.device:
        return None
    if value.dtype != torch.float32:
        return None
    return value


def _dimension_scalar(state: dict[str, Any], param: torch.nn.Parameter) -> torch.Tensor | None:
    value = state.get("dimension")
    if value is None:
        value = torch.tensor(param.numel(), device=param.device, dtype=torch.float32)
        state["dimension"] = value
    if not torch.is_tensor(value) or tuple(value.shape) != () or value.device != param.device:
        return None
    return value.to(dtype=torch.float32)


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
        "executor": "turbocore_plugin_adammini_training_executor_v0",
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
    "AdamMiniTrainingExecutor",
    "AdamMiniTrainingExecutorConfig",
    "build_plugin_adammini_training_executor",
]
