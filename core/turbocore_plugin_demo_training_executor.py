"""Default-off DeMo Triton TrainingLoop executor for selected plugin canaries."""

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
class DemoTrainingExecutorConfig:
    optimizer_kind: str = "demo"
    lr: float = 1.0e-3
    compression_decay: float = 0.0
    compression_top_k: int = 64
    compression_chunk: int = 64
    weight_decay: float = 0.0
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _demo_full_topk_sign_sgd_kernel(
        param_ptr,
        grad_ptr,
        delta_ptr,
        n_elements,
        lr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sign = tl.where(grad > 0.0, 1.0, tl.where(grad < 0.0, -1.0, 0.0))
        tl.store(param_ptr + offsets, param - lr * sign, mask=mask)
        tl.store(delta_ptr + offsets, 0.0, mask=mask)


class DemoTrainingExecutor:
    """Launch a real DeMo full-top-k first-step fp32 Triton kernel."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: DemoTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("DemoTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("demo_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("demo_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("demo_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"demo_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("demo_maximize_not_supported")
        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config)
            blockers = _unsupported_group_blockers(group_cfg, self.params[0])
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
            blockers.append("demo_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_demo_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "demo_native_step_failed"),
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
        if type(getattr(self.optimizer, "_base", self.optimizer)).__name__.lower() != "demo":
            return _case_failed("demo_training_executor_optimizer_kind_unsupported")
        if param.ndim != 1:
            return _case_failed("demo_training_executor_requires_1d_param")
        if param.dtype != torch.float32 or param.grad is None or param.grad.dtype != torch.float32:
            return _case_failed("demo_training_executor_requires_float32")
        if not param.is_contiguous() or not param.grad.is_contiguous():
            return _case_failed("demo_training_executor_requires_contiguous_tensors")
        step = int(group.get("step", 0) or 0)
        if step != 1:
            return _case_failed("demo_training_executor_only_first_step_supported_for_canary")
        demo_state = getattr(self.optimizer, "demo_state", None)
        if not isinstance(demo_state, dict):
            return _case_failed("demo_training_executor_demo_state_missing")
        state = demo_state.setdefault(param, {})
        delta = state.get("delta")
        if delta is None:
            delta = torch.zeros_like(param)
            state["delta"] = delta
        if not torch.is_tensor(delta) or tuple(delta.shape) != tuple(param.shape):
            return _case_failed("demo_training_executor_delta_state_missing")
        if delta.dtype != torch.float32 or not delta.is_contiguous():
            return _case_failed("demo_training_executor_requires_contiguous_float32_delta")
        before = param.detach().clone()
        grid = (triton.cdiv(int(param.numel()), int(group_cfg["block_size"])),)
        _demo_full_topk_sign_sgd_kernel[grid](
            param,
            param.grad.detach(),
            delta,
            int(param.numel()),
            float(group_cfg["lr"]),
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        nbytes = int(param.numel()) * (8 + 4)
        setattr(self.optimizer, "data_transmit", nbytes)
        setattr(self.optimizer, "data_receive", nbytes)
        param.grad.copy_(param.grad.sign())
        mutated = _max_abs_diff(before, param.detach()) > 0.0
        return {
            "schema_version": 1,
            "ok": mutated and torch.isfinite(param).all().item() and torch.isfinite(delta).all().item(),
            "param_shape": [int(dim) for dim in param.shape],
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_after": step,
            "kernel_executed": True,
            "training_parameters_mutated": bool(mutated),
            "state_keys": sorted(str(key) for key in state.keys()),
            "data_transmit": int(getattr(self.optimizer, "data_transmit", 0) or 0),
            "data_receive": int(getattr(self.optimizer, "data_receive", 0) or 0),
            "blocked_reasons": [] if mutated else ["demo_training_executor_parameters_not_mutated"],
        }


def build_plugin_demo_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: DemoTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> DemoTrainingExecutor:
    return DemoTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: DemoTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> DemoTrainingExecutorConfig:
    if isinstance(value, DemoTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    return DemoTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "demo"),
        lr=float(payload.get("lr", group.get("lr", 1.0e-3)) or 1.0e-3),
        compression_decay=float(payload.get("compression_decay", getattr(optimizer, "compression_decay", 0.0)) or 0.0),
        compression_top_k=int(payload.get("compression_top_k", getattr(optimizer, "compression_top_k", 64)) or 64),
        compression_chunk=int(payload.get("compression_chunk", getattr(optimizer, "compression_chunk", 64)) or 64),
        weight_decay=float(payload.get("weight_decay", getattr(optimizer, "weight_decay", 0.0)) or 0.0),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: DemoTrainingExecutorConfig) -> dict[str, Any]:
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "compression_decay": float(config.compression_decay),
        "compression_top_k": int(config.compression_top_k),
        "compression_chunk": int(config.compression_chunk),
        "weight_decay": float(config.weight_decay),
        "block_size": int(config.block_size),
    }


def _unsupported_group_blockers(group_cfg: Mapping[str, Any], param: torch.nn.Parameter) -> list[str]:
    blockers: list[str] = []
    if abs(float(group_cfg.get("compression_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("demo_compression_decay_not_supported_for_canary")
    if int(group_cfg.get("compression_top_k", 0) or 0) != int(param.numel()):
        blockers.append("demo_requires_full_top_k_canary")
    if int(group_cfg.get("compression_chunk", 0) or 0) != int(param.numel()):
        blockers.append("demo_requires_single_chunk_canary")
    if abs(float(group_cfg.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("demo_weight_decay_not_supported_for_canary")
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
        "executor": "turbocore_plugin_demo_training_executor_v0",
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


__all__ = ["DemoTrainingExecutor", "DemoTrainingExecutorConfig", "build_plugin_demo_training_executor"]
