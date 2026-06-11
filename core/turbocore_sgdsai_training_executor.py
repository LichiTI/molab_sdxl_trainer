"""Default-off SGDSaI native TrainingLoop executor."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from core.services.native_module_loader import native_with_entrypoints


ENTRYPOINTS = (
    "create_simple_optimizer_cuda_kernel_runtime_session_py",
    "step_simple_optimizer_cuda_kernel_runtime_session_py",
    "destroy_simple_optimizer_cuda_kernel_runtime_session_py",
)
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SGDSaITrainingExecutorConfig:
    lr: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 1e-2
    block_size: int = 128


class SGDSaITrainingExecutor:
    """Launch the SGDSaI flat fp32 kernel for a single representative route."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: SGDSaITrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("SGDSaITrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config)
        self.workspace_root = Path(workspace_root or REPO_ROOT)
        self.layout = _layout(self.params)
        self.param_flat = _flatten_tensors([param.detach() for param in self.params])
        self.grad_flat = torch.zeros_like(self.param_flat)
        self.momentum_flat = torch.zeros_like(self.param_flat)
        self._views = _views(self.param_flat, self.layout)
        self._momentum_views = _views(self.momentum_flat, self.layout)
        self._runtime_id: int | None = None
        self._native: Any | None = None

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not bool(request.get("training_dispatch", False)) or not bool(request.get("training_path_enabled", False)):
            return _blocked("sgdsai_training_executor_requires_training_dispatch")
        if self.param_flat.device.type != "cuda":
            return _blocked("cuda_required_for_sgdsai_training_executor")
        state = self._state_contract()
        if state.get("ok") is not True:
            return _blocked(str(state.get("reason", "sgdsai_state_contract_missing")), state_contract=state)
        started = time.perf_counter()
        self._sync_params_grads_and_state()
        native = self._load_native()
        if native is None:
            return _blocked("sgdsai_runtime_entrypoints_missing")
        runtime_id, create_report = self._ensure_runtime(native)
        if runtime_id is None:
            return _blocked("sgdsai_runtime_session_unavailable", create_report=create_report)
        try:
            payload = native.step_simple_optimizer_cuda_kernel_runtime_session_py(
                int(runtime_id),
                self.param_flat,
                self.grad_flat,
                self.momentum_flat,
                json.dumps(self._launch_config(float(state["gsnr"]))),
            )
        except Exception as exc:  # pragma: no cover - native/CUDA dependent
            return _blocked("sgdsai_native_step_call_failed", error=f"{type(exc).__name__}: {exc}")
        step_report = dict(payload) if isinstance(payload, Mapping) else {"ok": False, "reason": "invalid_native_step_payload"}
        if step_report.get("ok") is not True:
            return _blocked(str(step_report.get("reason", "sgdsai_native_step_failed")), step_report=step_report)
        _copy_flat_to_params(self.params, self._views)
        self.sync_optimizer_state_to_pytorch(reason="native_sgdsai_step")
        return {
            "schema_version": 1,
            "executor": "turbocore_sgdsai_training_executor_v0",
            "ok": True,
            "reason": "called",
            "optimizer_kind": "sgdsai",
            "training_dispatch": True,
            "training_path_enabled": True,
            "native_step_executed": True,
            "native_kernel_launched": step_report.get("kernel_executed") is True,
            "training_parameters_mutated": step_report.get("parameters_mutated") is True,
            "should_call_pytorch_optimizer_step": False,
            "pytorch_optimizer_state_synced": True,
            "step_report": step_report,
            "timing": {"elapsed_ms": _elapsed_ms(started)},
            "blocked_reasons": [],
        }

    def sync_optimizer_state_to_pytorch(self, *, reason: str) -> dict[str, Any]:
        with torch.no_grad():
            for param, view in zip(self.params, self._momentum_views):
                self.optimizer.state[param]["momentum_buffer"] = view.to(device=param.device, dtype=param.dtype).clone()
            for group in self.optimizer.param_groups:
                group["step"] = int(group.get("step", 0) or 0) + 1
        return {"schema_version": 1, "ok": True, "reason": reason, "optimizer_kind": "sgdsai"}

    def close(self) -> None:
        if self._native is not None and self._runtime_id is not None:
            try:
                self._native.destroy_simple_optimizer_cuda_kernel_runtime_session_py(int(self._runtime_id))
            except Exception:
                pass
        self._runtime_id = None

    def _state_contract(self) -> dict[str, Any]:
        if len(self.params) != 1:
            return {"ok": False, "reason": "sgdsai_representative_executor_requires_single_param"}
        group = self.optimizer.param_groups[0] if self.optimizer.param_groups else {}
        if bool(getattr(self.optimizer, "maximize", False)):
            return {"ok": False, "reason": "sgdsai_maximize_not_supported"}
        if group.get("weight_decouple", True) is not True:
            return {"ok": False, "reason": "sgdsai_coupled_weight_decay_not_supported"}
        state = self.optimizer.state.get(self.params[0], {})
        gsnr = state.get("gsnr")
        if gsnr is None:
            return {"ok": False, "reason": "sgdsai_warmup_gsnr_missing"}
        return {"ok": True, "gsnr": float(gsnr.detach().float().item() if torch.is_tensor(gsnr) else gsnr)}

    def _sync_params_grads_and_state(self) -> None:
        self.param_flat.copy_(_flatten_tensors([param.detach() for param in self.params]))
        grads = [param.grad if param.grad is not None else torch.zeros_like(param) for param in self.params]
        self.grad_flat.copy_(_flatten_tensors(grads))
        states = [self.optimizer.state[param].get("momentum_buffer", torch.zeros_like(param)) for param in self.params]
        self.momentum_flat.copy_(_flatten_tensors(states))

    def _load_native(self) -> Any | None:
        if self._native is None:
            self._native = native_with_entrypoints(*ENTRYPOINTS)
        return self._native

    def _ensure_runtime(self, native: Any) -> tuple[int | None, dict[str, Any]]:
        if self._runtime_id is not None:
            return int(self._runtime_id), {"ok": True, "runtime_session_id": int(self._runtime_id)}
        try:
            created = native.create_simple_optimizer_cuda_kernel_runtime_session_py(
                "sgdsai",
                str(self.workspace_root),
                _cuda_arch(self.param_flat.device),
            )
        except Exception as exc:  # pragma: no cover - native/CUDA dependent
            return None, {"ok": False, "reason": "sgdsai_runtime_create_failed", "error": f"{type(exc).__name__}: {exc}"}
        report = dict(created) if isinstance(created, Mapping) else {"ok": False, "reason": "invalid_runtime_create_payload"}
        if report.get("ok") is not True:
            return None, report
        self._runtime_id = int(report.get("runtime_session_id", 0) or 0)
        return int(self._runtime_id), report

    def _launch_config(self, gsnr: float) -> dict[str, Any]:
        return {
            "lr": float(self.config.lr),
            "momentum": float(self.config.momentum),
            "weight_decay": float(self.config.weight_decay),
            "alpha": float(gsnr),
            "block_size": int(self.config.block_size),
            "max_numel": int(self.param_flat.numel()),
            "training_dispatch": True,
            "training_path_enabled": True,
        }


def build_sgdsai_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: SGDSaITrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> SGDSaITrainingExecutor:
    return SGDSaITrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(value: SGDSaITrainingExecutorConfig | Mapping[str, Any] | None) -> SGDSaITrainingExecutorConfig:
    if isinstance(value, SGDSaITrainingExecutorConfig):
        return value
    payload = dict(value or {})
    return SGDSaITrainingExecutorConfig(
        lr=float(payload.get("lr", 1e-2)),
        momentum=float(payload.get("momentum", 0.9)),
        weight_decay=float(payload.get("weight_decay", 1e-2)),
        block_size=int(payload.get("block_size", 128)),
    )


def _layout(params: list[torch.Tensor]) -> list[tuple[tuple[int, ...], int, int]]:
    layout: list[tuple[tuple[int, ...], int, int]] = []
    offset = 0
    for param in params:
        count = int(param.numel())
        layout.append((tuple(int(dim) for dim in param.shape), offset, count))
        offset += count
    return layout


def _views(flat: torch.Tensor, layout: list[tuple[tuple[int, ...], int, int]]) -> list[torch.Tensor]:
    return [flat[offset : offset + count].view(shape) for shape, offset, count in layout]


def _flatten_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.detach().float().reshape(-1) for tensor in tensors]).contiguous()


def _copy_flat_to_params(params: list[torch.nn.Parameter], views: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, view in zip(params, views):
            param.copy_(view.to(device=param.device, dtype=param.dtype))


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 4)


def _blocked(reason: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "executor": "turbocore_sgdsai_training_executor_v0",
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
    payload.update(extra)
    return payload


__all__ = ["SGDSaITrainingExecutor", "SGDSaITrainingExecutorConfig", "build_sgdsai_training_executor"]
