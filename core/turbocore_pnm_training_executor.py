"""Default-off PNM native TrainingLoop executor."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from core.services.native_module_loader import native_with_entrypoints


ENTRYPOINT = "probe_pnm_training_tensor_binding_canary_py"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PNMTrainingExecutorConfig:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 1.0
    weight_decay: float = 0.0
    weight_decouple: bool = True
    fixed_decay: bool = False
    maximize: bool = False
    require_native_cuda: bool = True


class PNMTrainingExecutor:
    """Launch a narrow PNM native canary against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: PNMTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("PNMTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)
        self._native: Any | None = None

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not bool(request.get("training_dispatch", False)) or not bool(request.get("training_path_enabled", False)):
            return _blocked("pnm_training_executor_requires_training_dispatch")
        contract = self._state_contract()
        if contract.get("ok") is not True:
            return _blocked(str(contract.get("reason", "pnm_state_contract_missing")), state_contract=contract)
        native = self._load_native()
        if native is None:
            return _blocked("pnm_training_dispatch_entrypoint_missing")
        param = self.params[0]
        state = self.optimizer.state[param]
        grad = param.grad.detach().float().contiguous()
        pos = _state_tensor(state, "pos_momentum", param)
        neg = _state_tensor(state, "neg_momentum", param)
        step_index = int(self.optimizer.param_groups[0].get("step", 0) or 0) + 1
        started = time.perf_counter()
        try:
            launch = dict(
                getattr(native, ENTRYPOINT)(
                    param,
                    grad,
                    pos,
                    neg,
                    int(param.numel()),
                    float(self.config.lr),
                    float(self.config.beta1),
                    float(self.config.beta2),
                    float(self.config.weight_decay),
                    int(step_index),
                    str(self.workspace_root.resolve()),
                    _cuda_arch(param.device),
                )
            )
        except Exception as exc:  # pragma: no cover - native/CUDA dependent
            return _blocked("pnm_native_step_call_failed", error=f"{type(exc).__name__}: {exc}")
        ok = bool(
            launch.get("ok") is True
            and launch.get("kernel_executed") is True
            and launch.get("training_parameters_mutated") is True
            and launch.get("native_live_tensor_binding") is True
            and launch.get("training_dispatch") is False
            and launch.get("training_path_enabled") is False
        )
        if not ok:
            return _blocked("pnm_native_step_failed", launch=launch)
        state["pos_momentum"] = pos
        state["neg_momentum"] = neg
        self.optimizer.param_groups[0]["step"] = step_index
        return {
            "schema_version": 1,
            "executor": "turbocore_pnm_training_executor_v0",
            "ok": True,
            "reason": "called",
            "optimizer_kind": "pnm",
            "training_dispatch": True,
            "training_path_enabled": True,
            "native_step_executed": True,
            "native_kernel_launched": True,
            "training_parameters_mutated": True,
            "should_call_pytorch_optimizer_step": False,
            "pytorch_optimizer_state_synced": True,
            "step_after_native": step_index,
            "launch": launch,
            "timing": {"elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4)},
            "blocked_reasons": [],
        }

    def close(self) -> None:
        return None

    def _load_native(self) -> Any | None:
        if self._native is None:
            self._native = native_with_entrypoints(ENTRYPOINT)
        return self._native

    def _state_contract(self) -> dict[str, Any]:
        if len(self.params) != 1:
            return {"ok": False, "reason": "pnm_representative_executor_requires_single_param"}
        param = self.params[0]
        group = self.optimizer.param_groups[0] if self.optimizer.param_groups else {}
        if self.config.require_native_cuda and param.device.type != "cuda":
            return {"ok": False, "reason": "cuda_required_for_pnm_training_executor"}
        if param.dtype != torch.float32 or not param.is_contiguous():
            return {"ok": False, "reason": "pnm_executor_requires_contiguous_fp32_param"}
        if param.grad is None or param.grad.dtype != torch.float32:
            return {"ok": False, "reason": "pnm_executor_requires_fp32_grad"}
        if bool(group.get("weight_decouple", self.config.weight_decouple)) is not True:
            return {"ok": False, "reason": "pnm_coupled_weight_decay_not_supported"}
        if bool(group.get("fixed_decay", self.config.fixed_decay)):
            return {"ok": False, "reason": "pnm_fixed_decay_not_supported"}
        if bool(getattr(self.optimizer, "maximize", self.config.maximize)):
            return {"ok": False, "reason": "pnm_maximize_not_supported"}
        state = self.optimizer.state.get(param, {})
        if not torch.is_tensor(state.get("pos_momentum")) or not torch.is_tensor(state.get("neg_momentum")):
            return {"ok": False, "reason": "pnm_momentum_state_missing"}
        return {"ok": True}


def build_pnm_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: PNMTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> PNMTrainingExecutor:
    return PNMTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(value: PNMTrainingExecutorConfig | Mapping[str, Any] | None, optimizer: torch.optim.Optimizer) -> PNMTrainingExecutorConfig:
    if isinstance(value, PNMTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = payload.get("betas", group.get("betas", (0.9, 1.0)))
    return PNMTrainingExecutorConfig(
        lr=float(payload.get("lr", group.get("lr", 1e-3)) or 1e-3),
        beta1=float(betas[0]),
        beta2=float(betas[1]),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0)) or 0.0),
        weight_decouple=bool(payload.get("weight_decouple", group.get("weight_decouple", True))),
        fixed_decay=bool(payload.get("fixed_decay", group.get("fixed_decay", False))),
        maximize=bool(payload.get("maximize", getattr(optimizer, "maximize", False))),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _state_tensor(state: dict[str, Any], key: str, param: torch.nn.Parameter) -> torch.Tensor:
    value = state.get(key)
    if torch.is_tensor(value) and value.device == param.device and value.dtype == torch.float32 and value.is_contiguous():
        return value
    converted = torch.zeros_like(param.detach(), dtype=torch.float32) if not torch.is_tensor(value) else value.detach().to(device=param.device, dtype=torch.float32).contiguous()
    state[key] = converted
    return converted


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


def _blocked(reason: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "executor": "turbocore_pnm_training_executor_v0",
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


__all__ = ["PNMTrainingExecutor", "PNMTrainingExecutorConfig", "build_pnm_training_executor"]
