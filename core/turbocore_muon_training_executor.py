"""Default-off training executor for built-in Muon native canaries."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from core.services.native_module_loader import native_with_entrypoints


ENTRYPOINT = "probe_muon_training_tensor_binding_canary_py"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MuonTrainingExecutorConfig:
    optimizer_kind: str = "muon"
    lr: float = 2.0e-2
    momentum: float = 0.95
    ns_steps: int = 5
    nesterov: bool = True
    weight_decay: float = 0.0
    require_native_cuda: bool = True


class MuonTrainingExecutor:
    """Launch Muon native steps against explicit TrainingLoop canary state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: MuonTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("MuonTrainingExecutor requires trainable parameters")
        self._param_ids = {id(param) for param in self.params}
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)
        self._native: Any | None = None

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("muon_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("muon_training_executor_requires_cuda_params")
        native = self._load_native()
        if native is None:
            return _blocked("muon_training_dispatch_entrypoint_missing")
        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config)
            for param in group["params"]:
                if id(param) not in self._param_ids or param.grad is None:
                    continue
                cases.append(self._step_param(native, param, group, group_cfg))
        blockers = _dedupe(reason for case in cases for reason in case.get("blocked_reasons", []) or [])
        if not cases:
            blockers.append("muon_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_muon_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "muon_native_step_failed"),
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

    def _load_native(self) -> Any | None:
        if self._native is None:
            self._native = native_with_entrypoints(ENTRYPOINT)
        return self._native

    def _step_param(
        self,
        native: Any,
        param: torch.nn.Parameter,
        group: Mapping[str, Any],
        group_cfg: Mapping[str, Any],
    ) -> dict[str, Any]:
        if group.get("use_muon", True) is not True:
            return _case_failed("muon_training_executor_non_muon_group_present")
        if param.ndim != 2:
            return _case_failed("muon_training_executor_requires_2d_param")
        if param.dtype != torch.float32 or (param.grad is not None and param.grad.dtype != torch.float32):
            return _case_failed("muon_training_executor_requires_float32")
        if not param.is_contiguous():
            return _case_failed("muon_training_executor_requires_contiguous_param")
        state = self.optimizer.state[param]
        momentum_buffer = _momentum_buffer(state, param)
        try:
            launch = dict(
                getattr(native, ENTRYPOINT)(
                    param,
                    param.grad.detach().contiguous(),
                    momentum_buffer,
                    int(param.shape[0]),
                    int(param.shape[1]),
                    float(group_cfg["lr"]),
                    float(group_cfg["momentum"]),
                    int(group_cfg["ns_steps"]),
                    bool(group_cfg["nesterov"]),
                    str(self.workspace_root.resolve()),
                    _cuda_arch(param.device),
                )
            )
        except Exception as exc:  # pragma: no cover - native/CUDA dependent
            return _case_failed(f"muon_native_step_call_failed:{type(exc).__name__}: {exc}")
        ok = bool(
            launch.get("ok") is True
            and launch.get("kernel_executed") is True
            and launch.get("training_parameters_mutated") is True
            and launch.get("native_live_tensor_binding") is True
            and launch.get("training_dispatch") is False
            and launch.get("training_path_enabled") is False
        )
        if ok:
            state["momentum_buffer"] = momentum_buffer
            state["step"] = int(state.get("step", 0) or 0) + 1
        return {
            "schema_version": 1,
            "ok": ok,
            "param_shape": [int(dim) for dim in param.shape],
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_after": int(state.get("step", 0) or 0),
            "kernel_executed": launch.get("kernel_executed") is True,
            "training_parameters_mutated": launch.get("training_parameters_mutated") is True,
            "native_live_tensor_binding": launch.get("native_live_tensor_binding") is True,
            "state_keys": sorted(str(key) for key in state.keys()),
            "launch": launch,
            "blocked_reasons": [] if ok else _case_blockers(launch),
        }


def build_muon_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: MuonTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> MuonTrainingExecutor:
    return MuonTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: MuonTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> MuonTrainingExecutorConfig:
    if isinstance(value, MuonTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    return MuonTrainingExecutorConfig(
        optimizer_kind="muon",
        lr=float(payload.get("lr", group.get("lr", 2.0e-2)) or 2.0e-2),
        momentum=float(payload.get("momentum", group.get("momentum", 0.95)) or 0.95),
        ns_steps=int(payload.get("ns_steps", group.get("ns_steps", 5)) or 5),
        nesterov=bool(payload.get("nesterov", group.get("nesterov", True))),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0)) or 0.0),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: MuonTrainingExecutorConfig) -> dict[str, Any]:
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "momentum": float(group.get("momentum", config.momentum) or config.momentum),
        "ns_steps": int(group.get("ns_steps", config.ns_steps) or config.ns_steps),
        "nesterov": bool(group.get("nesterov", config.nesterov)),
    }


def _momentum_buffer(state: dict[str, Any], param: torch.nn.Parameter) -> torch.Tensor:
    existing = state.get("momentum_buffer")
    if torch.is_tensor(existing) and tuple(existing.shape) == tuple(param.shape):
        if existing.device == param.device and existing.dtype == torch.float32 and existing.is_contiguous():
            return existing
        converted = existing.detach().to(device=param.device, dtype=torch.float32).contiguous()
        state["momentum_buffer"] = converted
        return converted
    state["momentum_buffer"] = torch.zeros_like(param.detach(), dtype=torch.float32).contiguous()
    return state["momentum_buffer"]


def _case_blockers(launch: Mapping[str, Any]) -> list[str]:
    blockers = _strings(launch.get("blocked_reasons"))
    if not blockers and launch.get("reason"):
        blockers.append(str(launch.get("reason")))
    if launch.get("kernel_executed") is not True:
        blockers.append("muon_training_executor_kernel_not_executed")
    if launch.get("training_parameters_mutated") is not True:
        blockers.append("muon_training_executor_parameters_not_mutated")
    return _dedupe(blockers or ["muon_training_executor_native_step_failed"])


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
        "executor": "turbocore_muon_training_executor_v0",
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


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["MuonTrainingExecutor", "MuonTrainingExecutorConfig", "build_muon_training_executor"]
