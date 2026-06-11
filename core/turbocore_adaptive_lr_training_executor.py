"""Default-off training executor for adaptive-LR native canaries."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from core.services.native_module_loader import native_with_entrypoints


ENTRYPOINT = "probe_adaptive_lr_training_tensor_binding_canary_py"
REPO_ROOT = Path(__file__).resolve().parents[2]
KERNEL_KIND_BY_FAMILY = {
    "adaptive_lr_prodigy": "prodigy",
    "adaptive_lr_dadapt": "dadapt",
}


@dataclass(frozen=True)
class AdaptiveLrTrainingExecutorConfig:
    optimizer_kind: str = "prodigy"
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    global_d: float = 1.0
    dynamic_lr: float = 1.0
    max_numel: int = 1_048_576
    require_native_cuda: bool = True


class AdaptiveLrTrainingExecutor:
    """Launch adaptive-LR native steps against live toy TrainingLoop state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: AdaptiveLrTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("AdaptiveLrTrainingExecutor requires trainable parameters")
        self._param_ids = {id(param) for param in self.params}
        self.config = _normalize_config(config, optimizer)
        self.family = _family_for_kind(self.config.optimizer_kind)
        self.workspace_root = Path(workspace_root or REPO_ROOT)
        self._native: Any | None = None

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("adaptive_lr_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("adaptive_lr_training_executor_requires_cuda_params")
        native = self._load_native()
        if native is None:
            return _blocked("adaptive_lr_training_dispatch_entrypoint_missing")
        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config)
            for param in group["params"]:
                if id(param) not in self._param_ids or param.grad is None:
                    continue
                cases.append(self._step_param(native, param, group_cfg))
        blockers = _dedupe(reason for case in cases for reason in case.get("blocked_reasons", []) or [])
        if not cases:
            blockers.append("adaptive_lr_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_adaptive_lr_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "adaptive_lr_native_step_failed"),
            "optimizer_kind": self.config.optimizer_kind,
            "adaptive_lr_family": self.family,
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

    def _step_param(self, native: Any, param: torch.nn.Parameter, group_cfg: Mapping[str, Any]) -> dict[str, Any]:
        if param.dtype != torch.float32 or (param.grad is not None and param.grad.dtype != torch.float32):
            return _case_failed("adaptive_lr_native_step_requires_float32", "adaptive_lr_native_step_dtype_unsupported")
        state = self.optimizer.state[param]
        exp_avg = _state_tensor(state, param, ("exp_avg", "m1", "momentum"), fill=0.0)
        exp_avg_sq = _state_tensor(state, param, ("exp_avg_sq", "m2", "variance"), fill=0.0)
        adaptive_state = _adaptive_state_tensor(state, param, self.config)
        launch_config = {
            **dict(group_cfg),
            "numel": int(param.numel()),
            "global_d": float(adaptive_state[0].detach().item()),
            "dynamic_lr": float(adaptive_state[1].detach().item()),
            "max_numel": max(int(self.config.max_numel), int(param.numel())),
            "canary_probe_only": True,
            "training_tensor_binding": True,
            "training_dispatch": False,
            "training_path_enabled": False,
        }
        try:
            launch = dict(
                getattr(native, ENTRYPOINT)(
                    KERNEL_KIND_BY_FAMILY[self.family],
                    param,
                    param.grad.detach().contiguous(),
                    exp_avg,
                    exp_avg_sq,
                    adaptive_state,
                    json.dumps(launch_config),
                    str(self.workspace_root.resolve()),
                    _cuda_arch(param.device),
                )
            )
        except Exception as exc:  # pragma: no cover - native/CUDA dependent
            return _case_failed(f"{type(exc).__name__}: {exc}", "adaptive_lr_native_step_call_failed")
        if launch.get("ok") is not True:
            return _case_failed(str(launch.get("reason") or "adaptive_lr_native_step_failed"), "adaptive_lr_native_step_failed", launch)
        _write_state_tensor(state, param, exp_avg, ("exp_avg", "m1", "momentum"))
        _write_state_tensor(state, param, exp_avg_sq, ("exp_avg_sq", "m2", "variance"))
        state["adaptive_lr_native_state"] = adaptive_state.detach().clone()
        state["step"] = int(state.get("step", state.get("tick", 0)) or 0) + 1
        return {
            "schema_version": 1,
            "ok": True,
            "param_numel": int(param.numel()),
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "adaptive_lr_family": self.family,
            "step_after": int(state.get("step", 0) or 0),
            "kernel_executed": launch.get("kernel_executed") is True,
            "training_parameters_mutated": launch.get("parameters_mutated") is True,
            "launch": launch,
            "blocked_reasons": [],
        }


def build_adaptive_lr_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: AdaptiveLrTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> AdaptiveLrTrainingExecutor:
    return AdaptiveLrTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: AdaptiveLrTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> AdaptiveLrTrainingExecutorConfig:
    if isinstance(value, AdaptiveLrTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = payload.get("betas", group.get("betas", (0.9, 0.999)))
    global_d = payload.get("global_d", group.get("d", group.get("d0", 1.0)))
    return AdaptiveLrTrainingExecutorConfig(
        optimizer_kind=_normalize_kind(payload.get("optimizer_kind", "prodigy")),
        lr=float(payload.get("lr", group.get("lr", 1e-3)) or 1e-3),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(payload.get("eps", group.get("eps", 1e-8))),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0))),
        global_d=float(global_d or 1.0),
        dynamic_lr=float(payload.get("dynamic_lr", group.get("d_coef", 1.0)) or 1.0),
        max_numel=int(payload.get("max_numel", 1_048_576)),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: AdaptiveLrTrainingExecutorConfig) -> dict[str, Any]:
    betas = group.get("betas", config.betas)
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "eps": float(group.get("eps", config.eps)),
        "weight_decay": float(group.get("weight_decay", config.weight_decay)),
    }


def _state_tensor(
    state: Mapping[str, Any],
    param: torch.nn.Parameter,
    names: tuple[str, ...],
    *,
    fill: float,
) -> torch.Tensor:
    for name in names:
        value = state.get(name)
        if torch.is_tensor(value) and tuple(value.shape) == tuple(param.shape):
            return value.detach().to(device=param.device, dtype=torch.float32).contiguous()
    return torch.full_like(param.detach(), fill, dtype=torch.float32).contiguous()


def _write_state_tensor(state: dict[str, Any], param: torch.nn.Parameter, value: torch.Tensor, names: tuple[str, ...]) -> None:
    target = next((name for name in names if torch.is_tensor(state.get(name))), names[0])
    state[target] = value.detach().clone().reshape_as(param)


def _adaptive_state_tensor(
    state: Mapping[str, Any],
    param: torch.nn.Parameter,
    config: AdaptiveLrTrainingExecutorConfig,
) -> torch.Tensor:
    existing = state.get("adaptive_lr_native_state")
    if torch.is_tensor(existing) and int(existing.numel()) >= 4:
        return existing.detach().to(device=param.device, dtype=torch.float32).flatten()[:4].contiguous()
    return torch.tensor(
        [float(config.global_d), float(config.dynamic_lr), 0.0, 0.0],
        device=param.device,
        dtype=torch.float32,
    ).contiguous()


def _family_for_kind(kind: str) -> str:
    return "adaptive_lr_prodigy" if _normalize_kind(kind) == "prodigy" else "adaptive_lr_dadapt"


def _normalize_kind(value: Any) -> str:
    kind = str(value or "prodigy").strip().lower().replace("-", "_").replace(" ", "_")
    if kind in {"autoprodigy", "auto_prodigy", "prodigy", "prodigy_plus_schedule_free", "prodigyplusschedulefree"}:
        return "prodigy"
    if kind in {"dadapt", "dadaptation", "dadapt_adam_preprint", "dadaptadam", "dadapt_adam"}:
        return "dadapt"
    if kind.startswith("dadapt"):
        return "dadapt"
    return "prodigy"


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


def _case_failed(reason: str, blocker: str, launch: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "reason": reason,
        "kernel_executed": False,
        "training_parameters_mutated": False,
        "launch": dict(launch or {}),
        "blocked_reasons": [blocker],
    }


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "executor": "turbocore_adaptive_lr_training_executor_v0",
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
    "AdaptiveLrTrainingExecutor",
    "AdaptiveLrTrainingExecutorConfig",
    "build_adaptive_lr_training_executor",
]
