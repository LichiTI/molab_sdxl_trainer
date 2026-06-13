"""Default-off AdaMuon/AdaGO AdamW-fallback TrainingLoop executor for canaries."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from core.services.native_module_loader import native_with_entrypoints
from core.turbocore_native_tensor_binding import build_flat_adamw_native_binding_request
from core.turbocore_tensor_handle_registry import (
    TurboCoreTensorHandleRegistry,
    build_tensor_object_map_for_handles,
)


ENTRYPOINTS = (
    "create_flat_adamw_tensor_binding_session",
    "tensor_binding_session_cuda_adam_tensor_probe",
    "destroy_tensor_binding_session",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_KINDS = frozenset({"adamuon", "adago"})


@dataclass(frozen=True)
class PluginMuonFamilyAdamWFallbackTrainingExecutorConfig:
    optimizer_kind: str = "adamuon"
    lr: float = 3.0e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-10
    weight_decay: float = 0.0
    block_size: int = 128
    require_native_cuda: bool = True


class PluginMuonFamilyAdamWFallbackTrainingExecutor:
    """Launch AdaMuon/AdaGO use_muon=False AdamW fallback steps against live state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: PluginMuonFamilyAdamWFallbackTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("PluginMuonFamilyAdamWFallbackTrainingExecutor requires trainable parameters")
        self._param_ids = {id(param) for param in self.params}
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)
        self._native: Any | None = None

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked(self.config.optimizer_kind, "plugin_muon_family_adamw_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked(self.config.optimizer_kind, "plugin_muon_family_adamw_executor_requires_cuda_params")
        native = self._load_native()
        if native is None:
            return _blocked(self.config.optimizer_kind, "plugin_muon_family_adamw_dispatch_entrypoints_missing")
        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config)
            for param in group["params"]:
                if id(param) not in self._param_ids or param.grad is None:
                    continue
                cases.append(self._step_param(native, param, group, group_cfg))
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        blockers = _dedupe(reason for case in cases for reason in case.get("blocked_reasons", []) or [])
        if not cases:
            blockers.append("plugin_muon_family_adamw_executor_no_grad_params")
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_muon_family_adamw_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "plugin_muon_family_adamw_step_failed"),
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
            self._native = native_with_entrypoints(*ENTRYPOINTS)
        return self._native

    def _step_param(
        self,
        native: Any,
        param: torch.nn.Parameter,
        group: dict[str, Any],
        group_cfg: Mapping[str, Any],
    ) -> dict[str, Any]:
        kind = type(getattr(self.optimizer, "_base", self.optimizer)).__name__.lower()
        if kind != self.config.optimizer_kind or kind not in SUPPORTED_KINDS:
            return _case_failed("plugin_muon_family_adamw_optimizer_kind_unsupported")
        if group.get("use_muon") is not False:
            return _case_failed("plugin_muon_family_adamw_requires_non_muon_group")
        if param.ndim >= 2:
            return _case_failed("plugin_muon_family_adamw_requires_vector_param")
        if getattr(self.optimizer, "maximize", False):
            return _case_failed("plugin_muon_family_adamw_maximize_unsupported")
        if abs(float(group_cfg["weight_decay"])) > 0.0:
            return _case_failed("plugin_muon_family_adamw_weight_decay_unsupported")
        if param.dtype != torch.float32 or param.grad is None or param.grad.dtype != torch.float32:
            return _case_failed("plugin_muon_family_adamw_requires_float32")
        if not param.is_contiguous() or not param.grad.is_contiguous():
            return _case_failed("plugin_muon_family_adamw_requires_contiguous_param_grad")
        if torch.is_complex(param):
            return _case_failed("plugin_muon_family_adamw_complex_param_unsupported")

        state = self.optimizer.state[param]
        exp_avg, exp_avg_sq = _ensure_adam_state(state, param)
        if not exp_avg.is_contiguous() or not exp_avg_sq.is_contiguous():
            return _case_failed("plugin_muon_family_adamw_requires_contiguous_state")
        step_before = int(group.get("step", 0) or 0)
        registry = TurboCoreTensorHandleRegistry(namespace=f"plugin_{self.config.optimizer_kind}_adamw_executor")
        param_flat = param.detach().reshape(-1)
        grad_flat = param.grad.detach().reshape(-1)
        try:
            handles = registry.register_flat_adamw_buffers(
                param_flat=param_flat,
                grad_flat=grad_flat,
                exp_avg=exp_avg.reshape(-1),
                exp_avg_sq=exp_avg_sq.reshape(-1),
            )
            request = build_flat_adamw_native_binding_request(registry, handles)
            tensor_map = build_tensor_object_map_for_handles(registry, handles)
            session = dict(native.create_flat_adamw_tensor_binding_session(json.dumps(request), tensor_map))
            if session.get("ok") is not True:
                return _case_failed("plugin_muon_family_adamw_session_create_failed", session=session)
            session_id = int(session["session_id"])
            try:
                launch = dict(
                    native.tensor_binding_session_cuda_adam_tensor_probe(
                        session_id,
                        json.dumps(
                            {
                                **dict(group_cfg),
                                "step_index": step_before,
                                "block_size": int(self.config.block_size),
                                "max_numel": int(param_flat.numel()),
                                "training_tensor_binding": True,
                                "training_dispatch": False,
                                "training_path_enabled": False,
                            }
                        ),
                    )
                )
            finally:
                try:
                    native.destroy_tensor_binding_session(session_id)
                except Exception:
                    pass
        except Exception as exc:  # pragma: no cover - native/CUDA dependent
            return _case_failed(f"plugin_muon_family_adamw_call_failed:{type(exc).__name__}: {exc}")
        ok = bool(
            launch.get("ok") is True
            and launch.get("kernel_executed") is True
            and launch.get("parameters_mutated") is True
        )
        if ok:
            group["step"] = step_before + 1
        return {
            "schema_version": 1,
            "ok": ok,
            "param_numel": int(param.numel()),
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "use_muon": bool(group.get("use_muon")),
            "step_before": step_before,
            "step_after": int(group.get("step", 0) or 0),
            "kernel_executed": launch.get("kernel_executed") is True,
            "training_parameters_mutated": launch.get("parameters_mutated") is True,
            "state_keys": sorted(str(key) for key in state.keys()),
            "launch": launch,
            "blocked_reasons": [] if ok else _case_blockers(launch),
        }


def build_plugin_muon_family_adamw_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: PluginMuonFamilyAdamWFallbackTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> PluginMuonFamilyAdamWFallbackTrainingExecutor:
    return PluginMuonFamilyAdamWFallbackTrainingExecutor(
        optimizer=optimizer,
        params=params,
        config=config,
        workspace_root=workspace_root,
    )


def _normalize_config(
    value: PluginMuonFamilyAdamWFallbackTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> PluginMuonFamilyAdamWFallbackTrainingExecutorConfig:
    if isinstance(value, PluginMuonFamilyAdamWFallbackTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    kind = str(payload.get("optimizer_kind") or group.get("optimizer_kind") or type(optimizer).__name__).lower()
    betas = payload.get("betas", group.get("betas", (0.9, 0.999)))
    return PluginMuonFamilyAdamWFallbackTrainingExecutorConfig(
        optimizer_kind=kind,
        lr=float(payload.get("lr", group.get("lr", 3.0e-4))),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(payload.get("eps", group.get("eps", 1.0e-10))),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0)) or 0.0),
        block_size=int(payload.get("block_size", 128)),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(
    group: Mapping[str, Any],
    config: PluginMuonFamilyAdamWFallbackTrainingExecutorConfig,
) -> dict[str, Any]:
    betas = group.get("betas", config.betas)
    return {
        "lr": float(group.get("lr", config.lr)),
        "betas": [float(betas[0]), float(betas[1])],
        "eps": float(group.get("eps", config.eps)),
        "weight_decay": float(group.get("weight_decay", config.weight_decay) or 0.0),
    }


def _ensure_adam_state(state: dict[Any, Any], param: torch.nn.Parameter) -> tuple[torch.Tensor, torch.Tensor]:
    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    if not torch.is_tensor(exp_avg) or tuple(exp_avg.shape) != tuple(param.shape):
        exp_avg = torch.zeros_like(param.detach(), dtype=torch.float32).contiguous()
        state["exp_avg"] = exp_avg
    if not torch.is_tensor(exp_avg_sq) or tuple(exp_avg_sq.shape) != tuple(param.shape):
        exp_avg_sq = torch.zeros_like(param.detach(), dtype=torch.float32).contiguous()
        state["exp_avg_sq"] = exp_avg_sq
    return exp_avg, exp_avg_sq


def _case_blockers(launch: Mapping[str, Any]) -> list[str]:
    blockers = _strings(launch.get("blocked_reasons"))
    if not blockers and launch.get("reason"):
        blockers.append(str(launch.get("reason")))
    if launch.get("kernel_executed") is not True:
        blockers.append("plugin_muon_family_adamw_kernel_not_executed")
    if launch.get("parameters_mutated") is not True:
        blockers.append("plugin_muon_family_adamw_parameters_not_mutated")
    return _dedupe(blockers or ["plugin_muon_family_adamw_native_step_failed"])


def _case_failed(reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "reason": reason,
        "kernel_executed": False,
        "training_parameters_mutated": False,
        "blocked_reasons": [reason],
        **extra,
    }


def _blocked(optimizer_kind: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "executor": "turbocore_plugin_muon_family_adamw_training_executor_v0",
        "ok": False,
        "reason": reason,
        "optimizer_kind": optimizer_kind,
        "training_dispatch": True,
        "training_path_enabled": True,
        "native_step_executed": False,
        "native_kernel_launched": False,
        "training_parameters_mutated": False,
        "should_call_pytorch_optimizer_step": True,
        "blocked_reasons": [reason],
    }


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "PluginMuonFamilyAdamWFallbackTrainingExecutor",
    "PluginMuonFamilyAdamWFallbackTrainingExecutorConfig",
    "build_plugin_muon_family_adamw_training_executor",
]
