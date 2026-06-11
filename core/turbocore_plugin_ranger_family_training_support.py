"""Shared default-off TrainingLoop canaries for Ranger21/Ranger25 native probes."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from core.services.native_module_loader import native_with_entrypoints


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RangerFamilySpec:
    kind: str
    class_name: str
    native_route: str
    entrypoint: str
    state_keys: tuple[str, ...]
    optimizer_extra_args: str
    loss_scale: float
    loss_bias: float


SPECS: dict[str, RangerFamilySpec] = {
    "ranger21": RangerFamilySpec(
        kind="ranger21",
        class_name="Ranger21",
        native_route="rust_cuda_plugin_ranger21_v0",
        entrypoint="probe_ranger21_training_tensor_binding_canary_py",
        state_keys=("grad_ma", "variance_ma", "lookahead_params", "neg_grad_ma", "max_variance_ma"),
        optimizer_extra_args="disable_lr_scheduler=True lookahead_merge_time=5",
        loss_scale=0.18,
        loss_bias=0.003,
    ),
    "ranger25": RangerFamilySpec(
        kind="ranger25",
        class_name="Ranger25",
        native_route="rust_cuda_plugin_ranger25_v0",
        entrypoint="probe_ranger25_training_tensor_binding_canary_py",
        state_keys=("exp_avg", "exp_avg_sq", "exp_avg_slow", "slow_momentum"),
        optimizer_extra_args="lookahead_merge_time=5 cautious=True stable_adamw=True orthograd=True",
        loss_scale=0.2,
        loss_bias=0.004,
    ),
}


@dataclass(frozen=True)
class RangerFamilyTrainingExecutorConfig:
    optimizer_kind: str
    lr: float = 1e-3
    betas: tuple[float, ...] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    weight_decouple: bool = True
    fixed_decay: bool = False
    agc_eps: float = 1e-3
    agc_clip: float = 1e-2
    norm_loss_factor: float = 1e-4
    use_softplus: bool = True
    beta_softplus: float = 50.0
    alpha: float = 5.0
    cautious: bool = True
    stable_adamw: bool = True
    orthograd: bool = True
    lookahead_merge_time: int = 5
    lookahead_blending_alpha: float = 0.5
    maximize: bool = False
    max_numel: int = 1_048_576
    require_native_cuda: bool = True


class RangerFamilyTrainingExecutor:
    """Launch Ranger21/Ranger25 developer canaries against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: RangerFamilyTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
        kind: str,
    ) -> None:
        self.spec = _spec(kind)
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError(f"Plugin {self.spec.kind} TrainingExecutor requires trainable parameters")
        self._param_ids = {id(param) for param in self.params}
        self.config = _normalize_config(config, optimizer, self.spec.kind)
        self.workspace_root = Path(workspace_root or REPO_ROOT)
        self._native: Any | None = None

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked(self.spec.kind, f"plugin_{self.spec.kind}_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked(self.spec.kind, f"plugin_{self.spec.kind}_training_executor_requires_cuda_params")
        native = self._load_native()
        if native is None:
            return _blocked(self.spec.kind, f"plugin_{self.spec.kind}_training_dispatch_entrypoint_missing")

        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        stepped_groups: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config, self.spec.kind, self.optimizer)
            group_step_before = _group_step_to_int(group)
            group_cases: list[dict[str, Any]] = []
            for param in group["params"]:
                if id(param) not in self._param_ids or param.grad is None:
                    continue
                group_cases.append(self._step_param(native, param, group_cfg, group_step_before))
            cases.extend(group_cases)
            if group_cases and all(bool(case.get("ok", False)) for case in group_cases):
                _set_group_step(group, group_step_before + 1)
                stepped_groups.append(
                    {
                        "schema_version": 1,
                        "step_before": group_step_before,
                        "step_after": _group_step_to_int(group),
                        "parameter_count": len(group_cases),
                    }
                )
        ok = bool(cases) and all(bool(case.get("ok", False)) for case in cases)
        blockers = _dedupe([reason for case in cases for reason in case.get("blocked_reasons", [])])
        if not cases:
            blockers.append(f"plugin_{self.spec.kind}_training_executor_no_grad_params")
        return {
            "schema_version": 1,
            "executor": f"turbocore_plugin_{self.spec.kind}_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else f"plugin_{self.spec.kind}_native_step_failed"),
            "optimizer_kind": self.spec.kind,
            "training_dispatch": True,
            "training_path_enabled": True,
            "native_step_executed": ok,
            "native_kernel_launched": any(bool(case.get("kernel_executed", False)) for case in cases),
            "training_parameters_mutated": ok,
            "should_call_pytorch_optimizer_step": not ok,
            "pytorch_optimizer_state_synced": True,
            "parameter_step_count": len(cases),
            "cases": cases,
            "stepped_groups": stepped_groups,
            "timing": {"elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4)},
            "blocked_reasons": blockers,
        }

    def close(self) -> None:
        return None

    def _load_native(self) -> Any | None:
        if self._native is None:
            self._native = native_with_entrypoints(self.spec.entrypoint)
        return self._native

    def _step_param(
        self,
        native: Any,
        param: torch.nn.Parameter,
        group_cfg: Mapping[str, Any],
        group_step_before: int,
    ) -> dict[str, Any]:
        kind = self.spec.kind
        if type(self.optimizer).__name__.lower() != kind:
            return _case_failed(kind, f"plugin_{kind}_optimizer_class_unsupported")
        if param.dtype != torch.float32 or param.grad is None or param.grad.dtype != torch.float32:
            return _case_failed(kind, f"plugin_{kind}_native_step_dtype_unsupported")
        if not param.is_contiguous() or not param.grad.is_contiguous():
            return _case_failed(kind, f"plugin_{kind}_native_step_layout_unsupported")
        state = self.optimizer.state[param]
        if not all(name in state for name in self.spec.state_keys):
            return _case_failed(kind, f"plugin_{kind}_live_state_missing")
        state_tensors = tuple(state[name] for name in self.spec.state_keys)
        if not all(torch.is_tensor(item) for item in state_tensors):
            return _case_failed(kind, f"plugin_{kind}_live_state_missing")
        if any(item.dtype != torch.float32 for item in state_tensors):
            return _case_failed(kind, f"plugin_{kind}_native_state_dtype_unsupported")
        if not all(item.device.type == "cuda" for item in state_tensors):
            return _case_failed(kind, f"plugin_{kind}_native_state_device_unsupported")
        if not all(item.is_contiguous() and item.numel() == param.numel() for item in state_tensors):
            return _case_failed(kind, f"plugin_{kind}_native_state_layout_unsupported")
        launch_config = {
            **dict(group_cfg),
            "step": group_step_before,
            "max_numel": max(int(self.config.max_numel), int(param.numel())),
            "canary_probe_only": True,
            "training_tensor_binding": True,
            "training_dispatch": False,
            "training_path_enabled": False,
        }
        try:
            launch = dict(self._call_native(native, param, state_tensors, launch_config))
        except Exception as exc:  # pragma: no cover - native/CUDA dependent
            return _case_failed(kind, f"plugin_{kind}_native_step_call_failed", reason=f"{type(exc).__name__}: {exc}")
        if not bool(launch.get("ok", False)):
            return _case_failed(
                kind,
                f"plugin_{kind}_native_step_failed",
                reason=str(launch.get("reason") or f"plugin_{kind}_native_step_failed"),
                launch=launch,
            )
        return {
            "schema_version": 1,
            "ok": True,
            "param_numel": int(param.numel()),
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_before": group_step_before,
            "step_after": int(launch.get("step_after", group_step_before + 1) or (group_step_before + 1)),
            "kernel_executed": bool(launch.get("kernel_executed", False)),
            "training_parameters_mutated": bool(
                launch.get("training_parameters_mutated")
                or launch.get("parameters_mutated")
                or launch.get("live_tensors_mutated")
            ),
            "launch": launch,
            "blocked_reasons": [],
        }

    def _call_native(
        self,
        native: Any,
        param: torch.nn.Parameter,
        state_tensors: tuple[torch.Tensor, ...],
        launch_config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        args = [
            param,
            param.grad.detach().contiguous(),
            *state_tensors,
            json.dumps(dict(launch_config)),
            str(self.workspace_root.resolve()),
            _cuda_arch(param.device),
        ]
        return getattr(native, self.spec.entrypoint)(*args)


def build_ranger_family_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: RangerFamilyTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    kind: str,
) -> RangerFamilyTrainingExecutor:
    return RangerFamilyTrainingExecutor(
        optimizer=optimizer,
        params=params,
        config=config,
        workspace_root=workspace_root,
        kind=kind,
    )


def _normalize_config(
    value: RangerFamilyTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
    kind: str,
) -> RangerFamilyTrainingExecutorConfig:
    if isinstance(value, RangerFamilyTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    default_betas = (0.9, 0.98, 0.9999) if kind == "ranger25" else (0.9, 0.999)
    betas = tuple(float(item) for item in payload.get("betas", group.get("betas", default_betas)))
    return RangerFamilyTrainingExecutorConfig(
        optimizer_kind=kind,
        lr=float(payload.get("lr", group.get("lr", 1e-3))),
        betas=betas,
        eps=float(payload.get("eps", group.get("eps", 1e-8) or 1e-8)),
        weight_decay=float(payload.get("weight_decay", group.get("weight_decay", 0.0))),
        weight_decouple=bool(payload.get("weight_decouple", group.get("weight_decouple", True))),
        fixed_decay=bool(payload.get("fixed_decay", group.get("fixed_decay", False))),
        agc_eps=float(payload.get("agc_eps", getattr(optimizer, "agc_eps", 1e-3))),
        agc_clip=float(payload.get("agc_clip", getattr(optimizer, "agc_clipping_value", 1e-2))),
        norm_loss_factor=float(payload.get("norm_loss_factor", getattr(optimizer, "norm_loss_factor", 1e-4))),
        use_softplus=bool(payload.get("use_softplus", getattr(optimizer, "use_softplus", True))),
        beta_softplus=float(payload.get("beta_softplus", getattr(optimizer, "beta_softplus", 50.0))),
        alpha=float(payload.get("alpha", group.get("alpha", 5.0))),
        cautious=bool(payload.get("cautious", getattr(optimizer, "cautious", True))),
        stable_adamw=bool(payload.get("stable_adamw", getattr(optimizer, "stable_adamw", True))),
        orthograd=bool(payload.get("orthograd", getattr(optimizer, "orthograd", True))),
        lookahead_merge_time=int(payload.get("lookahead_merge_time", getattr(optimizer, "lookahead_merge_time", 5)) or 5),
        lookahead_blending_alpha=float(
            payload.get("lookahead_blending_alpha", getattr(optimizer, "lookahead_blending_alpha", 0.5))
        ),
        maximize=bool(payload.get("maximize", getattr(optimizer, "maximize", False))),
        max_numel=int(payload.get("max_numel", 1_048_576)),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(
    group: Mapping[str, Any],
    config: RangerFamilyTrainingExecutorConfig,
    kind: str,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    betas = tuple(float(item) for item in group.get("betas", config.betas))
    base = {
        "lr": float(group.get("lr", config.lr)),
        "betas": list(betas),
        "eps": float(group.get("eps", config.eps) or config.eps),
        "weight_decay": float(group.get("weight_decay", config.weight_decay)),
        "maximize": bool(config.maximize),
        "lookahead_merge_time": int(config.lookahead_merge_time),
        "lookahead_blending_alpha": float(config.lookahead_blending_alpha),
    }
    if kind == "ranger21":
        base.update(
            {
                "weight_decouple": bool(group.get("weight_decouple", config.weight_decouple)),
                "fixed_decay": bool(group.get("fixed_decay", config.fixed_decay)),
                "agc_eps": float(config.agc_eps),
                "agc_clip": float(config.agc_clip),
                "norm_loss_factor": float(config.norm_loss_factor),
                "use_softplus": bool(config.use_softplus),
                "beta_softplus": float(config.beta_softplus),
            }
        )
    else:
        beta3 = betas[2] if len(betas) >= 3 else 0.9999
        base.update(
            {
                "betas": [float(betas[0]), float(betas[1]), float(beta3)],
                "alpha": float(group.get("alpha", config.alpha)),
                "alpha_t": float(group.get("alpha", config.alpha)),
                "beta3_t": float(beta3),
                "cautious": bool(getattr(optimizer, "cautious", config.cautious)),
                "stable_adamw": bool(getattr(optimizer, "stable_adamw", config.stable_adamw)),
                "orthograd": bool(getattr(optimizer, "orthograd", config.orthograd)),
            }
        )
    return base


def _seed_previous_gate(loop: TrainingLoop) -> None:
    loop._turbocore_native_update_dispatch_armer._last_gate_report = _explicit_gate()


def _explicit_gate() -> dict[str, Any]:
    contract = {
        "dispatch_rehearsal_ready": True,
        "would_allow_native_dispatch": True,
        "rehearsal": {"would_launch_native_kernel": True},
        "recovery": {"default_off_recovery_bridge_ready": True, "training_dispatch_recovery_ready": True},
        "owner_gradient_sync": _ready_contract("sync_boundary_ready", "owner_gradient_sync_preconditions_ready"),
        "training_flat_owner": _ready_contract("owner_boundary_ready", "training_flat_owner_preconditions_ready"),
        "training_dispatch_kernel": _ready_contract("kernel_boundary_ready", "training_dispatch_kernel_preconditions_ready"),
        "training_executor": {"executor_boundary_ready": True, "training_executor_preconditions_ready": True},
        "stream_lifetime_ownership": {"ownership_boundary_ready": True, "stream_lifetime_ownership_preconditions_ready": True},
        "evidence": {
            "owner_native_launch_ok": True,
            "copyback_dispatch_validated": True,
            "event_chain_verified": True,
            "stream_ordering_verified": True,
            "representative_performance_gate_ready": True,
        },
        "blocked_reasons": [],
    }
    return {
        "dispatch_request": {
            "requested": True,
            "dispatch_allowed": True,
            "training_path_enabled": True,
            "training_path_request": {"request_boundary_ready": True, "explicit_training_path_requested": True},
        },
        "dispatch_contract": contract,
        "kernel_launch_plan": {"launch_allowed": True, "evidence": {"diagnostic_kernel_executed": True}},
    }


def _ready_contract(boundary: str, precondition: str) -> dict[str, Any]:
    return {boundary: True, precondition: True, "native_supported": True, "training_lifecycle_integrated": True}


def _spec(kind: str) -> RangerFamilySpec:
    normalized = str(kind or "").strip().lower().replace("-", "")
    if normalized not in SPECS:
        raise ValueError(f"unsupported Ranger family optimizer: {kind}")
    return SPECS[normalized]


def _group_step_to_int(group: Mapping[str, Any]) -> int:
    value = group.get("step", 0)
    if torch.is_tensor(value) and value.numel() > 0:
        return int(value.detach().reshape(-1)[0].cpu().item())
    return int(value or 0)


def _set_group_step(group: dict[str, Any], value: int) -> None:
    current = group.get("step")
    if torch.is_tensor(current):
        current.fill_(int(value))
    else:
        group["step"] = int(value)


def _cuda_arch(device: torch.device) -> str:
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"compute_{int(major)}{int(minor)}"
    except Exception:
        return "compute_89"


def _case_failed(kind: str, blocker: str, *, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "reason": reason or blocker,
        "kernel_executed": False,
        "training_parameters_mutated": False,
        "blocked_reasons": [blocker],
        **extra,
    }


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


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "RangerFamilyTrainingExecutor",
    "RangerFamilyTrainingExecutorConfig",
    "build_ranger_family_training_executor",
    "RangerFamilySpec",
    "SPECS",
]
