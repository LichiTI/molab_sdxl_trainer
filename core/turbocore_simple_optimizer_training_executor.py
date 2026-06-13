"""Default-off training executor for simple formula native optimizers."""

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
class SimpleOptimizerTrainingExecutorConfig:
    optimizer_kind: str = "lion"
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.99)
    momentum: float = 0.9
    centered: bool = False
    derivative: float = 10.0
    integral: float = 5.0
    nu: float = 1.0
    kappa: float = 1000.0
    xi: float = 10.0
    constant: float = 0.7
    alpha: float = 0.99
    beta: float = 0.9
    eps: float = 1e-8
    trust_coefficient: float = 1e-3
    dampening: float = 0.0
    weight_decay: float = 0.0
    weight_decouple: bool = True
    fixed_decay: bool = False
    delta: float = 0.1
    wd_ratio: float = 0.1
    nesterov: bool = False
    p_bound: float = 0.0
    block_size: int = 128
    require_native_cuda: bool = True


class SimpleOptimizerTrainingExecutor:
    """Own flat fp32 state and launch simple-formula native runtime steps."""

    def __init__(
        self,
        *,
        params: Iterable[torch.nn.Parameter],
        config: SimpleOptimizerTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("SimpleOptimizerTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config)
        self.workspace_root = Path(workspace_root or REPO_ROOT)
        self.layout = _layout(self.params)
        self.param_flat = _flatten_params(self.params)
        self.grad_flat = torch.zeros_like(self.param_flat)
        self.state_layout = _state_layout(self.config, int(self.param_flat.numel()), self.layout)
        self.state_flat = torch.zeros(
            int(self.state_layout[-1]["offset"] + self.state_layout[-1]["numel"]),
            device=self.param_flat.device,
            dtype=self.param_flat.dtype,
        )
        self._views = _views(self.param_flat, self.layout)
        self._runtime_id: int | None = None
        self._native: Any | None = None
        self._step_index = 0
        self._fromage_p_bound_initialized = False
        self._refresh_optimizer_state_metadata(initialize=True)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        request_payload = dict(request or {})
        if not bool(request_payload.get("training_dispatch", False)) or not bool(request_payload.get("training_path_enabled", False)):
            return _blocked("simple_optimizer_training_executor_requires_training_dispatch")
        if self.param_flat.device.type != "cuda":
            return _blocked("cuda_required_for_simple_optimizer_training_executor")
        started = time.perf_counter()
        self._sync_params_and_grads()
        self._refresh_optimizer_state_metadata(initialize=False)
        sync_ms = _elapsed_ms(started)
        native = self._load_native()
        if native is None:
            return _blocked("simple_optimizer_runtime_entrypoints_missing")
        runtime_id, create_report = self._ensure_runtime(native)
        if runtime_id is None:
            return _blocked("simple_optimizer_runtime_session_unavailable", create_report=create_report)
        step_started = time.perf_counter()
        try:
            payload = native.step_simple_optimizer_cuda_kernel_runtime_session_py(
                int(runtime_id),
                self.param_flat,
                self.grad_flat,
                self.state_flat,
                json.dumps(self._launch_config()),
            )
        except Exception as exc:  # pragma: no cover - native/CUDA dependent
            return _blocked("simple_optimizer_native_step_call_failed", error=f"{type(exc).__name__}: {exc}")
        step_report = dict(payload) if isinstance(payload, Mapping) else {"ok": False, "reason": "invalid_native_step_payload"}
        step_ms = _elapsed_ms(step_started)
        if not bool(step_report.get("ok", False)):
            return _blocked(str(step_report.get("reason", "simple_optimizer_native_step_failed")), step_report=step_report)
        copy_started = time.perf_counter()
        _copy_flat_to_params(self.params, self._views)
        copyback_ms = _elapsed_ms(copy_started)
        self._step_index += 1
        return {
            "schema_version": 1,
            "executor": "turbocore_simple_optimizer_training_executor_v0",
            "ok": True,
            "reason": "called",
            "optimizer_kind": self.config.optimizer_kind,
            "training_dispatch": True,
            "training_path_enabled": True,
            "native_step_executed": True,
            "native_kernel_launched": bool(step_report.get("kernel_executed", False)),
            "training_parameters_mutated": bool(step_report.get("parameters_mutated", False)),
            "should_call_pytorch_optimizer_step": False,
            "pytorch_optimizer_state_synced": False,
            "step_report": step_report,
            "timing": {
                "elapsed_ms": _elapsed_ms(started),
                "sync_ms": sync_ms,
                "native_step_ms": step_ms,
                "copyback_ms": copyback_ms,
            },
            "blocked_reasons": [],
        }

    def close(self) -> None:
        if self._native is not None and self._runtime_id is not None:
            try:
                self._native.destroy_simple_optimizer_cuda_kernel_runtime_session_py(int(self._runtime_id))
            except Exception:
                pass
        self._runtime_id = None

    def _sync_params_and_grads(self) -> None:
        self.param_flat.copy_(_flatten_params(self.params))
        grads = [param.grad for param in self.params]
        self.grad_flat.copy_(_flatten_tensors([grad if grad is not None else torch.zeros_like(param) for param, grad in zip(self.params, grads)]))

    def _refresh_optimizer_state_metadata(self, *, initialize: bool) -> None:
        if self.config.optimizer_kind == "fromage":
            _refresh_fromage_state_metadata(
                config=self.config,
                state_flat=self.state_flat,
                state_layout=self.state_layout,
                layout=self.layout,
                param_flat=self.param_flat,
                grad_flat=self.grad_flat,
                initialize_p_bound=initialize and not self._fromage_p_bound_initialized,
            )
            if initialize:
                self._fromage_p_bound_initialized = True

    def _load_native(self) -> Any | None:
        if self._native is None:
            self._native = native_with_entrypoints(*ENTRYPOINTS)
        return self._native

    def _ensure_runtime(self, native: Any) -> tuple[int | None, dict[str, Any]]:
        if self._runtime_id is not None:
            return int(self._runtime_id), {"ok": True, "runtime_session_id": int(self._runtime_id)}
        try:
            created = native.create_simple_optimizer_cuda_kernel_runtime_session_py(
                self.config.optimizer_kind,
                str(self.workspace_root),
                _cuda_arch(self.param_flat.device),
            )
        except Exception as exc:  # pragma: no cover - native/CUDA dependent
            return None, {"ok": False, "reason": "simple_optimizer_runtime_create_failed", "error": f"{type(exc).__name__}: {exc}"}
        report = dict(created) if isinstance(created, Mapping) else {"ok": False, "reason": "invalid_runtime_create_payload"}
        if not bool(report.get("ok", False)):
            return None, report
        self._runtime_id = int(report.get("runtime_session_id", 0) or 0)
        return int(self._runtime_id), report

    def _launch_config(self) -> dict[str, Any]:
        return {
            "optimizer_kind": self.config.optimizer_kind,
            "lr": float(self.config.lr),
            "betas": [float(self.config.betas[0]), float(self.config.betas[1])],
            "momentum": float(self.config.momentum),
            "centered": bool(self.config.centered),
            "derivative": float(self.config.derivative),
            "integral": float(self.config.integral),
            "nu": float(self.config.nu),
            "kappa": float(self.config.kappa),
            "xi": float(self.config.xi),
            "constant": float(self.config.constant),
            "alpha": float(self.config.alpha),
            "beta": float(self.config.beta),
            "eps": float(self.config.eps),
            "trust_coefficient": float(self.config.trust_coefficient),
            "dampening": float(self.config.dampening),
            "weight_decay": float(self.config.weight_decay),
            "weight_decouple": bool(self.config.weight_decouple),
            "fixed_decay": bool(self.config.fixed_decay),
            "delta": float(self.config.delta),
            "wd_ratio": float(self.config.wd_ratio),
            "nesterov": bool(self.config.nesterov),
            "p_bound": float(self.config.p_bound),
            "per_tensor_norm": self.config.optimizer_kind == "fromage",
            "tensor_count": len(self.layout),
            "param_group_offsets": _param_group_offsets(self.layout),
            "param_norm": _tensor_norm(self.param_flat),
            "grad_norm": _tensor_norm(self.grad_flat),
            "grad_abs_max": _tensor_absmax(self.grad_flat),
            "step_index": int(self._step_index + 1),
            "block_size": int(self.config.block_size),
            "parameter_numel": int(self.param_flat.numel()),
            "state_numel": int(self.state_flat.numel()),
            "state_roles": [str(item["role"]) for item in self.state_layout],
            "state_layout": [dict(item) for item in self.state_layout],
            "max_numel": int(self.param_flat.numel()),
            "training_dispatch": True,
            "training_path_enabled": True,
        }


def build_simple_optimizer_training_executor(
    *,
    params: Iterable[torch.nn.Parameter],
    config: SimpleOptimizerTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> SimpleOptimizerTrainingExecutor:
    return SimpleOptimizerTrainingExecutor(params=params, config=config, workspace_root=workspace_root)


def _normalize_config(value: SimpleOptimizerTrainingExecutorConfig | Mapping[str, Any] | None) -> SimpleOptimizerTrainingExecutorConfig:
    if isinstance(value, SimpleOptimizerTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    betas = payload.get("betas", (0.9, 0.99))
    kind = str(payload.get("optimizer_kind", payload.get("optimizer", "lion")) or "lion").strip().lower()
    kind = "sgd" if kind in {"torch_sgd", "plain_sgd"} else kind
    kind = "sgd_nesterov" if kind in {"sgdnesterov", "sgd-nesterov"} else kind
    kind = "sign_momentum" if kind in {"signmomentum", "sign-momentum", "signsgd", "tiger"} else kind
    if kind not in {
        "lion",
        "sgd",
        "sgd_nesterov",
        "sign_momentum",
        "qhm",
        "accsgd",
        "fromage",
        "rmsprop",
        "lars",
        "pid",
        "sgdp",
        "gravity",
        "aggmo",
        "asgd",
        "madgrad",
        "nero",
        "vsgd",
    }:
        raise ValueError(f"Unsupported simple optimizer kind: {kind}")
    return SimpleOptimizerTrainingExecutorConfig(
        optimizer_kind=kind,
        lr=float(payload.get("lr", 1e-3)),
        betas=(float(betas[0]), float(betas[1])),
        momentum=float(payload.get("momentum", 0.9)),
        centered=bool(payload.get("centered", False)),
        derivative=float(payload.get("derivative", 10.0)),
        integral=float(payload.get("integral", 5.0)),
        nu=float(payload.get("nu", 1.0)),
        kappa=float(payload.get("kappa", 1000.0)),
        xi=float(payload.get("xi", 10.0)),
        constant=float(payload.get("constant", 0.7)),
        alpha=float(payload.get("alpha", 0.99)),
        beta=float(payload.get("beta", 0.9)),
        eps=float(payload.get("eps", 1e-8)),
        trust_coefficient=float(payload.get("trust_coefficient", 1e-3)),
        dampening=float(payload.get("dampening", 0.0)),
        weight_decay=float(payload.get("weight_decay", 0.0)),
        weight_decouple=bool(payload.get("weight_decouple", True)),
        fixed_decay=bool(payload.get("fixed_decay", False)),
        delta=float(payload.get("delta", 0.1)),
        wd_ratio=float(payload.get("wd_ratio", 0.1)),
        nesterov=bool(payload.get("nesterov", False)),
        p_bound=float(payload.get("p_bound", 0.0) or 0.0),
        block_size=int(payload.get("block_size", 128)),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
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


def _state_layout(
    config: SimpleOptimizerTrainingExecutorConfig,
    parameter_numel: int,
    layout: list[tuple[tuple[int, ...], int, int]],
) -> list[dict[str, int | str]]:
    if config.optimizer_kind == "fromage":
        tensor_count = len(layout)
        sizes = [
            ("param_group_offsets", tensor_count + 1),
            ("per_tensor_param_norm", tensor_count),
            ("per_tensor_grad_norm", tensor_count),
            ("p_bound", tensor_count),
            ("per_tensor_post_norm", tensor_count),
        ]
        offset = 0
        rows: list[dict[str, int | str]] = []
        for role, numel in sizes:
            rows.append({"role": role, "offset": offset, "numel": numel})
            offset += numel
        return rows
    roles = _state_roles(config)
    return [
        {"role": role, "offset": index * parameter_numel, "numel": parameter_numel}
        for index, role in enumerate(roles)
    ]


def _state_roles(config: SimpleOptimizerTrainingExecutorConfig) -> list[str]:
    if config.optimizer_kind == "rmsprop":
        roles = ["square_avg"]
        if config.centered:
            roles.append("grad_avg")
        if float(config.momentum) != 0.0:
            roles.append("momentum_buffer")
        return roles
    if config.optimizer_kind == "pid" and float(config.momentum) > 0.0:
        return ["integral_buffer", "previous_grad", "momentum_buffer"]
    if config.optimizer_kind == "sgdp":
        return ["momentum"]
    return ["state_flat"]


def _param_group_offsets(layout: list[tuple[tuple[int, ...], int, int]]) -> list[int]:
    if not layout:
        return [0]
    offsets = [int(offset) for _, offset, _ in layout]
    last_offset = int(layout[-1][1])
    last_numel = int(layout[-1][2])
    offsets.append(last_offset + last_numel)
    return offsets


def _state_entry(state_layout: list[dict[str, int | str]], role: str) -> dict[str, int | str]:
    for item in state_layout:
        if item["role"] == role:
            return item
    raise KeyError(role)


def _refresh_fromage_state_metadata(
    *,
    config: SimpleOptimizerTrainingExecutorConfig,
    state_flat: torch.Tensor,
    state_layout: list[dict[str, int | str]],
    layout: list[tuple[tuple[int, ...], int, int]],
    param_flat: torch.Tensor,
    grad_flat: torch.Tensor,
    initialize_p_bound: bool,
) -> None:
    tensor_count = len(layout)
    offsets = _param_group_offsets(layout)
    with torch.no_grad():
        _copy_state_values(state_flat, state_layout, "param_group_offsets", offsets)
        _copy_state_values(
            state_flat,
            state_layout,
            "per_tensor_param_norm",
            [_tensor_norm(param_flat[offset : offset + count]) for _, offset, count in layout],
        )
        _copy_state_values(
            state_flat,
            state_layout,
            "per_tensor_grad_norm",
            [_tensor_norm(grad_flat[offset : offset + count]) for _, offset, count in layout],
        )
        _copy_state_values(state_flat, state_layout, "per_tensor_post_norm", [0.0] * tensor_count)
        if initialize_p_bound:
            p_bound_values = [0.0] * tensor_count
            if float(config.p_bound) > 0.0:
                p_bound_values = [
                    _tensor_norm(param_flat[offset : offset + count]) * float(config.p_bound)
                    for _, offset, count in layout
                ]
            _copy_state_values(state_flat, state_layout, "p_bound", p_bound_values)


def _copy_state_values(
    state_flat: torch.Tensor,
    state_layout: list[dict[str, int | str]],
    role: str,
    values: list[float | int],
) -> None:
    entry = _state_entry(state_layout, role)
    offset = int(entry["offset"])
    numel = int(entry["numel"])
    state_flat[offset : offset + numel].copy_(
        torch.tensor(values[:numel], device=state_flat.device, dtype=state_flat.dtype)
    )


def _flatten_params(params: list[torch.Tensor]) -> torch.Tensor:
    return _flatten_tensors([param.detach() for param in params])


def _flatten_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.detach().float().reshape(-1) for tensor in tensors]).contiguous()


def _copy_flat_to_params(params: list[torch.nn.Parameter], views: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, view in zip(params, views):
            param.copy_(view.to(device=param.device, dtype=param.dtype))


def _tensor_norm(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.float()).detach().item())


def _tensor_absmax(tensor: torch.Tensor) -> float:
    return float(tensor.float().abs().max().detach().item()) if int(tensor.numel()) else 0.0


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
        "executor": "turbocore_simple_optimizer_training_executor_v0",
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


__all__ = [
    "SimpleOptimizerTrainingExecutor",
    "SimpleOptimizerTrainingExecutorConfig",
    "build_simple_optimizer_training_executor",
]
