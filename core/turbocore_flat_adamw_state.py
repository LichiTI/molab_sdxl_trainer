"""Persistent flat AdamW state prototype for TurboCore research.

This module owns contiguous fp32 parameter, gradient, and moment buffers. It is
developer-facing only: no trainer route, no runtime dispatch, no automatic
fallback changes.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import torch

try:  # pragma: no cover - host/toolchain specific
    from core.turbocore_triton_optimizer import (
        triton_adamw_flat_available,
        triton_adamw_flat_v0_step_,
    )
except Exception:  # pragma: no cover
    triton_adamw_flat_available = None  # type: ignore[assignment]
    triton_adamw_flat_v0_step_ = None  # type: ignore[assignment]

try:  # pragma: no cover - native/CUDA/toolchain specific
    from core.turbocore_native_adamw_runtime_backend import NativeAdamWRuntimeBackend
except Exception:  # pragma: no cover
    NativeAdamWRuntimeBackend = None  # type: ignore[assignment]


class NativeAdamWRequiredError(RuntimeError):
    """Raised when a required native AdamW step is unavailable."""

    def __init__(self, reason: str, native_report: dict[str, Any] | None = None) -> None:
        super().__init__(str(reason or "native_cuda_adamw_required_but_unavailable"))
        self.native_report = dict(native_report or {})


@dataclass(frozen=True)
class FlatAdamWConfig:
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 0.0
    finite_check: bool = True
    prefer_native_cuda: bool = False
    native_training_dispatch: bool = False
    require_native_cuda: bool = False
    prefer_triton: bool = False
    block_size: int = 1024
    native_runtime_synchronization_policy: str = "context_synchronize"
    native_runtime_stream_guard_descriptor: dict[str, Any] = field(default_factory=dict)
    native_runtime_stream_lifetime_lease_evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["betas"] = [float(self.betas[0]), float(self.betas[1])]
        return payload


@dataclass(frozen=True)
class FlatAdamWStepReport:
    optimizer: str
    step_index: int
    numel: int
    finite: bool
    skipped: bool
    clipped: bool
    grad_norm_before_clip: float
    grad_norm_after_clip: float
    reason: str
    backend: str
    native_kernel_present: bool
    training_path_enabled: bool = False
    native_report: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FlatAdamWBufferLayout:
    shapes: tuple[tuple[int, ...], ...]
    numels: tuple[int, ...]
    offsets: tuple[int, ...]
    dtypes: tuple[str, ...]
    device: str

    @property
    def total_numel(self) -> int:
        return sum(int(item) for item in self.numels)

    def as_dict(self) -> dict[str, Any]:
        return {
            "shapes": [list(shape) for shape in self.shapes],
            "numels": [int(item) for item in self.numels],
            "offsets": [int(item) for item in self.offsets],
            "dtypes": list(self.dtypes),
            "device": self.device,
            "total_numel": self.total_numel,
        }


def _normalise_config(config: FlatAdamWConfig | dict[str, Any] | None) -> FlatAdamWConfig:
    if config is None:
        return FlatAdamWConfig()
    if isinstance(config, FlatAdamWConfig):
        return config
    betas = config.get("betas", (0.9, 0.999))
    return FlatAdamWConfig(
        lr=float(config.get("lr", 1e-4)),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(config.get("eps", 1e-8)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        max_grad_norm=float(config.get("max_grad_norm", 0.0)),
        finite_check=bool(config.get("finite_check", True)),
        prefer_native_cuda=bool(config.get("prefer_native_cuda", False)),
        native_training_dispatch=bool(config.get("native_training_dispatch", False)),
        require_native_cuda=bool(config.get("require_native_cuda", False)),
        prefer_triton=bool(config.get("prefer_triton", False)),
        block_size=int(config.get("block_size", 1024)),
        native_runtime_synchronization_policy=str(
            config.get("native_runtime_synchronization_policy", "context_synchronize")
            or "context_synchronize"
        ),
        native_runtime_stream_guard_descriptor=dict(config.get("native_runtime_stream_guard_descriptor", {}) or {}),
        native_runtime_stream_lifetime_lease_evidence=dict(
            config.get("native_runtime_stream_lifetime_lease_evidence", {}) or {}
        ),
    )


def _as_tensors(params: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    tensors = [tensor for tensor in params if isinstance(tensor, torch.Tensor)]
    if not tensors:
        raise ValueError("PersistentFlatAdamW requires at least one tensor")
    if len({str(tensor.device) for tensor in tensors}) != 1:
        raise ValueError("PersistentFlatAdamW requires all tensors on the same device")
    return tensors


def _build_layout(tensors: list[torch.Tensor]) -> FlatAdamWBufferLayout:
    offsets: list[int] = []
    numels: list[int] = []
    cursor = 0
    for tensor in tensors:
        offsets.append(cursor)
        count = int(tensor.numel())
        numels.append(count)
        cursor += count
    return FlatAdamWBufferLayout(
        shapes=tuple(tuple(int(dim) for dim in tensor.shape) for tensor in tensors),
        numels=tuple(numels),
        offsets=tuple(offsets),
        dtypes=tuple(str(tensor.dtype).replace("torch.", "") for tensor in tensors),
        device=str(tensors[0].device),
    )


def _flatten_fp32(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.detach().float().reshape(-1) for tensor in tensors]).contiguous()


def _vector_norm(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(tensor.float(), ord=2).detach().cpu().item())


def _foreach_copy_or_loop(targets: Iterable[torch.Tensor], sources: Iterable[torch.Tensor]) -> None:
    target_list = list(targets)
    source_list = list(sources)
    if not target_list:
        return
    foreach_copy = getattr(torch, "_foreach_copy_", None)
    if callable(foreach_copy):
        grouped: dict[tuple[str, torch.dtype, str, torch.dtype], tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        for target, source in zip(target_list, source_list):
            key = (str(target.device), target.dtype, str(source.device), source.dtype)
            group_targets, group_sources = grouped.setdefault(key, ([], []))
            group_targets.append(target)
            group_sources.append(source)
        try:
            for group_targets, group_sources in grouped.values():
                foreach_copy(group_targets, group_sources)
            return
        except Exception:
            pass
    for target, source in zip(target_list, source_list):
        target.copy_(source)


def _foreach_zero_or_loop(targets: Iterable[torch.Tensor]) -> None:
    target_list = list(targets)
    if not target_list:
        return
    foreach_zero = getattr(torch, "_foreach_zero_", None)
    if callable(foreach_zero):
        try:
            foreach_zero(target_list)
            return
        except Exception:
            pass
    for target in target_list:
        target.zero_()


class PersistentFlatAdamW:
    """Owns flat fp32 AdamW buffers for parity and layout experiments."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        config: FlatAdamWConfig | dict[str, Any] | None = None,
    ) -> None:
        tensors = _as_tensors(params)
        self.config = _normalise_config(config)
        self.layout = _build_layout(tensors)
        self.param_flat = _flatten_fp32(tensors)
        self.grad_flat = torch.zeros_like(self.param_flat)
        self.exp_avg = torch.zeros_like(self.param_flat)
        self.exp_avg_sq = torch.zeros_like(self.param_flat)
        self._param_views = tuple(
            self.param_flat[offset : offset + count].view(shape)
            for shape, offset, count in zip(self.layout.shapes, self.layout.offsets, self.layout.numels)
        )
        self._grad_views = tuple(
            self.grad_flat[offset : offset + count].view(shape)
            for shape, offset, count in zip(self.layout.shapes, self.layout.offsets, self.layout.numels)
        )
        self.step_index = 0
        self.last_backend = "not_stepped"
        self._native_backend: Any | None = None
        self._native_backend_error = ""

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "PersistentFlatAdamW":
        config = _normalise_config(state_dict.get("config") or {})
        layout_payload = dict(state_dict.get("layout") or {})
        shapes = [tuple(int(dim) for dim in shape) for shape in layout_payload.get("shapes", [])]
        if not shapes:
            raise ValueError("Flat AdamW state_dict is missing layout shapes")
        param_flat = state_dict["param_flat"].detach().clone().float().contiguous()
        params = []
        offset = 0
        for shape in shapes:
            count = 1
            for dim in shape:
                count *= int(dim)
            params.append(param_flat[offset : offset + count].view(shape).clone())
            offset += count
        owner = cls(params, config)
        owner.load_state_dict(state_dict)
        return owner

    def set_grads(self, grads: Iterable[torch.Tensor | None]) -> None:
        grad_list = list(grads)
        if len(grad_list) != len(self.layout.numels):
            raise ValueError("Gradient tensor count does not match flat AdamW layout")
        copy_targets: list[torch.Tensor] = []
        copy_sources: list[torch.Tensor] = []
        zero_targets: list[torch.Tensor] = []
        for target, grad in zip(self._grad_views, grad_list):
            if grad is None:
                zero_targets.append(target)
                continue
            copy_targets.append(target)
            copy_sources.append(grad.detach())
        _foreach_copy_or_loop(copy_targets, copy_sources)
        _foreach_zero_or_loop(zero_targets)

    def unpack_params(self) -> list[torch.Tensor]:
        result: list[torch.Tensor] = []
        for shape, offset, count in zip(self.layout.shapes, self.layout.offsets, self.layout.numels):
            result.append(self.param_flat[offset : offset + count].view(shape).detach().clone())
        return result

    def copy_params_to_(self, params: Iterable[torch.Tensor]) -> None:
        tensors = list(params)
        if len(tensors) != len(self.layout.numels):
            raise ValueError("Parameter tensor count does not match flat AdamW layout")
        with torch.no_grad():
            for tensor, shape in zip(tensors, self.layout.shapes):
                if tuple(tensor.shape) != tuple(shape):
                    raise ValueError("Parameter tensor shape does not match flat AdamW layout")
            _foreach_copy_or_loop(tensors, self._param_views)

    def zero_grad(self) -> None:
        self.grad_flat.zero_()

    def close(self) -> None:
        if self._native_backend is None:
            return
        close = getattr(self._native_backend, "close", None)
        if callable(close):
            close()
        self._native_backend = None

    def step(self) -> FlatAdamWStepReport:
        finite_check = bool(self.config.finite_check)
        needs_grad_norm = float(self.config.max_grad_norm or 0.0) > 0.0
        finite = bool(torch.isfinite(self.grad_flat).all().detach().cpu().item()) if finite_check else True
        grad_norm_before = _vector_norm(self.grad_flat) if needs_grad_norm else 0.0
        if self.config.finite_check and not finite:
            return FlatAdamWStepReport(
                optimizer="PersistentFlatAdamW",
                step_index=int(self.step_index),
                numel=int(self.param_flat.numel()),
                finite=False,
                skipped=True,
                clipped=False,
                grad_norm_before_clip=grad_norm_before,
                grad_norm_after_clip=grad_norm_before,
                reason="non_finite_gradient",
                backend=self.last_backend,
                native_kernel_present=False,
            )

        clipped = self._clip_grad_if_needed(grad_norm_before) if needs_grad_norm else False
        grad_norm_after = _vector_norm(self.grad_flat) if needs_grad_norm else grad_norm_before
        step_number = int(self.step_index) + 1
        backend, native_kernel_present, native_report = self._run_update(step_number)
        self.step_index = step_number
        self.last_backend = backend
        return FlatAdamWStepReport(
            optimizer="PersistentFlatAdamW",
            step_index=int(self.step_index),
            numel=int(self.param_flat.numel()),
            finite=finite,
            skipped=False,
            clipped=clipped,
            grad_norm_before_clip=grad_norm_before,
            grad_norm_after_clip=grad_norm_after,
            reason="stepped",
            backend=backend,
            native_kernel_present=native_kernel_present,
            native_report=native_report,
        )

    def _clip_grad_if_needed(self, grad_norm_before: float) -> bool:
        max_norm = float(self.config.max_grad_norm or 0.0)
        if max_norm <= 0.0 or grad_norm_before <= max_norm:
            return False
        scale = max_norm / max(grad_norm_before, 1e-12)
        self.grad_flat.mul_(scale)
        return True

    def _run_update(self, step_number: int) -> tuple[str, bool, dict[str, Any]]:
        native_report = self._run_native_cuda_update(step_number)
        if bool(native_report.get("ok", False)) and bool(native_report.get("kernel_executed", False)):
            return "rust_cuda_adamw_v0", True, dict(native_report)
        if bool(self.config.require_native_cuda):
            reason = str(native_report.get("reason", "native_cuda_adamw_required_but_unavailable") or "native_cuda_adamw_required_but_unavailable")
            raise NativeAdamWRequiredError(reason, dict(native_report))
        if self._can_use_triton():
            beta1, beta2 = self.config.betas
            triton_adamw_flat_v0_step_(
                self.param_flat,
                self.grad_flat,
                self.exp_avg,
                self.exp_avg_sq,
                step=int(step_number),
                lr=float(self.config.lr),
                beta1=float(beta1),
                beta2=float(beta2),
                eps=float(self.config.eps),
                weight_decay=float(self.config.weight_decay),
                block_size=int(self.config.block_size),
            )
            return "triton_adamw_flat_v0", True, dict(native_report)
        self._torch_flat_step(step_number)
        return "torch_flat_reference", False, dict(native_report)

    def _run_native_cuda_update(self, step_number: int) -> dict[str, Any]:
        if not bool(self.config.prefer_native_cuda):
            return {"ok": False, "reason": "native_cuda_not_requested"}
        if NativeAdamWRuntimeBackend is None:
            return {"ok": False, "reason": "native_adamw_runtime_backend_unavailable"}
        if self._native_backend is None:
            self._native_backend = NativeAdamWRuntimeBackend()
        report = self._native_backend.step(
            self,
            step_number=int(step_number),
            training_dispatch=bool(self.config.native_training_dispatch),
        )
        if not bool(report.get("ok", False)):
            self._native_backend_error = str(report.get("reason", "native_adamw_step_failed") or "native_adamw_step_failed")
        return report

    def _can_use_triton(self) -> bool:
        return (
            bool(self.config.prefer_triton)
            and triton_adamw_flat_available is not None
            and triton_adamw_flat_v0_step_ is not None
            and bool(triton_adamw_flat_available())
            and self.param_flat.device.type == "cuda"
            and self.param_flat.dtype is torch.float32
        )

    def _torch_flat_step(self, step_number: int) -> None:
        beta1, beta2 = self.config.betas
        self.exp_avg.mul_(float(beta1)).add_(self.grad_flat, alpha=1.0 - float(beta1))
        self.exp_avg_sq.mul_(float(beta2)).addcmul_(self.grad_flat, self.grad_flat, value=1.0 - float(beta2))
        if float(self.config.weight_decay):
            self.param_flat.mul_(1.0 - float(self.config.lr) * float(self.config.weight_decay))
        bias_correction1 = 1.0 - float(beta1) ** int(step_number)
        bias_correction2 = 1.0 - float(beta2) ** int(step_number)
        step_size = float(self.config.lr) / bias_correction1
        denom = self.exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(float(self.config.eps))
        self.param_flat.addcdiv_(self.exp_avg, denom, value=-step_size)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "backend": "persistent_flat_adamw_python_prototype",
            "training_path_enabled": False,
            "native_kernel_present": False,
            "step_index": int(self.step_index),
            "config": self.config.as_dict(),
            "layout": self.layout.as_dict(),
            "param_flat": self.param_flat.detach().clone(),
            "grad_flat": self.grad_flat.detach().clone(),
            "exp_avg": self.exp_avg.detach().clone(),
            "exp_avg_sq": self.exp_avg_sq.detach().clone(),
            "last_backend": self.last_backend,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if int(state_dict.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported PersistentFlatAdamW state schema")
        incoming_layout = dict(state_dict.get("layout") or {})
        if [list(shape) for shape in self.layout.shapes] != list(incoming_layout.get("shapes", [])):
            raise ValueError("Flat AdamW state layout does not match target owner")
        self.config = _normalise_config(state_dict.get("config") or self.config)
        self.step_index = int(state_dict.get("step_index", 0))
        self.param_flat.copy_(state_dict["param_flat"].detach().to(device=self.param_flat.device, dtype=torch.float32))
        self.grad_flat.copy_(state_dict["grad_flat"].detach().to(device=self.grad_flat.device, dtype=torch.float32))
        self.exp_avg.copy_(state_dict["exp_avg"].detach().to(device=self.exp_avg.device, dtype=torch.float32))
        self.exp_avg_sq.copy_(state_dict["exp_avg_sq"].detach().to(device=self.exp_avg_sq.device, dtype=torch.float32))
        self.last_backend = str(state_dict.get("last_backend") or "loaded")

    def snapshot(self) -> dict[str, Any]:
        state_bytes = sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in (self.param_flat, self.grad_flat, self.exp_avg, self.exp_avg_sq)
        )
        return {
            "schema_version": 1,
            "optimizer": "AdamW",
            "backend": "PersistentFlatAdamW",
            "step_index": int(self.step_index),
            "layout": self.layout.as_dict(),
            "state_bytes": int(state_bytes),
            "state_mb": round(state_bytes / 1024.0 / 1024.0, 3),
            "last_backend": self.last_backend,
            "training_path_enabled": False,
            "native_kernel_present": self.last_backend == "rust_cuda_adamw_v0",
            "native_backend_error": self._native_backend_error,
            "notes": [
                "developer_only_layout_prototype",
                "dense_gradients_only",
                "no_runtime_dispatch",
            ],
        }


def clone_flat_adamw_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state_dict)


__all__ = [
    "FlatAdamWBufferLayout",
    "FlatAdamWConfig",
    "FlatAdamWStepReport",
    "PersistentFlatAdamW",
    "clone_flat_adamw_state_dict",
]
