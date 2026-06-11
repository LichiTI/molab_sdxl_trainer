"""Native module weight residency helpers.

Warehouse layer-level residency for native Lulynx modules.  The first targets
are frozen Linear/Conv2d base weights during LoRA training: weights can live in
CPU pinned memory and be materialized only for the current forward/backward
math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..transfer_format import (
    PackedTensor,
    TransferFormatPolicy,
    decode_transfer_tensor,
    estimate_transfer_bytes,
    normalize_transfer_format,
    pack_tensor_for_transfer,
)
from ..pcie_cache_profiler import build_module_pcie_cache_profile
from ..pcie_cache_runtime import apply_pcie_cache_v0


VALID_WEIGHT_RESIDENCY_MODES = {"resident", "linear_cpu_pinned", "linear_conv_cpu_pinned"}


def normalize_weight_residency_mode(value: Any) -> str:
    mode = str(value or "resident").strip().lower().replace("-", "_")
    aliases = {
        "off": "resident",
        "gpu": "resident",
        "cpu_pinned": "linear_cpu_pinned",
        "linear_pinned": "linear_cpu_pinned",
        "conv_cpu_pinned": "linear_conv_cpu_pinned",
        "all_cpu_pinned": "linear_conv_cpu_pinned",
        "linear_conv_pinned": "linear_conv_cpu_pinned",
    }
    mode = aliases.get(mode, mode)
    return mode if mode in VALID_WEIGHT_RESIDENCY_MODES else "resident"


def _pin_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach()
    if value.device.type != "cpu":
        value = value.to("cpu", copy=True)
    if torch.cuda.is_available() and value.is_floating_point():
        try:
            value = value.pin_memory()
        except RuntimeError:
            pass
    return value


def _normalize_transfer_policy(value: TransferFormatPolicy | str | None) -> TransferFormatPolicy | None:
    if value is None:
        return None
    if isinstance(value, TransferFormatPolicy):
        return value
    fmt = normalize_transfer_format(value, default="raw_fp16")
    if fmt == "raw_fp16":
        return None
    return TransferFormatPolicy(format=fmt, experimental=True)


def _normalize_cuda_device(device: torch.device | str) -> torch.device:
    target = torch.device(device)
    if target.type == "cuda" and target.index is None and torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return target


class _FrozenLinearCpuPinnedFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        input_tensor: torch.Tensor,
        weight_cpu: torch.Tensor,
        bias_cpu: torch.Tensor | None,
        packed_weight: PackedTensor | None = None,
        prefetched_weight: torch.Tensor | None = None,
        prefetched_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = input_tensor.device
        dtype = input_tensor.dtype if input_tensor.is_floating_point() else weight_cpu.dtype
        weight = prefetched_weight
        if weight is None or weight.device != device or weight.dtype != dtype:
            if packed_weight is not None:
                weight = decode_transfer_tensor(packed_weight, device=device, compute_dtype=dtype)
            else:
                weight = weight_cpu.to(device=device, dtype=dtype, non_blocking=True)
        bias = prefetched_bias
        if bias_cpu is None:
            bias = None
        elif bias is None or bias.device != device or bias.dtype != dtype:
            bias = bias_cpu.to(device=device, dtype=dtype, non_blocking=True)
        ctx.save_for_backward(weight_cpu)
        ctx.packed_weight = packed_weight
        ctx.runtime_weight = weight
        ctx.input_shape = tuple(input_tensor.shape)
        return F.linear(input_tensor, weight, bias)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None, None, None, None, None]:
        (weight_cpu,) = ctx.saved_tensors
        packed_weight = getattr(ctx, "packed_weight", None)
        runtime_weight = getattr(ctx, "runtime_weight", None)
        if runtime_weight is not None and runtime_weight.device == grad_output.device and runtime_weight.dtype == grad_output.dtype:
            weight = runtime_weight
        elif packed_weight is not None:
            weight = decode_transfer_tensor(packed_weight, device=grad_output.device, compute_dtype=grad_output.dtype)
        else:
            weight = weight_cpu.to(device=grad_output.device, dtype=grad_output.dtype, non_blocking=True)
        grad_input = grad_output.reshape(-1, grad_output.shape[-1]).matmul(weight)
        grad_input = grad_input.reshape(ctx.input_shape)
        return grad_input, None, None, None, None, None


class LulynxManagedLinear(nn.Linear):
    """Linear layer with optional CPU-pinned frozen-weight residency."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any = None, dtype: Any = None) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.lulynx_weight_residency_mode = "resident"
        self.lulynx_weight_residency_active = False
        self._lulynx_prefetched_linear: tuple[torch.device, torch.dtype, torch.cuda.Event, torch.Tensor, torch.Tensor | None] | None = None
        self._lulynx_prefetch_submitted = 0
        self._lulynx_prefetch_consumed = 0
        self._lulynx_prefetch_missed = 0
        self._lulynx_prefetch_errors = 0
        self._lulynx_transfer_policy: TransferFormatPolicy | None = None
        self._lulynx_packed_weight: PackedTensor | None = None
        self._lulynx_transfer_pack_errors = 0
        self._lulynx_transfer_decode_errors = 0
        self._lulynx_gpu_cache: tuple[torch.device, torch.dtype, torch.Tensor, torch.Tensor | None, int] | None = None
        self._lulynx_gpu_cache_hits = 0
        self._lulynx_gpu_cache_misses = 0
        self._lulynx_gpu_cache_errors = 0

    def enable_cpu_pinned_residency(self, *, transfer_policy: TransferFormatPolicy | str | None = None) -> bool:
        if self.weight.requires_grad:
            return False
        self.clear_cpu_pinned_prefetch()
        self._lulynx_transfer_policy = None
        self._lulynx_packed_weight = None
        with torch.no_grad():
            self.weight.data = _pin_cpu_tensor(self.weight.data)
            if self.bias is not None:
                self.bias.data = _pin_cpu_tensor(self.bias.data)
        policy = _normalize_transfer_policy(transfer_policy)
        if policy is not None:
            try:
                self._lulynx_packed_weight = pack_tensor_for_transfer(self.weight.data, policy)
                self._lulynx_transfer_policy = policy
            except Exception:
                self._lulynx_packed_weight = None
                self._lulynx_transfer_policy = None
                self._lulynx_transfer_pack_errors += 1
        self.lulynx_weight_residency_mode = "linear_cpu_pinned"
        self.lulynx_weight_residency_active = True
        return True

    def disable_cpu_pinned_residency(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        self.clear_cpu_pinned_prefetch()
        self.clear_cpu_pinned_gpu_cache()
        self._lulynx_transfer_policy = None
        self._lulynx_packed_weight = None
        target_device = torch.device(device) if device is not None else None
        with torch.no_grad():
            if target_device is not None:
                self.weight.data = self.weight.data.to(device=target_device, dtype=dtype or self.weight.dtype)
                if self.bias is not None:
                    self.bias.data = self.bias.data.to(device=target_device, dtype=dtype or self.bias.dtype)
            elif dtype is not None:
                self.weight.data = self.weight.data.to(dtype=dtype)
                if self.bias is not None:
                    self.bias.data = self.bias.data.to(dtype=dtype)
        self.lulynx_weight_residency_mode = "resident"
        self.lulynx_weight_residency_active = False

    def clear_cpu_pinned_prefetch(self) -> None:
        self._lulynx_prefetched_linear = None

    def clear_cpu_pinned_gpu_cache(self) -> None:
        self._lulynx_gpu_cache = None

    def enable_cpu_pinned_gpu_cache(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> bool:
        if (
            not self.lulynx_weight_residency_active
            or self.weight.device.type != "cpu"
            or self.weight.requires_grad
            or not torch.cuda.is_available()
        ):
            return False
        target_device = _normalize_cuda_device(device)
        if target_device.type != "cuda":
            return False
        target_dtype = dtype or self.weight.dtype
        existing = self._lulynx_gpu_cache
        if existing is not None and existing[0] == target_device and existing[1] == target_dtype:
            return True
        try:
            weight = self._materialize_cpu_weight(device=target_device, dtype=target_dtype)
            bias = self.bias.to(device=target_device, dtype=target_dtype, non_blocking=True) if self.bias is not None else None
            cache_bytes = int(weight.numel() * max(weight.element_size(), 1))
            if bias is not None:
                cache_bytes += int(bias.numel() * max(bias.element_size(), 1))
            self._lulynx_gpu_cache = (target_device, target_dtype, weight, bias, cache_bytes)
            return True
        except Exception:
            self._lulynx_gpu_cache = None
            self._lulynx_gpu_cache_errors += 1
            return False

    def _consume_cpu_pinned_gpu_cache(
        self,
        input_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        cached = self._lulynx_gpu_cache
        if cached is None or input_tensor.device.type != "cuda":
            return None, None
        device, dtype, weight, bias, _cache_bytes = cached
        expected_dtype = input_tensor.dtype if input_tensor.is_floating_point() else self.weight.dtype
        if device != input_tensor.device:
            self._lulynx_gpu_cache_misses += 1
            return None, None
        if dtype != expected_dtype:
            try:
                self._lulynx_gpu_cache_hits += 1
                converted_bias = bias.to(dtype=expected_dtype) if bias is not None else None
                return weight.to(dtype=expected_dtype), converted_bias
            except Exception:
                self._lulynx_gpu_cache_errors += 1
                return None, None
        self._lulynx_gpu_cache_hits += 1
        return weight, bias

    def get_cpu_pinned_gpu_cache_stats(self) -> dict[str, Any]:
        cached = self._lulynx_gpu_cache
        cache_bytes = int(cached[4]) if cached is not None else 0
        return {
            "enabled": cached is not None,
            "cache_bytes": cache_bytes,
            "cache_mb": round(float(cache_bytes) / (1024 * 1024), 3),
            "hits": int(self._lulynx_gpu_cache_hits),
            "misses": int(self._lulynx_gpu_cache_misses),
            "errors": int(self._lulynx_gpu_cache_errors),
        }

    def _materialize_cpu_weight(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._lulynx_packed_weight is not None:
            try:
                return decode_transfer_tensor(self._lulynx_packed_weight, device=device, compute_dtype=dtype)
            except Exception:
                self._lulynx_transfer_decode_errors += 1
        return self.weight.to(device=device, dtype=dtype, non_blocking=True)

    def prefetch_cpu_pinned_residency(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> bool:
        if (
            not self.lulynx_weight_residency_active
            or self.weight.device.type != "cpu"
            or self.weight.requires_grad
            or not torch.cuda.is_available()
        ):
            return False
        target_device = _normalize_cuda_device(device)
        if target_device.type != "cuda":
            return False
        target_dtype = dtype or self.weight.dtype
        existing = self._lulynx_prefetched_linear
        if existing is not None and existing[0] == target_device and existing[1] == target_dtype:
            return False
        self.clear_cpu_pinned_prefetch()
        try:
            active_stream = stream or torch.cuda.current_stream(target_device)
            with torch.cuda.stream(active_stream):
                weight = self._materialize_cpu_weight(device=target_device, dtype=target_dtype)
                bias = self.bias.to(device=target_device, dtype=target_dtype, non_blocking=True) if self.bias is not None else None
                event = torch.cuda.Event(blocking=False)
                event.record(active_stream)
            self._lulynx_prefetched_linear = (target_device, target_dtype, event, weight, bias)
            self._lulynx_prefetch_submitted += 1
            return True
        except Exception:
            self.clear_cpu_pinned_prefetch()
            self._lulynx_prefetch_errors += 1
            return False

    def _consume_cpu_pinned_prefetch(
        self,
        input_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        cached = self._lulynx_prefetched_linear
        if cached is None or input_tensor.device.type != "cuda":
            self._lulynx_prefetch_missed += 1
            return None, None
        device, dtype, event, weight, bias = cached
        expected_dtype = input_tensor.dtype if input_tensor.is_floating_point() else self.weight.dtype
        if device != input_tensor.device or dtype != expected_dtype:
            self.clear_cpu_pinned_prefetch()
            self._lulynx_prefetch_missed += 1
            return None, None
        torch.cuda.current_stream(input_tensor.device).wait_event(event)
        self._lulynx_prefetched_linear = None
        self._lulynx_prefetch_consumed += 1
        return weight, bias

    def get_cpu_pinned_prefetch_stats(self) -> dict[str, int]:
        return {
            "submitted": int(self._lulynx_prefetch_submitted),
            "consumed": int(self._lulynx_prefetch_consumed),
            "missed": int(self._lulynx_prefetch_missed),
            "errors": int(self._lulynx_prefetch_errors),
            "pending": 1 if self._lulynx_prefetched_linear is not None else 0,
        }

    def get_transfer_format_stats(self) -> dict[str, Any]:
        packed = self._lulynx_packed_weight
        fmt = self._lulynx_transfer_policy.format if self._lulynx_transfer_policy is not None else "raw_fp16"
        bias_bytes = 0
        if self.bias is not None:
            bias_bytes = int(self.bias.numel() * max(self.bias.element_size(), 1))
        transfer_bytes = int(packed.transfer_bytes if packed is not None else estimate_transfer_bytes(tuple(self.weight.shape), "raw_fp16"))
        cache_stats = self.get_cpu_pinned_gpu_cache_stats()
        return {
            "format": fmt,
            "packed": packed is not None,
            "transfer_bytes": int(transfer_bytes + bias_bytes),
            "transfer_mb": round(float(transfer_bytes + bias_bytes) / (1024 * 1024), 3),
            "pack_errors": int(self._lulynx_transfer_pack_errors),
            "decode_errors": int(self._lulynx_transfer_decode_errors),
            "gpu_cache_enabled": bool(cache_stats.get("enabled")),
            "gpu_cache_mb": float(cache_stats.get("cache_mb", 0.0) or 0.0),
            "gpu_cache_hits": int(cache_stats.get("hits", 0) or 0),
            "gpu_cache_misses": int(cache_stats.get("misses", 0) or 0),
            "gpu_cache_errors": int(cache_stats.get("errors", 0) or 0),
        }

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if (
            self.lulynx_weight_residency_active
            and self.weight.device.type == "cpu"
            and not self.weight.requires_grad
            and input_tensor.device.type != "cpu"
        ):
            cached_weight, cached_bias = self._consume_cpu_pinned_gpu_cache(input_tensor)
            if cached_weight is not None:
                return _FrozenLinearCpuPinnedFn.apply(
                    input_tensor,
                    self.weight,
                    self.bias,
                    None,
                    cached_weight,
                    cached_bias,
                )
            prefetched_weight, prefetched_bias = self._consume_cpu_pinned_prefetch(input_tensor)
            return _FrozenLinearCpuPinnedFn.apply(
                input_tensor,
                self.weight,
                self.bias,
                self._lulynx_packed_weight,
                prefetched_weight,
                prefetched_bias,
            )
        return F.linear(input_tensor, self.weight, self.bias)


class _FrozenConv2dCpuPinnedFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        input_tensor: torch.Tensor,
        weight_cpu: torch.Tensor,
        bias_cpu: torch.Tensor | None,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
    ) -> torch.Tensor:
        device = input_tensor.device
        dtype = input_tensor.dtype if input_tensor.is_floating_point() else weight_cpu.dtype
        weight = weight_cpu.to(device=device, dtype=dtype, non_blocking=True)
        bias = bias_cpu.to(device=device, dtype=dtype, non_blocking=True) if bias_cpu is not None else None
        ctx.save_for_backward(weight_cpu)
        ctx.input_shape = tuple(input_tensor.shape)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = int(groups)
        return F.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None, None, None, None, None, None]:
        (weight_cpu,) = ctx.saved_tensors
        weight = weight_cpu.to(device=grad_output.device, dtype=grad_output.dtype, non_blocking=True)
        grad_input = torch.nn.grad.conv2d_input(
            ctx.input_shape,
            weight,
            grad_output,
            stride=ctx.stride,
            padding=ctx.padding,
            dilation=ctx.dilation,
            groups=ctx.groups,
        )
        return grad_input, None, None, None, None, None, None


class LulynxManagedConv2d(nn.Conv2d):
    """Conv2d layer with optional CPU-pinned frozen-weight residency."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lulynx_weight_residency_mode = "resident"
        self.lulynx_weight_residency_active = False

    def enable_cpu_pinned_residency(self) -> bool:
        if self.weight.requires_grad:
            return False
        with torch.no_grad():
            self.weight.data = _pin_cpu_tensor(self.weight.data)
            if self.bias is not None:
                self.bias.data = _pin_cpu_tensor(self.bias.data)
        self.lulynx_weight_residency_mode = "linear_conv_cpu_pinned"
        self.lulynx_weight_residency_active = True
        return True

    def disable_cpu_pinned_residency(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        target_device = torch.device(device) if device is not None else None
        with torch.no_grad():
            if target_device is not None:
                self.weight.data = self.weight.data.to(device=target_device, dtype=dtype or self.weight.dtype)
                if self.bias is not None:
                    self.bias.data = self.bias.data.to(device=target_device, dtype=dtype or self.bias.dtype)
            elif dtype is not None:
                self.weight.data = self.weight.data.to(dtype=dtype)
                if self.bias is not None:
                    self.bias.data = self.bias.data.to(dtype=dtype)
        self.lulynx_weight_residency_mode = "resident"
        self.lulynx_weight_residency_active = False

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if (
            self.lulynx_weight_residency_active
            and self.weight.device.type == "cpu"
            and not self.weight.requires_grad
            and input_tensor.device.type != "cpu"
        ):
            return _FrozenConv2dCpuPinnedFn.apply(
                input_tensor,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return F.conv2d(input_tensor, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


@dataclass
class WeightResidencyReport:
    mode: str
    managed_linear_count: int = 0
    active_linear_count: int = 0
    managed_conv2d_count: int = 0
    active_conv2d_count: int = 0
    skipped_small_count: int = 0
    min_parameter_count: int = 0
    cpu_parameter_mb: float = 0.0
    transfer_format: str = "off"
    transfer_packed_linear_count: int = 0
    transfer_h2d_mb: float = 0.0
    transfer_pack_errors: int = 0
    transfer_decode_errors: int = 0
    pcie_delta_cache: dict[str, Any] | None = None
    pcie_cache_v0: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "managed_linear_count": int(self.managed_linear_count),
            "active_linear_count": int(self.active_linear_count),
            "managed_conv2d_count": int(self.managed_conv2d_count),
            "active_conv2d_count": int(self.active_conv2d_count),
            "skipped_small_count": int(self.skipped_small_count),
            "min_parameter_count": int(self.min_parameter_count),
            "cpu_parameter_mb": round(float(self.cpu_parameter_mb), 3),
            "transfer_format": self.transfer_format,
            "transfer_packed_linear_count": int(self.transfer_packed_linear_count),
            "transfer_h2d_mb": round(float(self.transfer_h2d_mb), 3),
            "transfer_pack_errors": int(self.transfer_pack_errors),
            "transfer_decode_errors": int(self.transfer_decode_errors),
            "pcie_delta_cache": dict(self.pcie_delta_cache or {}),
            "pcie_cache_v0": dict(self.pcie_cache_v0 or {}),
        }


def _module_parameter_count(module: nn.Module) -> int:
    return sum(int(param.numel()) for param in module.parameters(recurse=False))


def _over_threshold(module: nn.Module, min_parameter_count: int) -> bool:
    return _module_parameter_count(module) >= max(int(min_parameter_count or 0), 0)


def apply_weight_residency(
    module: nn.Module,
    *,
    mode: str = "resident",
    min_parameter_count: int = 0,
    transfer_format: str | None = None,
    pcie_delta_cache_enabled: bool = False,
    pcie_delta_cache_mode: str = "observe",
    pcie_delta_cache_budget_mb: float = 256.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> WeightResidencyReport:
    normalized = normalize_weight_residency_mode(mode)
    min_parameter_count = max(int(min_parameter_count or 0), 0)
    requested_transfer_format = "off"
    if str(transfer_format or "").strip().lower() not in {"", "off", "none", "disabled"}:
        requested_transfer_format = normalize_transfer_format(transfer_format, default="raw_fp16")
    transfer_policy = _normalize_transfer_policy(transfer_format)
    report = WeightResidencyReport(
        mode=normalized,
        min_parameter_count=min_parameter_count,
        transfer_format=requested_transfer_format,
        pcie_delta_cache={"enabled": False, "mode": "observe", "reason": "disabled"},
        pcie_cache_v0={"enabled": False, "mode": "observe", "reason": "disabled"},
    )
    cache_units: list[tuple[str, LulynxManagedLinear, int]] = []
    for name, child in module.named_modules():
        if isinstance(child, LulynxManagedLinear):
            report.managed_linear_count += 1
            if normalized in {"linear_cpu_pinned", "linear_conv_cpu_pinned"} and _over_threshold(child, min_parameter_count):
                active = child.enable_cpu_pinned_residency(transfer_policy=transfer_policy)
            else:
                child.disable_cpu_pinned_residency()
                active = False
                if normalized in {"linear_cpu_pinned", "linear_conv_cpu_pinned"}:
                    report.skipped_small_count += 1
            if active:
                report.active_linear_count += 1
                cache_units.append((str(name), child, _module_parameter_count(child)))
                transfer_stats = child.get_transfer_format_stats()
                if transfer_stats.get("packed"):
                    report.transfer_packed_linear_count += 1
                report.transfer_h2d_mb += float(transfer_stats.get("transfer_mb", 0.0) or 0.0)
                report.transfer_pack_errors += int(transfer_stats.get("pack_errors", 0) or 0)
                report.transfer_decode_errors += int(transfer_stats.get("decode_errors", 0) or 0)
                for param in child.parameters(recurse=False):
                    if param.device.type == "cpu":
                        report.cpu_parameter_mb += param.numel() * max(param.element_size(), 1) / (1024 * 1024)
            continue
        if isinstance(child, LulynxManagedConv2d):
            report.managed_conv2d_count += 1
            if normalized == "linear_conv_cpu_pinned" and _over_threshold(child, min_parameter_count):
                active = child.enable_cpu_pinned_residency()
            else:
                child.disable_cpu_pinned_residency()
                active = False
                if normalized == "linear_conv_cpu_pinned":
                    report.skipped_small_count += 1
            if active:
                report.active_conv2d_count += 1
                for param in child.parameters(recurse=False):
                    if param.device.type == "cpu":
                        report.cpu_parameter_mb += param.numel() * max(param.element_size(), 1) / (1024 * 1024)
    cache_profile = build_module_pcie_cache_profile(
        cache_units,
        enabled=bool(pcie_delta_cache_enabled) and report.active_linear_count > 0,
        family="sdxl",
        mode=normalized,
    )
    cache_payload = cache_profile.as_dict()
    if not bool(pcie_delta_cache_enabled):
        cache_payload["reason"] = "disabled"
    elif report.active_linear_count <= 0:
        cache_payload["reason"] = "no active CPU-pinned Linear units"
    else:
        cache_payload["reason"] = "observe_only"
    report.pcie_delta_cache = cache_payload
    cache_v0 = apply_pcie_cache_v0(
        cache_units,
        enabled=bool(pcie_delta_cache_enabled) and str(pcie_delta_cache_mode or "observe").strip().lower() == "cache_v0",
        mode=str(pcie_delta_cache_mode or "observe"),
        budget_mb=float(pcie_delta_cache_budget_mb or 0.0),
        device=device,
        dtype=dtype,
    )
    report.pcie_cache_v0 = cache_v0.as_dict()
    return report

