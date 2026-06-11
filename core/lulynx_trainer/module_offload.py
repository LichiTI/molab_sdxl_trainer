"""Warehouse module-level CPU offload for frozen Linear / Conv modules."""

from __future__ import annotations

import fnmatch
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module_offload_contract import (
    MODULE_OFFLOAD_SCOPE_BACKBONE,
    MODULE_OFFLOAD_SCOPE_ORDER,
    MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_1,
    MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_2,
    ModuleOffloadConfigView,
    resolve_module_offload_config,
)


SUPPORTED_MODULE_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
PREFETCH_SUPPORTED_MODULE_TYPES = (nn.Linear,)


@dataclass(frozen=True)
class ModuleOffloadCandidate:
    scope: str
    path: str
    module: nn.Module = field(repr=False)
    module_type: str
    param_bytes: int

    @property
    def param_mb(self) -> float:
        return float(self.param_bytes) / (1024.0 * 1024.0)


@dataclass
class ModuleOffloadScopePlan:
    scope: str
    ratio: int
    candidate_count: int = 0
    selected_count: int = 0
    selected_param_bytes: int = 0
    selected_paths: List[str] = field(default_factory=list)
    selected_candidates: List[ModuleOffloadCandidate] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ratio": self.ratio,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "selected_param_bytes": self.selected_param_bytes,
            "selected_param_mb": float(self.selected_param_bytes) / (1024.0 * 1024.0),
            "selected_paths": list(self.selected_paths),
        }


@dataclass
class ModuleOffloadPlan:
    config: ModuleOffloadConfigView
    scopes: Dict[str, ModuleOffloadScopePlan]
    enabled: bool
    source: str = "runtime"
    reason: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def selected_count(self) -> int:
        return sum(scope.selected_count for scope in self.scopes.values())

    @property
    def candidate_count(self) -> int:
        return sum(scope.candidate_count for scope in self.scopes.values())

    @property
    def selected_param_bytes(self) -> int:
        return sum(scope.selected_param_bytes for scope in self.scopes.values())

    def iter_selected_candidates(self) -> Iterator[ModuleOffloadCandidate]:
        for scope_name in MODULE_OFFLOAD_SCOPE_ORDER:
            scope = self.scopes.get(scope_name)
            if scope is None:
                continue
            for candidate in scope.selected_candidates:
                yield candidate

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": "module_offload" if self.enabled else "none",
            "source": self.source,
            "reason": self.reason,
            "warnings": list(self.warnings),
            "ratio": self.config.ratio,
            "profile": self.config.profile,
            "profile_enabled": self.config.profile_enabled,
            "min_param_mb": self.config.min_param_mb,
            "include_patterns": self.config.include_patterns,
            "exclude_patterns": self.config.exclude_patterns,
            "verify_state": self.config.verify_state,
            "prefetch_enabled": self.config.prefetch_enabled,
            "prefetch_mode": self.config.prefetch_mode,
            "backbone_ratio": self.config.effective_backbone_ratio,
            "text_encoder_ratio": self.config.effective_text_encoder_ratio,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "selected_param_bytes": self.selected_param_bytes,
            "estimated_transfer_mb": float(self.selected_param_bytes) / (1024.0 * 1024.0),
            "scopes": {
                scope_name: self.scopes[scope_name].as_dict()
                for scope_name in MODULE_OFFLOAD_SCOPE_ORDER
                if scope_name in self.scopes
            },
        }


def _local_param_bytes(module: nn.Module) -> int:
    total = 0
    for param in module.parameters(recurse=False):
        total += param.numel() * param.element_size()
    return total


def _is_supported_module(module: nn.Module) -> bool:
    if type(module) not in SUPPORTED_MODULE_TYPES:
        return False
    if getattr(module, "_lora_leaf", False):
        return False
    if getattr(module, "weight", None) is None:
        return False
    direct_params = list(module.named_parameters(recurse=False))
    if not direct_params:
        return False
    if any(name not in {"weight", "bias"} for name, _ in direct_params):
        return False
    if any(param.requires_grad for _, param in direct_params):
        return False
    return True


def _split_patterns(patterns: str) -> List[str]:
    return [
        part.strip().lower()
        for part in str(patterns or "").replace("\n", ",").replace(";", ",").split(",")
        if part.strip()
    ]


def _matches_any(candidate: ModuleOffloadCandidate, patterns: List[str]) -> bool:
    if not patterns:
        return False
    values = (
        candidate.path.lower(),
        f"{candidate.scope}.{candidate.path}".lower(),
        candidate.module_type.lower(),
    )
    return any(pattern in value or fnmatch.fnmatch(value, pattern) for pattern in patterns for value in values)


def _filter_candidates(
    candidates: List[ModuleOffloadCandidate],
    *,
    min_param_mb: float,
    include_patterns: str,
    exclude_patterns: str,
) -> List[ModuleOffloadCandidate]:
    min_bytes = int(max(float(min_param_mb or 0.0), 0.0) * 1024.0 * 1024.0)
    include = _split_patterns(include_patterns)
    exclude = _split_patterns(exclude_patterns)
    filtered: List[ModuleOffloadCandidate] = []
    for candidate in candidates:
        if min_bytes and candidate.param_bytes < min_bytes:
            continue
        if include and not _matches_any(candidate, include):
            continue
        if exclude and _matches_any(candidate, exclude):
            continue
        filtered.append(candidate)
    return filtered


def _iter_scope_candidates(root: Optional[nn.Module], scope_name: str) -> List[ModuleOffloadCandidate]:
    if root is None:
        return []
    candidates: List[ModuleOffloadCandidate] = []
    seen: set[int] = set()
    for path, module in root.named_modules():
        if id(module) in seen:
            continue
        if not _is_supported_module(module):
            continue
        seen.add(id(module))
        stable_path = path or "__root__"
        candidates.append(
            ModuleOffloadCandidate(
                scope=scope_name,
                path=stable_path,
                module=module,
                module_type=type(module).__name__,
                param_bytes=_local_param_bytes(module),
            )
        )
    candidates.sort(key=lambda item: (-item.param_bytes, item.path))
    return candidates


def select_module_offload_count(total: int, ratio: int) -> int:
    if total <= 0 or ratio <= 0:
        return 0
    return max(1, min(total, int(math.ceil(total * float(ratio) / 100.0))))


def _build_scope_plan(scope_name: str, ratio: int, candidates: List[ModuleOffloadCandidate]) -> ModuleOffloadScopePlan:
    selected_count = select_module_offload_count(len(candidates), ratio)
    selected = candidates[:selected_count]
    return ModuleOffloadScopePlan(
        scope=scope_name,
        ratio=ratio,
        candidate_count=len(candidates),
        selected_count=len(selected),
        selected_param_bytes=sum(candidate.param_bytes for candidate in selected),
        selected_paths=[candidate.path for candidate in selected],
        selected_candidates=list(selected),
    )


def build_module_offload_plan(
    backbone: Optional[nn.Module],
    text_encoder_1: Optional[nn.Module],
    text_encoder_2: Optional[nn.Module],
    config: ModuleOffloadConfigView | Dict[str, Any] | Any,
) -> ModuleOffloadPlan:
    view = config if isinstance(config, ModuleOffloadConfigView) else resolve_module_offload_config(config)
    scope_candidates = {
        MODULE_OFFLOAD_SCOPE_BACKBONE: _filter_candidates(
            _iter_scope_candidates(backbone, MODULE_OFFLOAD_SCOPE_BACKBONE),
            min_param_mb=view.min_param_mb,
            include_patterns=view.include_patterns,
            exclude_patterns=view.exclude_patterns,
        ),
        MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_1: _filter_candidates(
            _iter_scope_candidates(text_encoder_1, MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_1),
            min_param_mb=view.min_param_mb,
            include_patterns=view.include_patterns,
            exclude_patterns=view.exclude_patterns,
        ),
        MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_2: _filter_candidates(
            _iter_scope_candidates(text_encoder_2, MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_2),
            min_param_mb=view.min_param_mb,
            include_patterns=view.include_patterns,
            exclude_patterns=view.exclude_patterns,
        ),
    }
    scopes = {
        MODULE_OFFLOAD_SCOPE_BACKBONE: _build_scope_plan(
            MODULE_OFFLOAD_SCOPE_BACKBONE,
            view.effective_backbone_ratio,
            scope_candidates[MODULE_OFFLOAD_SCOPE_BACKBONE],
        ),
        MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_1: _build_scope_plan(
            MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_1,
            view.effective_text_encoder_ratio,
            scope_candidates[MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_1],
        ),
        MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_2: _build_scope_plan(
            MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_2,
            view.effective_text_encoder_ratio,
            scope_candidates[MODULE_OFFLOAD_SCOPE_TEXT_ENCODER_2],
        ),
    }
    enabled = any(scope.selected_count > 0 for scope in scopes.values())
    reason = ""
    if view.requested and not enabled:
        reason = "module_offload requested but no eligible frozen Linear/Conv modules were found."
    return ModuleOffloadPlan(config=view, scopes=scopes, enabled=enabled, reason=reason)


@dataclass
class _ManagedModule:
    candidate: ModuleOffloadCandidate
    module: nn.Module
    cpu_weight: torch.Tensor
    cpu_bias: Optional[torch.Tensor]
    original_forward: Callable[..., Any]


class ModuleResidencyManager:
    """Keep selected frozen modules on CPU and materialize temporary working tensors on demand."""

    def __init__(self, plan: ModuleOffloadPlan, device: torch.device | str = "cuda") -> None:
        self.plan = plan
        self.device = torch.device(device)
        self._records: List[_ManagedModule] = []
        self._record_by_module_id: Dict[int, _ManagedModule] = {}
        self._step_depth = 0
        self._active_staging_tensors: List[torch.Tensor] = []
        self._materialize_count = 0
        self._materialize_bytes = 0
        self._materialize_seconds = 0.0
        self._materialize_by_module: Dict[str, Dict[str, float]] = {}
        self._prefetch_enabled = bool(plan.config.prefetch_enabled)
        self._prefetch_degraded_reason = ""
        self._record_index_by_module_id: Dict[int, int] = {}
        self._install()

    @property
    def managed_module_count(self) -> int:
        return len(self._records)

    def estimate_vram_savings_mb(self) -> float:
        return float(self.plan.selected_param_bytes) / (1024.0 * 1024.0)

    def _pin_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.device.type != "cuda" or tensor.device.type != "cpu":
            return tensor
        try:
            return tensor.pin_memory()
        except RuntimeError:
            return tensor

    def _clone_authority_tensor(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        authority = tensor.detach().to(device="cpu").contiguous()
        return self._pin_tensor(authority)

    def _install(self) -> None:
        for candidate in self.plan.iter_selected_candidates():
            module = candidate.module
            cpu_weight = self._clone_authority_tensor(getattr(module, "weight", None))
            cpu_bias = self._clone_authority_tensor(getattr(module, "bias", None))
            if cpu_weight is None:
                continue
            module.weight.data = cpu_weight
            if getattr(module, "bias", None) is not None and cpu_bias is not None:
                module.bias.data = cpu_bias
            original_forward = module.forward
            record = _ManagedModule(
                candidate=candidate,
                module=module,
                cpu_weight=cpu_weight,
                cpu_bias=cpu_bias,
                original_forward=original_forward,
            )
            module.forward = MethodType(self._build_patched_forward(record), module)
            self._records.append(record)
            self._record_by_module_id[id(module)] = record
            self._record_index_by_module_id[id(module)] = len(self._records) - 1

    def _record_materialize(self, record: _ManagedModule, bytes_count: int, seconds: float) -> None:
        self._materialize_count += 1
        self._materialize_bytes += int(bytes_count)
        self._materialize_seconds += max(float(seconds), 0.0)
        key = f"{record.candidate.scope}.{record.candidate.path}"
        bucket = self._materialize_by_module.setdefault(key, {"count": 0.0, "bytes": 0.0, "seconds": 0.0})
        bucket["count"] += 1.0
        bucket["bytes"] += float(bytes_count)
        bucket["seconds"] += max(float(seconds), 0.0)

    def _materialize_tensor(
        self,
        cpu_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        record: Optional[_ManagedModule] = None,
    ) -> torch.Tensor:
        target_dtype = cpu_tensor.dtype
        if cpu_tensor.is_floating_point() and input_tensor.is_floating_point():
            target_dtype = input_tensor.dtype
        start = time.perf_counter()
        tensor = cpu_tensor.to(
            device=input_tensor.device,
            dtype=target_dtype,
            non_blocking=(input_tensor.device.type == "cuda" and cpu_tensor.device.type == "cpu"),
        )
        seconds = time.perf_counter() - start
        if record is not None:
            self._record_materialize(record, tensor.numel() * tensor.element_size(), seconds)
        if self._step_depth > 0:
            self._active_staging_tensors.append(tensor)
        return tensor

    def _try_prefetch_next(self, record: _ManagedModule, input_tensor: torch.Tensor) -> None:
        if not self._prefetch_enabled or self._prefetch_degraded_reason:
            return
        if record.candidate.scope != MODULE_OFFLOAD_SCOPE_BACKBONE or type(record.module) not in PREFETCH_SUPPORTED_MODULE_TYPES:
            return
        if input_tensor.device.type != "cuda":
            self._prefetch_degraded_reason = "prefetch requires CUDA input tensors; using normal module_offload"
            return
        idx = self._record_index_by_module_id.get(id(record.module))
        if idx is None or idx + 1 >= len(self._records):
            return
        next_record = self._records[idx + 1]
        if next_record.candidate.scope != MODULE_OFFLOAD_SCOPE_BACKBONE or type(next_record.module) not in PREFETCH_SUPPORTED_MODULE_TYPES:
            return
        try:
            self._materialize_tensor(next_record.cpu_weight, input_tensor, None)
            if next_record.cpu_bias is not None:
                self._materialize_tensor(next_record.cpu_bias, input_tensor, None)
        except Exception as exc:
            self._prefetch_degraded_reason = f"prefetch failed ({exc}); using normal module_offload"

    @staticmethod
    def _normalize_padding(padding: Any, dims: int) -> Any:
        if isinstance(padding, str):
            return padding
        if isinstance(padding, int):
            return (padding,) * dims
        return tuple(padding)

    @staticmethod
    def _reversed_padding_repeated_twice(padding: Any, dims: int) -> tuple[int, ...]:
        padding_tuple = ModuleResidencyManager._normalize_padding(padding, dims)
        if isinstance(padding_tuple, str):
            return tuple()
        expanded: List[int] = []
        for value in reversed(tuple(padding_tuple)):
            expanded.extend((int(value), int(value)))
        return tuple(expanded)

    def _conv_forward(
        self,
        module: nn.Module,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if type(module) is nn.Conv1d:
            conv_fn = F.conv1d
            dims = 1
        elif type(module) is nn.Conv2d:
            conv_fn = F.conv2d
            dims = 2
        else:
            conv_fn = F.conv3d
            dims = 3
        padding = self._normalize_padding(module.padding, dims)
        if getattr(module, "padding_mode", "zeros") != "zeros":
            input_tensor = F.pad(
                input_tensor,
                self._reversed_padding_repeated_twice(module.padding, dims),
                mode=module.padding_mode,
            )
            padding = 0
        return conv_fn(
            input_tensor,
            weight,
            bias,
            module.stride,
            padding,
            module.dilation,
            module.groups,
        )

    def _run_module_forward(
        self,
        record: _ManagedModule,
        module: nn.Module,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        weight = self._materialize_tensor(record.cpu_weight, input_tensor, record)
        bias = self._materialize_tensor(record.cpu_bias, input_tensor, record) if record.cpu_bias is not None else None
        self._try_prefetch_next(record, input_tensor)
        if record.candidate.module_type == "Linear":
            return F.linear(input_tensor, weight, bias)
        return self._conv_forward(module, input_tensor, weight, bias)

    def stats_dict(self) -> Dict[str, Any]:
        top = sorted(
            self._materialize_by_module.items(),
            key=lambda item: (item[1].get("bytes", 0.0), item[1].get("seconds", 0.0)),
            reverse=True,
        )[:10]
        return {
            "materialize_count": self._materialize_count,
            "materialize_bytes": self._materialize_bytes,
            "materialize_mb": float(self._materialize_bytes) / (1024.0 * 1024.0),
            "materialize_seconds": self._materialize_seconds,
            "top_modules": [
                {
                    "path": name,
                    "count": int(stats.get("count", 0.0)),
                    "bytes": int(stats.get("bytes", 0.0)),
                    "seconds": float(stats.get("seconds", 0.0)),
                }
                for name, stats in top
            ],
            "prefetch_enabled": self._prefetch_enabled and not bool(self._prefetch_degraded_reason),
            "prefetch_requested": bool(self.plan.config.prefetch_enabled),
            "prefetch_degraded_reason": self._prefetch_degraded_reason,
        }

    def _reset_step_stats(self) -> None:
        self._materialize_count = 0
        self._materialize_bytes = 0
        self._materialize_seconds = 0.0
        self._materialize_by_module.clear()

    def _build_patched_forward(self, record: _ManagedModule) -> Callable[..., Any]:
        def _patched_forward(module_self: nn.Module, input_tensor: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            if args or kwargs:
                return self._run_module_forward(record, module_self, input_tensor)
            return self._run_module_forward(record, module_self, input_tensor)

        return _patched_forward

    @contextmanager
    def step_context(self) -> Iterable[None]:
        self._step_depth += 1
        if self._step_depth == 1:
            self._active_staging_tensors.clear()
            self._reset_step_stats()
        try:
            yield
        finally:
            self._step_depth -= 1
            if self._step_depth <= 0:
                self._step_depth = 0
                self._active_staging_tensors.clear()

    def close(self) -> None:
        for record in self._records:
            try:
                record.module.forward = record.original_forward
            except Exception:
                pass
        self._records.clear()
        self._record_by_module_id.clear()
        self._active_staging_tensors.clear()

