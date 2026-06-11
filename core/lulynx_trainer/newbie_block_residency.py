"""Block-scoped residency helpers for native Newbie DiT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .dit_residency_planner import (
    build_dit_residency_plan,
    clear_dit_block_prefetch_controller,
    install_dit_block_prefetch_controller,
    normalize_dit_block_residency,
)
from .native_unet.weight_residency import LulynxManagedLinear
from .pcie_cache_profiler import build_dit_pcie_cache_profile
from .pcie_cache_runtime import apply_pcie_cache_v0


VALID_NEWBIE_BLOCK_RESIDENCY_MODES = {"resident", "streaming_offload", "block_cpu_pinned"}


def normalize_newbie_block_residency(value: Any) -> str:
    return normalize_dit_block_residency(value)


@dataclass
class NewbieBlockResidencyReport:
    mode: str
    block_count: int = 0
    strategy: str = "resident"
    managed_linear_count: int = 0
    planned_linear_count: int = 0
    active_linear_count: int = 0
    lora_wrapped_linear_count: int = 0
    skipped_small_count: int = 0
    hot_resident_count: int = 0
    edge_resident_count: int = 0
    cold_candidate_count: int = 0
    sparse_swap_enabled: bool = False
    sparse_warm_prefetch_count: int = 0
    sparse_cold_on_demand_count: int = 0
    requested_min_parameter_count: int = 0
    min_parameter_count: int = 0
    auto_min_parameter_count: bool = False
    auto_threshold_candidate_count: int = 0
    auto_threshold_total_parameter_count: int = 0
    planned_cpu_parameter_mb: float = 0.0
    cpu_parameter_mb: float = 0.0
    transfer_format: str = "off"
    transfer_packed_linear_count: int = 0
    transfer_h2d_mb: float = 0.0
    transfer_pack_errors: int = 0
    transfer_decode_errors: int = 0
    prefetch_enabled: bool = False
    prefetch_depth: int = 1
    prefetch: dict[str, Any] | None = None
    pcie_delta_cache: dict[str, Any] | None = None
    pcie_cache_v0: dict[str, Any] | None = None
    unit_sample: list[dict[str, Any]] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "block_count": int(self.block_count),
            "strategy": self.strategy,
            "managed_linear_count": int(self.managed_linear_count),
            "planned_linear_count": int(self.planned_linear_count),
            "active_linear_count": int(self.active_linear_count),
            "lora_wrapped_linear_count": int(self.lora_wrapped_linear_count),
            "skipped_small_count": int(self.skipped_small_count),
            "hot_resident_count": int(self.hot_resident_count),
            "edge_resident_count": int(self.edge_resident_count),
            "cold_candidate_count": int(self.cold_candidate_count),
            "sparse_swap_enabled": bool(self.sparse_swap_enabled),
            "sparse_warm_prefetch_count": int(self.sparse_warm_prefetch_count),
            "sparse_cold_on_demand_count": int(self.sparse_cold_on_demand_count),
            "requested_min_parameter_count": int(self.requested_min_parameter_count),
            "min_parameter_count": int(self.min_parameter_count),
            "auto_min_parameter_count": bool(self.auto_min_parameter_count),
            "auto_threshold_candidate_count": int(self.auto_threshold_candidate_count),
            "auto_threshold_total_parameter_count": int(self.auto_threshold_total_parameter_count),
            "planned_cpu_parameter_mb": round(float(self.planned_cpu_parameter_mb), 3),
            "cpu_parameter_mb": round(float(self.cpu_parameter_mb), 3),
            "transfer_format": self.transfer_format,
            "transfer_packed_linear_count": int(self.transfer_packed_linear_count),
            "transfer_h2d_mb": round(float(self.transfer_h2d_mb), 3),
            "transfer_pack_errors": int(self.transfer_pack_errors),
            "transfer_decode_errors": int(self.transfer_decode_errors),
            "prefetch_enabled": bool(self.prefetch_enabled),
            "prefetch_depth": int(self.prefetch_depth),
            "prefetch": dict(self.prefetch or {}),
            "pcie_delta_cache": dict(self.pcie_delta_cache or {}),
            "pcie_cache_v0": dict(self.pcie_cache_v0 or {}),
            "unit_sample": list(self.unit_sample or []),
        }


def _cpu_parameter_mb(module: nn.Module) -> float:
    total = 0.0
    for param in module.parameters(recurse=False):
        if param.device.type == "cpu":
            total += param.numel() * max(param.element_size(), 1) / (1024 * 1024)
    return total


def _move_tensor_data(tensor: torch.Tensor, *, device: torch.device, dtype: torch.dtype | None) -> torch.Tensor:
    target_dtype = dtype if dtype is not None and (tensor.is_floating_point() or tensor.is_complex()) else None
    if target_dtype is not None:
        return tensor.to(device=device, dtype=target_dtype, non_blocking=True)
    return tensor.to(device=device, non_blocking=True)


def _block_modules(model: nn.Module) -> list[nn.Module]:
    unet = getattr(model, "unet", model)
    blocks = getattr(unet, "_block_modules", None)
    if blocks:
        return list(blocks)
    for root_name in ("layers", "context_refiner", "transformer_blocks", "blocks", "double_blocks", "single_blocks"):
        root = getattr(unet, root_name, None)
        if root is None:
            continue
        try:
            return [root[index] for index in range(len(root))]
        except Exception:
            try:
                return list(root)
            except Exception:
                return [root]
    return []


def apply_newbie_block_residency(
    model: nn.Module,
    *,
    mode: str = "resident",
    min_parameter_count: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    prefetch_enabled: bool = False,
    prefetch_depth: int = 1,
    transfer_format: str | None = None,
    sparse_swap_enabled: bool = False,
    sparse_swap_budget_mb: float | None = None,
    sparse_swap_warm_fraction: float = 0.35,
    pcie_delta_cache_enabled: bool = False,
    pcie_delta_cache_mode: str = "observe",
    pcie_delta_cache_budget_mb: float = 256.0,
) -> NewbieBlockResidencyReport:
    normalized = normalize_newbie_block_residency(mode)
    requested_min_parameter_count = max(int(min_parameter_count or 0), 0)
    requested_prefetch_depth = max(int(prefetch_depth or 0), 0)
    blocks = _block_modules(model)
    plan = build_dit_residency_plan(
        blocks,
        family="newbie",
        mode=normalized,
        requested_min_parameter_count=requested_min_parameter_count,
        sparse_swap_enabled=bool(sparse_swap_enabled),
        sparse_swap_budget_mb=sparse_swap_budget_mb,
        sparse_swap_warm_fraction=sparse_swap_warm_fraction,
    )
    plan_payload = plan.as_dict()
    report = NewbieBlockResidencyReport(
        mode=normalized,
        block_count=len(blocks),
        strategy=plan.strategy,
        managed_linear_count=plan.managed_linear_count,
        planned_linear_count=plan.planned_linear_count,
        lora_wrapped_linear_count=plan.lora_wrapped_linear_count,
        skipped_small_count=plan.skipped_small_count,
        hot_resident_count=plan.hot_resident_count,
        edge_resident_count=plan.edge_resident_count,
        cold_candidate_count=plan.cold_candidate_count,
        sparse_swap_enabled=bool(sparse_swap_enabled) and normalized == "streaming_offload",
        sparse_warm_prefetch_count=plan.sparse_warm_prefetch_count,
        sparse_cold_on_demand_count=plan.sparse_cold_on_demand_count,
        requested_min_parameter_count=requested_min_parameter_count,
        min_parameter_count=plan.min_parameter_count,
        auto_min_parameter_count=plan.auto_min_parameter_count,
        auto_threshold_candidate_count=plan.auto_threshold_candidate_count,
        auto_threshold_total_parameter_count=plan.auto_threshold_total_parameter_count,
        planned_cpu_parameter_mb=plan.planned_cpu_parameter_mb,
        transfer_format=str(transfer_format or "off"),
        prefetch_enabled=bool(prefetch_enabled),
        prefetch_depth=requested_prefetch_depth,
        prefetch={"enabled": False, "reason": "disabled"},
        pcie_delta_cache={"enabled": False, "mode": "observe", "reason": "disabled"},
        pcie_cache_v0={"enabled": False, "mode": "observe", "reason": "disabled"},
        unit_sample=plan_payload.get("unit_sample", []),
    )
    cache_units: list[tuple[str, Any, int, int]] = []

    for unit in plan.units:
        active = False
        if unit.cpu_pinned:
            active = unit.module.enable_cpu_pinned_residency(transfer_policy=transfer_format)
        else:
            unit.module.disable_cpu_pinned_residency(device=device, dtype=dtype)
        if active:
            cache_units.append((str(unit.module_name), unit.module, int(unit.parameter_count), int(unit.block_index)))
            report.active_linear_count += 1
            report.cpu_parameter_mb += _cpu_parameter_mb(unit.module)
            transfer_stats = unit.module.get_transfer_format_stats()
            if transfer_stats.get("packed"):
                report.transfer_packed_linear_count += 1
            report.transfer_h2d_mb += float(transfer_stats.get("transfer_mb", 0.0) or 0.0)
            report.transfer_pack_errors += int(transfer_stats.get("pack_errors", 0) or 0)
            report.transfer_decode_errors += int(transfer_stats.get("decode_errors", 0) or 0)

    if bool(prefetch_enabled) and normalized == "streaming_offload" and report.active_linear_count > 0:
        controller = install_dit_block_prefetch_controller(
            model,
            blocks,
            plan,
            enabled=True,
            depth=requested_prefetch_depth,
            device=device,
            dtype=dtype,
            install_hooks=False,
        )
        block_owner = getattr(model, "unet", model)
        if controller is not None and block_owner is not model:
            try:
                setattr(block_owner, "_lulynx_dit_prefetch_controller", controller)
            except Exception:
                pass
        report.prefetch = controller.as_dict() if controller is not None else {
            "enabled": False,
            "reason": "prefetch controller installation failed",
        }
    else:
        clear_dit_block_prefetch_controller(model)
        block_owner = getattr(model, "unet", model)
        if block_owner is not model:
            clear_dit_block_prefetch_controller(block_owner)
        if bool(prefetch_enabled) and normalized != "streaming_offload":
            report.prefetch = {"enabled": False, "reason": "prefetch requires streaming_offload"}
        elif bool(prefetch_enabled) and report.active_linear_count <= 0:
            report.prefetch = {"enabled": False, "reason": "no active CPU-pinned Linear units"}

    cache_profile = build_dit_pcie_cache_profile(
        plan,
        enabled=bool(pcie_delta_cache_enabled) and report.active_linear_count > 0,
        family="newbie",
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

    if normalized in {"block_cpu_pinned", "streaming_offload"} and report.active_linear_count > 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def move_newbie_nonresident_tensors(
    model: nn.Module,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> None:
    """Move normal module tensors to the training device without touching CPU-resident Linear weights."""

    target_device = torch.device(device)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LulynxManagedLinear) and bool(getattr(module, "lulynx_weight_residency_active", False)):
                continue
            for param in module.parameters(recurse=False):
                param.data = _move_tensor_data(param.data, device=target_device, dtype=dtype)
                if param.grad is not None:
                    param.grad.data = _move_tensor_data(param.grad.data, device=target_device, dtype=dtype)
            for buffer_name, buffer in module.named_buffers(recurse=False):
                if buffer is None:
                    continue
                module._buffers[buffer_name] = _move_tensor_data(buffer, device=target_device, dtype=dtype)
