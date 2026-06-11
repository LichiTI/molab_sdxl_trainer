"""Warehouse precision swap planning contracts.

The first implementation is a compatibility layer over diffusers-style module
graphs.  Native model implementations can later emit the same plan directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


def _normalize_resolution(resolution: Any) -> Tuple[int, int]:
    if isinstance(resolution, int):
        return (int(resolution), int(resolution))
    if isinstance(resolution, float):
        value = int(resolution)
        return (value, value)
    if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
        try:
            return (int(resolution[0]), int(resolution[1]))
        except (TypeError, ValueError):
            pass
    return (1024, 1024)


@dataclass(frozen=True)
class PrecisionSwapUnit:
    name: str
    module: torch.nn.Module
    stage: str
    order: int
    parameter_mb: float
    recompute_safe: bool = True
    activation_hint_mb: float = 0.0
    score: float = 0.0

    def as_profile_dict(self, *, selected: bool = False) -> Dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "order": self.order,
            "parameter_mb": round(float(self.parameter_mb), 3),
            "activation_hint_mb": round(float(self.activation_hint_mb), 3),
            "score": round(float(self.score), 3),
            "recompute_safe": bool(self.recompute_safe),
            "selected": bool(selected),
        }


@dataclass(frozen=True)
class PrecisionSwapPlan:
    family: str
    units: List[PrecisionSwapUnit] = field(default_factory=list)
    selected_names: List[str] = field(default_factory=list)
    selected_indices: List[int] = field(default_factory=list)
    strategy: str = "balanced"
    backend: str = "suffix_block_swap"
    profile_source: str = "static"
    resolution: Tuple[int, int] = (0, 0)
    runtime_observations: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    selection_reason: str = ""

    @property
    def selected_parameter_mb(self) -> float:
        selected = set(self.selected_names)
        return sum(unit.parameter_mb for unit in self.units if unit.name in selected)

    @property
    def compatible_blocks_to_swap(self) -> int:
        """Number of trailing units that the current BlockSwapOffloader can swap."""

        return len(self.selected_indices)

    @property
    def total_parameter_mb(self) -> float:
        return sum(unit.parameter_mb for unit in self.units)

    @property
    def selected_activation_hint_mb(self) -> float:
        selected = set(self.selected_names)
        return sum(unit.activation_hint_mb for unit in self.units if unit.name in selected)

    def as_profile_dict(self) -> Dict[str, Any]:
        selected = set(self.selected_names)
        return {
            "family": self.family,
            "backend": self.backend,
            "strategy": self.strategy,
            "profile_source": self.profile_source,
            "resolution": list(self.resolution),
            "units_total": len(self.units),
            "selected_count": len(self.selected_names),
            "selected_names": list(self.selected_names),
            "selected_indices": list(self.selected_indices),
            "compatible_blocks_to_swap": self.compatible_blocks_to_swap,
            "total_parameter_mb": round(float(self.total_parameter_mb), 3),
            "selected_parameter_mb": round(float(self.selected_parameter_mb), 3),
            "selected_activation_hint_mb": round(float(self.selected_activation_hint_mb), 3),
            "units": [unit.as_profile_dict(selected=unit.name in selected) for unit in self.units],
            "runtime_observations": dict(self.runtime_observations),
            "warnings": list(self.warnings),
            "selection_reason": self.selection_reason,
        }


def _module_parameter_mb(module: torch.nn.Module) -> float:
    total = 0
    for param in module.parameters(recurse=True):
        total += param.numel() * max(param.element_size(), 1)
    return total / (1024 * 1024)


def _static_activation_hint_mb(stage: str, order: int, resolution: Tuple[int, int]) -> float:
    """Cheap Warehouse activation pressure hint for planning and diagnostics.

    This is intentionally an estimate, not a measured profiler result. Runtime
    measurements can be written into ``runtime_observations`` later without
    changing the public profile shape.
    """

    width, height = resolution
    pixels = max(int(width or 0), 1) * max(int(height or 0), 1)
    scale = pixels / float(1024 * 1024)
    stage_weight = {"down": 0.65, "mid": 0.9, "up": 1.0}.get(stage, 0.5)
    order_weight = 1.0 + max(order, 0) * 0.03
    return 128.0 * scale * stage_weight * order_weight


def _selection_score(unit: PrecisionSwapUnit) -> float:
    """Rank block residency candidates by stable static pressure hints."""

    stage_bonus = {"up": 1.18, "mid": 1.08, "down": 0.94}.get(unit.stage, 1.0)
    return (unit.parameter_mb * 0.55 + unit.activation_hint_mb * 0.45) * stage_bonus


def _joint_residency_score(unit: PrecisionSwapUnit) -> float:
    """Rank block-swap candidates when layer residency already owns weights.

    In this mode the largest parameter blocks are already cold CPU-pinned at
    layer granularity.  Selecting them again as block-swap units can increase
    checkpoint/recompute peaks because many temporary layer weights overlap.
    Prefer light blocks with useful activation pressure instead.
    """

    parameter_penalty = max(float(unit.parameter_mb), 1.0)
    stage_bonus = {"up": 1.12, "mid": 1.04, "down": 0.96}.get(unit.stage, 1.0)
    return (float(unit.activation_hint_mb) * stage_bonus) / (parameter_penalty ** 0.5)


def iter_diffusers_sdxl_units(
    unet: torch.nn.Module,
    *,
    resolution: Tuple[int, int] = (1024, 1024),
) -> Iterable[PrecisionSwapUnit]:
    """Yield SDXL UNet units from a diffusers-compatible module graph."""

    order = 0
    for stage, attr in (("down", "down_blocks"), ("mid", "mid_block"), ("up", "up_blocks")):
        value = getattr(unet, attr, None)
        modules = list(value) if isinstance(value, (list, torch.nn.ModuleList)) else ([value] if value is not None else [])
        for index, module in enumerate(modules):
            name = f"{stage}.{index}" if stage != "mid" else "mid.0"
            parameter_mb = _module_parameter_mb(module)
            activation_hint_mb = _static_activation_hint_mb(stage, order, resolution)
            unit = PrecisionSwapUnit(
                name=name,
                module=module,
                stage=stage,
                order=order,
                parameter_mb=parameter_mb,
                recompute_safe=True,
                activation_hint_mb=activation_hint_mb,
            )
            yield PrecisionSwapUnit(
                name=unit.name,
                module=unit.module,
                stage=unit.stage,
                order=unit.order,
                parameter_mb=unit.parameter_mb,
                recompute_safe=unit.recompute_safe,
                activation_hint_mb=unit.activation_hint_mb,
                score=_selection_score(unit),
            )
            order += 1


def build_sdxl_precision_swap_plan(
    unet: torch.nn.Module,
    *,
    strategy: str = "balanced",
    max_units: Optional[int] = None,
    resolution: Tuple[int, int] = (1024, 1024),
    residency_mode: str = "resident",
) -> PrecisionSwapPlan:
    resolution = _normalize_resolution(resolution)
    units = list(iter_diffusers_sdxl_units(unet, resolution=resolution))
    normalized_strategy = str(strategy or "balanced").strip().lower()
    normalized_residency = str(residency_mode or "resident").strip().lower()
    joint_residency = normalized_residency not in {"", "resident", "off", "none"}
    warnings: List[str] = []
    if max_units is None:
        if normalized_strategy == "off":
            max_units = 0
        elif joint_residency:
            max_units = 0
        elif normalized_strategy == "aggressive":
            max_units = max(1, min(4, len(units)))
        else:
            max_units = max(1, min(2, len(units)))
    if joint_residency and normalized_strategy != "off":
        if int(max_units or 0) <= 0:
            warnings.append(
                "layer-level residency is enabled; precision swap stays advisory because block swap currently increases checkpoint/recompute peaks in this mode"
            )
        else:
            warnings.append(
                "layer-level residency is enabled with an explicit precision-swap unit count; this is experimental and may increase checkpoint/recompute peaks"
            )

    count = max(0, min(int(max_units or 0), max(len(units) - 1, 0)))
    if joint_residency:
        light_threshold_mb = 256.0
        eligible = [index for index, unit in enumerate(units) if unit.parameter_mb <= light_threshold_mb]
        if not eligible:
            eligible = list(range(len(units)))
        ranked = sorted(eligible, key=lambda index: _joint_residency_score(units[index]), reverse=True)
        count = min(count, len(ranked))
    else:
        ranked = sorted(range(len(units)), key=lambda index: _selection_score(units[index]), reverse=True)
    selected_indices = sorted(ranked[:count]) if count else []
    selected = [units[index].name for index in selected_indices]
    selected_ranked = [units[index].name for index in ranked[:count]] if count else []
    if joint_residency and count:
        reason = f"selected top {count} light units by joint layer-residency score"
    elif joint_residency:
        reason = "selected no units because layer-level residency and block swap are not beneficial together yet"
    else:
        reason = f"selected top {count} units by static pressure score"
    return PrecisionSwapPlan(
        family="sdxl",
        units=units,
        selected_names=selected,
        selected_indices=selected_indices,
        strategy=normalized_strategy,
        backend="selected_block_swap",
        profile_source="static_pressure_v2",
        resolution=resolution,
        warnings=warnings,
        selection_reason=reason,
        runtime_observations={"selected_ranked": selected_ranked} if selected_ranked else {},
    )

