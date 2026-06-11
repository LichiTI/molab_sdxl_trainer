"""Smoke tests for shared DiT residency planning."""

from __future__ import annotations

import torch.nn as nn

try:
    from .anima_block_residency import apply_anima_block_residency
    from .dit_residency_planner import build_dit_residency_plan
    from .native_unet.weight_residency import LulynxManagedLinear
    from .newbie_block_residency import apply_newbie_block_residency
except ImportError:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from core.lulynx_trainer.anima_block_residency import apply_anima_block_residency
    from core.lulynx_trainer.dit_residency_planner import build_dit_residency_plan
    from core.lulynx_trainer.native_unet.weight_residency import LulynxManagedLinear
    from core.lulynx_trainer.newbie_block_residency import apply_newbie_block_residency


def _managed_linear(in_features: int, out_features: int) -> LulynxManagedLinear:
    layer = LulynxManagedLinear(in_features, out_features, bias=False)
    layer.weight.requires_grad_(False)
    return layer


class _SyntheticBlock(nn.Module):
    def __init__(self, *, edge: bool = False) -> None:
        super().__init__()
        self.mlp = _managed_linear(1024, 1024)
        self.self_attn = _managed_linear(1024, 1024)
        self.small = _managed_linear(32, 32)
        if edge:
            self.edge_mlp = _managed_linear(1024, 1024)


class _SyntheticAnimaModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Module()
        self.net.blocks = nn.ModuleList([
            _SyntheticBlock(edge=True),
            _SyntheticBlock(),
            _SyntheticBlock(edge=True),
        ])


class _SyntheticNewbieModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            _SyntheticBlock(edge=True),
            _SyntheticBlock(),
            _SyntheticBlock(edge=True),
        ])


def test_hot_aware_streaming_plan() -> None:
    model = _SyntheticAnimaModel()
    blocks = list(model.net.blocks)
    plan = build_dit_residency_plan(
        blocks,
        family="anima",
        mode="streaming_offload",
        requested_min_parameter_count=0,
    )
    assert plan.mode == "streaming_offload"
    assert plan.strategy == "hot_aware_streaming_auto_threshold"
    assert plan.auto_min_parameter_count is True
    assert plan.edge_resident_count > 0
    assert plan.hot_resident_count > 0
    assert plan.planned_linear_count == 1
    planned = [unit for unit in plan.units if unit.cpu_pinned]
    assert planned[0].module_name == "mlp"
    print("PASS: test_hot_aware_streaming_plan")


def test_anima_apply_uses_shared_plan() -> None:
    model = _SyntheticAnimaModel()
    report = apply_anima_block_residency(model, mode="streaming_offload", min_parameter_count=0)
    payload = report.as_dict()
    assert payload["strategy"] == "hot_aware_streaming_auto_threshold"
    assert payload["planned_linear_count"] == 1
    assert payload["active_linear_count"] == 1
    assert payload["auto_min_parameter_count"] is True
    assert payload["unit_sample"]
    active = [
        module for module in model.modules()
        if isinstance(module, LulynxManagedLinear) and module.lulynx_weight_residency_active
    ]
    assert len(active) == 1
    print("PASS: test_anima_apply_uses_shared_plan")


def test_anima_prefetch_reports_disabled_without_cuda_device() -> None:
    model = _SyntheticAnimaModel()
    report = apply_anima_block_residency(
        model,
        mode="streaming_offload",
        min_parameter_count=0,
        prefetch_enabled=True,
        prefetch_depth=1,
    )
    payload = report.as_dict()
    prefetch = payload["prefetch"]
    assert payload["prefetch_enabled"] is True
    assert payload["prefetch_depth"] == 1
    assert prefetch["enabled"] is False
    assert prefetch["reason"]
    assert "CUDA" in prefetch["reason"] or "device" in prefetch["reason"]
    print("PASS: test_anima_prefetch_reports_disabled_without_cuda_device")


def test_sparse_swap_splits_prefetch_and_on_demand() -> None:
    blocks = [_SyntheticBlock() for _ in range(5)]
    plan = build_dit_residency_plan(
        blocks,
        family="anima",
        mode="streaming_offload",
        requested_min_parameter_count=262_144,
        edge_blocks=1,
        hot_tokens=("self_attn",),
        sparse_swap_enabled=True,
        sparse_swap_warm_fraction=0.5,
    )
    assert plan.strategy == "hot_aware_streaming_sparse_swap"
    assert plan.sparse_warm_prefetch_count > 0
    assert plan.sparse_cold_on_demand_count > 0
    assert plan.planned_linear_count == plan.sparse_warm_prefetch_count + plan.sparse_cold_on_demand_count
    warm = [unit for unit in plan.units if unit.sparse_decision == "warm_prefetch"]
    cold = [unit for unit in plan.units if unit.sparse_decision == "cold_on_demand"]
    assert all(unit.cpu_pinned for unit in warm + cold)
    print("PASS: test_sparse_swap_splits_prefetch_and_on_demand")


def test_newbie_prefetch_reports_disabled_without_cuda_device() -> None:
    model = _SyntheticNewbieModel()
    report = apply_newbie_block_residency(
        model,
        mode="streaming_offload",
        min_parameter_count=0,
        prefetch_enabled=True,
        prefetch_depth=1,
    )
    payload = report.as_dict()
    prefetch = payload["prefetch"]
    assert payload["strategy"] == "hot_aware_streaming_auto_threshold"
    assert payload["planned_linear_count"] == 1
    assert payload["active_linear_count"] == 1
    assert payload["prefetch_enabled"] is True
    assert payload["prefetch_depth"] == 1
    assert prefetch["enabled"] is False
    assert prefetch["reason"]
    assert "CUDA" in prefetch["reason"] or "device" in prefetch["reason"]
    print("PASS: test_newbie_prefetch_reports_disabled_without_cuda_device")


if __name__ == "__main__":
    test_hot_aware_streaming_plan()
    test_anima_apply_uses_shared_plan()
    test_anima_prefetch_reports_disabled_without_cuda_device()
    test_sparse_swap_splits_prefetch_and_on_demand()
    test_newbie_prefetch_reports_disabled_without_cuda_device()
    print("PASS: dit_residency_planner_smoke")
