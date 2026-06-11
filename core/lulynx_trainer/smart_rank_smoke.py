"""Smoke test for SmartRank: rank inference from weight SVD analysis."""
from __future__ import annotations

import sys
import os
import importlib.util
from pathlib import Path
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))

# Load smart_rank via importlib from core/training_components/smart_rank.py
_sr_path = os.path.join(_HERE, "..", "..", "core", "training_components", "smart_rank.py")
_sr_spec = importlib.util.spec_from_file_location(
    "core.training_components.smart_rank",
    _sr_path,
)
_sr_mod = importlib.util.module_from_spec(_sr_spec)
sys.modules["core.training_components.smart_rank"] = _sr_mod
_sr_spec.loader.exec_module(_sr_mod)

infer_rank_from_svd = _sr_mod.infer_rank_from_svd
advise_rank = _sr_mod.advise_rank
advise_rank_from_weight = _sr_mod.advise_rank_from_weight
SmartRankController = _sr_mod.SmartRankController

import torch
import torch.nn as nn

_BACKEND_ROOT = Path(_HERE).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.lulynx_trainer.lora_injector import LoRAInjector
from core.lulynx_trainer.training_loop import TrainingLoop


def test_infer_rank_within_bounds():
    """infer_rank_from_svd returns a rank within [min_rank, max_rank]."""
    W = torch.randn(64, 32)
    rank = infer_rank_from_svd(W, min_rank=4, max_rank=32)
    assert 4 <= rank <= 32, f"Rank {rank} outside [4, 32]"


def test_infer_rank_higher_for_high_rank_weight():
    """A weight matrix with broad singular-value spread gets a higher inferred rank
    than a weight matrix with concentrated energy in few components."""
    # Low-rank weight: only a few effective directions
    torch.manual_seed(0)
    low_rank_W = torch.randn(64, 3) @ torch.randn(3, 32)  # effective rank ~3

    # High-rank weight: full-rank random
    high_rank_W = torch.randn(64, 32)

    rank_low = infer_rank_from_svd(low_rank_W, min_rank=4, max_rank=32)
    rank_high = infer_rank_from_svd(high_rank_W, min_rank=4, max_rank=32)

    assert rank_low <= rank_high, (
        f"Low-rank weight should have <= rank than high-rank: {rank_low} vs {rank_high}"
    )


def test_infer_rank_aligned_to_4():
    """Returned rank should be a multiple of 4."""
    W = torch.randn(64, 32)
    for min_r, max_r in [(4, 64), (8, 128), (4, 32)]:
        rank = infer_rank_from_svd(W, min_rank=min_r, max_rank=max_r)
        assert rank % 4 == 0, f"Rank {rank} not aligned to 4 (min={min_r}, max={max_r})"
        assert min_r <= rank <= max_r


def test_infer_rank_near_zero_weight():
    """Near-zero weight matrix should return min_rank."""
    W = torch.zeros(64, 32)
    rank = infer_rank_from_svd(W, min_rank=4, max_rank=32)
    assert rank == 4, f"Expected min_rank=4 for zero matrix, got {rank}"


def test_infer_rank_energy_threshold_effect():
    """Higher energy_threshold should produce a higher (or equal) inferred rank."""
    W = torch.randn(64, 32)
    rank_80 = infer_rank_from_svd(W, min_rank=4, max_rank=64, energy_threshold=0.8)
    rank_99 = infer_rank_from_svd(W, min_rank=4, max_rank=64, energy_threshold=0.99)
    assert rank_80 <= rank_99, (
        f"rank with 80% energy ({rank_80}) should be <= rank with 99% ({rank_99})"
    )



def test_advise_rank_report_only_for_high_rank():
    """High rank without metrics returns report-only advice and never enables mutation."""
    advice = advise_rank(current_rank=96, min_rank=4, max_rank=128)
    data = advice.to_dict()
    assert data["current_rank"] == 96
    assert data["suggested_rank"] < 96
    assert data["severity"] in {"watch", "info"}
    assert any("Report-only" in note for note in data["notes"])


def test_advise_rank_from_stable_rank_prunes():
    """Stable-rank signal below current rank suggests a lower aligned rank."""
    advice = advise_rank(current_rank=64, stable_rank=18.0, min_rank=4, max_rank=128)
    data = advice.to_dict()
    assert data["suggested_rank"] == 20
    assert data["source"] == "stable_rank"
    assert data["severity"] == "watch"


def test_advise_rank_from_weight():
    """Weight-based advice wraps infer_rank_from_svd without mutating weights."""
    torch.manual_seed(1)
    W = torch.randn(64, 4) @ torch.randn(4, 32)
    advice = advise_rank_from_weight(W, current_rank=64, min_rank=4, max_rank=64)
    data = advice.to_dict()
    assert data["suggested_rank"] <= 64
    assert data["source"] == "svd"


class _TinySDXLAttentionBlock(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)


class _TinySDXLUNet(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.attn = _TinySDXLAttentionBlock(dim)


class _FakeAuditor:
    def __init__(self, layer_name: str) -> None:
        self.layer_name = layer_name
        self.step_calls = 0

    def step(self, **_kwargs) -> None:
        self.step_calls += 1

    def get_last_report(self):
        return {"layers": {self.layer_name: {"stable_rank": 1.0}}}


def test_sdxl_training_loop_smart_rank_prunes_lora_layer():
    """SDXL-style LoRA layer is pruned through TrainingLoop audit wiring."""
    torch.manual_seed(2)
    unet = _TinySDXLUNet()
    injector = LoRAInjector(rank=8, alpha=8.0, target_modules=["to_q"], model_arch="sdxl")
    injected = injector.inject_unet(unet)
    assert list(injected.keys()) == ["unet.attn.to_q"]

    layer = injector.injected_layers["unet.attn.to_q"]
    with torch.no_grad():
        layer.lora.lora_up.weight.zero_()
        layer.lora.lora_down.weight.zero_()
        layer.lora.lora_up.weight[:, 0] = torch.linspace(0.1, 0.8, layer.lora.lora_up.weight.shape[0])
        layer.lora.lora_down.weight[0, :] = torch.linspace(0.2, 0.9, layer.lora.lora_down.weight.shape[1])
    assert layer.lora.rank == 8

    loop = TrainingLoop.__new__(TrainingLoop)
    loop.auditor = _FakeAuditor("unet.attn.to_q")
    loop.global_step = 1
    loop.current_epoch = 0
    loop.total_steps = 10
    loop._last_loss = 0.25
    loop.lr_scheduler = SimpleNamespace(get_last_lr=lambda: [1e-4])
    loop.lora_injector = injector
    loop.smart_rank_controller = SmartRankController(injector, min_rank=4, max_rank=8, interval=1)
    changed = {"count": 0}
    loop.on_params_changed = lambda: changed.__setitem__("count", changed["count"] + 1)

    TrainingLoop._run_audit(loop)

    assert loop.auditor.step_calls == 1
    assert changed["count"] == 1
    assert layer.lora.rank == 4
    assert layer.lora.lora_down.out_features == 4
    assert layer.lora.lora_up.in_features == 4
    assert len(injector.get_trainable_params()) == 2


def test_infer_rank_rejects_non_2d():
    """infer_rank_from_svd raises ValueError for non-2-D tensors."""
    W_3d = torch.randn(2, 64, 32)
    try:
        infer_rank_from_svd(W_3d)
        assert False, "Should have raised ValueError for 3-D tensor"
    except ValueError as e:
        assert "2-D" in str(e)


if __name__ == "__main__":
    print("SmartRank Smoke Tests")
    print("=" * 40)
    test_infer_rank_within_bounds()
    print("PASS: infer_rank_within_bounds")
    test_infer_rank_higher_for_high_rank_weight()
    print("PASS: infer_rank_higher_for_high_rank_weight")
    test_infer_rank_aligned_to_4()
    print("PASS: infer_rank_aligned_to_4")
    test_infer_rank_near_zero_weight()
    print("PASS: infer_rank_near_zero_weight")
    test_infer_rank_energy_threshold_effect()
    print("PASS: infer_rank_energy_threshold_effect")
    test_infer_rank_rejects_non_2d()
    print("PASS: infer_rank_rejects_non_2d")
    test_advise_rank_report_only_for_high_rank()
    print("PASS: advise_rank_report_only_for_high_rank")
    test_advise_rank_from_stable_rank_prunes()
    print("PASS: advise_rank_from_stable_rank_prunes")
    test_advise_rank_from_weight()
    print("PASS: advise_rank_from_weight")
    test_sdxl_training_loop_smart_rank_prunes_lora_layer()
    print("PASS: sdxl_training_loop_smart_rank_prunes_lora_layer")
    print("=" * 40)
    print("All SmartRank smoke tests passed!")
