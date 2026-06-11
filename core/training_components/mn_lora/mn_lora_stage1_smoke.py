"""Smoke tests for MN-LoRA stage-1 wiring fixes."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

BACKEND_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.training_components.mn_lora.mn_optimizer import MNLoRAOptimizer
from core.training_components.mn_lora.mn_presets import SDXL_PRESET, split_mnlora_preset
from core.training_components.mn_lora.proxy_regularizer import ProxyRegularizer
from core.training_components.mn_lora.v_matrix_cache import CacheConfig, CacheMode, VMatrixCache


def test_proxy_regularizer_backpropagates() -> None:
    model = nn.Linear(3, 2, bias=False)
    proxy = {
        "layers": {
            "weight": {
                "l2_norm": float(model.weight.detach().norm()) + 0.5,
                "mean": float(model.weight.detach().mean()) + 0.1,
            }
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "proxy.json"
        path.write_text(json.dumps(proxy), encoding="utf-8")
        regularizer = ProxyRegularizer(str(path), lambda_norm=1.0, lambda_mean=1.0, update_interval=1)
        loss = regularizer.compute_loss(model, step=1)
        assert loss.requires_grad, "Proxy regularizer loss must remain attached to autograd"
        loss.backward()
        assert model.weight.grad is not None
        assert float(model.weight.grad.abs().sum()) > 0.0


def test_proxy_regularizer_accepts_frobenius_alias_and_reports_telemetry() -> None:
    model = nn.Linear(3, 2, bias=False)
    proxy = {
        "layers": {
            "weight": {
                "frobenius": float(model.weight.detach().norm()) + 0.25,
            }
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "proxy.json"
        path.write_text(json.dumps(proxy), encoding="utf-8")
        regularizer = ProxyRegularizer(str(path), lambda_norm=1.0, lambda_mean=0.0, update_interval=1)
        loss = regularizer.compute_loss(model, step=1)
        assert loss.requires_grad
        assert float(loss.detach()) > 0.0
        telemetry = regularizer.get_telemetry_snapshot()
        assert telemetry["computed"] is True
        assert telemetry["matched_layers"] == 1
        assert telemetry["proxy_layers"] == 1


def test_optimizer_initializes_gsp_on_first_step() -> None:
    param = nn.Parameter(torch.randn(4, 4))
    optimizer = MNLoRAOptimizer(
        torch.optim.SGD([param], lr=0.01),
        enable_tgwd=False,
        enable_gsp=True,
        enable_pilot=False,
        gsp_config={
            "k_ratio": 0.5,
            "update_interval": 100,
            "lazy_update": True,
            "adaptive_sparse_enabled": False,
        },
        param_names={id(param): "block.lora_down.weight"},
    )
    param.grad = torch.ones_like(param)
    optimizer.step()
    assert optimizer.gsp is not None
    assert "block.lora_down.weight" in optimizer.gsp.V_cache
    assert "block.lora_down.weight" in optimizer.gsp.S_cache
    assert optimizer.gsp._global_step == 1


def test_v_matrix_cache_uses_subspace_rank_dimension() -> None:
    cache = VMatrixCache(CacheConfig(mode=CacheMode.TIERED, rank_threshold=64))
    V = torch.randn(100, 4)
    cache.put("layer", V)
    stats = cache.get_stats()
    assert stats["gpu_layers"] == 1, "V cache should classify rank by V.shape[1], not input dim"
    assert stats["cpu_layers"] == 0


def test_preset_splitter_maps_component_keys() -> None:
    split = split_mnlora_preset(SDXL_PRESET)
    assert split["gsp_config"]["k_ratio"] == SDXL_PRESET["k_ratio"]
    assert split["tgwd_config"]["base_lambda"] == SDXL_PRESET["tgwd_base_decay"]
    assert split["pilot_config"]["strategy"] == SDXL_PRESET["pilot_strategy"]


def main() -> int:
    test_proxy_regularizer_backpropagates()
    print("  Proxy regularizer autograd -- PASS")
    test_proxy_regularizer_accepts_frobenius_alias_and_reports_telemetry()
    print("  Proxy frobenius alias telemetry -- PASS")
    test_optimizer_initializes_gsp_on_first_step()
    print("  GSP first-step initialization -- PASS")
    test_v_matrix_cache_uses_subspace_rank_dimension()
    print("  V cache rank dimension -- PASS")
    test_preset_splitter_maps_component_keys()
    print("  Preset splitter -- PASS")
    print("MN-LoRA stage-1 smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
