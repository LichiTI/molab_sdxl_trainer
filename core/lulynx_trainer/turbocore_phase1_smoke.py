"""Smoke tests for TurboCore phase-1 math-equivalent scaffolding."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_phase1 import (  # noqa: E402
    NativeWorkspacePool,
    collect_lora_optimizer_params,
    lora_delta_reference,
    phase1_capability_stub,
    static_route_step_key,
)
from core.turbocore_resolver import TurboCoreExecutionResolver  # noqa: E402


def test_lora_delta_reference_matches_explicit_formula() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 5, 12)
    down = torch.randn(4, 12)
    up = torch.randn(8, 4)
    base = torch.randn(2, 5, 8)
    scale = 0.5

    expected = base + torch.nn.functional.linear(
        torch.nn.functional.linear(x, down),
        up,
    ) * scale
    actual = lora_delta_reference(x, down, up, scale=scale, base_output=base)

    assert torch.allclose(actual, expected, atol=1e-6), "LoRA delta reference mismatch"


def test_collect_lora_optimizer_params_filters_trainable_adapter_names() -> None:
    module = nn.Module()
    module.lora_down = nn.Linear(8, 4, bias=False)
    module.lora_up = nn.Linear(4, 8, bias=False)
    module.base = nn.Linear(8, 8, bias=False)
    module.base.weight.requires_grad_(False)

    selected = collect_lora_optimizer_params(module.named_parameters())

    assert len(selected) == 2
    assert any(param is module.lora_down.weight for param in selected)
    assert any(param is module.lora_up.weight for param in selected)


def test_static_route_step_key_is_stable() -> None:
    assert static_route_step_key(model_type="anima", training_type="lora", cached=True) == "anima_lora_cached"
    assert static_route_step_key(model_type="sdxl", training_type="lora", cached=False) == "sdxl_lora_live"
    assert static_route_step_key(model_type="newbie", training_type="lora", live_text_encoder=True) == "newbie_lora_live_text"


def test_workspace_pool_reuses_owned_buffers() -> None:
    pool = NativeWorkspacePool(max_cached_buffers=2)
    first = pool.acquire((2, 3), dtype=torch.float32, device="cpu")
    pool.release(first)
    second = pool.acquire((2, 3), dtype=torch.float32, device="cpu")

    assert second is first
    assert pool.stats()["hits"] == 1
    assert pool.stats()["misses"] == 1


def test_phase1_feature_contract_order_and_resolver_name() -> None:
    stub = phase1_capability_stub()
    assert stub["order"] == ["lora_fused", "native_optimizer", "static_route_step", "workspace_pool", "data_pipeline"]

    resolved = TurboCoreExecutionResolver().resolve_from_config(
        {
            "execution_core": "turbo",
            "turbocore_features": ["static_route_step"],
            "turbocore_allow_fallback": True,
        },
        model_type="anima",
        training_type="lora",
    )
    assert resolved.effective_execution_core == "standard"
    assert resolved.requested_features == ["static_route_step"]
    assert resolved.disabled_features[0]["reason"] == "turbocore_not_implemented"


if __name__ == "__main__":
    test_lora_delta_reference_matches_explicit_formula()
    test_collect_lora_optimizer_params_filters_trainable_adapter_names()
    test_static_route_step_key_is_stable()
    test_workspace_pool_reuses_owned_buffers()
    test_phase1_feature_contract_order_and_resolver_name()
    print("TurboCore phase-1 smoke tests passed.")
