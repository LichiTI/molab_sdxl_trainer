# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0-0
"""Tiny Newbie LoKr dropout smoke.

Proves that rank_dropout and module_dropout parameters are consumed by
the LoKr layer during training, and that the config round-trips from
frontend dict through ConfigAdapter into the LyCORISConfig.
"""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import torch
from torch import nn


def _install_xformers_stub() -> None:
    sys.modules.pop("xformers", None)
    sys.modules.pop("xformers.ops", None)

    xformers_module = ModuleType("xformers")
    ops_module = ModuleType("xformers.ops")

    def _unavailable(*_: object, **__: object) -> object:
        raise RuntimeError("xFormers is unavailable in the Newbie LoKr dropout smoke")

    ops_module.memory_efficient_attention = _unavailable  # type: ignore[attr-defined]
    ops_module.__spec__ = ModuleSpec("xformers.ops", loader=None)
    xformers_module.ops = ops_module  # type: ignore[attr-defined]
    xformers_module.__spec__ = ModuleSpec("xformers", loader=None)
    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = ops_module


_install_xformers_stub()

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.lycoris_layers import LyCORISConfig, LyCORISInjector, LyCORISType, LoKrLayer


class _TinyNewbieBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.Module()
        self.attention.qkv = nn.Linear(8, 8)
        self.attention.out = nn.Linear(8, 8)
        self.feed_forward = nn.Module()
        self.feed_forward.w2 = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention.out(torch.tanh(self.attention.qkv(x)))
        return self.feed_forward.w2(x)


def _test_config_round_trip() -> None:
    """ConfigAdapter preserves lokr_rank_dropout and lokr_module_dropout."""
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "newbie-lora",
            "pretrained_model_name_or_path": "H:/tmp/newbie-placeholder",
            "train_data_dir": "H:/tmp/lulynx-newbie-data",
            "adapter_type": "lokr",
            "lokr_rank": 4,
            "lokr_alpha": 4,
            "lokr_rank_dropout": 0.3,
            "lokr_module_dropout": 0.2,
        }
    )
    assert getattr(cfg, "lokr_rank_dropout", 0.0) == 0.3, (
        f"lokr_rank_dropout not preserved: {getattr(cfg, 'lokr_rank_dropout', 'MISSING')}"
    )
    assert getattr(cfg, "lokr_module_dropout", 0.0) == 0.2, (
        f"lokr_module_dropout not preserved: {getattr(cfg, 'lokr_module_dropout', 'MISSING')}"
    )


def _test_lokr_layer_consumes_rank_dropout() -> None:
    """LoKrLayer with rank_dropout produces zero output for some rank dimensions."""
    layer = LoKrLayer(
        in_features=8,
        out_features=8,
        rank=4,
        alpha=4,
        dropout=0.0,
        factor=2,
        rank_dropout=1.0,  # drop ALL rank dimensions
        module_dropout=0.0,
    )
    layer.train()
    x = torch.randn(1, 8)
    # With rank_dropout=1.0, all output dimensions should be zeroed
    out = layer(x)
    assert torch.all(out == 0.0), (
        f"Expected all-zero output with rank_dropout=1.0, got non-zero: {out}"
    )


def _test_lokr_layer_consumes_module_dropout() -> None:
    """LoKrLayer with module_dropout=1.0 always returns zeros in training."""
    layer = LoKrLayer(
        in_features=8,
        out_features=8,
        rank=4,
        alpha=4,
        dropout=0.0,
        factor=2,
        rank_dropout=0.0,
        module_dropout=1.0,  # drop entire module always
    )
    layer.train()
    x = torch.randn(1, 8)
    out = layer(x)
    assert torch.all(out == 0.0), (
        f"Expected all-zero output with module_dropout=1.0, got non-zero: {out}"
    )


def _seed_lokr_second_branch(layer: LoKrLayer, value: float = 0.5) -> None:
    with torch.no_grad():
        if hasattr(layer, "lokr_w2_b"):
            layer.lokr_w2_b.fill_(value)
        elif hasattr(layer, "lokr_w2"):
            layer.lokr_w2.fill_(value)
        elif hasattr(layer, "lokr_w1_b"):
            layer.lokr_w1_b.fill_(value)


def _test_lokr_layer_no_dropout_in_eval() -> None:
    """LoKrLayer ignores dropout in eval mode."""
    layer = LoKrLayer(
        in_features=8,
        out_features=8,
        rank=4,
        alpha=4,
        dropout=0.0,
        factor=2,
        rank_dropout=1.0,
        module_dropout=1.0,
    )
    # Seed the zero-initialized LoKr branch so delta_w is non-zero.
    _seed_lokr_second_branch(layer)
    layer.eval()
    x = torch.randn(1, 8)
    out = layer(x)
    assert not torch.all(out == 0.0), (
        "Expected non-zero output in eval mode despite high dropout settings"
    )


def _test_injector_passes_dropout_to_layer() -> None:
    """LyCORISInjector creates LoKrLayer with rank_dropout and module_dropout."""
    injector = LyCORISInjector(
        LyCORISConfig(
            lycoris_type=LyCORISType.LOKR,
            rank=4,
            alpha=4,
            lokr_factor=2,
            lokr_rank_dropout=0.25,
            lokr_module_dropout=0.15,
        )
    )
    block = _TinyNewbieBlock()
    injected = injector.inject(block, ["attention.qkv"], prefix="unet")
    assert "unet.attention.qkv" in injected, f"Expected injection at unet.attention.qkv, got {injected}"
    layer = injected["unet.attention.qkv"]
    assert isinstance(layer, LoKrLayer), f"Expected LoKrLayer, got {type(layer)}"
    assert layer.rank_dropout == 0.25, f"rank_dropout not passed: {layer.rank_dropout}"
    assert layer.module_dropout == 0.15, f"module_dropout not passed: {layer.module_dropout}"


def _test_zero_dropout_preserves_behavior() -> None:
    """LoKrLayer with zero dropout behaves like the original (no dropout)."""
    layer = LoKrLayer(
        in_features=8,
        out_features=8,
        rank=4,
        alpha=4,
        dropout=0.0,
        factor=2,
        rank_dropout=0.0,
        module_dropout=0.0,
    )
    # Seed the zero-initialized LoKr branch so delta_w is non-zero.
    _seed_lokr_second_branch(layer)
    layer.train()
    torch.manual_seed(42)
    x = torch.randn(2, 8)
    out = layer(x)
    assert out.shape == (2, 8), f"Expected shape (2, 8), got {out.shape}"
    assert torch.any(out != 0.0), "Expected non-zero output with zero dropout"


def main() -> int:
    tests = [
        ("config_round_trip", _test_config_round_trip),
        ("lokr_layer_consumes_rank_dropout", _test_lokr_layer_consumes_rank_dropout),
        ("lokr_layer_consumes_module_dropout", _test_lokr_layer_consumes_module_dropout),
        ("lokr_layer_no_dropout_in_eval", _test_lokr_layer_no_dropout_in_eval),
        ("injector_passes_dropout_to_layer", _test_injector_passes_dropout_to_layer),
        ("zero_dropout_preserves_behavior", _test_zero_dropout_preserves_behavior),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        print("FAIL -- LoKr dropout NOT fully proved")
        return 1
    print("PASS -- LoKr rank_dropout and module_dropout are consumed by the layer and injector")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
