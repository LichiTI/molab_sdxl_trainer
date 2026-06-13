# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Anima GLoRA Phase 2 extended tier.

Phase 2 layers all defaults to OFF.  These tests cover the optional knobs:

- rank-dropout: zeros random rows of ΔW while training, identity at eval.
- module-dropout: whole-layer skip returns zero output while training.
- bias adaptation: enabled only when the base module has a bias; off by
  default for ``bias=False`` Linear / Conv2d targets.
- no-materialize forward parity: chained matmuls must match F.linear of the
  materialized ΔW (to within fp64 tolerance).
- tucker conv parity: the three-segment B path must materialize back to a
  proper [out, in/groups, kH, kW] kernel.
- backward compatibility: default kwargs reproduce the Phase 1 layer's
  state-dict keys (no spurious bm/c1/c2 tensors when not enabled).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = BACKEND_ROOT / "core"
TRAINER_ROOT = CORE_ROOT / "lulynx_trainer"


def _ensure_namespace(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name: str, path: Path):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ensure_namespace("core", CORE_ROOT)
_ensure_namespace("core.lulynx_trainer", TRAINER_ROOT)
_load_module("core.safe_pickle", CORE_ROOT / "safe_pickle.py")
_load_module("core.configs", CORE_ROOT / "configs.py")
_load_module("core.lulynx_trainer.config", TRAINER_ROOT / "config.py")
_load_module("core.lulynx_trainer.config_adapter", TRAINER_ROOT / "config_adapter.py")
generalized_mod = _load_module("core.lulynx_trainer.generalized_adapters", TRAINER_ROOT / "generalized_adapters.py")
lycoris_mod = _load_module("core.lulynx_trainer.lycoris_layers", TRAINER_ROOT / "lycoris_layers.py")

GLoRALinearLayer = generalized_mod.GLoRALinearLayer
GLoRAConv2dLayer = generalized_mod.GLoRAConv2dLayer
collect_glora_layer_state = generalized_mod.collect_glora_layer_state
LyCORISConfig = lycoris_mod.LyCORISConfig
LyCORISInjector = lycoris_mod.LyCORISInjector
LyCORISType = lycoris_mod.LyCORISType


def _seed_layer(layer: nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in layer.parameters():
            param.copy_(torch.randn_like(param) * 0.2)


# ---------------------------------------------------------------------------


def test_default_layer_has_no_phase2_tensors() -> bool:
    """Backward-compat: default Linear layer carries no bm/c1/c2 tensors."""
    base = nn.Linear(8, 16, bias=False)
    layer = GLoRALinearLayer(8, 16, rank=2, alpha=2.0, org_weight=base.weight)
    state = collect_glora_layer_state(layer, "demo")
    keys = set(state)
    assert "demo.a1.weight" in keys and "demo.b2.weight" in keys
    assert "demo.bm.weight" not in keys, "bm must not appear without tucker"
    assert "demo.c1.weight" not in keys, "c1 must not appear without bias"
    assert "demo.c2.weight" not in keys
    print("PASS: test_default_layer_has_no_phase2_tensors")
    return True


def test_bias_path_enabled_only_when_base_has_bias() -> bool:
    base_no_bias = nn.Linear(8, 16, bias=False)
    layer_no = GLoRALinearLayer(8, 16, rank=2, alpha=2.0, org_weight=base_no_bias.weight)
    assert not layer_no._has_bias_delta(), "bias must not be enabled when base has no bias"

    base_with_bias = nn.Linear(8, 16, bias=True)
    layer_yes = GLoRALinearLayer(
        8, 16, rank=2, alpha=2.0,
        org_weight=base_with_bias.weight,
        org_bias=base_with_bias.bias,
    )
    assert layer_yes._has_bias_delta(), "bias must engage when org_bias is passed"
    # At init, Δbias must be zero (c2 = 0).
    assert torch.count_nonzero(layer_yes.get_delta_bias()) == 0, "Δbias must be 0 at init"
    print("PASS: test_bias_path_enabled_only_when_base_has_bias")
    return True


def test_no_materialize_forward_matches_materialized() -> bool:
    """no-materialize chained matmul must equal F.linear(x, ΔW)."""
    base = nn.Linear(8, 16, bias=False)
    layer = GLoRALinearLayer(
        8, 16, rank=2, alpha=2.0,
        org_weight=base.weight,
        no_materialize_forward=True,
    ).double()
    _seed_layer(layer, 4242)
    layer.eval()  # disable rank/module dropout
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    assert layer._can_use_no_materialize_forward(x), "fast path should be eligible here"

    fast = layer(x)
    reference = F.linear(x, layer.get_delta_weight())
    torch.testing.assert_close(fast, reference, rtol=1e-10, atol=1e-10)
    print("PASS: test_no_materialize_forward_matches_materialized")
    return True


def test_rank_dropout_zeros_random_rows() -> bool:
    base = nn.Linear(8, 16, bias=False)
    layer = GLoRALinearLayer(
        8, 16, rank=4, alpha=2.0,
        org_weight=base.weight,
        rank_dropout=0.5,
    )
    _seed_layer(layer, 7)
    layer.train()
    x = torch.randn(2, 8)
    out = layer(x)
    assert out.shape == (2, 16)
    # In eval mode rank dropout must be off.
    layer.eval()
    out_eval = layer(x)
    reference = F.linear(x, layer.get_delta_weight())
    torch.testing.assert_close(out_eval, reference, rtol=1e-5, atol=1e-5)
    print("PASS: test_rank_dropout_zeros_random_rows")
    return True


def test_module_dropout_returns_zeros_when_training() -> bool:
    base = nn.Linear(8, 16, bias=False)
    layer = GLoRALinearLayer(
        8, 16, rank=2, alpha=2.0,
        org_weight=base.weight,
        module_dropout=1.0,  # always drop
    )
    _seed_layer(layer, 88)
    layer.train()
    x = torch.randn(2, 3, 8)
    out = layer(x)
    assert out.shape == (2, 3, 16)
    assert torch.count_nonzero(out) == 0, "module_dropout=1.0 must produce zero output in training"
    # In eval mode no skip.
    layer.eval()
    out_eval = layer(x)
    assert torch.count_nonzero(out_eval) > 0, "eval mode must not drop the module"
    print("PASS: test_module_dropout_returns_zeros_when_training")
    return True


def test_tucker_conv_parity_against_materialized_delta() -> bool:
    """Tucker B path must produce a kernel that F.conv2d accepts and matches forward."""
    base = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)
    layer = GLoRAConv2dLayer(
        in_channels=4, out_channels=8,
        kernel_size=(3, 3), padding=(1, 1),
        rank=2, alpha=2.0,
        org_weight=base.weight,
        use_tucker=True,
    ).double()
    _seed_layer(layer, 314)
    layer.eval()
    x = torch.randn(2, 4, 8, 8, dtype=torch.float64)

    fast = layer(x)
    # Reference: materialize the delta and conv with it.
    delta = layer.get_delta_weight()
    reference = F.conv2d(x, delta, padding=(1, 1))
    torch.testing.assert_close(fast, reference, rtol=1e-10, atol=1e-10)
    # Tucker should carry the bm tensor.
    state = collect_glora_layer_state(layer, "conv")
    assert "conv.bm.weight" in state, "tucker layer must export bm tensor"
    print("PASS: test_tucker_conv_parity_against_materialized_delta")
    return True


def test_injector_propagates_phase2_knobs() -> bool:
    """LyCORISInjector + LyCORISConfig must forward the Phase 2 fields."""
    module = nn.Linear(8, 16, bias=True)

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = module
    model = _Tiny()

    cfg = LyCORISConfig(
        lycoris_type=LyCORISType.GLORA,
        rank=2,
        alpha=2.0,
        glora_rank_dropout=0.25,
        glora_module_dropout=0.1,
        glora_no_materialize_forward=True,
        glora_train_bias=True,
    )
    injector = LyCORISInjector(cfg)
    injected = injector.inject(model, ["q_proj"], prefix="")
    layer = next(iter(injected.values()))
    assert isinstance(layer, GLoRALinearLayer)
    assert layer.rank_dropout == 0.25
    assert layer.module_dropout == 0.1
    assert layer.no_materialize_forward is True
    assert layer._has_bias_delta() is True, "bias path must engage when train_bias=True and base has bias"
    print("PASS: test_injector_propagates_phase2_knobs")
    return True


def test_injector_disables_bias_when_train_bias_false() -> bool:
    module = nn.Linear(8, 16, bias=True)

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = module
    model = _Tiny()

    cfg = LyCORISConfig(
        lycoris_type=LyCORISType.GLORA,
        rank=2,
        alpha=2.0,
        glora_train_bias=False,
    )
    injector = LyCORISInjector(cfg)
    injected = injector.inject(model, ["q_proj"], prefix="")
    layer = next(iter(injected.values()))
    assert not layer._has_bias_delta(), "train_bias=False must disable bias adaptation"
    print("PASS: test_injector_disables_bias_when_train_bias_false")
    return True


def test_phase1_defaults_remain_bit_compatible() -> bool:
    """With all Phase 2 knobs off, ΔW and state_dict match the Phase 1 layer."""
    torch.manual_seed(2026)
    base = nn.Linear(8, 16, bias=False)
    layer = GLoRALinearLayer(8, 16, rank=2, alpha=2.0, org_weight=base.weight)
    _seed_layer(layer, 9001)
    delta = layer.get_delta_weight()
    state = collect_glora_layer_state(layer, "d")
    # Reconstruct ΔW from saved tensors alone (no Phase 2 tensors should be needed).
    a1 = state["d.a1.weight"]
    a2 = state["d.a2.weight"]
    b1 = state["d.b1.weight"]
    b2 = state["d.b2.weight"]
    expected = (base.weight @ (a1 @ a2) + (b1 @ b2)) * layer.scaling
    torch.testing.assert_close(delta, expected, rtol=1e-6, atol=1e-6)
    # No Phase 2 tensors leaked into save state.
    keys = set(state)
    assert "d.bm.weight" not in keys
    assert "d.c1.weight" not in keys
    assert "d.c2.weight" not in keys
    print("PASS: test_phase1_defaults_remain_bit_compatible")
    return True


def test_bias_aware_forward_adds_delta_bias() -> bool:
    """When bias path is enabled and parameters are non-zero, forward must add Δbias."""
    base = nn.Linear(8, 16, bias=True)
    layer = GLoRALinearLayer(
        8, 16, rank=2, alpha=2.0,
        org_weight=base.weight,
        org_bias=base.bias,
    ).double()
    _seed_layer(layer, 555)
    layer.eval()
    x = torch.randn(3, 8, dtype=torch.float64)
    out = layer(x)
    expected_weight_term = F.linear(x, layer.get_delta_weight())
    expected = expected_weight_term + layer.get_delta_bias()
    torch.testing.assert_close(out, expected, rtol=1e-10, atol=1e-10)
    print("PASS: test_bias_aware_forward_adds_delta_bias")
    return True


# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        test_default_layer_has_no_phase2_tensors,
        test_bias_path_enabled_only_when_base_has_bias,
        test_no_materialize_forward_matches_materialized,
        test_rank_dropout_zeros_random_rows,
        test_module_dropout_returns_zeros_when_training,
        test_tucker_conv_parity_against_materialized_delta,
        test_injector_propagates_phase2_knobs,
        test_injector_disables_bias_when_train_bias_false,
        test_phase1_defaults_remain_bit_compatible,
        test_bias_aware_forward_adds_delta_bias,
    ]
    results = []
    for test_fn in tests:
        try:
            results.append((test_fn.__name__, test_fn()))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"FAIL: {test_fn.__name__} — {exc}")
            results.append((test_fn.__name__, False))

    passed = sum(1 for _, ok in results if ok)
    print("\n" + "=" * 60)
    print("Anima GLoRA Phase 2 Smoke Test Results")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n{passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
