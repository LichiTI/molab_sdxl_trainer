# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Anima GLoKr (Phase 3 — Kronecker Generalized adapter).

GLoKr is project-original research: no upstream parity, no published
implementation.  These tests pin down the math contract end-to-end:

- ΔW = W·A + B with A = A1⊗A2 and B = B1⊗B2 (Kronecker shapes match what
  torch.kron produces).
- ΔW = 0 at initialization (Kronecker zero factor → zero product).
- no-materialize forward equals F.linear(x, ΔW) to fp64 tolerance — this is
  the math-correctness backstop on the Kronecker mat-vec identity.
- Save → load → forward is bit-faithful on the trainable tensors.
- Merge into the base weight reproduces the composite forward.
- Factor auto-selection picks a divisor of both dims; non-divisor request
  falls back gracefully.
- LyCORISInjector + LyCORISConfig + adapter contract resolve glokr correctly.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
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
config_adapter_mod = _load_module("core.lulynx_trainer.config_adapter", TRAINER_ROOT / "config_adapter.py")
generalized_mod = _load_module("core.lulynx_trainer.generalized_adapters", TRAINER_ROOT / "generalized_adapters.py")
glokr_mod = _load_module("core.lulynx_trainer.glokr_layer", TRAINER_ROOT / "glokr_layer.py")
lycoris_mod = _load_module("core.lulynx_trainer.lycoris_layers", TRAINER_ROOT / "lycoris_layers.py")
merge_export_mod = _load_module("core.lulynx_trainer.merge_export", TRAINER_ROOT / "merge_export.py")
contract_mod = _load_module("core.lulynx_trainer.method_adapter_contract", TRAINER_ROOT / "method_adapter_contract.py")

ConfigAdapter = config_adapter_mod.ConfigAdapter
GLoKrLinearLayer = glokr_mod.GLoKrLinearLayer
collect_glokr_layer_state = glokr_mod.collect_glokr_layer_state
_select_glokr_factor = glokr_mod._select_glokr_factor
LyCORISConfig = lycoris_mod.LyCORISConfig
LyCORISInjector = lycoris_mod.LyCORISInjector
LyCORISType = lycoris_mod.LyCORISType
merge_lycoris_into_base = merge_export_mod.merge_lycoris_into_base
resolve_adapter_method = contract_mod.resolve_adapter_method


class _TinyAnimaRoot(nn.Module):
    """Anima-like nested module with Linear targets whose dims are easily factorizable."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Module()
        self.net.blocks = nn.ModuleList([_block(dim)])


def _block(dim: int) -> nn.Module:
    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.q_proj = nn.Linear(dim, dim, bias=False)
    block.self_attn.k_proj = nn.Linear(dim, dim, bias=False)
    block.mlp = nn.Module()
    block.mlp.layer1 = nn.Linear(dim, dim * 2, bias=False)
    return block


def _seed_layer(layer: nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in layer.parameters():
            param.copy_(torch.randn_like(param) * 0.2)


# ---------------------------------------------------------------------------


def test_glokr_factor_selection_picks_divisor() -> bool:
    # 16 = 4×4 = 2×8 = 8×2 …  auto should pick a divisor of both.
    assert _select_glokr_factor(16, 16, -1) >= 2
    assert _select_glokr_factor(16, 32, -1) in {2, 4, 8, 16}
    # Non-divisor request falls back gracefully.
    f = _select_glokr_factor(16, 16, 7)
    assert 16 % f == 0 and 16 % f == 0
    # Coprime dims fall back to 1.
    assert _select_glokr_factor(13, 17, -1) == 1
    print("PASS: test_glokr_factor_selection_picks_divisor")
    return True


def test_glokr_delta_shape_matches_kron_definition() -> bool:
    """ΔW shape must be [out, in] and equal W·(A1⊗A2) + (B1⊗B2) (post-scaling)."""
    base = nn.Linear(16, 16, bias=False)
    layer = GLoKrLinearLayer(16, 16, rank=4, alpha=2.0, factor=4, org_weight=base.weight).double()
    _seed_layer(layer, 1234)
    A = torch.kron(layer.glokr_a1, layer.glokr_a2)
    B = torch.kron(layer.glokr_b1, layer.glokr_b2)
    assert A.shape == (16, 16), A.shape
    assert B.shape == (16, 16), B.shape
    expected = (base.weight.double() @ A + B) * layer.scaling
    delta = layer.get_delta_weight()
    torch.testing.assert_close(delta, expected, rtol=1e-10, atol=1e-10)
    print("PASS: test_glokr_delta_shape_matches_kron_definition")
    return True


def test_glokr_initialization_yields_zero_delta() -> bool:
    base = nn.Linear(16, 16, bias=False)
    layer = GLoKrLinearLayer(16, 16, rank=4, alpha=2.0, factor=4, org_weight=base.weight)
    delta = layer.get_delta_weight()
    assert torch.count_nonzero(delta) == 0, "Kronecker product with zero factor must be 0"
    x = torch.randn(3, 16)
    out = layer(x)
    assert torch.allclose(out, torch.zeros_like(out)), "forward must be 0 at init"
    print("PASS: test_glokr_initialization_yields_zero_delta")
    return True


def test_glokr_forward_matches_materialized_delta() -> bool:
    """Forward (materialized path) equals F.linear(x, get_delta_weight())."""
    base = nn.Linear(16, 16, bias=False)
    layer = GLoKrLinearLayer(16, 16, rank=4, alpha=2.0, factor=4, org_weight=base.weight).double()
    _seed_layer(layer, 555)
    layer.eval()
    x = torch.randn(2, 3, 16, dtype=torch.float64)
    out = layer(x)
    expected = F.linear(x, layer.get_delta_weight())
    torch.testing.assert_close(out, expected, rtol=1e-10, atol=1e-10)
    print("PASS: test_glokr_forward_matches_materialized_delta")
    return True


def test_glokr_no_materialize_parity_with_materialized_delta() -> bool:
    """The Kronecker-aware chained matmul must equal F.linear(x, ΔW).

    This is the math backstop on the (M1 ⊗ M2) · x  =  vec(M1 · X · M2ᵀ) identity
    against torch.kron's index convention.  If the reshape order or transpose
    is wrong, this fails immediately.
    """
    base = nn.Linear(16, 32, bias=False)  # asymmetric to catch dim swaps
    layer = GLoKrLinearLayer(
        16, 32, rank=4, alpha=2.0, factor=4,
        org_weight=base.weight, no_materialize_forward=True,
    ).double()
    _seed_layer(layer, 7777)
    layer.eval()
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    assert layer._can_use_no_materialize_forward(x), "fast path should be eligible"
    fast = layer(x)
    reference = F.linear(x, layer.get_delta_weight())
    torch.testing.assert_close(fast, reference, rtol=1e-10, atol=1e-10)
    print("PASS: test_glokr_no_materialize_parity_with_materialized_delta")
    return True


def test_glokr_injection_produces_gradients() -> bool:
    model = _TinyAnimaRoot(dim=16)
    config = LyCORISConfig(
        lycoris_type=LyCORISType.GLOKR, rank=4, alpha=2.0, glokr_factor=4,
    )
    injector = LyCORISInjector(config)
    injected = injector.inject(model, ["self_attn.q_proj", "mlp.layer1"], prefix="unet")
    assert injected, "expected at least one injected GLoKr layer"
    assert all(isinstance(layer, GLoKrLinearLayer) for layer in injected.values())

    for layer in injected.values():
        _seed_layer(layer, 13)
    x = torch.randn(2, 3, 16)
    y = model.net.blocks[0].self_attn.q_proj(x).sum() + model.net.blocks[0].mlp.layer1(x).sum()
    y.backward()
    trainable = [p for p in injector.get_trainable_parameters() if p.grad is not None]
    assert trainable, "expected GLoKr gradients"
    print("PASS: test_glokr_injection_produces_gradients")
    return True


def test_glokr_save_load_roundtrip() -> bool:
    model = _TinyAnimaRoot(dim=16)
    config = LyCORISConfig(lycoris_type=LyCORISType.GLOKR, rank=4, alpha=2.0, glokr_factor=4)
    injector = LyCORISInjector(config)
    injector.inject(model, ["self_attn.q_proj"], prefix="unet")
    layer = next(iter(injector.injected_layers.values()))
    _seed_layer(layer, 4321)

    expected = {k: v.clone() for k, v in injector.get_lora_state_dict().items()}
    probe = torch.randn(1, 2, 16)
    expected_out = layer(probe)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "glokr.safetensors")
        injector.save(path)

        reloaded_model = _TinyAnimaRoot(dim=16)
        with torch.no_grad():
            for src, dst in zip(model.modules(), reloaded_model.modules()):
                if isinstance(src, nn.Linear) and isinstance(dst, nn.Linear):
                    dst.weight.copy_(src.weight)
        reloaded = LyCORISInjector(config)
        reloaded.inject(reloaded_model, ["self_attn.q_proj"], prefix="unet")
        reloaded.load_lora(path)
        actual = reloaded.get_lora_state_dict()
        actual_layer = next(iter(reloaded.injected_layers.values()))

    for key, value in expected.items():
        if key.endswith(".alpha"):
            continue
        assert key in actual, f"missing {key}"
        torch.testing.assert_close(actual[key], value, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_layer(probe), expected_out, rtol=1e-5, atol=1e-5)
    print("PASS: test_glokr_save_load_roundtrip")
    return True


def test_glokr_merge_into_base_matches_composite_forward() -> bool:
    model = _TinyAnimaRoot(dim=16)
    config = LyCORISConfig(lycoris_type=LyCORISType.GLOKR, rank=4, alpha=2.0, glokr_factor=4)
    injector = LyCORISInjector(config)
    injector.inject(model, ["self_attn.q_proj"], prefix="unet")
    layer = next(iter(injector.injected_layers.values()))
    _seed_layer(layer, 991)

    x = torch.randn(2, 16)
    expected = model.net.blocks[0].self_attn.q_proj(x)
    merged = merge_lycoris_into_base(model, injector)
    assert merged == 1, merged
    actual = model.net.blocks[0].self_attn.q_proj(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    print("PASS: test_glokr_merge_into_base_matches_composite_forward")
    return True


def test_glokr_parameter_count_is_smaller_than_glora() -> bool:
    """Document the headline win: at moderate dim, GLoKr uses far fewer trainable params."""
    GLoRALinearLayer = generalized_mod.GLoRALinearLayer

    in_dim, out_dim, rank = 128, 128, 8
    base = nn.Linear(in_dim, out_dim, bias=False)
    glora = GLoRALinearLayer(in_dim, out_dim, rank=rank, alpha=rank, org_weight=base.weight)
    glokr = GLoKrLinearLayer(in_dim, out_dim, rank=rank, alpha=rank, factor=-1, org_weight=base.weight)
    glora_params = sum(p.numel() for p in glora.parameters())
    glokr_params = sum(p.numel() for p in glokr.parameters())
    assert glokr_params < glora_params, f"GLoKr should be lighter: glokr={glokr_params} glora={glora_params}"
    # On 128×128 with auto factor √128≈11, GLoKr should land well under half of GLoRA.
    assert glokr_params * 2 < glora_params, (
        f"GLoKr expected at least 2× lighter than GLoRA at 128×128: glokr={glokr_params}, glora={glora_params}"
    )
    print(f"PASS: test_glokr_parameter_count_is_smaller_than_glora (glora={glora_params}, glokr={glokr_params})")
    return True


def test_glokr_method_contract_resolves_with_research_warning() -> bool:
    spec = resolve_adapter_method(
        {"model_arch": "anima", "network_module": "lycoris.locon", "lycoris_algo": "glokr"}
    )
    assert spec.method == "glokr"
    assert spec.backend == "lycoris"
    assert spec.lycoris_algo == "glokr"
    assert spec.safe_merge is False, "GLoKr declares base-weight-dependent merge"
    assert any("project-original" in w or "research" in w for w in spec.warnings), \
        f"GLoKr must warn about research-grade status: warnings={spec.warnings}"
    print("PASS: test_glokr_method_contract_resolves_with_research_warning")
    return True


def test_glokr_config_alias_lands_on_glokr_route() -> bool:
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "anima-lora",
            "adapter_type": "lycoris-glokr",
            "network_dim": 8,
            "network_alpha": 4,
            "glokr_factor": 4,
        }
    )
    assert getattr(cfg.network_module, "value", cfg.network_module) == "lycoris.locon"
    assert cfg.lycoris_algo == "glokr"
    assert cfg.glokr_factor == 4
    print("PASS: test_glokr_config_alias_lands_on_glokr_route")
    return True


def test_glokr_bias_path_engages_only_when_base_has_bias() -> bool:
    base_no_bias = nn.Linear(16, 16, bias=False)
    layer_no = GLoKrLinearLayer(16, 16, rank=4, alpha=2.0, factor=4, org_weight=base_no_bias.weight)
    assert not layer_no._has_bias_delta()

    base_with_bias = nn.Linear(16, 16, bias=True)
    layer_yes = GLoKrLinearLayer(
        16, 16, rank=4, alpha=2.0, factor=4,
        org_weight=base_with_bias.weight, org_bias=base_with_bias.bias,
    )
    assert layer_yes._has_bias_delta()
    assert torch.count_nonzero(layer_yes.get_delta_bias()) == 0, "Δbias must be 0 at init"
    print("PASS: test_glokr_bias_path_engages_only_when_base_has_bias")
    return True


# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        test_glokr_factor_selection_picks_divisor,
        test_glokr_delta_shape_matches_kron_definition,
        test_glokr_initialization_yields_zero_delta,
        test_glokr_forward_matches_materialized_delta,
        test_glokr_no_materialize_parity_with_materialized_delta,
        test_glokr_injection_produces_gradients,
        test_glokr_save_load_roundtrip,
        test_glokr_merge_into_base_matches_composite_forward,
        test_glokr_parameter_count_is_smaller_than_glora,
        test_glokr_method_contract_resolves_with_research_warning,
        test_glokr_config_alias_lands_on_glokr_route,
        test_glokr_bias_path_engages_only_when_base_has_bias,
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
    print("Anima GLoKr Smoke Test Results")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n{passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
