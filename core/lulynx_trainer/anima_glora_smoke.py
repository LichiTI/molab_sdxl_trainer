# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Anima GLoRA (Phase 1 — ΔW = W·A + B standard).

Covers:

- config aliases land on the GLoRA route,
- injection produces trainable parameters with non-zero gradients,
- ``get_delta_weight`` matches the forward output numerically (parity),
- save → load round-trip is bit-for-bit on the trainable tensors,
- merging the adapter into the base model produces the same output as the
  pre-merge composite forward,
- ``native`` and ``lora_compatible`` exports stay round-trippable.
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
targets_mod = _load_module("core.lulynx_trainer.anima_targets", TRAINER_ROOT / "anima_targets.py")
generalized_mod = _load_module("core.lulynx_trainer.generalized_adapters", TRAINER_ROOT / "generalized_adapters.py")
lycoris_mod = _load_module("core.lulynx_trainer.lycoris_layers", TRAINER_ROOT / "lycoris_layers.py")
export_rules_mod = _load_module("core.lulynx_trainer.glora_export_rules", TRAINER_ROOT / "glora_export_rules.py")
merge_export_mod = _load_module("core.lulynx_trainer.merge_export", TRAINER_ROOT / "merge_export.py")
contract_mod = _load_module("core.lulynx_trainer.method_adapter_contract", TRAINER_ROOT / "method_adapter_contract.py")

ConfigAdapter = config_adapter_mod.ConfigAdapter
get_anima_dit_targets = targets_mod.get_anima_dit_targets
GLoRALinearLayer = generalized_mod.GLoRALinearLayer
GLoRAConv2dLayer = generalized_mod.GLoRAConv2dLayer
LyCORISConfig = lycoris_mod.LyCORISConfig
LyCORISInjector = lycoris_mod.LyCORISInjector
LyCORISType = lycoris_mod.LyCORISType
export_glora_state_dict = export_rules_mod.export_glora_state_dict
merge_lycoris_into_base = merge_export_mod.merge_lycoris_into_base
resolve_adapter_method = contract_mod.resolve_adapter_method


class _TinyAnimaBlock(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(dim, dim, bias=False)
        self.self_attn.k_proj = nn.Linear(dim, dim, bias=False)
        self.self_attn.v_proj = nn.Linear(dim, dim, bias=False)
        self.self_attn.output_proj = nn.Linear(dim, dim, bias=False)

        self.mlp = nn.Module()
        self.mlp.layer1 = nn.Linear(dim, dim * 2, bias=False)
        self.mlp.layer2 = nn.Linear(dim * 2, dim, bias=False)


class _TinyAnimaNet(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyAnimaBlock(dim)])


class _TinyAnimaRoot(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.net = _TinyAnimaNet(dim)


class _TinyConvRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_in = nn.Conv2d(4, 8, kernel_size=1, bias=False)


def _make_glora_injector(rank: int = 2, alpha: float = 2.0, dropout: float = 0.0) -> LyCORISInjector:
    config = LyCORISConfig(
        lycoris_type=LyCORISType.GLORA,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    return LyCORISInjector(config)


def _seed_layer(layer: nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in layer.parameters():
            param.copy_(torch.randn_like(param) * 0.2)


# ---------------------------------------------------------------------------


def test_glora_method_contract_resolves() -> bool:
    spec = resolve_adapter_method(
        {"model_arch": "anima", "network_module": "lycoris.locon", "lycoris_algo": "glora"}
    )
    assert spec.method == "glora", spec.method
    assert spec.backend == "lycoris", spec.backend
    assert spec.lycoris_algo == "glora", spec.lycoris_algo
    assert spec.safe_merge is False, "GLoRA must declare base-weight-dependent merge"
    print("PASS: test_glora_method_contract_resolves")
    return True


def test_glora_config_alias_lands_on_glora_route() -> bool:
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "anima-lora",
            "adapter_type": "lycoris-glora",
            "network_dim": 4,
            "network_alpha": 2,
        }
    )
    assert getattr(cfg.model_type, "value", cfg.model_type) == "anima"
    assert getattr(cfg.network_module, "value", cfg.network_module) == "lycoris.locon"
    assert cfg.lycoris_algo == "glora"
    assert cfg.network_dim == 4
    print("PASS: test_glora_config_alias_lands_on_glora_route")
    return True


def test_glora_initialization_is_identity_delta() -> bool:
    """At init, ΔW must be zero: a2 and b2 are zero, so W·A + B = 0."""
    base = nn.Linear(8, 16, bias=False)
    layer = GLoRALinearLayer(8, 16, rank=4, alpha=2.0, org_weight=base.weight)
    delta = layer.get_delta_weight()
    assert torch.count_nonzero(delta) == 0, "ΔW must be zero at init"
    x = torch.randn(3, 8)
    out = layer(x)
    assert torch.allclose(out, torch.zeros_like(out)), "layer output must be zero at init"
    print("PASS: test_glora_initialization_is_identity_delta")
    return True


def test_glora_forward_matches_materialized_delta() -> bool:
    """Forward path must equal F.linear(x, get_delta_weight())."""
    base = nn.Linear(8, 16, bias=False)
    layer = GLoRALinearLayer(8, 16, rank=2, alpha=2.0, org_weight=base.weight).double()
    _seed_layer(layer, 1234)
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    fast = layer(x)
    reference = F.linear(x, layer.get_delta_weight())
    torch.testing.assert_close(fast, reference, rtol=1e-9, atol=1e-9)
    print("PASS: test_glora_forward_matches_materialized_delta")
    return True


def test_glora_injection_produces_gradients() -> bool:
    model = _TinyAnimaRoot()
    injector = _make_glora_injector(rank=2, alpha=2.0)
    injected = injector.inject(model, get_anima_dit_targets(include_llm_adapter=False), prefix="unet")
    assert injected, "expected at least one injected GLoRA layer"
    assert all(isinstance(layer, GLoRALinearLayer) for layer in injected.values()), "all injected must be GLoRA"

    # Push out of init by perturbing params so gradients are non-trivial.
    for layer in injected.values():
        _seed_layer(layer, 42)

    x = torch.randn(2, 3, 8)
    y = (
        model.net.blocks[0].self_attn.q_proj(x).sum()
        + model.net.blocks[0].mlp.layer1(x).sum()
    )
    y.backward()
    trainable = [param for param in injector.get_trainable_parameters() if param.grad is not None]
    assert trainable, "expected GLoRA trainable gradients after backward"
    print("PASS: test_glora_injection_produces_gradients")
    return True


def test_glora_save_load_roundtrip() -> bool:
    model = _TinyAnimaRoot()
    injector = _make_glora_injector(rank=2, alpha=2.0)
    injector.inject(model, ["self_attn.q_proj"], prefix="unet")
    layer = next(iter(injector.injected_layers.values()))
    _seed_layer(layer, 9999)

    expected_state = {key: value.clone() for key, value in injector.get_lora_state_dict().items()}
    probe = torch.randn(1, 2, 8)
    expected_output = layer(probe)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "glora.safetensors")
        injector.save(path)

        reloaded_model = _TinyAnimaRoot()
        with torch.no_grad():
            for src, dst in zip(model.modules(), reloaded_model.modules()):
                if isinstance(src, nn.Linear) and isinstance(dst, nn.Linear):
                    dst.weight.copy_(src.weight)
        reloaded_injector = _make_glora_injector(rank=2, alpha=2.0)
        reloaded_injector.inject(reloaded_model, ["self_attn.q_proj"], prefix="unet")
        reloaded_injector.load_lora(path)
        actual_state = reloaded_injector.get_lora_state_dict()
        actual_layer = next(iter(reloaded_injector.injected_layers.values()))

    for key, value in expected_state.items():
        if key.endswith(".alpha"):
            continue
        assert key in actual_state, f"missing {key}"
        torch.testing.assert_close(actual_state[key], value, rtol=1e-6, atol=1e-6)
    actual_output = actual_layer(probe)
    torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-5)
    print("PASS: test_glora_save_load_roundtrip")
    return True


def test_glora_merge_into_base_matches_composite_forward() -> bool:
    model = _TinyAnimaRoot()
    injector = _make_glora_injector(rank=2, alpha=2.0)
    injector.inject(model, ["self_attn.q_proj"], prefix="unet")
    layer = next(iter(injector.injected_layers.values()))
    _seed_layer(layer, 7777)

    x = torch.randn(2, 8)
    expected = model.net.blocks[0].self_attn.q_proj(x)
    merged = merge_lycoris_into_base(model, injector)
    assert merged == 1, f"expected 1 merge, got {merged}"
    actual = model.net.blocks[0].self_attn.q_proj(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    print("PASS: test_glora_merge_into_base_matches_composite_forward")
    return True


def test_glora_conv2d_injection_and_backward() -> bool:
    model = _TinyConvRoot()
    injector = _make_glora_injector(rank=2, alpha=2.0)
    injected = injector.inject(model, ["proj_in"], prefix="unet")
    assert "unet.proj_in" in injected
    layer = injected["unet.proj_in"]
    assert isinstance(layer, GLoRAConv2dLayer)

    _seed_layer(layer, 314)
    x = torch.randn(2, 4, 8, 8)
    out = model.proj_in(x)
    out.sum().backward()
    grads = [param.grad for param in injector.get_trainable_parameters() if param.grad is not None]
    assert grads, "expected Conv2d GLoRA gradients"
    print("PASS: test_glora_conv2d_injection_and_backward")
    return True


def test_glora_native_export_roundtrips() -> bool:
    model = _TinyAnimaRoot()
    injector = _make_glora_injector(rank=2, alpha=2.0)
    injector.inject(model, ["self_attn.q_proj"], prefix="unet")
    for param in injector.get_trainable_parameters():
        nn.init.constant_(param, 0.1)
    layer = next(iter(injector.injected_layers.values()))
    probe = torch.randn(1, 2, 8)
    expected = layer(probe)

    exported, metadata = export_glora_state_dict(
        injector.get_lora_state_dict(),
        {"ss_network_module": "networks.lora_anima"},
        export_mode="native",
    )
    base = "lora_unet_net_blocks_0_self_attn_q_proj"
    for suffix in ("a1.weight", "a2.weight", "b1.weight", "b2.weight", "alpha"):
        assert f"{base}.{suffix}" in exported, f"missing {suffix} in native export"
    assert metadata["ss_glora_export_mode"] == "native"
    assert metadata["ss_glora_native_export"] == "true"

    # Re-load using the export key layout (strip "lora_" prefix to match base name).
    reload_state = {}
    for key, value in exported.items():
        if key.startswith(f"{base}.") and not key.endswith(".alpha"):
            # Renormalize back to the injector's base name (no lora_ prefix).
            reload_state[key.replace("lora_", "", 1)] = value
        elif key.endswith(".alpha"):
            reload_state[key.replace("lora_", "", 1)] = value
    reloaded_model = _TinyAnimaRoot()
    with torch.no_grad():
        reloaded_model.net.blocks[0].self_attn.q_proj.weight.copy_(
            model.net.blocks[0].self_attn.q_proj.weight
        )
    reloaded = _make_glora_injector(rank=2, alpha=2.0)
    reloaded.inject(reloaded_model, ["self_attn.q_proj"], prefix="unet")
    loaded, total = reloaded.load_lora_state_dict(reload_state)
    assert loaded == total and total > 0, (loaded, total)
    actual = next(iter(reloaded.injected_layers.values()))(probe)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    print("PASS: test_glora_native_export_roundtrips")
    return True


def test_glora_lora_compatible_export_bakes_full_delta() -> bool:
    model = _TinyAnimaRoot()
    injector = _make_glora_injector(rank=2, alpha=2.0)
    injector.inject(model, ["self_attn.q_proj"], prefix="unet")
    for param in injector.get_trainable_parameters():
        nn.init.constant_(param, 0.05)

    layer = next(iter(injector.injected_layers.values()))
    expected_delta = layer.get_delta_weight().detach().clone()
    base_name = next(iter(injector.injected_layers.keys())).replace(".", "_")
    base_weights = {base_name: model.net.blocks[0].self_attn.q_proj.weight}

    exported, metadata = export_glora_state_dict(
        injector.get_lora_state_dict(),
        {"ss_network_module": "networks.lora_anima"},
        export_mode="lora_compatible",
        base_weights=base_weights,
    )
    assert metadata["ss_glora_export_mode"] == "lora_compatible"
    assert metadata["ss_glora_compatible_export"] == "true"
    keys = set(exported)
    assert not any(".a1.weight" in key or ".b2.weight" in key for key in keys), \
        "lora_compatible must strip raw GLoRA tensors"
    export_base = f"lora_{base_name}"
    up = exported[f"{export_base}.lora_up.weight"]
    down = exported[f"{export_base}.lora_down.weight"]
    torch.testing.assert_close(down, torch.eye(up.shape[1], dtype=up.dtype), rtol=0, atol=0)
    torch.testing.assert_close(up, expected_delta, rtol=1e-5, atol=1e-5)
    print("PASS: test_glora_lora_compatible_export_bakes_full_delta")
    return True


# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        test_glora_method_contract_resolves,
        test_glora_config_alias_lands_on_glora_route,
        test_glora_initialization_is_identity_delta,
        test_glora_forward_matches_materialized_delta,
        test_glora_injection_produces_gradients,
        test_glora_save_load_roundtrip,
        test_glora_merge_into_base_matches_composite_forward,
        test_glora_conv2d_injection_and_backward,
        test_glora_native_export_roundtrips,
        test_glora_lora_compatible_export_bakes_full_delta,
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
    print("Anima GLoRA Smoke Test Results")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n{passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
