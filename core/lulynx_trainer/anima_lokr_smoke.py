# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test for Anima LoKr routing, shape safety, and roundtrip."""
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
lycoris_mod = _load_module("core.lulynx_trainer.lycoris_layers", TRAINER_ROOT / "lycoris_layers.py")
export_rules_mod = _load_module("core.lulynx_trainer.lokr_export_rules", TRAINER_ROOT / "lokr_export_rules.py")
weight_format_mod = _load_module("core.lulynx_trainer.lokr_weight_format", TRAINER_ROOT / "lokr_weight_format.py")
train_norm_compat_mod = _load_module("core.lulynx_trainer.anima_train_norm_compat", TRAINER_ROOT / "anima_train_norm_compat.py")
merge_export_mod = _load_module("core.lulynx_trainer.merge_export", TRAINER_ROOT / "merge_export.py")

ConfigAdapter = config_adapter_mod.ConfigAdapter
get_anima_dit_targets = targets_mod.get_anima_dit_targets
LyCORISConfig = lycoris_mod.LyCORISConfig
LyCORISInjector = lycoris_mod.LyCORISInjector
LyCORISType = lycoris_mod.LyCORISType
LoKrLayer = lycoris_mod.LoKrLayer
LoKrConv2dLayer = lycoris_mod.LoKrConv2dLayer
export_lokr_state_dict = export_rules_mod.export_lokr_state_dict
collect_lokr_weight_layouts = weight_format_mod.collect_lokr_weight_layouts
export_anima_train_norm_state_dict = train_norm_compat_mod.export_anima_train_norm_state_dict
merge_lycoris_into_base = merge_export_mod.merge_lycoris_into_base


class _TinyAnimaBlock(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(dim, dim, bias=False)
        self.self_attn.k_proj = nn.Linear(dim, dim, bias=False)
        self.self_attn.v_proj = nn.Linear(dim, dim, bias=False)
        self.self_attn.output_proj = nn.Linear(dim, dim, bias=False)

        self.cross_attn = nn.Module()
        self.cross_attn.q_proj = nn.Linear(dim, dim, bias=False)
        self.cross_attn.k_proj = nn.Linear(dim, dim, bias=False)
        self.cross_attn.v_proj = nn.Linear(dim, dim, bias=False)
        self.cross_attn.output_proj = nn.Linear(dim, dim, bias=False)

        self.mlp = nn.Module()
        self.mlp.layer1 = nn.Linear(dim, dim * 2, bias=False)
        self.mlp.layer2 = nn.Linear(dim * 2, dim, bias=False)
        self.norm = nn.LayerNorm(dim)


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


def _make_lokr_injector(**kwargs) -> LyCORISInjector:
    config = LyCORISConfig(
        lycoris_type=LyCORISType.LOKR,
        rank=kwargs.pop("rank", 2),
        alpha=kwargs.pop("alpha", 2),
        lokr_factor=kwargs.pop("lokr_factor", -1),
        lokr_rank_dropout=kwargs.pop("lokr_rank_dropout", 0.0),
        lokr_module_dropout=kwargs.pop("lokr_module_dropout", 0.0),
        train_norm=kwargs.pop("train_norm", False),
        lokr_full_matrix=kwargs.pop("lokr_full_matrix", False),
        lokr_decompose_both=kwargs.pop("lokr_decompose_both", False),
        lokr_unbalanced_factorization=kwargs.pop("lokr_unbalanced_factorization", False),
        lokr_no_materialize_strategy=kwargs.pop("lokr_no_materialize_strategy", "auto"),
    )
    if kwargs:
        raise AssertionError(f"Unexpected config kwargs: {kwargs}")
    return LyCORISInjector(config)


def _single_lokr_layer(injector: LyCORISInjector) -> LoKrLayer:
    layer = next(iter(injector.injected_layers.values()))
    assert isinstance(layer, LoKrLayer)
    return layer


def _single_norm_layer(injector: LyCORISInjector):
    for layer in injector.injected_layers.values():
        if type(layer).__name__ == "_NormAdapter":
            return layer
    raise AssertionError("Expected a norm adapter")


def _seed_lokr_params(layer: LoKrLayer, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in layer.parameters():
            param.copy_(torch.randn_like(param) * 0.2)


def test_anima_lokr_config_aliases() -> bool:
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "anima-lora",
            "adapter_type": "lycoris-lokr",
            "lokr_rank": 4,
            "lokr_alpha": 3,
            "lokr_dropout": 0.25,
            "lokr_factor": 6,
            "lokr_train_norm": True,
            "full_matrix": True,
            "decompose_both": "true",
            "unbalanced_factorization": "1",
            "no_materialize_strategy": "matmul",
            "lokr_export_mode": "native",
        }
    )
    assert getattr(cfg.model_type, "value", cfg.model_type) == "anima"
    assert getattr(cfg.network_module, "value", cfg.network_module) == "lycoris.locon"
    assert cfg.lycoris_algo == "lokr"
    assert cfg.network_dim == 4
    assert cfg.network_alpha == 3
    assert cfg.network_dropout == 0.25
    assert cfg.lycoris_lokr_factor == 6
    assert cfg.lokr_train_norm is True
    assert cfg.lokr_full_matrix is True
    assert cfg.lokr_decompose_both is True
    assert cfg.lokr_unbalanced_factorization is True
    assert cfg.lokr_no_materialize_strategy == "matmul"
    assert cfg.lokr_export_mode == "native"
    print("PASS: test_anima_lokr_config_aliases")
    return True


def test_anima_lokr_injection_and_backward() -> bool:
    model = _TinyAnimaRoot()
    injector = _make_lokr_injector(lokr_factor=5, lokr_rank_dropout=0.5, train_norm=True)
    injected = injector.inject(model, get_anima_dit_targets(include_llm_adapter=False), prefix="unet")
    assert any(name.endswith("self_attn.q_proj") for name in injected)
    assert any(name.endswith("mlp.layer1") for name in injected)
    assert any(isinstance(layer, LoKrLayer) and layer.factor in {1, 2, 4, 8} for layer in injected.values())
    assert any(name.endswith("norm") for name in injected)

    x = torch.randn(2, 3, 8)
    y = model.net.blocks[0].self_attn.q_proj(x).sum() + model.net.blocks[0].mlp.layer1(x).sum()
    y.backward()
    grads = [param.grad for param in injector.get_trainable_parameters() if param.grad is not None]
    assert grads, "Expected LoKr trainable gradients after backward"
    print("PASS: test_anima_lokr_injection_and_backward")
    return True


def test_lokr_module_dropout_output_shape() -> bool:
    layer = LoKrLayer(8, 16, rank=2, alpha=2, factor=4, module_dropout=1.0)
    layer.train()
    x = torch.randn(2, 3, 8)
    y = layer(x)
    assert y.shape == (2, 3, 16), y.shape
    assert torch.count_nonzero(y) == 0
    print("PASS: test_lokr_module_dropout_output_shape")
    return True


def test_lokr_hotfix_layout_modes() -> bool:
    direct = LoKrLayer(8, 16, rank=100000, alpha=100000, factor=4)
    assert direct.full_matrix is True
    assert hasattr(direct, "lokr_w1") and hasattr(direct, "lokr_w2")
    assert not hasattr(direct, "lokr_w1_a") and not hasattr(direct, "lokr_w2_a")
    assert direct.scaling == 1.0
    assert direct.get_delta_weight().shape == (16, 8)

    decomposed = LoKrLayer(8, 16, rank=1, alpha=0.25, factor=4, decompose_both=True)
    assert decomposed.w1_decomposed is True
    assert decomposed.w2_decomposed is True
    assert hasattr(decomposed, "lokr_w1_a") and hasattr(decomposed, "lokr_w1_b")
    assert hasattr(decomposed, "lokr_w2_a") and hasattr(decomposed, "lokr_w2_b")
    assert decomposed.scaling == 0.25
    assert decomposed.get_delta_weight().shape == (16, 8)

    unbalanced = LoKrLayer(8, 16, rank=1, alpha=1, factor=4, decompose_both=True, unbalanced_factorization=True)
    assert unbalanced.unbalanced_factorization is True
    assert unbalanced.get_delta_weight().shape == (16, 8)
    print("PASS: test_lokr_hotfix_layout_modes")
    return True


def test_lokr_no_materialize_forward_backward_parity() -> bool:
    cases = [
        (
            "mixed_direct_decomposed",
            LoKrLayer(8, 16, rank=2, alpha=2, factor=4, no_materialize_forward=True, no_materialize_strategy="legacy"),
            (2, 3, 8),
        ),
        (
            "decomposed_both_matmul",
            LoKrLayer(8, 16, rank=1, alpha=1, factor=4, decompose_both=True, no_materialize_forward=True, no_materialize_strategy="matmul"),
            (4, 8),
        ),
        (
            "full_matrix_native_auto",
            LoKrLayer(8, 16, rank=100000, alpha=100000, factor=4, no_materialize_forward=True, no_materialize_strategy="auto"),
            (2, 8),
        ),
        (
            "unbalanced_decomposed",
            LoKrLayer(12, 24, rank=1, alpha=1, factor=4, decompose_both=True, unbalanced_factorization=True, no_materialize_forward=True, no_materialize_strategy="matmul"),
            (2, 5, 12),
        ),
    ]

    for index, (name, layer, input_shape) in enumerate(cases):
        layer = layer.double()
        layer.train()
        _seed_lokr_params(layer, 1000 + index)
        state_keys_before = tuple(layer.state_dict().keys())

        torch.manual_seed(2000 + index)
        x_fast = torch.randn(input_shape, dtype=torch.float64, requires_grad=True)
        x_ref = x_fast.detach().clone().requires_grad_(True)
        assert layer._can_use_no_materialize_forward(x_fast), f"fast path disabled for {name}"

        out_fast = layer(x_fast)
        out_ref = F.linear(x_ref, layer.get_delta_weight())
        upstream = torch.randn_like(out_fast)

        torch.testing.assert_close(out_fast, out_ref, rtol=1e-7, atol=1e-8)

        params = tuple(layer.parameters())
        fast_grads = torch.autograd.grad((out_fast * upstream).sum(), (x_fast, *params))
        ref_grads = torch.autograd.grad((out_ref * upstream).sum(), (x_ref, *params))
        for fast_grad, ref_grad in zip(fast_grads, ref_grads):
            torch.testing.assert_close(fast_grad, ref_grad, rtol=1e-7, atol=1e-8)

        assert tuple(layer.state_dict().keys()) == state_keys_before, f"state_dict keys changed for {name}"

    default_layer = LoKrLayer(8, 16, rank=2, alpha=2, factor=4)
    assert not default_layer._can_use_no_materialize_forward(torch.randn(2, 8))

    dropped = LoKrLayer(8, 16, rank=2, alpha=2, factor=4, rank_dropout=0.5, no_materialize_forward=True)
    dropped.train()
    assert not dropped._can_use_no_materialize_forward(torch.randn(2, 8))

    print("PASS: test_lokr_no_materialize_forward_backward_parity")
    return True


def test_lokr_no_materialize_strategy_auto_prefers_benchmarked_layouts() -> bool:
    small = LoKrLayer(1024, 1024, rank=8, alpha=8, factor=16, no_materialize_forward=True, no_materialize_strategy="auto")
    large = LoKrLayer(2048, 2048, rank=8, alpha=8, factor=16, no_materialize_forward=True, no_materialize_strategy="auto")
    wide = LoKrLayer(2048, 2048, rank=8, alpha=8, factor=32, no_materialize_forward=True, no_materialize_strategy="auto")

    assert small.get_resolved_no_materialize_strategy() == "legacy"
    assert large.get_resolved_no_materialize_strategy() == "matmul"
    assert wide.get_resolved_no_materialize_strategy() == "matmul"
    print("PASS: test_lokr_no_materialize_strategy_auto_prefers_benchmarked_layouts")
    return True


def test_anima_lokr_roundtrip() -> bool:
    model = _TinyAnimaRoot()
    injector = _make_lokr_injector(lokr_factor=4, train_norm=True, lokr_decompose_both=True, rank=1, alpha=1)
    injector.inject(model, ["self_attn.q_proj", "norm"], prefix="unet")
    expected = injector.get_lora_state_dict()
    for param in injector.get_trainable_parameters():
        nn.init.constant_(param, 0.123)
    expected = {key: value.clone() for key, value in injector.get_lora_state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "anima_lokr.safetensors")
        injector.save(path)

        reloaded_model = _TinyAnimaRoot()
        reloaded = _make_lokr_injector(lokr_factor=4, train_norm=True, lokr_decompose_both=True, rank=1, alpha=1)
        reloaded.inject(reloaded_model, ["self_attn.q_proj", "norm"], prefix="unet")
        reloaded.load_lora(path)
        actual = reloaded.get_lora_state_dict()

    for key, expected_value in expected.items():
        if key.endswith(".alpha"):
            continue
        assert key in actual, key
        assert torch.allclose(actual[key], expected_value), key

    y = reloaded_model.net.blocks[0].self_attn.q_proj(torch.randn(1, 2, 8))
    assert torch.isfinite(y).all()
    print("PASS: test_anima_lokr_roundtrip")
    return True


def test_lokr_native_export_roundtrip_without_rank() -> bool:
    model = _TinyAnimaRoot()
    injector = _make_lokr_injector(lokr_factor=4, train_norm=False, lokr_decompose_both=True, rank=1, alpha=1)
    injector.inject(model, ["self_attn.q_proj"], prefix="unet")
    for param in injector.get_trainable_parameters():
        nn.init.constant_(param, 0.2)

    source_layer = _single_lokr_layer(injector)
    source_input = torch.randn(2, 8)
    expected = source_layer(source_input)
    exported, metadata = export_lokr_state_dict(
        injector.get_lora_state_dict(),
        {"ss_network_module": "networks.lora_anima"},
        export_mode="native",
    )
    base = "lora_unet_net_blocks_0_self_attn_q_proj"
    assert f"{base}.lokr_w1" in exported
    assert f"{base}.lokr_w2" in exported
    assert f"{base}.lokr_rank" not in exported
    assert metadata["ss_lokr_export_mode"] == "native"
    assert metadata["ss_lokr_rank_exported"] == "false"

    reloaded_model = _TinyAnimaRoot()
    reloaded = _make_lokr_injector(lokr_factor=4, train_norm=False, lokr_decompose_both=True, rank=1, alpha=1)
    reloaded.inject(reloaded_model, ["self_attn.q_proj"], prefix="unet")
    loaded, total = reloaded.load_lora_state_dict(exported, metadata=metadata)
    assert loaded == total and total > 0, (loaded, total)

    actual = _single_lokr_layer(reloaded)(source_input)
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
    print("PASS: test_lokr_native_export_roundtrip_without_rank")
    return True


def test_lokr_direct_native_layout_loads() -> bool:
    layer = LoKrLayer(8, 16, rank=100000, alpha=100000, factor=4)
    with torch.no_grad():
        layer.lokr_w1.fill_(0.25)
        layer.lokr_w2.fill_(0.5)
    state_dict = {
        "lora_unet_demo.lokr_w1": layer.lokr_w1.detach().clone(),
        "lora_unet_demo.lokr_w2": layer.lokr_w2.detach().clone(),
        "lora_unet_demo.alpha": torch.tensor(100000.0),
    }
    metadata = {
        "ss_lokr_export_mode": "native",
        "ss_lokr_native_export": "true",
        "ss_lokr_rank_exported": "false",
    }

    target = LoKrLayer(8, 16, rank=100000, alpha=100000, factor=4)
    injector = _make_lokr_injector(rank=100000, alpha=100000, lokr_factor=4)
    injector._injected_layers = {"unet_demo": target}
    loaded, total = injector.load_lora_state_dict(state_dict, metadata=metadata)
    assert loaded == total and total == 2, (loaded, total)

    x = torch.randn(1, 8)
    assert torch.allclose(target(x), layer(x), atol=1e-5, rtol=1e-5)
    print("PASS: test_lokr_direct_native_layout_loads")
    return True


def test_lokr_lora_compatible_export_strips_lokr_keys() -> bool:
    model = _TinyAnimaRoot()
    injector = _make_lokr_injector(lokr_factor=4, rank=1, alpha=1, lokr_decompose_both=True)
    injector.inject(model, ["self_attn.q_proj"], prefix="unet")
    for param in injector.get_trainable_parameters():
        nn.init.constant_(param, 0.125)

    exported, metadata = export_lokr_state_dict(
        injector.get_lora_state_dict(),
        {"ss_network_module": "networks.lora_anima", "lycoris_lokr_factor": "4"},
        export_mode="lora_compatible",
    )
    keys = set(exported)
    assert any(key.endswith(".lora_down.weight") for key in keys)
    assert any(key.endswith(".lora_up.weight") for key in keys)
    assert not any(".lokr_" in key for key in keys)
    assert metadata["ss_lokr_export_mode"] == "lora_compatible"
    assert metadata["ss_lokr_compatible_export"] == "true"
    assert "lycoris_lokr_factor" not in metadata
    print("PASS: test_lokr_lora_compatible_export_strips_lokr_keys")
    return True


def test_lokr_decomposed_native_layout_loads_without_rank() -> bool:
    source = LoKrLayer(8, 16, rank=1, alpha=1, factor=4, decompose_both=True)
    with torch.no_grad():
        source.lokr_w1_a.fill_(0.1)
        source.lokr_w1_b.fill_(0.2)
        source.lokr_w2_a.fill_(0.3)
        source.lokr_w2_b.fill_(0.4)
    state_dict = {
        "lora_unet_demo.lokr_w1_a": source.lokr_w1_a.detach().clone(),
        "lora_unet_demo.lokr_w1_b": source.lokr_w1_b.detach().clone(),
        "lora_unet_demo.lokr_w2_a": source.lokr_w2_a.detach().clone(),
        "lora_unet_demo.lokr_w2_b": source.lokr_w2_b.detach().clone(),
        "lora_unet_demo.alpha": torch.tensor(1.0),
    }
    metadata = {
        "ss_lokr_export_mode": "native",
        "ss_lokr_native_export": "true",
        "ss_lokr_rank_exported": "false",
    }

    target = LoKrLayer(8, 16, rank=1, alpha=1, factor=4, decompose_both=True)
    injector = _make_lokr_injector(rank=1, alpha=1, lokr_factor=4, lokr_decompose_both=True)
    injector._injected_layers = {"unet_demo": target}
    loaded, total = injector.load_lora_state_dict(state_dict, metadata=metadata)
    assert loaded == total and total == 4, (loaded, total)

    x = torch.randn(1, 8)
    assert torch.allclose(target(x), source(x), atol=1e-5, rtol=1e-5)
    print("PASS: test_lokr_decomposed_native_layout_loads_without_rank")
    return True


def test_lokr_weight_layout_detection_contract() -> bool:
    state_dict = {
        "lora_unet_direct.lokr_w1": torch.zeros(4, 2),
        "lora_unet_direct.lokr_w2": torch.zeros(4, 4),
        "lora_unet_direct.alpha": torch.tensor(4.0),
        "lora_unet_decomp.lokr_w1_a": torch.zeros(4, 1),
        "lora_unet_decomp.lokr_w1_b": torch.zeros(1, 2),
        "lora_unet_decomp.lokr_w2_a": torch.zeros(4, 1),
        "lora_unet_decomp.lokr_w2_b": torch.zeros(1, 4),
        "lora_unet_decomp.alpha": torch.tensor(1.0),
    }
    layouts = collect_lokr_weight_layouts(state_dict)
    assert layouts["unet_direct"].layout_kind == "direct"
    assert layouts["unet_direct"].has_rank_key is False
    assert layouts["unet_direct"].rank == 4
    assert layouts["unet_decomp"].layout_kind == "decomposed"
    assert layouts["unet_decomp"].rank == 1
    assert layouts["unet_decomp"].factor == 4
    print("PASS: test_lokr_weight_layout_detection_contract")
    return True


def test_lokr_export_metadata_contract() -> bool:
    model = _TinyAnimaRoot()
    injector = _make_lokr_injector(
        lokr_factor=4,
        rank=1,
        alpha=1,
        lokr_decompose_both=True,
        lokr_unbalanced_factorization=True,
    )
    injector.inject(model, ["self_attn.q_proj"], prefix="unet")
    for param in injector.get_trainable_parameters():
        nn.init.constant_(param, 0.15)

    metadata_seed = {
        "ss_network_module": "networks.lora_anima",
        "lycoris_lokr_factor": "4",
        "lycoris_factor": "4",
        "lokr_rank_dropout": "0.0",
        "lokr_module_dropout": "0.0",
        "lokr_full_matrix": "false",
        "lokr_decompose_both": "true",
        "lokr_unbalanced_factorization": "true",
        "lokr_export_mode": "native",
    }

    native_exported, native_metadata = export_lokr_state_dict(
        injector.get_lora_state_dict(),
        dict(metadata_seed),
        export_mode="native",
    )
    assert native_metadata["ss_lokr_export_mode"] == "native"
    assert native_metadata["ss_lokr_native_export"] == "true"
    assert native_metadata["ss_lokr_scale_export_format"] == "comfyui_baked_single_scale"
    assert "ss_lokr_compatible_export" not in native_metadata
    assert all(key not in native_metadata for key in metadata_seed if key != "ss_network_module")
    assert any(key.endswith(".lokr_w1") for key in native_exported)
    assert any(key.endswith(".lokr_w2") for key in native_exported)

    compatible_exported, compatible_metadata = export_lokr_state_dict(
        injector.get_lora_state_dict(),
        dict(metadata_seed),
        export_mode="lora_compatible",
    )
    assert compatible_metadata["ss_lokr_export_mode"] == "lora_compatible"
    assert compatible_metadata["ss_lokr_native_export"] == "false"
    assert compatible_metadata["ss_lokr_compatible_export"] == "true"
    assert compatible_metadata["ss_lokr_scale_export_format"] == "lora_identity_down"
    assert all(key not in compatible_metadata for key in metadata_seed if key != "ss_network_module")
    assert not any(".lokr_" in key for key in compatible_exported)
    assert any(key.endswith(".lora_down.weight") for key in compatible_exported)
    assert any(key.endswith(".lora_up.weight") for key in compatible_exported)
    print("PASS: test_lokr_export_metadata_contract")
    return True


def test_lokr_train_norm_export_writes_comfyui_diff() -> bool:
    model = _TinyAnimaRoot()
    injector = _make_lokr_injector(train_norm=True, rank=1, alpha=1, lokr_factor=4)
    injector.inject(model, ["norm"], prefix="unet")
    layer = _single_norm_layer(injector)
    with torch.no_grad():
        layer.scale.fill_(0.25)
        layer.bias.fill_(0.5)

    exported, metadata = export_anima_train_norm_state_dict(
        injector.get_lora_state_dict(),
        {},
        injector=injector,
    )
    assert metadata["ss_train_norm_export_format"] == "comfyui_diff"
    assert metadata["ss_train_norm_exported_count"] == "2"
    assert "unet_net_blocks_0_norm.norm_scale" not in exported
    assert "unet_net_blocks_0_norm.norm_bias" not in exported
    assert "lora_unet_net_blocks_0_norm.diff" in exported
    assert "lora_unet_net_blocks_0_norm.diff_b" in exported
    torch.testing.assert_close(exported["lora_unet_net_blocks_0_norm.diff"], torch.full_like(exported["lora_unet_net_blocks_0_norm.diff"], 0.25))
    torch.testing.assert_close(exported["lora_unet_net_blocks_0_norm.diff_b"], torch.full_like(exported["lora_unet_net_blocks_0_norm.diff_b"], 0.5))
    print("PASS: test_lokr_train_norm_export_writes_comfyui_diff")
    return True


def test_lokr_train_norm_loads_comfyui_diff_and_old_weight_format() -> bool:
    diff_model = _TinyAnimaRoot()
    diff_injector = _make_lokr_injector(train_norm=True, rank=1, alpha=1, lokr_factor=4)
    diff_injector.inject(diff_model, ["norm"], prefix="unet")
    diff_layer = _single_norm_layer(diff_injector)

    diff_state = {
        "lora_unet_net_blocks_0_norm.diff": torch.full_like(diff_layer.scale, 0.25),
        "lora_unet_net_blocks_0_norm.diff_b": torch.full_like(diff_layer.bias, 0.5),
    }
    loaded, total = diff_injector.load_lora_state_dict(diff_state)
    assert loaded == total == 2, (loaded, total)
    torch.testing.assert_close(diff_layer.base_weight * (1.0 + diff_layer.scale), torch.full_like(diff_layer.base_weight, 1.25))
    torch.testing.assert_close(diff_layer.base_bias + diff_layer.bias, torch.full_like(diff_layer.bias, 0.5))

    old_model = _TinyAnimaRoot()
    old_injector = _make_lokr_injector(train_norm=True, rank=1, alpha=1, lokr_factor=4)
    old_injector.inject(old_model, ["norm"], prefix="unet")
    old_layer = _single_norm_layer(old_injector)
    old_state = {
        "lora_unet_net_blocks_0_norm.weight": torch.full_like(old_layer.scale, 1.25),
        "lora_unet_net_blocks_0_norm.bias": torch.full_like(old_layer.bias, 0.5),
    }
    loaded, total = old_injector.load_lora_state_dict(old_state)
    assert loaded == total == 2, (loaded, total)
    torch.testing.assert_close(old_layer.base_weight * (1.0 + old_layer.scale), torch.full_like(old_layer.base_weight, 1.25))
    torch.testing.assert_close(old_layer.base_bias + old_layer.bias, torch.full_like(old_layer.bias, 0.5))
    print("PASS: test_lokr_train_norm_loads_comfyui_diff_and_old_weight_format")
    return True


def test_lokr_native_merge_accepts_decomposed_weights() -> bool:
    source_model = _TinyAnimaRoot()
    source_injector = _make_lokr_injector(lokr_factor=4, rank=1, alpha=1, lokr_decompose_both=True)
    source_injector.inject(source_model, ["self_attn.q_proj"], prefix="unet")
    for param in source_injector.get_trainable_parameters():
        nn.init.constant_(param, 0.2)

    exported, metadata = export_lokr_state_dict(
        source_injector.get_lora_state_dict(),
        {"ss_network_module": "networks.lora_anima"},
        export_mode="native",
    )

    merge_model = _TinyAnimaRoot()
    merge_injector = _make_lokr_injector(lokr_factor=4, rank=1, alpha=1, lokr_decompose_both=True)
    merge_injector.inject(merge_model, ["self_attn.q_proj"], prefix="unet")
    loaded, total = merge_injector.load_lora_state_dict(exported, metadata=metadata)
    assert loaded == total and total > 0, (loaded, total)

    x = torch.randn(2, 8)
    expected = merge_model.net.blocks[0].self_attn.q_proj(x)
    merged = merge_lycoris_into_base(merge_model, merge_injector)
    assert merged == 1, merged
    actual = merge_model.net.blocks[0].self_attn.q_proj(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    print("PASS: test_lokr_native_merge_accepts_decomposed_weights")
    return True


def test_lokr_conv2d_injection_and_roundtrip() -> bool:
    model = _TinyConvRoot()
    injector = _make_lokr_injector(rank=2, alpha=2, lokr_factor=4)
    injected = injector.inject(model, ["proj_in"], prefix="unet")
    assert "unet.proj_in" in injected
    layer = injected["unet.proj_in"]
    assert isinstance(layer, LoKrConv2dLayer)

    for param in injector.get_trainable_parameters():
        nn.init.constant_(param, 0.1)

    x = torch.randn(2, 4, 8, 8)
    source_out = model.proj_in(x)
    source_out.sum().backward()
    grads = [param.grad for param in injector.get_trainable_parameters() if param.grad is not None]
    assert grads, "Expected Conv2d LoKr trainable gradients after backward"

    state = {key: value.clone() for key, value in injector.get_lora_state_dict().items()}
    reloaded_model = _TinyConvRoot()
    with torch.no_grad():
        reloaded_model.proj_in.weight.copy_(model.proj_in.weight)
    reloaded = _make_lokr_injector(rank=2, alpha=2, lokr_factor=4)
    reloaded.inject(reloaded_model, ["proj_in"], prefix="unet")
    loaded, total = reloaded.load_lora_state_dict(state)
    assert loaded == total and total > 0, (loaded, total)

    target_out = reloaded_model.proj_in(x.detach())
    torch.testing.assert_close(target_out, source_out.detach(), rtol=1e-5, atol=1e-5)
    print("PASS: test_lokr_conv2d_injection_and_roundtrip")
    return True


def main() -> int:
    tests = [
        test_anima_lokr_config_aliases,
        test_anima_lokr_injection_and_backward,
        test_lokr_module_dropout_output_shape,
        test_lokr_hotfix_layout_modes,
        test_lokr_no_materialize_forward_backward_parity,
        test_lokr_no_materialize_strategy_auto_prefers_benchmarked_layouts,
        test_anima_lokr_roundtrip,
        test_lokr_native_export_roundtrip_without_rank,
        test_lokr_direct_native_layout_loads,
        test_lokr_lora_compatible_export_strips_lokr_keys,
        test_lokr_decomposed_native_layout_loads_without_rank,
        test_lokr_weight_layout_detection_contract,
        test_lokr_export_metadata_contract,
        test_lokr_train_norm_export_writes_comfyui_diff,
        test_lokr_train_norm_loads_comfyui_diff_and_old_weight_format,
        test_lokr_native_merge_accepts_decomposed_weights,
        test_lokr_conv2d_injection_and_roundtrip,
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
    print("Anima LoKr Smoke Test Results")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n{passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
