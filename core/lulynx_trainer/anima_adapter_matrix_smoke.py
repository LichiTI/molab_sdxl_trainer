# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test for Anima adapter matrix config-to-injector wiring."""
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
_ensure_namespace("core.lulynx", CORE_ROOT / "lulynx")
_load_module("core.safe_pickle", CORE_ROOT / "safe_pickle.py")
_load_module("core.configs", CORE_ROOT / "configs.py")
_load_module("core.memory_vortex_v2", CORE_ROOT / "memory_vortex_v2.py")
_load_module("core.lulynx.dora_layer", CORE_ROOT / "lulynx" / "dora_layer.py")
_load_module("core.lulynx_trainer.config", TRAINER_ROOT / "config.py")
_load_module("core.lulynx_trainer.model_family", TRAINER_ROOT / "model_family.py")
config_adapter_mod = _load_module("core.lulynx_trainer.config_adapter", TRAINER_ROOT / "config_adapter.py")
targets_mod = _load_module("core.lulynx_trainer.anima_targets", TRAINER_ROOT / "anima_targets.py")
lora_mod = _load_module("core.lulynx_trainer.lora_injector", TRAINER_ROOT / "lora_injector.py")
lycoris_mod = _load_module("core.lulynx_trainer.lycoris_layers", TRAINER_ROOT / "lycoris_layers.py")
lora_fa_mod = _load_module("core.lulynx_trainer.lora_fa_layer", TRAINER_ROOT / "lora_fa_layer.py")
vera_mod = _load_module("core.lulynx_trainer.vera_layer", TRAINER_ROOT / "vera_layer.py")
tlora_mod = _load_module("core.lulynx_trainer.tlora", TRAINER_ROOT / "tlora.py")
flexrank_mod = _load_module("core.lulynx_trainer.flexrank_lora", TRAINER_ROOT / "flexrank_lora.py")

ConfigAdapter = config_adapter_mod.ConfigAdapter
get_anima_dit_targets = targets_mod.get_anima_dit_targets
LoRAInjector = lora_mod.LoRAInjector
LoRALayer = lora_mod.LoRALayer
LoRALinear = lora_mod.LoRALinear
HydraLoRALinear = _load_module("core.lulynx_trainer.hydralora", TRAINER_ROOT / "hydralora.py").HydraLoRALinear
FeRALinear = _load_module("core.lulynx_trainer.fera", TRAINER_ROOT / "fera.py").FeRALinear
LoHaLayer = lycoris_mod.LoHaLayer
LoKrLayer = lycoris_mod.LoKrLayer
FullRankAdapter = lycoris_mod.FullRankAdapter
IA3Adapter = lycoris_mod.IA3Adapter
DiagOFTAdapter = lycoris_mod.DiagOFTAdapter
LyCORISConfig = lycoris_mod.LyCORISConfig
LyCORISInjector = lycoris_mod.LyCORISInjector
LyCORISType = lycoris_mod.LyCORISType
LoRAFALinear = lora_fa_mod.LoRAFALinear
VeRALinear = vera_mod.VeRALinear
TLoRALinear = tlora_mod.TLoRALinear
FlexRankLoRALinear = flexrank_mod.FlexRankLoRALinear


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


class _TinyAnimaRoot(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Module()
        self.net.blocks = nn.ModuleList([_TinyAnimaBlock(dim)])


def _anima_cfg(**overrides):
    data = {
        "schema_id": "anima-lora",
        "network_dim": 4,
        "network_alpha": 4,
    }
    data.update(overrides)
    return ConfigAdapter.from_frontend_dict(data)


def _network_module(cfg) -> str:
    return str(getattr(cfg.network_module, "value", cfg.network_module))


def _inject_lora_variant(cfg):
    model = _TinyAnimaRoot()
    network_module = _network_module(cfg)
    injector = LoRAInjector(
        rank=cfg.network_dim,
        alpha=cfg.network_alpha,
        dropout=cfg.network_dropout,
        pissa_enabled=cfg.pissa_enabled,
        pissa_niter=cfg.pissa_init_iters,
        svd_algo=cfg.pissa_svd_algo,
        dora_enabled=cfg.use_dora or cfg.dora_enabled,
        model_arch="anima",
        tlora_enabled=network_module == "networks.tlora" or network_module.endswith("tlora"),
        tlora_min_rank=cfg.tlora_min_rank,
        tlora_rank_schedule=cfg.tlora_rank_schedule,
        tlora_orthogonal_init=cfg.tlora_orthogonal_init,
        tlora_total_steps=1000,
        vera_enabled=network_module == "networks.vera" or cfg.vera_enabled,
        vera_d_initial=cfg.vera_d_initial,
        vera_prng_key=cfg.vera_prng_key,
        lora_fa_enabled=network_module == "networks.lora_fa" or cfg.lora_fa_enabled,
        flexrank_enabled=network_module == "networks.flexrank_lora" or cfg.flexrank_lora_enabled,
        flexrank_rank_range_min=cfg.flexrank_lora_rank_range_min,
        hydralora_enabled=cfg.hydralora_enabled,
        hydralora_num_experts=cfg.hydralora_num_experts,
        hydralora_routing=cfg.hydralora_routing,
        hydralora_top_k=cfg.hydralora_top_k,
        hydralora_sparse_top_k=cfg.hydralora_sparse_top_k,
        fera_enabled=cfg.fera_enabled,
        fera_gate_init=cfg.fera_gate_init,
    )
    injected = injector.inject(model, ["self_attn.q_proj"], prefix="unet")
    assert len(injected) == 1
    return next(iter(injected.values()))


def _inject_lycoris_variant(cfg):
    model = _TinyAnimaRoot()
    injector = LyCORISInjector(
        LyCORISConfig(
            lycoris_type=LyCORISType(cfg.lycoris_algo),
            rank=cfg.network_dim,
            alpha=cfg.network_alpha,
            dropout=cfg.network_dropout,
            lokr_factor=cfg.lycoris_lokr_factor,
            lokr_rank_dropout=cfg.lokr_rank_dropout,
            lokr_module_dropout=cfg.lokr_module_dropout,
            lokr_no_materialize_forward=cfg.lokr_no_materialize_forward,
            train_norm=cfg.lycoris_train_norm or cfg.lokr_train_norm,
        )
    )
    injected = injector.inject(model, ["self_attn.q_proj"], prefix="unet")
    assert len(injected) == 1
    return next(iter(injected.values()))


def test_lora_family_matrix() -> bool:
    matrix = [
        ("lora", LoRALinear),
        ("lora_plus", LoRALinear),
        ("dora", LoRALinear),
        ("lora_fa", LoRAFALinear),
        ("vera", VeRALinear),
        ("tlora", TLoRALinear),
        ("flexrank", FlexRankLoRALinear),
        ("hydralora", HydraLoRALinear),
        ("fera", FeRALinear),
    ]
    for lora_type, expected_type in matrix:
        cfg = _anima_cfg(lora_type=lora_type, tlora_rank_schedule="linear")
        if lora_type == "hydralora":
            cfg.hydralora_sparse_top_k = True
        layer = _inject_lora_variant(cfg)
        assert isinstance(layer, expected_type), (lora_type, type(layer), expected_type)
        if lora_type == "lora_plus":
            assert cfg.lora_plus_enabled is True
        if lora_type == "dora":
            assert getattr(layer, "use_dora", False) is True
            assert type(layer.lora).__name__ == "DoRALinear"
        if lora_type == "lora_fa":
            assert layer.lora_down.weight.requires_grad is False
            assert layer.lora_up.weight.requires_grad is True
        if lora_type == "vera":
            assert layer.vera_lambda_d.shape[0] == cfg.network_dim
        if lora_type == "tlora":
            assert layer.schedule == "linear"
        if lora_type == "flexrank":
            assert layer.min_rank == cfg.flexrank_lora_rank_range_min
        if lora_type == "fera":
            assert torch.allclose(layer.residual_gate, torch.zeros_like(layer.residual_gate))
            x = torch.randn(2, 3, 8)
            y = layer(x)
            assert y.shape == (2, 3, 8)
            state_dict = LoRAInjector(rank=cfg.network_dim, alpha=cfg.network_alpha, model_arch="anima", fera_enabled=True).inject(
                _TinyAnimaRoot(), ["self_attn.q_proj"], prefix="unet"
            )
            assert len(state_dict) == 1
        if lora_type == "hydralora":
            assert layer.config.num_experts == cfg.hydralora_num_experts
            assert layer.config.sparse_top_k is True
            x = torch.randn(2, 3, 8)
            y = layer(x)
            assert y.shape == (2, 3, 8)
            state = LoRAInjector(
                rank=cfg.network_dim,
                alpha=cfg.network_alpha,
                model_arch="anima",
                hydralora_enabled=True,
                hydralora_num_experts=cfg.hydralora_num_experts,
                hydralora_routing=cfg.hydralora_routing,
                hydralora_top_k=cfg.hydralora_top_k,
                hydralora_sparse_top_k=True,
            )
            model = _TinyAnimaRoot()
            state.inject(model, ["self_attn.q_proj"], prefix="unet")
            state_dict = state.get_lora_state_dict()
            assert any("hydralora_lora_down" in key for key in state_dict)
    print("PASS: test_lora_family_matrix")
    return True


def test_standard_lora_fast_path_forward_backward_parity() -> bool:
    layer = LoRALayer(8, 6, rank=3, alpha=3, dropout=0.0).double()
    torch.manual_seed(8100)
    with torch.no_grad():
        layer.lora_down.weight.normal_(std=0.2)
        layer.lora_up.weight.normal_(std=0.2)

    state_keys_before = tuple(layer.state_dict().keys())
    x_fast = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    x_ref = x_fast.detach().clone().requires_grad_(True)
    assert layer._can_use_fast_path(x_fast)

    out_fast = layer(x_fast)
    out_ref = layer.lora_up(layer.lora_down(x_ref)) * layer.scaling
    upstream = torch.randn_like(out_fast)
    torch.testing.assert_close(out_fast, out_ref, rtol=1e-7, atol=1e-8)

    params = tuple(layer.parameters())
    fast_grads = torch.autograd.grad((out_fast * upstream).sum(), (x_fast, *params))
    ref_grads = torch.autograd.grad((out_ref * upstream).sum(), (x_ref, *params))
    for fast_grad, ref_grad in zip(fast_grads, ref_grads):
        torch.testing.assert_close(fast_grad, ref_grad, rtol=1e-7, atol=1e-8)

    dropout_layer = LoRALayer(8, 6, rank=3, alpha=3, dropout=0.1)
    assert not dropout_layer._can_use_fast_path(torch.randn(2, 8))
    assert tuple(layer.state_dict().keys()) == state_keys_before
    print("PASS: test_standard_lora_fast_path_forward_backward_parity")
    return True


def test_lycoris_family_matrix() -> bool:
    matrix = [
        ("loha", LoHaLayer),
        ("locon", lora_mod.LoRALayer),
        ("lokr", LoKrLayer),
        ("ia3", IA3Adapter),
        ("full", FullRankAdapter),
        ("diag_oft", DiagOFTAdapter),
    ]
    for lora_type, expected_type in matrix:
        cfg = _anima_cfg(lora_type=lora_type, lokr_factor=-1)
        if lora_type == "lokr":
            cfg.lokr_no_materialize_forward = True
        assert _network_module(cfg) == "lycoris.locon"
        expected_algo = "diag-oft" if lora_type == "diag_oft" else lora_type
        assert cfg.lycoris_algo == expected_algo
        layer = _inject_lycoris_variant(cfg)
        assert isinstance(layer, expected_type), (lora_type, type(layer), expected_type)
        if lora_type == "lokr":
            assert layer.no_materialize_forward is True
    print("PASS: test_lycoris_family_matrix")
    return True


def test_extended_lycoris_forward_backward_and_roundtrip() -> bool:
    matrix = ["loha", "locon", "lokr", "ia3", "full", "diag_oft"]
    for lora_type in matrix:
        cfg = _anima_cfg(lora_type=lora_type, lokr_factor=-1)
        model = _TinyAnimaRoot()
        injector = LyCORISInjector(
            LyCORISConfig(
                lycoris_type=LyCORISType(cfg.lycoris_algo),
                rank=cfg.network_dim,
                alpha=cfg.network_alpha,
                dropout=cfg.network_dropout,
                lokr_factor=cfg.lycoris_lokr_factor,
            )
        )
        injected = injector.inject(model, ["self_attn.q_proj"], prefix="unet")
        assert len(injected) == 1
        x = torch.randn(2, 3, 8)
        y = model.net.blocks[0].self_attn.q_proj(x).sum()
        y.backward()
        grads = [param.grad for param in injector.get_trainable_parameters() if param.grad is not None]
        assert grads, f"Expected gradients for {lora_type}"
        expected = {key: value.detach().clone() for key, value in injector.get_lora_state_dict().items()}

        reloaded_model = _TinyAnimaRoot()
        reloaded = LyCORISInjector(
            LyCORISConfig(
                lycoris_type=LyCORISType(cfg.lycoris_algo),
                rank=cfg.network_dim,
                alpha=cfg.network_alpha,
                dropout=cfg.network_dropout,
                lokr_factor=cfg.lycoris_lokr_factor,
            )
        )
        reloaded.inject(reloaded_model, ["self_attn.q_proj"], prefix="unet")
        reloaded.load_lora_state_dict(expected)
        actual = reloaded.get_lora_state_dict()
        for key, expected_value in expected.items():
            if key.endswith(".alpha"):
                continue
            assert key in actual, key
            assert torch.allclose(actual[key], expected_value), (lora_type, key)
    print("PASS: test_extended_lycoris_forward_backward_and_roundtrip")
    return True


def test_lycoris_residency_params_cover_linear_and_conv() -> bool:
    class _MixedAdapterTargets(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(8, 8, bias=False)
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)

    matrix = ["loha", "locon", "lokr", "ia3", "full", "diag_oft"]
    conv_capable = {"locon", "lokr", "ia3", "diag_oft"}
    for lora_type in matrix:
        algo = "diag-oft" if lora_type == "diag_oft" else lora_type
        model = _MixedAdapterTargets()
        injector = LyCORISInjector(
            LyCORISConfig(
                lycoris_type=LyCORISType(algo),
                rank=4,
                alpha=1.0,
                conv_dim=2,
                conv_alpha=1.0,
                lokr_factor=-1,
            )
        )
        injected = injector.inject(model, ["proj", "conv"], prefix="unet")
        expected_count = 2 if lora_type in conv_capable else 1
        assert len(injected) == expected_count, (lora_type, list(injected))

        residency = injector.get_residency_params()
        trainable = injector.get_trainable_parameters()
        assert residency, f"Expected residency params for {lora_type}"
        assert {id(param) for param in trainable}.issubset({id(param) for param in residency}), lora_type
        assert len(residency) == len({id(param) for param in residency}), lora_type
    print("PASS: test_lycoris_residency_params_cover_linear_and_conv")
    return True


def test_loha_no_materialize_forward_backward_parity() -> bool:
    layer = LoHaLayer(8, 16, rank=3, alpha=3).double()
    layer.train()
    torch.manual_seed(9100)
    with torch.no_grad():
        for param in layer.parameters():
            param.copy_(torch.randn_like(param) * 0.2)
    state_keys_before = tuple(layer.state_dict().keys())

    torch.manual_seed(9200)
    x_fast = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    x_ref = x_fast.detach().clone().requires_grad_(True)
    assert layer._can_use_no_materialize_forward(x_fast)

    out_fast = layer(x_fast)
    out_ref = F.linear(x_ref, layer.get_delta_weight())
    upstream = torch.randn_like(out_fast)
    torch.testing.assert_close(out_fast, out_ref, rtol=1e-7, atol=1e-8)

    params = tuple(layer.parameters())
    fast_grads = torch.autograd.grad((out_fast * upstream).sum(), (x_fast, *params))
    ref_grads = torch.autograd.grad((out_ref * upstream).sum(), (x_ref, *params))
    for fast_grad, ref_grad in zip(fast_grads, ref_grads):
        torch.testing.assert_close(fast_grad, ref_grad, rtol=1e-7, atol=1e-8)

    assert tuple(layer.state_dict().keys()) == state_keys_before
    assert not layer._can_use_no_materialize_forward(torch.randn(2, 7, dtype=torch.float64))
    print("PASS: test_loha_no_materialize_forward_backward_parity")
    return True


def test_extended_aliases() -> bool:
    cfg = _anima_cfg(lora_type="lycoris_diag_oft")
    assert cfg.lycoris_algo == "diag-oft"
    assert _network_module(cfg) == "lycoris.locon"

    cfg = _anima_cfg(lora_type="lycoris_ia3")
    assert cfg.lycoris_algo == "ia3"
    assert _network_module(cfg) == "lycoris.locon"

    cfg = _anima_cfg(lora_type="oft")
    assert cfg.lycoris_algo == "diag-oft"
    assert _network_module(cfg) == "lycoris.locon"

    cfg = _anima_cfg(network_module="lycoris.locon", lycoris_algo="diag_oft")
    assert cfg.lycoris_algo == "diag-oft"

    cfg = _anima_cfg(network_module="lycoris.locon", lycoris_algo="dora")
    assert _network_module(cfg) == "networks.lora"
    assert cfg.dora_enabled is True
    cfg = _anima_cfg(lora_type="hydra_lora", hydra_sparse_top_k="true")
    assert cfg.hydralora_enabled is True
    assert cfg.hydralora_sparse_top_k is True
    cfg = _anima_cfg(lora_type="lokr", no_materialize_forward="true")
    assert cfg.lycoris_algo == "lokr"
    assert cfg.lokr_no_materialize_forward is True
    print("PASS: test_extended_aliases")
    return True


def test_pissa_and_weight_aliases() -> bool:
    cfg = _anima_cfg(
        lora_type="lora",
        pissa_init=True,
        pissa_niter=2,
        pissa_method="full",
        pissa_oversample=4,
        pissa_apply_conv2d="false",
        network_weights="H:/tmp/anima_adapter.safetensors",
        dit_adapter_path="H:/tmp/anima_dit_adapter.safetensors",
    )
    assert cfg.adapter_init_strategy == "pissa"
    assert cfg.pissa_enabled is True
    assert cfg.use_pissa is True
    assert cfg.pissa_init_iters == 2
    assert cfg.pissa_svd_algo == "full"
    assert cfg.pissa_oversample == 4
    assert cfg.pissa_apply_conv2d is False
    assert cfg.network_weights_path == "H:/tmp/anima_adapter.safetensors"
    assert cfg.anima_dit_adapter_path == "H:/tmp/anima_dit_adapter.safetensors"

    olora_cfg = _anima_cfg(lora_type="lora", adapter_init_strategy="o-lora", adapter_init_export_mode="standard")
    assert olora_cfg.adapter_init_strategy == "olora"
    assert olora_cfg.adapter_init_export_mode == "lora_compatible"
    loftq_cfg = _anima_cfg(lora_type="lora", adapter_init_strategy="loft-q", loftq_bits=3, loftq_quant_type="global")
    assert loftq_cfg.adapter_init_strategy == "loftq"
    assert loftq_cfg.loftq_bits == 3
    assert loftq_cfg.loftq_quant_type == "tensorwise"

    layer = _inject_lora_variant(cfg)
    assert isinstance(layer, LoRALinear)
    assert layer.pissa_niter == 2
    assert layer.svd_algo == "full"
    assert torch.count_nonzero(layer.lora.lora_up.weight) > 0
    print("PASS: test_pissa_and_weight_aliases")
    return True


def main() -> int:
    tests = [
        test_lora_family_matrix,
        test_standard_lora_fast_path_forward_backward_parity,
        test_lycoris_family_matrix,
        test_extended_lycoris_forward_backward_and_roundtrip,
        test_lycoris_residency_params_cover_linear_and_conv,
        test_loha_no_materialize_forward_backward_parity,
        test_extended_aliases,
        test_pissa_and_weight_aliases,
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
    print("Anima Adapter Matrix Smoke Test Results")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n{passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
