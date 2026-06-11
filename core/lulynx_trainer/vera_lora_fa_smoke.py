"""Smoke tests for VeRA and LoRA-FA adapter layers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn


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
_ensure_namespace("core.lulynx", CORE_ROOT / "lulynx")
_ensure_namespace("core.lulynx_trainer", TRAINER_ROOT)
_load_module("core.safe_pickle", CORE_ROOT / "safe_pickle.py")
_load_module("core.memory_vortex_v2", CORE_ROOT / "memory_vortex_v2.py")
_load_module("core.lulynx.dora_layer", CORE_ROOT / "lulynx" / "dora_layer.py")
_load_module("core.lulynx_trainer.model_family", TRAINER_ROOT / "model_family.py")
_load_module("core.lulynx_trainer.tlora", TRAINER_ROOT / "tlora.py")
_load_module("core.lulynx_trainer.hydralora", TRAINER_ROOT / "hydralora.py")
_load_module("core.lulynx_trainer.fera", TRAINER_ROOT / "fera.py")
vera_mod = _load_module("core.lulynx_trainer.vera_layer", TRAINER_ROOT / "vera_layer.py")
lora_fa_mod = _load_module("core.lulynx_trainer.lora_fa_layer", TRAINER_ROOT / "lora_fa_layer.py")
lora_injector_mod = _load_module("core.lulynx_trainer.lora_injector", TRAINER_ROOT / "lora_injector.py")

VeRALinear = vera_mod.VeRALinear
VeRASharedBuffers = vera_mod.VeRASharedBuffers
LoRAFALinear = lora_fa_mod.LoRAFALinear
LoRAInjector = lora_injector_mod.LoRAInjector


def _make_linear(in_f=64, out_f=128):
    return nn.Linear(in_f, out_f, bias=False)


# ── VeRA tests ─────────────────────────────────────────────

def test_vera_shared_buffers_deterministic():
    """Same PRNG key produces identical shared buffers."""
    b1 = VeRASharedBuffers(rank=4, prng_key=42)
    b2 = VeRASharedBuffers(rank=4, prng_key=42)
    b1.ensure(64, 128)
    b2.ensure(64, 128)
    assert torch.equal(b1.shared_A.data, b2.shared_A.data), "A buffers differ"
    assert torch.equal(b1.shared_B.data, b2.shared_B.data), "B buffers differ"
    print("  [PASS] vera_shared_buffers_deterministic")


def test_vera_shared_buffers_grow():
    """Buffers grow when a larger layer is registered."""
    b = VeRASharedBuffers(rank=4, prng_key=0)
    b.ensure(32, 64)
    a_small = b.shared_A.data.clone()
    b.ensure(64, 128)
    # Top-left corner preserved
    assert torch.equal(b.shared_A.data[:, :32], a_small[:, :32]), "A corner lost after grow"
    assert b._max_in == 64
    assert b._max_out == 128
    print("  [PASS] vera_shared_buffers_grow")


def test_vera_linear_forward():
    """VeRALinear forward produces correct shape and modifies output."""
    buffers = VeRASharedBuffers(rank=4, prng_key=7)
    layer = nn.Linear(64, 128, bias=False)
    vera = VeRALinear(layer, buffers, d_initial=0.1, alpha=4.0)
    x = torch.randn(2, 64)
    out = vera(x)
    assert out.shape == (2, 128), f"Wrong output shape: {out.shape}"
    # Output should differ from original (lambda_b starts at zero but lambda_d is non-zero,
    # so there is a small contribution)
    orig_out = layer(x)
    # After zero-init lambda_b, the delta is zero => output == original
    assert torch.allclose(out, orig_out, atol=1e-5), "VeRA output should match original at init (lambda_b=0)"
    print("  [PASS] vera_linear_forward")


def test_vera_trainable_params():
    """Only lambda_d and lambda_b are trainable."""
    buffers = VeRASharedBuffers(rank=4, prng_key=7)
    vera = VeRALinear(nn.Linear(64, 128, bias=False), buffers)
    trainable = [n for n, p in vera.named_parameters() if p.requires_grad]
    assert set(trainable) == {"vera_lambda_d", "vera_lambda_b"}, f"Wrong trainable: {trainable}"
    print("  [PASS] vera_trainable_params")


def test_vera_export():
    """Export produces standard LoRA down/up weights."""
    buffers = VeRASharedBuffers(rank=4, prng_key=7)
    vera = VeRALinear(nn.Linear(64, 128, bias=False), buffers)
    weights = vera.export_standard_lora_weights()
    assert "lora_down.weight" in weights
    assert "lora_up.weight" in weights
    assert weights["lora_down.weight"].shape == (4, 64)
    assert weights["lora_up.weight"].shape == (128, 4)
    print("  [PASS] vera_export")


def test_vera_injector():
    """LoRAInjector with vera_enabled creates VeRALinear layers."""
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
    injector = LoRAInjector(
        rank=4, alpha=1.0, vera_enabled=True, vera_d_initial=0.05, vera_prng_key=99,
    )
    injected = injector.inject(model, ["0", "2"])
    assert len(injected) == 2, f"Expected 2 VeRA layers, got {len(injected)}"
    from core.lulynx_trainer.vera_layer import VeRALinear
    for layer in injector.injected_layers.values():
        assert isinstance(layer, VeRALinear), f"Expected VeRALinear, got {type(layer)}"
    # Shared buffers should be created
    assert injector._vera_buffers is not None
    assert injector._vera_buffers._A is not None
    print("  [PASS] vera_injector")


def test_vera_residency_params():
    """CPU residency should include shared VeRA matrices once."""
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
    injector = LoRAInjector(
        rank=4, alpha=1.0, vera_enabled=True, vera_d_initial=0.05, vera_prng_key=99,
    )
    injector.inject(model, ["0", "2"])
    params = injector.get_residency_params()
    unique_ids = {id(param) for param in params}
    assert len(unique_ids) == 6, f"Expected 6 unique residency params, got {len(unique_ids)}"
    assert id(injector._vera_buffers.shared_A) in unique_ids
    assert id(injector._vera_buffers.shared_B) in unique_ids
    print("  [PASS] vera_residency_params")


# ── LoRA-FA tests ──────────────────────────────────────────

def test_lora_fa_frozen_A():
    """lora_down (A) is frozen, lora_up (B) is trainable."""
    layer = nn.Linear(64, 128, bias=False)
    fa = LoRAFALinear(layer, rank=4, alpha=1.0)
    assert not fa.lora_down.weight.requires_grad, "A should be frozen"
    assert fa.lora_up.weight.requires_grad, "B should be trainable"
    print("  [PASS] lora_fa_frozen_A")


def test_lora_fa_forward():
    """LoRA-FA forward produces correct shape."""
    fa = LoRAFALinear(nn.Linear(64, 128, bias=False), rank=4, alpha=4.0)
    x = torch.randn(2, 64)
    out = fa(x)
    assert out.shape == (2, 128)
    # At init B=0, so output should match original
    orig = fa.original(x)
    assert torch.allclose(out, orig, atol=1e-5), "LoRA-FA output should match original at init"
    print("  [PASS] lora_fa_forward")


def test_lora_fa_trainable_params():
    """get_trainable_params returns only lora_up parameters."""
    fa = LoRAFALinear(nn.Linear(64, 128, bias=False), rank=4)
    params = fa.get_trainable_params()
    assert len(params) == 1, f"Expected 1 trainable param tensor, got {len(params)}"
    assert params[0] is fa.lora_up.weight
    print("  [PASS] lora_fa_trainable_params")


def test_lora_fa_injector():
    """LoRAInjector with lora_fa_enabled creates LoRAFALinear layers."""
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
    injector = LoRAInjector(rank=4, alpha=1.0, lora_fa_enabled=True)
    injected = injector.inject(model, ["0", "2"])
    assert len(injected) == 2, f"Expected 2 LoRA-FA layers, got {len(injected)}"
    from core.lulynx_trainer.lora_fa_layer import LoRAFALinear
    for layer in injector.injected_layers.values():
        assert isinstance(layer, LoRAFALinear), f"Expected LoRAFALinear, got {type(layer)}"
    print("  [PASS] lora_fa_injector")


def test_lora_fa_residency_params():
    """CPU residency should include frozen LoRA-FA down projections."""
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
    injector = LoRAInjector(rank=4, alpha=1.0, lora_fa_enabled=True)
    injector.inject(model, ["0", "2"])
    params = injector.get_residency_params()
    unique_ids = {id(param) for param in params}
    assert len(unique_ids) == 4, f"Expected 4 unique residency params, got {len(unique_ids)}"
    downs = [layer.lora_down.weight for layer in injector.injected_layers.values()]
    for down in downs:
        assert id(down) in unique_ids, "Frozen LoRA-FA down projection missing from residency params"
    print("  [PASS] lora_fa_residency_params")


def test_lora_fa_state_dict():
    """Injector state dict includes standard LoRA keys."""
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
    injector = LoRAInjector(rank=4, alpha=1.0, lora_fa_enabled=True)
    injector.inject(model, ["0", "2"])
    sd = injector.get_lora_state_dict()
    down_keys = [k for k in sd if "lora_down" in k]
    up_keys = [k for k in sd if "lora_up" in k]
    assert len(down_keys) == 2, f"Expected 2 down keys, got {len(down_keys)}"
    assert len(up_keys) == 2, f"Expected 2 up keys, got {len(up_keys)}"
    print("  [PASS] lora_fa_state_dict")


if __name__ == "__main__":
    print("VeRA / LoRA-FA Smoke Tests")
    print("=" * 40)
    test_vera_shared_buffers_deterministic()
    test_vera_shared_buffers_grow()
    test_vera_linear_forward()
    test_vera_trainable_params()
    test_vera_export()
    test_vera_injector()
    test_vera_residency_params()
    test_lora_fa_frozen_A()
    test_lora_fa_forward()
    test_lora_fa_trainable_params()
    test_lora_fa_injector()
    test_lora_fa_residency_params()
    test_lora_fa_state_dict()
    print("=" * 40)
    print("All VeRA / LoRA-FA smoke tests passed!")
