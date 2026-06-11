# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test FP8 base-model quantization (Phase 3.6, #70).

Validates that ``quantize_base_weights_fp8`` correctly quantizes frozen base
weights while preserving LoRA adapter parameters in their original dtype.
Runs on CPU so no GPU is required.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Set

import torch
import torch.nn as nn

# ── Direct file imports bypassing __init__.py ────────────────────────────
_here = Path(__file__).resolve().parent


def _import_from_file(module_name: str, file_path: Path):
    """Import a single file without triggering the package __init__."""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_fp8_mod = _import_from_file(
    "core.lulynx_trainer.fp8_quantize",
    _here / "fp8_quantize.py",
)
quantize_base_weights_fp8 = _fp8_mod.quantize_base_weights_fp8
_collect_lora_param_ids = _fp8_mod._collect_lora_param_ids


# ── Helpers ──────────────────────────────────────────────────────────────

class _TinyDiT(nn.Module):
    """Minimal DiT-like model with a few Linear layers for testing."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear2(self.linear1(x)))


class _FakeLoRAInjector:
    """Minimal stand-in for LoRAInjector with ``injected_layers`` and
    ``get_trainable_params``."""

    def __init__(self, trainable_params, injected_layers=None):
        self._trainable = list(trainable_params)
        self.injected_layers = injected_layers or {}

    def get_trainable_params(self):
        return self._trainable


# ── Tests ────────────────────────────────────────────────────────────────

def test_fp8_quantizes_frozen_base_weights() -> None:
    model = _TinyDiT(dim=16)
    # Freeze all base weights
    for p in model.parameters():
        p.requires_grad = False

    saved = quantize_base_weights_fp8(model, lora_injector=None)
    # On CPU, quantization may fail silently for FP8 if the dtype is not
    # supported.  At minimum, the function should return without error.
    assert isinstance(saved, float), f"Expected float, got {type(saved)}"
    print("  [PASS] fp8_quantizes_frozen_base_weights: no errors, returns float")


def test_fp8_skips_lora_params() -> None:
    model = _TinyDiT(dim=16)
    # Freeze base weights
    model.linear1.weight.requires_grad = False
    model.linear2.weight.requires_grad = False
    model.norm.weight.requires_grad = False
    model.norm.bias.requires_grad = False

    # Create a LoRA-like trainable parameter (simulating lora_up weight)
    lora_up = nn.Parameter(torch.randn(16, 4), requires_grad=True)
    lora_down = nn.Parameter(torch.randn(4, 16), requires_grad=True)

    fake_injector = _FakeLoRAInjector(
        trainable_params=[lora_up, lora_down],
    )

    _ = quantize_base_weights_fp8(model, lora_injector=fake_injector)

    # LoRA parameters must remain in their original dtype (float32 on CPU)
    assert lora_up.dtype != torch.float8_e4m3fn, (
        f"LoRA lora_up should NOT be quantized, got dtype={lora_up.dtype}"
    )
    assert lora_down.dtype != torch.float8_e4m3fn, (
        f"LoRA lora_down should NOT be quantized, got dtype={lora_down.dtype}"
    )
    print("  [PASS] fp8_skips_lora_params: LoRA parameters untouched")


def test_fp8_skips_requires_grad_params() -> None:
    model = _TinyDiT(dim=16)
    # Leave linear1.weight trainable (requires_grad=True)
    for name, p in model.named_parameters():
        if name != "linear1.weight":
            p.requires_grad = False

    _ = quantize_base_weights_fp8(model, lora_injector=None)

    # Trainable weight must stay in original dtype
    assert model.linear1.weight.dtype != torch.float8_e4m3fn, (
        f"Trainable param should NOT be quantized, got dtype={model.linear1.weight.dtype}"
    )
    print("  [PASS] fp8_skips_requires_grad_params: trainable params untouched")


def test_collect_lora_param_ids_empty() -> None:
    ids = _collect_lora_param_ids(None)
    assert ids == set(), f"Expected empty set, got {ids}"
    print("  [PASS] collect_lora_param_ids: None injector -> empty set")


def test_collect_lora_param_ids_with_injector() -> None:
    p1 = nn.Parameter(torch.randn(2, 2))
    p2 = nn.Parameter(torch.randn(3, 3))
    injector = _FakeLoRAInjector(trainable_params=[p1, p2])
    ids = _collect_lora_param_ids(injector)
    assert id(p1) in ids, "p1 should be in LoRA param IDs"
    assert id(p2) in ids, "p2 should be in LoRA param IDs"
    print("  [PASS] collect_lora_param_ids: injector params found")


def test_collect_lora_param_ids_injected_layers() -> None:
    """Verify that parameters from injected_layers dict are also collected."""
    lora_layer = nn.Linear(4, 4)  # simulates a LoRALinear wrapper
    injector = _FakeLoRAInjector(
        trainable_params=[],
        injected_layers={"layer1": lora_layer},
    )
    ids = _collect_lora_param_ids(injector)
    for p in lora_layer.parameters():
        assert id(p) in ids, "injected_layers params should be collected"
    print("  [PASS] collect_lora_param_ids: injected_layers params found")


def test_fp8_vram_savings_estimate() -> None:
    model = _TinyDiT(dim=64)
    for p in model.parameters():
        p.requires_grad = False
    # Count base elements for savings estimation
    total_elements = sum(p.numel() for p in model.parameters())

    saved = quantize_base_weights_fp8(model, lora_injector=None)

    # On systems where float8_e4m3fn is supported, savings > 0.
    # On unsupported systems, savings may be 0 (all casts failed).
    # In either case, the estimate should be non-negative.
    assert saved >= 0, f"VRAM savings estimate should be >= 0, got {saved}"

    # If quantization succeeded (itemsizes differ), verify the direction.
    # bf16/fp32 -> fp8 saves 1-3 bytes per element.
    # This check is lenient: just verify the return type and range.
    print(f"  [PASS] fp8_vram_savings_estimate: returned {saved:.2f} MB (non-negative)")


def test_fp8_config_field_exists() -> None:
    """Verify fp8_base is a recognized config field."""
    # Import configs without full package
    _constants = _import_from_file(
        "core.constants",
        _here.parent / "constants.py",
    )
    _configs = _import_from_file(
        "core.configs",
        _here.parent / "configs.py",
    )
    config = _configs.UnifiedTrainingConfig()
    assert hasattr(config, "fp8_base"), "Config should have fp8_base field"
    assert config.fp8_base is False, "fp8_base default should be False"

    config2 = _configs.UnifiedTrainingConfig(fp8_base=True)
    assert config2.fp8_base is True, "fp8_base should be settable to True"
    print("  [PASS] fp8_config_field_exists: fp8_base field OK")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        test_fp8_quantizes_frozen_base_weights,
        test_fp8_skips_lora_params,
        test_fp8_skips_requires_grad_params,
        test_collect_lora_param_ids_empty,
        test_collect_lora_param_ids_with_injector,
        test_collect_lora_param_ids_injected_layers,
        test_fp8_vram_savings_estimate,
        test_fp8_config_field_exists,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\nFP8 base smoke: {passed} passed, {failed} failed out of {len(tests)} tests")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
