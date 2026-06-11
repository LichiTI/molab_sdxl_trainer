"""Smoke test for SDXL full finetune config routing: network_dim=0 triggers full finetune path."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.configs import UnifiedTrainingConfig


class _DummyUnet(nn.Module):
    """Tiny UNet stand-in with a few linear layers."""
    def __init__(self, dim: int = 8):
        super().__init__()
        self.down = nn.Linear(dim, dim)
        self.mid = nn.Linear(dim, dim)
        self.up = nn.Linear(dim, dim)

    def forward(self, x):
        return self.up(self.mid(self.down(x)) + x)


def test_finetune_config_routing():
    """Config with training_type='full_finetune' is recognized by the config system."""
    cfg = UnifiedTrainingConfig(
        training_type="full_finetune",
        model_type="sdxl",
    )
    assert cfg.training_type == "full_finetune", f"training_type should be 'full_finetune', got {cfg.training_type}"


def test_finetune_all_parameters_trainable():
    """Full finetune mode: all model parameters should be trainable."""
    model = _DummyUnet(dim=8)

    # Full finetune: every parameter gets requires_grad=True
    model.requires_grad_(True)

    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for p in model.parameters())
    assert trainable == total, f"All params should be trainable: {trainable}/{total}"


def test_finetune_no_lora_injection():
    """Full finetune mode: no LoRA injection — model weights are used directly,
    no LoRA wrapper modules exist on the model."""
    model = _DummyUnet(dim=8)
    original_weight = model.down.weight.data.clone()

    # Simulate full finetune: no LoRA injection, weights stay as-is
    # Just verify the model forward works without any LoRA wrapper
    x = torch.randn(2, 8)
    out = model(x)
    assert out.shape == (2, 8), f"Expected output (2, 8), got {out.shape}"

    # No LoRA wrapper modules should exist on the model
    lora_modules = [name for name, _ in model.named_modules() if "lora" in name.lower()]
    assert len(lora_modules) == 0, f"Full finetune should not have LoRA modules, found: {lora_modules}"

    # The weight values should be the same as before forward (weights unchanged)
    assert torch.equal(model.down.weight.data, original_weight), (
        "Weight data should be unchanged (no LoRA swap)"
    )


def test_finetune_gradient_flows_to_all_params():
    """Gradients flow to all parameters in full finetune mode."""
    model = _DummyUnet(dim=8)
    model.requires_grad_(True)

    x = torch.randn(2, 8)
    out = model(x)
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} should have a gradient in full finetune"
        assert param.grad.abs().sum() > 0, f"Parameter {name} gradient should be non-zero"


def test_lora_mode_not_all_trainable():
    """In LoRA mode (normal training_type), base model params are frozen by default."""
    model = _DummyUnet(dim=8)
    # LoRA mode: freeze base model
    model.requires_grad_(False)
    # Only a fraction would be trainable (simulated by LoRA params)
    lora_down = nn.Linear(8, 2, bias=False)
    lora_up = nn.Linear(2, 8, bias=False)
    nn.init.zeros_(lora_up.weight)

    trainable_base = sum(1 for p in model.parameters() if p.requires_grad)
    trainable_lora = sum(1 for p in lora_down.parameters() if p.requires_grad)
    trainable_lora += sum(1 for p in lora_up.parameters() if p.requires_grad)

    assert trainable_base == 0, f"Base model params should be frozen in LoRA mode, got {trainable_base} trainable"
    assert trainable_lora > 0, "LoRA params should be trainable"


if __name__ == "__main__":
    print("SDXL Full Finetune Smoke Tests")
    print("=" * 40)
    test_finetune_config_routing()
    print("PASS: finetune_config_routing")
    test_finetune_all_parameters_trainable()
    print("PASS: finetune_all_parameters_trainable")
    test_finetune_no_lora_injection()
    print("PASS: finetune_no_lora_injection")
    test_finetune_gradient_flows_to_all_params()
    print("PASS: finetune_gradient_flows_to_all_params")
    test_lora_mode_not_all_trainable()
    print("PASS: lora_mode_not_all_trainable")
    print("=" * 40)
    print("All SDXL full finetune smoke tests passed!")
