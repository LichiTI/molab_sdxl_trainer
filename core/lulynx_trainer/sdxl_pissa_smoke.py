# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test SDXL PiSSA init path: route -> config -> injector consumption."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.config import LulynxConfig
from core.lulynx_trainer.lora_injector import LoRAInjector


class _TinyAttention(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])


class _TinyTransformerBlock(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.attentions = nn.ModuleList([_TinyAttention(dim)])
        self.ff = nn.Module()
        self.ff.net = nn.ModuleList([nn.Module(), nn.Module(), nn.Linear(dim, dim)])
        self.ff.net[0].proj = nn.Linear(dim, dim)


class _TinyUnet(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.down_blocks = nn.ModuleList([_TinyTransformerBlock(dim)])
        self.mid_block = _TinyTransformerBlock(dim)
        self.up_blocks = nn.ModuleList([_TinyTransformerBlock(dim)])


class _SingleLinear(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_q(x)


class _SingleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_in(x)


def main() -> int:
    # 1. Route alias preservation: pissa_init -> pissa_enabled, pissa_niter -> pissa_init_iters
    parsed = ConfigAdapter.from_frontend_dict({
        "schema_id": "sdxl-lora",
        "pissa_init": True,
        "pissa_method": "svd",
        "pissa_niter": 3,
        "pissa_oversample": 5,
        "pissa_apply_conv2d": "false",
        "pissa_export_mode": "LoRA无损兼容导出",
    })
    assert parsed.adapter_init_strategy == "pissa", f"Expected adapter_init_strategy=pissa, got {parsed.adapter_init_strategy}"
    assert parsed.pissa_enabled is True, f"Expected pissa_enabled=True, got {parsed.pissa_enabled}"
    assert parsed.use_pissa is True, f"Expected use_pissa=True, got {parsed.use_pissa}"
    assert parsed.pissa_init_iters == 3, f"Expected pissa_init_iters=3, got {parsed.pissa_init_iters}"
    assert parsed.pissa_svd_algo == "full", f"Expected pissa_svd_algo=full, got {parsed.pissa_svd_algo}"
    assert parsed.pissa_oversample == 5, f"Expected pissa_oversample=5, got {parsed.pissa_oversample}"
    assert parsed.pissa_apply_conv2d is False, f"Expected pissa_apply_conv2d=False, got {parsed.pissa_apply_conv2d}"
    assert parsed.pissa_export_mode == "lora_compatible", f"Expected compatible export mode, got {parsed.pissa_export_mode}"

    # 2. Injector marks layers for PiSSA init when enabled
    unet = _TinyUnet()
    injector = LoRAInjector(rank=2, alpha=2, pissa_enabled=True, pissa_niter=3, model_arch="sdxl")
    injected = injector.inject_unet(unet)
    assert injected, "Expected LoRA layers to be injected"

    # Verify all injected layers were marked for PiSSA
    for name, layer in injector.injected_layers.items():
        original = layer.original
        assert getattr(original, "_pissa_init", False), f"Layer {name} not marked for PiSSA init"

    # 3. Verify PiSSA SVD initialization runs without error
    # LoRALinear.__init__ applies PiSSA when _pissa_init is set
    for name, layer in injector.injected_layers.items():
        lora = layer.lora
        # After PiSSA init, lora_down and lora_up weights should be non-zero
        # (PiSSA uses SVD decomposition of the original weight)
        down_norm = float(lora.lora_down.weight.data.norm())
        up_norm = float(lora.lora_up.weight.data.norm())
        # Both should be non-trivial after SVD init
        assert down_norm > 0, f"Layer {name} lora_down weight is zero after PiSSA init"
        assert up_norm > 0, f"Layer {name} lora_up weight is zero after PiSSA init"

    # 4. Verify injector WITHOUT pissa_enabled does NOT mark layers
    unet2 = _TinyUnet()
    injector2 = LoRAInjector(rank=2, alpha=2, pissa_enabled=False, model_arch="sdxl")
    injector2.inject_unet(unet2)
    for name, layer in injector2.injected_layers.items():
        original = layer.original
        assert not getattr(original, "_pissa_init", False), f"Layer {name} incorrectly marked for PiSSA"

    # 5. OLoRA uses the same request field and preserves the initial forward pass.
    torch.manual_seed(42)
    base = _SingleLinear()
    reference = _SingleLinear()
    reference.load_state_dict(base.state_dict())
    x = torch.randn(3, 4, 8)
    expected = reference(x)
    olora_parsed = ConfigAdapter.from_frontend_dict({
        "schema_id": "sdxl-lora",
        "adapter_init_strategy": "olora",
    })
    assert olora_parsed.adapter_init_strategy == "olora", olora_parsed.adapter_init_strategy
    olora_injector = LoRAInjector(rank=2, alpha=2, adapter_init_strategy="olora", model_arch="sdxl")
    olora_injector.inject(base, ["to_q"], prefix="unet")
    actual = base(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    olora_layer = next(iter(olora_injector.injected_layers.values()))
    assert float(olora_layer.lora.lora_down.weight.detach().norm()) > 0, "OLoRA lora_down should be non-zero"
    assert float(olora_layer.lora.lora_up.weight.detach().norm()) > 0, "OLoRA lora_up should be non-zero"
    raw_state = olora_injector.get_lora_state_dict()
    export_state = olora_injector.export_adapter_init_state_dict(raw_state, "lora_compatible")
    raw_down = raw_state["unet_to_q.lora_down.weight"]
    export_down = export_state["unet_to_q.lora_down.weight"]
    assert export_down.shape[0] == raw_down.shape[0] * 2, (export_down.shape, raw_down.shape)

    # 6. LoftQ fake-quant residual init is reachable from the same strategy field.
    torch.manual_seed(99)
    loftq_base = _SingleLinear()
    loftq_reference = _SingleLinear()
    loftq_reference.load_state_dict(loftq_base.state_dict())
    loftq_parsed = ConfigAdapter.from_frontend_dict({
        "schema_id": "sdxl-lora",
        "adapter_init_strategy": "loft-q",
        "loftq_bits": 4,
        "loftq_quant_type": "per_channel",
    })
    assert loftq_parsed.adapter_init_strategy == "loftq", loftq_parsed.adapter_init_strategy
    assert loftq_parsed.loftq_bits == 4, loftq_parsed.loftq_bits
    assert loftq_parsed.loftq_quant_type == "rowwise", loftq_parsed.loftq_quant_type
    W = loftq_reference.to_q.weight.detach().float()
    levels = float((1 << (4 - 1)) - 1)
    scale = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / levels
    quantized = torch.round(W / scale).clamp(-levels, levels) * scale
    raw_quant_error = float((W - quantized).norm())
    loftq_injector = LoRAInjector(
        rank=2,
        alpha=2,
        adapter_init_strategy="loftq",
        loftq_bits=4,
        loftq_quant_type="rowwise",
        model_arch="sdxl",
    )
    loftq_injector.inject(loftq_base, ["to_q"], prefix="unet")
    loftq_layer = next(iter(loftq_injector.injected_layers.values()))
    approx = loftq_layer.original.weight.detach().float() + loftq_layer.lora.get_weight_matrix().detach().float()
    loftq_error = float((W - approx).norm())
    assert raw_quant_error > 0, "Expected fake quantization to introduce error"
    assert loftq_error < raw_quant_error, (loftq_error, raw_quant_error)
    assert loftq_layer.applied_adapter_init_strategy == "loftq"
    loftq_export = loftq_injector.export_adapter_init_state_dict(loftq_injector.get_lora_state_dict(), "lora_compatible")
    assert loftq_export["unet_to_q.lora_down.weight"].shape[0] == raw_down.shape[0] * 2

    # 7. PiSSA Conv2d init is real runtime behavior when pissa_apply_conv2d is requested.
    torch.manual_seed(123)
    conv_base = _SingleConv()
    conv_reference = _SingleConv()
    conv_reference.load_state_dict(conv_base.state_dict())
    conv_x = torch.randn(2, 3, 8, 8)
    conv_expected = conv_reference(conv_x)
    conv_injector = LoRAInjector(
        rank=2,
        alpha=2,
        pissa_enabled=True,
        pissa_apply_conv2d=True,
        svd_algo="full",
        model_arch="sdxl",
    )
    conv_injected = conv_injector.inject(conv_base, ["proj_in"], prefix="unet")
    assert len(conv_injected) == 1, f"Expected one Conv2d PiSSA injection, got {list(conv_injected)}"
    conv_actual = conv_base(conv_x)
    torch.testing.assert_close(conv_actual, conv_expected, rtol=1e-5, atol=1e-6)
    conv_layer = next(iter(conv_injector.injected_layers.values()))
    assert conv_layer.applied_adapter_init_strategy == "pissa"
    assert conv_layer.lora.lora_down.weight.ndim == 4
    assert conv_layer.lora.lora_up.weight.shape[2:] == (1, 1)
    assert float(conv_layer.lora.lora_down.weight.detach().norm()) > 0
    assert float(conv_layer.lora.lora_up.weight.detach().norm()) > 0
    conv_state = conv_injector.get_lora_state_dict()
    conv_export = conv_injector.export_adapter_init_state_dict(conv_state, "lora_compatible")
    assert conv_export["unet_proj_in.lora_down.weight"].shape[0] == conv_state["unet_proj_in.lora_down.weight"].shape[0] * 2
    assert conv_export["unet_proj_in.lora_up.weight"].shape[1] == conv_state["unet_proj_in.lora_up.weight"].shape[1] * 2

    print("SDXL adapter init smoke passed: PiSSA Linear/Conv2d, OLoRA, and LoftQ init/export paths work")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
