# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test SDXL-specific memory runtime features."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.lora_injector import LoRAInjector
from core.lulynx_trainer.memory_optimizations import AdapterCPUResidency, cpu_offload_checkpoint


class _TinySDXLAttentionBlock(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_out[0](self.to_q(x) + self.to_k(x) + self.to_v(x))


class _TinySDXLUNet(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.attn1 = _TinySDXLAttentionBlock(dim)

    def forward(self, sample: torch.Tensor, **_kwargs):
        return SimpleNamespace(sample=self.attn1(sample))


def test_route_config_aliases() -> None:
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "sdxl-lora",
            "vram_swap_to_ram": "yes",
            "cpu_offload_checkpointing": "on",
        }
    )
    assert cfg.vram_swap_to_ram is True
    assert cfg.cpu_offload_checkpointing is True


def test_sdxl_lora_adapter_cpu_residency_step_context() -> None:
    torch.manual_seed(101)
    unet = _TinySDXLUNet()
    injector = LoRAInjector(rank=2, alpha=2.0, target_modules=["to_q", "to_k", "to_v"], model_arch="sdxl")
    injected = injector.inject_unet(unet)
    assert len(injected) == 3, f"expected three SDXL attention projections injected, got {len(injected)}"

    trainable_params = injector.get_trainable_params()
    residency_params = trainable_params + injector.get_residency_params()
    assert trainable_params, "SDXL LoRA injector should expose trainable adapter params"

    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    residency = AdapterCPUResidency(device=device)
    registered = residency.register_parameters(residency_params)
    assert registered >= len(trainable_params)

    residency.to_cpu()
    assert all(param.device.type == "cpu" for param in trainable_params)

    with residency.step_context():
        assert residency.is_active
        assert all(param.device.type == device.type for param in trainable_params)
        sample = torch.randn(2, 4, 8, device=device)
        unet.to(device)
        out = unet(sample).sample
        assert out.shape == sample.shape

    assert not residency.is_active
    assert all(param.device.type == "cpu" for param in trainable_params)


def test_sdxl_cpu_offload_checkpoint_unet_forward_backward() -> None:
    torch.manual_seed(202)
    unet = _TinySDXLUNet()
    sample = torch.randn(2, 4, 8, requires_grad=True)

    def _forward(**kwargs):
        return unet(**kwargs)

    out = cpu_offload_checkpoint(_forward, sample=sample).sample
    assert out.shape == sample.shape
    loss = out.square().mean()
    loss.backward()
    assert sample.grad is not None
    assert torch.isfinite(sample.grad).all()
    first_weight = unet.attn1.to_q.weight
    assert first_weight.grad is not None
    assert torch.isfinite(first_weight.grad).all()


def main() -> int:
    test_route_config_aliases()
    test_sdxl_lora_adapter_cpu_residency_step_context()
    test_sdxl_cpu_offload_checkpoint_unet_forward_backward()
    print("SDXL memory runtime smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
