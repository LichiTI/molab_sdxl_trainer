"""Smoke tests for shared BHND attention kernel adapters."""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import torch

from core.lulynx_trainer.attention_kernel_adapters import (
    forward_only_attention_bhnd,
    sdpa_attention_bhnd,
    torch_attention_bhnd,
)


def test_sdpa_and_torch_shapes() -> None:
    q = torch.randn(2, 3, 8, 16)
    k = torch.randn(2, 3, 8, 16)
    v = torch.randn(2, 3, 8, 16)
    out_sdpa = sdpa_attention_bhnd(q, k, v)
    out_torch = torch_attention_bhnd(q, k, v)
    assert out_sdpa.shape == q.shape
    assert out_torch.shape == q.shape
    assert torch.isfinite(out_sdpa).all()
    assert torch.isfinite(out_torch).all()
    print("PASS: sdpa/torch BHND adapters return finite matching shapes")


def test_forward_only_recompute_backward() -> None:
    q = torch.randn(1, 2, 8, 16, requires_grad=True)
    k = torch.randn(1, 2, 8, 16, requires_grad=True)
    v = torch.randn(1, 2, 8, 16, requires_grad=True)

    def forward_fn(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return sdpa_attention_bhnd(q_in, k_in, v_in)

    out = forward_only_attention_bhnd(q, k, v, forward_fn=forward_fn)
    assert out.requires_grad
    loss = out.square().mean()
    loss.backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    print("PASS: forward-only attention shim recomputes finite q/k/v gradients")


if __name__ == "__main__":
    test_sdpa_and_torch_shapes()
    test_forward_only_recompute_backward()
    print("\nAll shared attention kernel adapter smoke tests passed!")
