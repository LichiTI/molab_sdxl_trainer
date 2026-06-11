"""Smoke tests for native LoRA activation recompute."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


_HERE = Path(__file__).resolve().parent
_BACKEND_CORE = _HERE.parent
_BACKEND = _BACKEND_CORE.parent
_REPO = _BACKEND.parent
for _path in (_REPO, _BACKEND, _BACKEND_CORE):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from core.lulynx_trainer.lora_injector import LoRALayer, LoRAInjector, LoRALinear  # noqa: E402


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-6) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=1e-5):
        diff = (actual - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch: max_diff={diff}")


def test_lora_branch_forward_backward_parity() -> None:
    torch.manual_seed(11)
    plain = LoRALayer(7, 5, rank=3, alpha=2.0, activation_recompute=False).double()
    recompute = LoRALayer(7, 5, rank=3, alpha=2.0, activation_recompute=True).double()
    recompute.load_state_dict(plain.state_dict())

    x_plain = torch.randn(2, 4, 7, dtype=torch.float64, requires_grad=True)
    x_recompute = x_plain.detach().clone().requires_grad_(True)
    grad = torch.randn(2, 4, 5, dtype=torch.float64)

    out_plain = plain(x_plain)
    out_recompute = recompute(x_recompute)
    _assert_close("forward", out_recompute, out_plain)

    out_plain.backward(grad)
    out_recompute.backward(grad)
    _assert_close("input grad", x_recompute.grad, x_plain.grad)
    _assert_close("down grad", recompute.lora_down.weight.grad, plain.lora_down.weight.grad)
    _assert_close("up grad", recompute.lora_up.weight.grad, plain.lora_up.weight.grad)


def test_injector_wires_recompute_to_standard_lora_only() -> None:
    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    injector = LoRAInjector(
        rank=2,
        alpha=2,
        target_modules=["0"],
        activation_recompute=True,
    )
    injected = injector.inject(model, ["0"], prefix="net")
    layer = next(iter(injected.values()))
    if not isinstance(layer, LoRALinear):
        raise AssertionError(f"expected LoRALinear, got {type(layer)!r}")
    if not bool(getattr(layer.lora, "activation_recompute", False)):
        raise AssertionError("activation_recompute was not propagated to LoRALayer")


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    test_lora_branch_forward_backward_parity()
    print("PASS: lora branch recompute forward/backward parity")
    test_injector_wires_recompute_to_standard_lora_only()
    print("PASS: injector wires activation recompute")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
