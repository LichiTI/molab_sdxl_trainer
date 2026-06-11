# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for DoRA optimized effective-weight construction."""

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
_ensure_namespace("core.lulynx_trainer", CORE_ROOT / "lulynx_trainer")
dora_mod = _load_module("core.lulynx.dora_layer", CORE_ROOT / "lulynx" / "dora_layer.py")

DoRALinear = dora_mod.DoRALinear


class CountingLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return super().forward(x)


def _reference_dora_weight(layer: DoRALinear) -> torch.Tensor:
    lora_weight = layer.lora_B @ layer.lora_A
    weight_eff = layer.base_weight + layer.scaling * lora_weight
    norm = torch.linalg.norm(weight_eff, dim=1, keepdim=True)
    return layer.m.unsqueeze(1) * (weight_eff / (norm + 1e-6))


def test_dora_optimized_forward_backward_parity() -> bool:
    torch.manual_seed(1234)
    base = nn.Linear(8, 6, bias=True).double()
    layer = DoRALinear(base, rank=3, alpha=3).double()
    with torch.no_grad():
        layer.lora_A.normal_(std=0.2)
        layer.lora_B.normal_(std=0.2)
        layer.m.copy_(torch.rand_like(layer.m) + 0.5)

    state_keys_before = tuple(layer.state_dict().keys())
    x_fast = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    x_ref = x_fast.detach().clone().requires_grad_(True)

    out_fast = layer(x_fast)
    out_ref = F.linear(x_ref, _reference_dora_weight(layer), layer.base_bias)
    torch.testing.assert_close(out_fast, out_ref, rtol=1e-7, atol=1e-8)

    upstream = torch.randn_like(out_fast)
    params = (layer.lora_A, layer.lora_B, layer.m)
    fast_grads = torch.autograd.grad((out_fast * upstream).sum(), (x_fast, *params))
    ref_grads = torch.autograd.grad((out_ref * upstream).sum(), (x_ref, *params))
    for fast_grad, ref_grad in zip(fast_grads, ref_grads):
        torch.testing.assert_close(fast_grad, ref_grad, rtol=1e-7, atol=1e-8)

    assert tuple(layer.state_dict().keys()) == state_keys_before
    print("PASS: test_dora_optimized_forward_backward_parity")
    return True


def test_dora_wrapper_skips_redundant_original_matmul() -> bool:
    lora_mod = _load_module(
        "core.lulynx_trainer.lora_injector",
        CORE_ROOT / "lulynx_trainer" / "lora_injector.py",
    )
    LoRALinear = lora_mod.LoRALinear

    torch.manual_seed(4321)
    base = CountingLinear(8, 6, bias=True).double()
    wrapper = LoRALinear(base, rank=3, alpha=3, use_dora=True).double()
    with torch.no_grad():
        wrapper.lora.lora_A.normal_(std=0.2)
        wrapper.lora.lora_B.normal_(std=0.2)
        wrapper.lora.m.copy_(torch.rand_like(wrapper.lora.m) + 0.5)

    x = torch.randn(2, 5, 8, dtype=torch.float64)
    out = wrapper(x)
    expected = F.linear(x, _reference_dora_weight(wrapper.lora), wrapper.lora.base_bias)
    torch.testing.assert_close(out, expected, rtol=1e-7, atol=1e-8)
    assert base.forward_calls == 0, f"DoRA wrapper redundantly called original layer {base.forward_calls} time(s)"
    print("PASS: test_dora_wrapper_skips_redundant_original_matmul")
    return True


def main() -> int:
    tests = [
        test_dora_optimized_forward_backward_parity,
        test_dora_wrapper_skips_redundant_original_matmul,
    ]
    results = []
    for test_fn in tests:
        try:
            results.append((test_fn.__name__, test_fn()))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"FAIL: {test_fn.__name__} -- {exc}")
            results.append((test_fn.__name__, False))

    passed = sum(1 for _, ok in results if ok)
    print("\n" + "=" * 60)
    print("DoRA Optimization Smoke Test Results")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n{passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
