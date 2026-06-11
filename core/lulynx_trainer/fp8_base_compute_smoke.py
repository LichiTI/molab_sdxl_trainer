# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test for FP8 base GEMM compute (Ada tensor cores). Requires CUDA.

Run with the flashattention env on an Ada (or newer) GPU:
    backend/env/python-flashattention/python.exe \
        backend/core/lulynx_trainer/fp8_base_compute_smoke.py

Checks: (1) torch._scaled_mm is available; (2) the FP8 base GEMM matches the
bf16 reference within e4m3 tolerance and isn't far worse than the weight-only
fp8 baseline; (3) an unsupported (non-16-aligned) shape falls back to an exact
bf16 dequant matmul; (4) LoRALinear composes the fp8 base + bf16 LoRA correctly,
and the storage-only (no-compute) path still forwards.  Emits the scorecard.
"""

from __future__ import annotations

import os
import sys

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.dirname(_BACKEND)
for _p in (_BACKEND, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.lulynx_trainer.fp8_quantize import fp8_base_linear_forward, _resolve_scaled_mm
from core.lulynx_trainer.fp8_base_compute_scorecard import (
    build_fp8_base_compute_scorecard,
    FP8_REL_TOLERANCE,
)

DEV = "cuda"
_FP8 = torch.float8_e4m3fn


def _rel(a, b):
    return (a.float() - b.float()).abs().max().item() / (b.float().abs().max().item() + 1e-6)


def check_numerical() -> tuple[bool, float]:
    print("== FP8 base GEMM vs bf16 reference ==")
    ok = True
    worst = 0.0
    for (M, cin, cout) in [(256, 2048, 2048), (128, 1536, 4096), (512, 2048, 512)]:
        lin = nn.Linear(cin, cout, bias=True).to(DEV, torch.bfloat16)
        x = torch.randn(M, cin, device=DEV, dtype=torch.bfloat16)
        y_ref = F.linear(x, lin.weight, lin.bias)               # bf16 truth

        w_deq = lin.weight.data.to(_FP8).to(torch.bfloat16)     # weight-only fp8 error
        y_deq = F.linear(x, w_deq, lin.bias)
        base_err = _rel(y_deq, y_ref)

        lin.weight.data = lin.weight.data.to(_FP8)              # store fp8
        lin._fp8_base_compute = True
        y_fp8 = fp8_base_linear_forward(lin, x)
        fp8_err = _rel(y_fp8, y_ref)
        worst = max(worst, fp8_err)

        passed = fp8_err < FP8_REL_TOLERANCE and fp8_err <= base_err * 3 + 0.03
        ok &= passed
        print(f"  ({M},{cin},{cout}): fp8_err={fp8_err:.3f} weight_only={base_err:.3f} {'OK' if passed else 'FAIL'}")
    return ok, worst


def check_fallback() -> bool:
    print("== unsupported shape → exact bf16 dequant fallback ==")
    # in_features not a multiple of 16 → scaled_mm path skipped
    lin = nn.Linear(2050, 256, bias=True).to(DEV, torch.bfloat16)
    x = torch.randn(8, 2050, device=DEV, dtype=torch.bfloat16)
    lin.weight.data = lin.weight.data.to(_FP8)
    lin._fp8_base_compute = True
    out = fp8_base_linear_forward(lin, x)
    expected = F.linear(x, lin.weight.to(x.dtype), lin.bias.to(x.dtype))
    ok = torch.equal(out, expected)
    print(f"  fallback exact match={ok}  {'OK' if ok else 'FAIL'}")
    return ok


def check_lora_integration() -> bool:
    print("== LoRALinear fp8 base + bf16 LoRA ==")
    from core.lulynx_trainer.lora_injector import LoRALinear

    cin, cout, rank = 2048, 2048, 32
    torch.manual_seed(0)
    orig = nn.Linear(cin, cout, bias=True).to(DEV, torch.bfloat16)
    layer = LoRALinear(orig, rank=rank, alpha=2 * rank, dropout=0.0).to(DEV, torch.bfloat16)
    with torch.no_grad():
        layer.lora.lora_down.weight.normal_(0.0, cin ** -0.5)
        layer.lora.lora_up.weight.normal_(0.0, rank ** -0.5)

    x = torch.randn(64, cin, device=DEV, dtype=torch.bfloat16)
    y_ref = layer(x)  # bf16 base + lora

    # fp8 storage-only (no compute): dequant base path, must still run + be close
    layer.original.weight.data = layer.original.weight.data.to(_FP8)
    y_storage = layer(x)
    storage_err = _rel(y_storage, y_ref)

    # fp8 compute: tensor-core base GEMM + same lora
    layer.original._fp8_base_compute = True
    y_compute = layer(x)
    compute_err = _rel(y_compute, y_ref)

    ok = storage_err < FP8_REL_TOLERANCE and compute_err < FP8_REL_TOLERANCE
    print(f"  storage(dequant) err={storage_err:.3f}  compute(scaled_mm) err={compute_err:.3f}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    is_ada = cap >= (8, 9)
    print(f"GPU: {name}  sm_{cap[0]}{cap[1]}  ada_or_newer={is_ada}")

    scaled_mm_ok = _resolve_scaled_mm() is not None
    print(f"scaled_mm available: {scaled_mm_ok}")
    num_ok, worst = check_numerical()
    fb_ok = check_fallback()
    lora_ok = check_lora_integration()

    scorecard = build_fp8_base_compute_scorecard(
        scaled_mm_available=scaled_mm_ok,
        numerical_verified=num_ok,
        fp8_rel_error=worst,
        fallback_verified=fb_ok,
        lora_integration_verified=lora_ok,
        gpu_name=name,
        is_ada_or_newer=is_ada,
    )
    print("\n== scorecard ==")
    for k, v in scorecard.items():
        print(f"  {k}: {v}")

    all_ok = scaled_mm_ok and num_ok and fb_ok and lora_ok
    print("\nRESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT", f"| scorecard.ok={scorecard['ok']}")
    sys.exit(0 if all_ok else 1)
