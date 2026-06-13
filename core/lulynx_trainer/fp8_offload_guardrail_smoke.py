"""CPU/logic smoke for the fp8-base robustness收口 (plan: 让"勾选 fp8"成为稳健默认路径).

Covers two things that are unit-testable without a real training run:

  (a) ``fp8_quantize._fp8_compute_supported`` rejects a cross-device (weight on
      CPU, activation on cuda) pair so ``fp8_base_linear_forward`` falls back to
      the device-safe bf16 dequant instead of crashing in ``_scaled_mm``.
      (Only exercised when CUDA is present; skipped otherwise.)

  (b) ``vram_guardrails.apply_low_vram_guardrails`` suppresses the auto
      CPU-offload residency switch when ``fp8_base`` is enabled (it would
      duplicate/conflict with fp8), while still switching it when fp8 is off.

Run:
  backend/env/python-flashattention/python.exe \
    backend/core/lulynx_trainer/fp8_offload_guardrail_smoke.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.dirname(_BACKEND)
for _p in (_BACKEND, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from core.lulynx_trainer.fp8_quantize import _fp8_compute_supported, fp8_base_linear_forward
from core.lulynx_trainer.vram_guardrails import apply_low_vram_guardrails

_FP8 = getattr(torch, "float8_e4m3fn", None)


def _make_dit_runtime() -> dict:
    return {
        "available": True,
        "residency_key": "anima_block_residency",
        "checkpoint_key": "anima_block_checkpointing",
        "prefetch_key": "anima_block_prefetch",
        "prefetch_depth_key": "anima_block_prefetch_depth",
        "mode": "resident",
        "recommendation": "streaming_offload",
        "risk": True,
    }


def _vram_report() -> dict:
    return {
        "family": "anima",
        "safety": "tight",  # triggers the guardrail
        "available_gb": 16.0,
        "estimated_gb": 15.0,
        "usage_ratio": 0.95,
        "dit_runtime": _make_dit_runtime(),
        "recommendations": [],
        "action_plan": {"steps": []},
    }


def _base_config() -> SimpleNamespace:
    # SimpleNamespace (not dict) because apply_low_vram_guardrails uses setattr.
    return SimpleNamespace(
        model_arch="anima",
        anima_block_residency="resident",
        vram_smart_sensing_streaming_enabled=True,
        vram_smart_sensing_sparse_swap_enabled=True,
        vram_smart_sensing_delta_cache_enabled=False,
        enhanced_protection_mode=False,
    )


def check_guardrail_suppression() -> None:
    # fp8 OFF -> auto-offload switch happens
    cfg_off = _base_config()
    cfg_off.fp8_base = False
    cfg_off.fp8_base_compute = False
    res_off = apply_low_vram_guardrails(
        cfg_off, vram_report=_vram_report(), smart_enabled=True, auto_enabled=True
    )
    assert res_off.get("triggered"), "expected guardrail to trigger on safety=tight"
    assert getattr(cfg_off, "anima_block_residency") == "streaming_offload", (
        f"fp8-off: expected residency switched to streaming_offload, "
        f"got {cfg_off.anima_block_residency!r}"
    )
    assert res_off["changes"].get("anima_block_residency") == "streaming_offload"
    print("  [b1] fp8 OFF -> residency auto-switched to streaming_offload  OK")

    # fp8 ON -> auto-offload switch suppressed, residency stays resident
    cfg_on = _base_config()
    cfg_on.fp8_base = True
    cfg_on.fp8_base_compute = True
    res_on = apply_low_vram_guardrails(
        cfg_on, vram_report=_vram_report(), smart_enabled=True, auto_enabled=True
    )
    assert getattr(cfg_on, "anima_block_residency") == "resident", (
        f"fp8-on: expected residency to stay resident, got {cfg_on.anima_block_residency!r}"
    )
    assert "anima_block_residency" not in res_on["changes"], (
        "fp8-on: residency must not be auto-changed"
    )
    reasons = {s.get("reason") for s in res_on.get("skipped", [])}
    assert "fp8_base_resident_preferred" in reasons, (
        f"fp8-on: expected fp8_base_resident_preferred in skipped reasons, got {reasons}"
    )
    # sparse_swap / prefetch must not have been turned on either (gated on the
    # residency switch we suppressed).
    assert "sparse_swap_enabled" not in res_on["changes"]
    assert "anima_block_prefetch" not in res_on["changes"]
    print("  [b2] fp8 ON  -> offload suppressed (fp8_base_resident_preferred)  OK")

    # fp8_base_compute alone (without fp8_base) also suppresses.
    cfg_compute = _base_config()
    cfg_compute.fp8_base = False
    cfg_compute.fp8_base_compute = True
    res_compute = apply_low_vram_guardrails(
        cfg_compute, vram_report=_vram_report(), smart_enabled=True, auto_enabled=True
    )
    assert getattr(cfg_compute, "anima_block_residency") == "resident"
    print("  [b3] fp8_base_compute alone -> offload suppressed  OK")


def check_cross_device_fallback() -> None:
    if _FP8 is None:
        print("  [a] skipped: torch has no float8_e4m3fn")
        return
    if not torch.cuda.is_available():
        print("  [a] skipped: CUDA not available (cross-device branch needs cuda)")
        return

    linear = torch.nn.Linear(32, 16, bias=True)
    # frozen base weight cast to fp8 and left on CPU (simulates block offload)
    linear.weight.data = linear.weight.data.to(_FP8)
    assert linear.weight.device.type == "cpu"
    x = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)

    # The cross-device pair must be rejected by the tensor-core gate ...
    assert _fp8_compute_supported(linear, x) is False, (
        "cross-device (cpu weight, cuda x) must not be reported _scaled_mm-supported"
    )
    # ... and the forward must run via the device-safe dequant fallback.
    out = fp8_base_linear_forward(linear, x)
    assert out.shape == (4, 16), f"unexpected output shape {tuple(out.shape)}"
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all(), "fallback produced non-finite output"
    print("  [a] cross-device fp8 weight -> device-safe fallback, no crash  OK")


def main() -> int:
    print("== fp8 offload-guardrail smoke ==")
    check_cross_device_fallback()
    check_guardrail_suppression()
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
