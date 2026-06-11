# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Promotion scorecard for FP8 base GEMM compute (Ada tensor cores).

Opt-in via ``fp8_base_compute`` on top of fp8 storage; the default (non-fp8)
path is bit-identical to legacy.  This is the forward-safe enablement for the
otherwise storage-only native fp8 base path.  Promotion gates on: ``_scaled_mm``
available, the FP8 GEMM matches the bf16 reference within e4m3 tolerance, the
unsupported-shape fallback is exact, and the LoRALinear integration composes the
base + LoRA paths correctly.

Clean-room Lulynx module; shares no source with any reference implementation.
"""

from __future__ import annotations

from typing import Any

# e4m3 has ~2 mantissa bits; per-tensor activation + weight quant gives a few-%
# output error after the GEMM averaging — the honest correctness bar.
FP8_REL_TOLERANCE = 0.10


def build_fp8_base_compute_scorecard(
    *,
    scaled_mm_available: bool = False,
    numerical_verified: bool = False,
    fp8_rel_error: float | None = None,
    fallback_verified: bool = False,
    lora_integration_verified: bool = False,
    gpu_name: str | None = None,
    is_ada_or_newer: bool | None = None,
    rel_tolerance: float = FP8_REL_TOLERANCE,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not scaled_mm_available:
        blockers.append("scaled_mm_unavailable")
    if not numerical_verified:
        blockers.append("fp8_gemm_outside_tolerance")
    if not fallback_verified:
        blockers.append("dequant_fallback_not_exact")
    if not lora_integration_verified:
        blockers.append("lora_integration_not_verified")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "fp8_base_compute_v0",
        "gate": "fp8_base_gemm_tensor_core",
        "ok": ready,
        "optimization_ready": ready,
        "promotion_ready": ready,
        "training_path_enabled": True,
        "trainer_wiring_allowed": True,
        "default_behavior_changed": False,
        "scaled_mm_available": bool(scaled_mm_available),
        "numerical_verified": bool(numerical_verified),
        "fp8_rel_error": (float(fp8_rel_error) if fp8_rel_error is not None else None),
        "rel_tolerance": float(rel_tolerance),
        "fallback_verified": bool(fallback_verified),
        "lora_integration_verified": bool(lora_integration_verified),
        "gpu": gpu_name,
        "is_ada_or_newer": is_ada_or_newer,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "ready: fp8_base_compute makes the native fp8 base forward-safe on FP8 tensor cores"
            if ready
            else "resolve blockers: " + ", ".join(blockers)
        ),
        "notes": [
            "Opt-in via fp8_base_compute; complements fp8 storage and the Triton LoRA split design.",
            "Frozen base GEMM runs on FP8 tensor cores; LoRA path stays bf16. Falls back to bf16 dequant if unsupported.",
            "Default (non-fp8) base forward is unchanged.",
        ],
    }


__all__ = ["build_fp8_base_compute_scorecard", "FP8_REL_TOLERANCE"]
