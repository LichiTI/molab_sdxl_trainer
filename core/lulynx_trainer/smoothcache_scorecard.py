# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Promotion scorecard for the Lulynx SmoothCache subsystem.

SmoothCache is a **DiT inference / preview accelerator** (it speeds up the
sampling / validation-preview denoising loop, not the trainer step).  This v1
mirrors the Spectrum / T-GATE posture: an error-guided observe-only probe is
wired into the live DiT block loop and the Anima sampler (default off), while
the reuse execution layer (``run_with_smoothcache``) ships as a verified library
primitive that is **not** auto-wired into the live block loop.

Gates on: the calibrator produces a schedule, the schedule grows monotonically
with the error threshold, the execution layer is bit-identical to a plain block
call at ``alpha == 0`` (no caching), and the observe-only probe is wired.

Clean-room Lulynx module; references no external caching source.
"""

from __future__ import annotations

from typing import Any


def build_smoothcache_scorecard(
    *,
    calibration_verified: bool = False,
    schedule_monotonic: bool = False,
    reuse_parity_at_alpha0: bool = False,
    observe_probe_wired: bool = False,
    theoretical_speedup: float = 1.0,
    total_steps: int = 0,
    error_threshold: float = 0.08,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not calibration_verified:
        blockers.append("calibrator_did_not_produce_schedule")
    if not schedule_monotonic:
        blockers.append("schedule_not_monotonic_in_threshold")
    if not reuse_parity_at_alpha0:
        blockers.append("execution_not_bit_identical_at_alpha0")
    if not observe_probe_wired:
        blockers.append("observe_probe_not_wired")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "smoothcache_v0",
        "gate": "smoothcache_subsystem",
        "ok": ready,
        "subsystem_ready": ready,
        # Honest scope: observe-only probe wired; reuse execution is a library
        # primitive, not auto-wired into the live DiT block loop.
        "observe_probe_wired": bool(observe_probe_wired),
        "execution_layer_wired": False,
        "wired_into_trainer": False,
        "default_behavior_changed": False,
        # Preview/sampling accelerator, not a trainer-step speedup.
        "trainer_step_speedup": False,
        "calibration_verified": bool(calibration_verified),
        "schedule_monotonic": bool(schedule_monotonic),
        "reuse_parity_at_alpha0": bool(reuse_parity_at_alpha0),
        "theoretical_step_speedup": float(theoretical_speedup),
        "total_steps": int(total_steps),
        "error_threshold": float(error_threshold),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "subsystem verified; next: add to deepcache_spectrum_cache_ab seam comparison"
            if ready
            else "resolve blockers: " + ", ".join(blockers)
        ),
        "notes": [
            "SmoothCache is inference/preview accel; do not count as a trainer speedup.",
            "Error-guided per-block schedule from relative-L1 inter-step change (calibration pass).",
            "Observe-only probe wired into live DiT/sampler; reuse execution layer is opt-in only.",
            "At alpha==0 / no schedule the probe never reuses, so output is bit-identical.",
        ],
    }


__all__ = ["build_smoothcache_scorecard"]
