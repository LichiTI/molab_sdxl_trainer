# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test: P4 DP-DMD/Turbo + SPD opt-in reserve seam.

Proves both opt-in surfaces are genuine default-off reserves:

* SPD: disabled == a plain per-step loop (bitwise); a unit-scale SPD config ==
  disabled (bitwise); a real multi-resolution config really changes the latent
  trajectory; the result is always at base resolution and finite.
* DP-DMD: disabled returns the base loss unchanged (bitwise, no added terms);
  enabled adds the real distillation terms and gradients flow.

Run:
    backend/env/python-flashattention/python.exe \
        backend/core/lulynx_trainer/dp_dmd_spd_reserve_smoke.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import torch

from core.lulynx_trainer.spd_inference import SpdInferenceConfig
from core.lulynx_trainer.dp_dmd_turbo import DpDmdTurboConfig
from core.lulynx_trainer.dp_dmd_spd_reserve_seam import (
    compose_dp_dmd_turbo_loss,
    dp_dmd_spd_reserve_readiness,
    run_spd_multiresolution_denoise,
)


def _denoise_fn(latents, *, level, scale, step_index):
    # Deterministic, resolution-sensitive, non-linear pseudo-denoise step.
    return latents - 0.1 * torch.tanh(latents) + 0.001 * float(step_index)


def _rel_drift(a, b):
    return float((a - b).norm() / (b.norm() + 1e-8))


def test_spd_disabled_matches_plain_loop():
    torch.manual_seed(0)
    latents = torch.randn(1, 4, 8, 8)
    cfg = SpdInferenceConfig(scale_factors=(0.5, 1.0), steps_per_level=(2, 3))  # total 5 steps
    out = run_spd_multiresolution_denoise(_denoise_fn, latents, config=cfg, enabled=False)
    manual = latents
    for step in range(5):
        manual = _denoise_fn(manual, level=0, scale=1.0, step_index=step)
    assert torch.equal(out, manual), "SPD disabled must equal a plain base-resolution loop"
    assert out.shape == latents.shape
    print("PASS: SPD disabled == plain per-step loop (bitwise), base shape preserved")


def test_spd_unit_scale_equals_disabled():
    torch.manual_seed(1)
    latents = torch.randn(1, 4, 8, 8)
    cfg = SpdInferenceConfig(scale_factors=(1.0,), steps_per_level=(5,))
    on = run_spd_multiresolution_denoise(_denoise_fn, latents, config=cfg, enabled=True)
    off = run_spd_multiresolution_denoise(_denoise_fn, latents, config=cfg, enabled=False)
    assert torch.equal(on, off), "unit-scale SPD must be bitwise-identical to disabled"
    print("PASS: SPD with all scales=1.0 is bitwise-parity with disabled")


def test_spd_multiresolution_changes_trajectory():
    torch.manual_seed(2)
    latents = torch.randn(1, 4, 8, 8)
    cfg = SpdInferenceConfig(scale_factors=(0.5, 1.0), steps_per_level=(2, 2))
    on = run_spd_multiresolution_denoise(_denoise_fn, latents, config=cfg, enabled=True)
    off = run_spd_multiresolution_denoise(_denoise_fn, latents, config=cfg, enabled=False)
    assert on.shape == latents.shape, "SPD output must return to base resolution"
    assert torch.isfinite(on).all(), "SPD output must be finite"
    assert _rel_drift(on, off) > 1e-3, "real multi-resolution SPD should change the trajectory"
    print(f"PASS: SPD multi-resolution changes the latent trajectory (drift={_rel_drift(on, off):.3e})")


def test_dp_dmd_disabled_is_base_loss():
    torch.manual_seed(3)
    student = torch.randn(2, 4, requires_grad=True)
    teacher = torch.randn(2, 4)
    base = (student.square().mean())
    out = compose_dp_dmd_turbo_loss(student, teacher, enabled=False, base_loss=base)
    assert out["dp_dmd_applied"] is False
    assert torch.equal(out["total"], base), "disabled DP-DMD must leave the base loss unchanged"
    assert float(out["consistency"]) == 0.0 and float(out["fake_critic"]) == 0.0
    print("PASS: DP-DMD disabled returns the base loss unchanged (no added terms)")


def test_dp_dmd_enabled_adds_terms_and_backprops():
    torch.manual_seed(4)
    student = torch.randn(2, 4, requires_grad=True)
    teacher = torch.randn(2, 4)
    uncond = torch.randn(2, 4)
    base = student.square().mean()
    cfg = DpDmdTurboConfig(guidance_scale=2.0)
    out = compose_dp_dmd_turbo_loss(
        student, teacher, config=cfg, enabled=True, base_loss=base, teacher_uncond_pred=uncond
    )
    assert out["dp_dmd_applied"] is True
    assert float(out["consistency"].detach()) > 0.0, "enabled DP-DMD should add a real consistency term"
    assert not torch.equal(out["total"], base), "enabled DP-DMD must change the total loss"
    out["total"].backward()
    assert student.grad is not None and float(student.grad.abs().sum()) > 0.0, "no gradient through DP-DMD"
    print("PASS: DP-DMD enabled adds real terms and gradients flow")


def test_readiness_flags():
    report = dp_dmd_spd_reserve_readiness()
    assert report["wired"] is True
    for flag in (
        "runtime_activation_enabled",
        "request_fields_emitted",
        "trainer_wiring_allowed",
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
    ):
        assert report[flag] is False, flag
    print("PASS: readiness reports wired reserves with all activation/promotion gates False")


def main():
    test_spd_disabled_matches_plain_loop()
    test_spd_unit_scale_equals_disabled()
    test_spd_multiresolution_changes_trajectory()
    test_dp_dmd_disabled_is_base_loss()
    test_dp_dmd_enabled_adds_terms_and_backprops()
    test_readiness_flags()
    print("\n[dp_dmd_spd_reserve_smoke] 6/6 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
