# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke for the sampler-quality execution layer (CPU only).

CNS (``cns_sampling``) and SMC-CFG (``smc_cfg``) are already wired into the live
Anima/Newbie ER-SDE sampler.  This smoke exercises the underlying primitives
directly (no model needed) to prove default-off parity + functional correctness:

CNS:
  * empty gamma_path -> recolorer is None (disabled).
  * strength=0 -> recolor returns white noise unchanged (parity).
  * strength=1 -> output keeps shape + spatial variance budget, changes signal.
SMC-CFG:
  * disabled / alpha=0 -> build returns None (sampler uses standard CFG).
  * alpha>0 -> combine differs from standard CFG (correction applied).
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import torch  # noqa: E402

from core.lulynx_trainer.cns_sampling import CNSCalibration, build_cns_recolorer  # noqa: E402
from core.lulynx_trainer.smc_cfg import (  # noqa: E402
    SMCCFGConfig,
    SMCCFGState,
    build_smc_cfg_state,
    standard_cfg,
)


def _spatial_std(x: torch.Tensor) -> torch.Tensor:
    c = x - x.mean(dim=(-2, -1), keepdim=True)
    return c.square().mean(dim=(-2, -1), keepdim=True).sqrt()


def _synthetic_calibration() -> CNSCalibration:
    torch.manual_seed(0)
    gamma = torch.rand(1, 2, 4)          # (A, T, F)
    aspects = torch.tensor([[64.0, 64.0]])
    sigmas = torch.tensor([1.0, 0.5, 0.1])  # T + 1
    return CNSCalibration.from_arrays(gamma, aspects, sigmas)


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    torch.manual_seed(0)
    white = torch.randn(1, 4, 64, 64)

    # --- CNS ----------------------------------------------------------------
    checks.append(("cns_disabled_none", build_cns_recolorer(gamma_path="") is None, "empty path -> None"))

    recolor0 = build_cns_recolorer(calibration=_synthetic_calibration(), strength=0.0)
    out0 = recolor0.recolor(white, sigma=0.5)
    checks.append(("cns_strength0_parity", torch.equal(out0, white), f"max={float((out0-white).abs().max()):.2e}"))

    recolor1 = build_cns_recolorer(calibration=_synthetic_calibration(), strength=1.0)
    out1 = recolor1.recolor(white, sigma=0.5)
    var_ratio = float((_spatial_std(out1) / _spatial_std(white).clamp_min(1e-8)).mean())
    preserves_var = abs(var_ratio - 1.0) < 0.05
    changes = not torch.allclose(out1, white, atol=1e-4)
    checks.append(("cns_shape", tuple(out1.shape) == tuple(white.shape), str(tuple(out1.shape))))
    checks.append(("cns_preserves_variance", preserves_var, f"var_ratio={var_ratio:.4f}"))
    checks.append(("cns_changes_signal", changes, "colored != white"))

    # --- SMC-CFG ------------------------------------------------------------
    checks.append(("smc_disabled_none", build_smc_cfg_state(SMCCFGConfig(enabled=False)) is None, "disabled -> None"))
    checks.append(("smc_alpha0_none", build_smc_cfg_state(SMCCFGConfig(enabled=True, alpha=0.0)) is None, "alpha=0 -> None"))

    cond = torch.randn(1, 4, 8, 8)
    uncond = torch.randn(1, 4, 8, 8)
    gs = 5.0
    std = standard_cfg(cond, uncond, gs)
    state = SMCCFGState(lam=5.0, alpha=0.2)
    combined = state.combine(cond, uncond, gs)
    checks.append(("smc_standard_shape", tuple(std.shape) == tuple(cond.shape), str(tuple(std.shape))))
    checks.append(("smc_combine_changes", not torch.allclose(combined, std, atol=1e-5), "SMC != standard CFG"))

    from core.lulynx_trainer.sampler_quality_scorecard import build_sampler_quality_scorecard

    r = {name: passed for name, passed, _ in checks}
    card = build_sampler_quality_scorecard(
        cns_parity_off=r["cns_strength0_parity"],
        cns_recolor_preserves_variance=r["cns_preserves_variance"],
        cns_recolor_changes_signal=r["cns_changes_signal"],
        smc_parity_off=r["smc_disabled_none"] and r["smc_alpha0_none"],
        smc_combine_changes_signal=r["smc_combine_changes"],
    )
    checks.append(("scorecard_ok", card["ok"], f"default_changed={card['default_behavior_changed']}"))

    ok = all(passed for _, passed, _ in checks)
    print("=== sampler_quality smoke (CNS + SMC-CFG) ===")
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print(f"scorecard ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
