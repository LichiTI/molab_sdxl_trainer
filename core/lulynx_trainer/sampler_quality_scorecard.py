# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Scorecard for the sampler-quality execution layer (CNS + SMC-CFG).

Both primitives are already wired into the live Anima/Newbie ER-SDE sampler
(``cns_recolorer.recolor`` at the noise-injection seam, ``smc_cfg_state.combine``
at the CFG-combine seam).  This scorecard records that they are default-off,
parity-safe when disabled, and functionally active when enabled.  Real
preview-quality must be judged on a real GPU by the user.
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_sampler_quality_scorecard(
    *,
    cns_parity_off: bool,
    cns_recolor_preserves_variance: bool,
    cns_recolor_changes_signal: bool,
    smc_parity_off: bool,
    smc_combine_changes_signal: bool,
    quality_gate_verified: bool = False,
) -> Dict[str, Any]:
    blocked: List[str] = []
    if not cns_parity_off:
        blocked.append("CNS strength=0 / no gamma diverged from white noise")
    if not cns_recolor_preserves_variance:
        blocked.append("CNS recolor did not preserve the spatial variance budget")
    if not cns_recolor_changes_signal:
        blocked.append("CNS recolor (strength>0) did not change the noise")
    if not smc_parity_off:
        blocked.append("SMC-CFG disabled / alpha=0 did not fall back to standard CFG")
    if not smc_combine_changes_signal:
        blocked.append("SMC-CFG combine (alpha>0) did not correct standard CFG")

    ok = not blocked
    return {
        "ok": ok,
        "wired_into_trainer": False,
        "execution_layer_wired": True,
        "wiring_mode": "live ER-SDE sampler (CNS noise seam + SMC-CFG combine seam), default off",
        "default_behavior_changed": False,
        "cns_parity_off": cns_parity_off,
        "cns_recolor_preserves_variance": cns_recolor_preserves_variance,
        "cns_recolor_changes_signal": cns_recolor_changes_signal,
        "smc_parity_off": smc_parity_off,
        "smc_combine_changes_signal": smc_combine_changes_signal,
        "quality_gate_verified": quality_gate_verified,
        "blocked_reasons": blocked,
        "notes": (
            "CNS is ER-SDE-only (stochastic noise recoloring); empty gamma_path "
            "or strength<=0 returns white noise unchanged. SMC-CFG falls back to "
            "standard CFG when disabled or alpha<=0. Both are inference/preview "
            "sampler-quality features, not trainer speedups. Real preview-quality "
            "and parameter tuning need a real-GPU quality gate (user-run)."
        ),
    }


__all__ = ["build_sampler_quality_scorecard"]
