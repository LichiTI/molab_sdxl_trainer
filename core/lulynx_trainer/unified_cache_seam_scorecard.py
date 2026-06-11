# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Scorecard for the unified DiT block cache seam (roadmap 624)."""

from __future__ import annotations

from typing import Any, Dict, List


def build_unified_cache_seam_scorecard(
    *,
    parity_disabled: bool,
    parity_no_context: bool,
    dispatch_routes_correct: bool,
    reuse_on_cacheable_step: bool,
    backends_supported: List[str],
    quality_gate_verified: bool = False,
) -> Dict[str, Any]:
    """Summarize default-off safety + correctness of the cache seam.

    ``quality_gate_verified`` stays False here: real preview-quality and actual
    speedup must be measured on a real GPU by the user (CPU smoke only proves
    parity + dispatch + reuse logic).
    """
    blocked_reasons: List[str] = []
    if not parity_disabled:
        blocked_reasons.append("disabled seam (backend=none) diverged from baseline")
    if not parity_no_context:
        blocked_reasons.append("enabled seam without an active step context diverged from baseline")
    if not dispatch_routes_correct:
        blocked_reasons.append("backend dispatch routed to the wrong execution primitive")
    if not reuse_on_cacheable_step:
        blocked_reasons.append("seam did not reuse a cached block on a cacheable step")

    ok = not blocked_reasons
    return {
        "ok": ok,
        "wired_into_trainer": False,
        "execution_layer_wired": True,
        "wiring_mode": "opt-in live _run_blocks (default off)",
        "default_behavior_changed": False,
        "observe_probe_wired": True,
        "parity_disabled": parity_disabled,
        "parity_no_context": parity_no_context,
        "dispatch_routes_correct": dispatch_routes_correct,
        "reuse_on_cacheable_step": reuse_on_cacheable_step,
        "backends_supported": list(backends_supported),
        "quality_gate_verified": quality_gate_verified,
        "blocked_reasons": blocked_reasons,
        "notes": (
            "Block-level seam drives Spectrum/SmoothCache only; T-GATE is "
            "cross-attention granularity and stays the observe probe + library "
            "primitive. Default backend=none is bitwise-identical to baseline. "
            "Inference/preview acceleration only -- not a trainer speedup. Real "
            "preview-quality/speedup needs a real-GPU quality gate (user-run)."
        ),
    }


__all__ = ["build_unified_cache_seam_scorecard"]
