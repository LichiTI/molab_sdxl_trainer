# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Promotion scorecard for Immiscible diffusion (L2 minibatch noise assignment).

Opt-in via ``immiscible_diffusion_enabled``; default behavior (no assignment on
DDPM, cosine OT on flow when ``flow_use_ot``) is unchanged.  Promotion gates on:
the L2 assignment is a valid permutation, it actually lowers paired transport
cost, the integration hook routes correctly (and is a no-op when disabled), and
the stage plan records the feature.

Clean-room Lulynx module; shares no source with any reference implementation.
"""

from __future__ import annotations

from typing import Any


def build_immiscible_scorecard(
    *,
    bijection_verified: bool = False,
    l2_reduced: bool = False,
    assigned_cost: float | None = None,
    random_cost: float | None = None,
    routing_verified: bool = False,
    disabled_parity_verified: bool = False,
    plan_feature_verified: bool = False,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not bijection_verified:
        blockers.append("assignment_not_a_permutation")
    if not l2_reduced:
        blockers.append("paired_l2_not_reduced")
    if not routing_verified:
        blockers.append("integration_routing_not_verified")
    if not disabled_parity_verified:
        blockers.append("disabled_parity_not_verified")
    if not plan_feature_verified:
        blockers.append("stage_plan_feature_not_recorded")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "immiscible_diffusion_v0",
        "gate": "immiscible_l2_minibatch_assignment",
        "ok": ready,
        "optimization_ready": ready,
        "promotion_ready": ready,
        "training_path_enabled": True,
        "trainer_wiring_allowed": True,
        "default_behavior_changed": False,
        "bijection_verified": bool(bijection_verified),
        "l2_reduced": bool(l2_reduced),
        "assigned_cost": (float(assigned_cost) if assigned_cost is not None else None),
        "random_cost": (float(random_cost) if random_cost is not None else None),
        "routing_verified": bool(routing_verified),
        "disabled_parity_verified": bool(disabled_parity_verified),
        "plan_feature_verified": bool(plan_feature_verified),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "ready: immiscible diffusion may ship default-off via immiscible_diffusion_enabled"
            if ready
            else "resolve blockers: " + ", ".join(blockers)
        ),
        "notes": [
            "Opt-in via immiscible_diffusion_enabled; DDPM gains L2 assignment, flow keeps cosine unless metric='l2'.",
            "When disabled, the noise path is bit-identical to legacy (flow_use_ot cosine still honored).",
        ],
    }


__all__ = ["build_immiscible_scorecard"]
