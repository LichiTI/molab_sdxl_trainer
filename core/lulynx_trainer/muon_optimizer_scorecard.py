# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Promotion scorecard for the Muon optimizer (2D LoRA-factor orthogonalization).

Muon is an opt-in optimizer route (``OptimizerType.MUON``); it never changes
default behavior.  Promotion gates on: the Newton-Schulz iteration actually
orthogonalizes (singular values → 1), the hybrid 2D/non-2D routing is correct,
a short LoRA run trains (loss decreases), and optimizer state round-trips
through save/load.

Clean-room Lulynx module; shares no source with any reference implementation.
"""

from __future__ import annotations

from typing import Any


def build_muon_scorecard(
    *,
    orthogonality_verified: bool = False,
    orthogonality_error: float | None = None,
    hybrid_routing_verified: bool = False,
    muon_param_count: int = 0,
    adamw_param_count: int = 0,
    loss_decreased: bool = False,
    initial_loss: float | None = None,
    final_loss: float | None = None,
    resume_verified: bool = False,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not orthogonality_verified:
        blockers.append("newton_schulz_not_orthogonal")
    if not hybrid_routing_verified:
        blockers.append("hybrid_routing_not_verified")
    if muon_param_count <= 0:
        blockers.append("no_2d_params_routed_to_muon")
    if not loss_decreased:
        blockers.append("loss_did_not_decrease")
    if not resume_verified:
        blockers.append("optimizer_state_resume_not_verified")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "muon_optimizer_v0",
        "gate": "muon_lora_orthogonalized_momentum",
        "ok": ready,
        "optimization_ready": ready,
        "promotion_ready": ready,
        "training_path_enabled": True,
        "trainer_wiring_allowed": True,
        "default_behavior_changed": False,
        "optimizer_type": "Muon",
        "orthogonality_verified": bool(orthogonality_verified),
        "orthogonality_error": (float(orthogonality_error) if orthogonality_error is not None else None),
        "hybrid_routing_verified": bool(hybrid_routing_verified),
        "muon_param_count": int(muon_param_count),
        "adamw_param_count": int(adamw_param_count),
        "loss_decreased": bool(loss_decreased),
        "initial_loss": (float(initial_loss) if initial_loss is not None else None),
        "final_loss": (float(final_loss) if final_loss is not None else None),
        "resume_verified": bool(resume_verified),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "ready: Muon may ship default-off as OptimizerType.MUON"
            if ready
            else "resolve blockers: " + ", ".join(blockers)
        ),
        "notes": [
            "Opt-in via optimizer='Muon'; default optimizer behavior is unchanged.",
            "2D params (LoRA factors) use Newton-Schulz orthogonalized momentum; 1D params fall back to AdamW (no decay).",
            "v1 orthogonalizes each 2D param locally (correct for single-GPU / DDP-replicated runs).",
        ],
    }


__all__ = ["build_muon_scorecard"]
