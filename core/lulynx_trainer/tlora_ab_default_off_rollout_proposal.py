"""Default-off rollout proposal gate for reviewed T-LoRA A/B evidence."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_tlora_ab_default_off_rollout_proposal(
    *,
    quality_decision: Mapping[str, Any],
    rollout_proposal: Mapping[str, Any],
) -> dict[str, Any]:
    decision = dict(quality_decision)
    proposal = dict(rollout_proposal)
    blockers: list[str] = []

    if decision.get("scorecard") != "tlora_ab_quality_review_decision_v0":
        blockers.append("unexpected_quality_decision")
    if not bool(decision.get("quality_review_ready", decision.get("ok", False))):
        blockers.append("quality_review_not_ready")
    if not bool(decision.get("promotion_review_ready", False)):
        blockers.append("promotion_review_not_ready")
    if _unsafe_flags(decision, proposal):
        blockers.append("unsafe_child_flag")
    if not str(proposal.get("proposal_id") or "").strip():
        blockers.append("proposal_id_missing")
    if not str(proposal.get("owner") or "").strip():
        blockers.append("owner_missing")
    if not str(proposal.get("reviewer") or "").strip():
        blockers.append("reviewer_missing")
    if not str(proposal.get("rollback_plan") or "").strip():
        blockers.append("rollback_plan_missing")
    if not str(proposal.get("canary_scope") or "").strip():
        blockers.append("canary_scope_missing")
    if not str(proposal.get("activation_boundary") or "").strip():
        blockers.append("activation_boundary_missing")
    if proposal.get("default_enable_allowed") is not False:
        blockers.append("default_enable_boundary_missing")
    if proposal.get("auto_rollout_allowed") is not False:
        blockers.append("auto_rollout_boundary_missing")
    if not bool(proposal.get("acknowledge_default_off", False)):
        blockers.append("default_off_acknowledgement_missing")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "tlora_ab_default_off_rollout_proposal_v0",
        "ok": ready,
        "rollout_proposal_ready": ready,
        "activation_boundary_recorded": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "auto_rollout_allowed": False,
        "default_rollout_allowed": False,
        "proposal": _summary(
            proposal,
            (
                "proposal_id",
                "owner",
                "reviewer",
                "rollback_plan",
                "canary_scope",
                "activation_boundary",
                "acknowledge_default_off",
            ),
        ),
        "quality_review_ready": bool(decision.get("quality_review_ready", decision.get("ok", False))),
        "promotion_review_ready": bool(decision.get("promotion_review_ready", False)),
        "case_count": int(decision.get("case_count") or 0),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "hold T-LoRA rollout proposal default-off until explicit runtime activation review"
            if ready
            else "complete default-off T-LoRA rollout proposal fields before activation review"
        ),
    }


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    return any(
        bool(payload.get("training_path_enabled", False))
        or bool(payload.get("default_behavior_changed", False))
        or bool(payload.get("promotion_ready", False))
        or bool(payload.get("default_enable_allowed", False))
        or bool(payload.get("auto_rollout_allowed", False))
        for payload in payloads
    )


def _summary(payload: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if key in payload}


__all__ = ["build_tlora_ab_default_off_rollout_proposal"]
