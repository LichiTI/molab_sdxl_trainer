"""Default-off rollout proposal for reviewed DiT compute reducer results."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


REQUIRED_PROPOSAL_FIELDS = (
    "proposal_id",
    "owner",
    "reviewer",
    "reducer_scope",
    "rollback_plan",
    "quality_monitoring_plan",
    "canary_scope",
    "activation_boundary",
)


def build_dit_compute_reducer_default_off_rollout_proposal(
    *,
    quality_decision: Mapping[str, Any],
    rollout_proposal: Mapping[str, Any],
) -> dict[str, Any]:
    decision = dict(quality_decision)
    proposal = dict(rollout_proposal)
    passed = tuple(str(item) for item in decision.get("passed_reducers", ()) if str(item).strip())
    blockers: list[str] = []

    if decision.get("scorecard") != "dit_compute_reducer_quality_review_decision_v0":
        blockers.append("unexpected_quality_decision")
    if not bool(decision.get("quality_review_ready", decision.get("ok", False))):
        blockers.append("quality_review_not_ready")
    if not bool(decision.get("promotion_review_ready", False)):
        blockers.append("promotion_review_not_ready")
    if _unsafe_flags(decision, proposal):
        blockers.append("unsafe_child_flag")
    if not passed:
        blockers.append("passed_reducers_missing")
    for name in REQUIRED_PROPOSAL_FIELDS:
        if not str(proposal.get(name) or "").strip():
            blockers.append(f"proposal_field_missing:{name}")
    if proposal.get("default_enable_allowed") is not False:
        blockers.append("default_enable_boundary_missing")
    if proposal.get("auto_rollout_allowed") is not False:
        blockers.append("auto_rollout_boundary_missing")
    if proposal.get("trainer_wiring_allowed") is not False:
        blockers.append("trainer_wiring_boundary_missing")
    if not bool(proposal.get("acknowledge_default_off", False)):
        blockers.append("default_off_acknowledgement_missing")
    if not bool(proposal.get("requires_later_runtime_activation_review", False)):
        blockers.append("later_runtime_activation_review_missing")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_default_off_rollout_proposal_v0",
        "ok": ready,
        "rollout_proposal_ready": ready,
        "activation_boundary_recorded": ready,
        "passed_reducers": list(passed),
        "passed_reducer_count": len(passed),
        "proposal": _summary(
            proposal,
            (
                "proposal_id",
                "owner",
                "reviewer",
                "reducer_scope",
                "rollback_plan",
                "quality_monitoring_plan",
                "canary_scope",
                "activation_boundary",
                "acknowledge_default_off",
            ),
        ),
        "trainer_wiring_allowed": False,
        "trainer_wiring_executed": False,
        "ab_dispatch_allowed": False,
        "ab_dispatch_executed": False,
        "ab_execution_allowed": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "run_dispatch_executed": False,
        "runs_dispatched": False,
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "quality_review_ready": bool(decision.get("quality_review_ready", decision.get("ok", False))),
        "promotion_review_ready": bool(decision.get("promotion_review_ready", False)),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "hold compute reducer rollout proposal default-off until explicit runtime activation review"
            if ready
            else "complete default-off compute reducer rollout proposal fields before activation review"
        ),
    }


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
        "default_enable_allowed",
        "default_rollout_allowed",
        "auto_rollout_allowed",
        "trainer_wiring_allowed",
        "trainer_wiring_executed",
        "ab_dispatch_allowed",
        "ab_dispatch_executed",
        "ab_execution_allowed",
        "training_launch_allowed",
        "training_launch_executed",
        "run_dispatch_executed",
        "runs_dispatched",
        "runtime_activation_enabled",
        "request_fields_emitted",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


def _summary(payload: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if key in payload}


__all__ = ["build_dit_compute_reducer_default_off_rollout_proposal"]
