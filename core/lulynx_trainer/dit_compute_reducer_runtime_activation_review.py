"""Runtime activation review boundary for DiT compute reducer rollout proposals."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


APPROVED_REVIEW_DECISIONS = {"approved", "approve", "signed"}
EXPECTED_SCOPE = "dit_compute_reducer_runtime_activation_review"


def build_dit_compute_reducer_runtime_activation_review(
    *,
    rollout_proposal: Mapping[str, Any],
    activation_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    proposal = dict(rollout_proposal)
    review = dict(activation_review or {})
    blockers: list[str] = []

    if proposal.get("scorecard") != "dit_compute_reducer_default_off_rollout_proposal_v0":
        blockers.append("unexpected_rollout_proposal")
    if not bool(proposal.get("rollout_proposal_ready", proposal.get("ok", False))):
        blockers.append("rollout_proposal_not_ready")
    if not bool(proposal.get("activation_boundary_recorded", False)):
        blockers.append("activation_boundary_not_recorded")
    if _unsafe_flags(proposal, review):
        blockers.append("unsafe_child_flag")
    if not review:
        blockers.append("activation_review_missing")
    if str(review.get("decision") or "").strip().lower() not in APPROVED_REVIEW_DECISIONS:
        blockers.append("activation_review_not_approved")
    if not str(review.get("signed_review_id") or "").strip():
        blockers.append("signed_review_id_missing")
    if not str(review.get("reviewer") or "").strip():
        blockers.append("reviewer_missing")
    if not str(review.get("reviewed_at") or "").strip():
        blockers.append("reviewed_at_missing")
    if str(review.get("requested_scope") or "").strip() != EXPECTED_SCOPE:
        blockers.append("requested_scope_mismatch")
    if not str(review.get("proposal_digest") or review.get("artifact_digest") or "").strip():
        blockers.append("proposal_digest_missing")
    for key in (
        "acknowledge_default_off",
        "acknowledge_no_runtime_activation",
        "acknowledge_no_request_fields_emitted",
        "acknowledge_no_trainer_wiring",
        "acknowledge_no_training_launch",
        "acknowledge_manual_activation_required",
    ):
        if not bool(review.get(key, False)):
            blockers.append(f"{key}_missing")
    for key in (
        "approve_runtime_activation_allowed",
        "approve_runtime_activation_enabled",
        "approve_request_fields_emitted",
        "approve_trainer_wiring_allowed",
        "approve_training_launch_allowed",
        "approve_default_rollout_allowed",
        "approve_auto_rollout_allowed",
    ):
        if review.get(key) is not False:
            blockers.append(f"{key}_must_be_false")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_runtime_activation_review_v0",
        "ok": ready,
        "runtime_activation_review_ready": ready,
        "activation_review_signed": ready,
        "runtime_activation_allowed": False,
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "trainer_wiring_allowed": False,
        "trainer_wiring_executed": False,
        "ab_dispatch_allowed": False,
        "ab_dispatch_executed": False,
        "ab_execution_allowed": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "run_dispatch_executed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "proposal_ready": bool(proposal.get("rollout_proposal_ready", proposal.get("ok", False))),
        "activation_boundary_recorded": bool(proposal.get("activation_boundary_recorded", False)),
        "passed_reducer_count": int(proposal.get("passed_reducer_count") or 0),
        "passed_reducers": list(proposal.get("passed_reducers", ()) or ()),
        "review": _summary(
            review,
            (
                "signed_review_id",
                "decision",
                "reviewer",
                "reviewed_at",
                "requested_scope",
                "proposal_digest",
                "artifact_digest",
            ),
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare separate request-field emission contract while keeping compute reducers default-off"
            if ready
            else "complete signed compute reducer runtime activation review before request wiring"
        ),
    }


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
        "default_enable_allowed",
        "runtime_activation_allowed",
        "runtime_activation_enabled",
        "request_fields_emitted",
        "request_adapter_registered",
        "trainer_wiring_allowed",
        "trainer_wiring_executed",
        "ab_dispatch_allowed",
        "ab_dispatch_executed",
        "ab_execution_allowed",
        "training_launch_allowed",
        "training_launch_executed",
        "run_dispatch_executed",
        "runs_dispatched",
        "default_rollout_allowed",
        "auto_rollout_allowed",
        "approve_runtime_activation_allowed",
        "approve_runtime_activation_enabled",
        "approve_request_fields_emitted",
        "approve_trainer_wiring_allowed",
        "approve_training_launch_allowed",
        "approve_default_rollout_allowed",
        "approve_auto_rollout_allowed",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


def _summary(payload: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if key in payload}


__all__ = ["build_dit_compute_reducer_runtime_activation_review"]
