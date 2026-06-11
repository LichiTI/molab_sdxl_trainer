"""Runtime activation review boundary for T-LoRA A/B rollout proposals."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


APPROVED_REVIEW_DECISIONS = {"approved", "approve", "signed"}
EXPECTED_SCOPE = "tlora_ab_runtime_activation_review"


def build_tlora_ab_runtime_activation_review(
    *,
    rollout_proposal: Mapping[str, Any],
    activation_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    proposal = dict(rollout_proposal)
    review = dict(activation_review or {})
    blockers: list[str] = []

    if proposal.get("scorecard") != "tlora_ab_default_off_rollout_proposal_v0":
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
        "acknowledge_no_training_launch",
        "acknowledge_manual_activation_required",
    ):
        if not bool(review.get(key, False)):
            blockers.append(f"{key}_missing")
    for key in (
        "approve_runtime_activation_allowed",
        "approve_runtime_activation_enabled",
        "approve_request_fields_emitted",
        "approve_training_launch_allowed",
        "approve_default_rollout_allowed",
        "approve_auto_rollout_allowed",
    ):
        if review.get(key) is not False:
            blockers.append(f"{key}_must_be_false")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "tlora_ab_runtime_activation_review_v0",
        "ok": ready,
        "runtime_activation_review_ready": ready,
        "activation_review_signed": ready,
        "runtime_activation_allowed": False,
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "proposal_ready": bool(proposal.get("rollout_proposal_ready", proposal.get("ok", False))),
        "activation_boundary_recorded": bool(proposal.get("activation_boundary_recorded", False)),
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
            "prepare separate request-field emission contract while keeping T-LoRA default-off"
            if ready
            else "complete signed default-off runtime activation review before request wiring"
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
        "training_launch_allowed",
        "runs_dispatched",
        "default_rollout_allowed",
        "auto_rollout_allowed",
        "approve_runtime_activation_allowed",
        "approve_runtime_activation_enabled",
        "approve_request_fields_emitted",
        "approve_training_launch_allowed",
        "approve_default_rollout_allowed",
        "approve_auto_rollout_allowed",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


def _summary(payload: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if key in payload}


__all__ = ["build_tlora_ab_runtime_activation_review"]
