"""V5 manual wider-canary review gate for TurboCore native AdamW."""

from __future__ import annotations

from typing import Any, Mapping


def build_v5_manual_wider_canary_review(
    *,
    stability_gate: Mapping[str, Any] | None = None,
    owner_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manual review package without enabling default rollout."""

    stability = _as_dict(stability_gate)
    review = _as_dict(owner_review)
    aggregate = _as_dict(stability.get("aggregate"))
    rollback = _rollback_policy(review)
    progress_gates = {
        "p3_stability_gate_ready": bool(stability.get("stability_gate_ready", False)),
        "replicate_run_count_met": int(stability.get("run_count", 0) or 0) >= int(
            stability.get("min_replicate_runs", 3) or 3
        ),
        "manual_owner_review_present": bool(review),
        "manual_owner_approved": bool(review.get("approve_manual_wider_canary", False)),
        "default_and_auto_confirmed_off": _review_confirmed_default_off(review),
        "rollback_policy_ready": _rollback_ready(rollback),
        "timing_sync_policy_acknowledged": bool(review.get("acknowledge_runtime_synchronization", False)),
        "scope_limited_to_manual_wider_canary": _scope_ok(review),
        "default_behavior_unchanged": True,
    }
    ready = all(progress_gates.values())
    blocked = _blockers(progress_gates, stability)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v5_manual_wider_canary_review_v0",
        "gate": "v5_manual_wider_canary_review",
        "ok": ready,
        "promotion_review_ready": ready,
        "promotion_decision": "manual_wider_canary_review_ready" if ready else _hold_decision(progress_gates),
        "manual_wider_canary_allowed": ready,
        "explicit_single_run_canary_allowed": bool(stability.get("stability_gate_ready", False)),
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "manual_review_required": True,
        "stability_summary": {
            "run_count": int(stability.get("run_count", 0) or 0),
            "ready_run_count": int(stability.get("ready_run_count", 0) or 0),
            "speedup_samples": list(aggregate.get("speedup_samples", []) or []),
            "min_speedup": aggregate.get("min_speedup"),
            "mean_speedup": aggregate.get("mean_speedup"),
            "speedup_spread_ratio": aggregate.get("speedup_spread_ratio"),
            "blocked_reasons": list(stability.get("blocked_reasons", []) or []),
        },
        "owner_review": _review_summary(review),
        "rollback_policy": rollback,
        "progress_gates": progress_gates,
        "blocked_reasons": blocked,
        "promotion_blockers": blocked,
        "recommended_next_step": _recommended_next_step(ready, progress_gates),
        "notes": [
            "This review gate does not enable default dispatch.",
            "Manual wider canary requires explicit owner approval and rollback readiness.",
            "Auto rollout remains blocked even when manual wider canary review passes.",
        ],
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(review),
        "reviewer": str(review.get("reviewer", "") or ""),
        "reviewed_at": str(review.get("reviewed_at", "") or ""),
        "approve_manual_wider_canary": bool(review.get("approve_manual_wider_canary", False)),
        "confirmed_default_auto_off": _review_confirmed_default_off(review),
        "acknowledge_runtime_synchronization": bool(review.get("acknowledge_runtime_synchronization", False)),
        "requested_scope": str(review.get("requested_scope", "") or ""),
    }


def _rollback_policy(review: Mapping[str, Any]) -> dict[str, Any]:
    source = _as_dict(review.get("rollback_policy"))
    return {
        "schema_version": 1,
        "policy": "v5_manual_wider_canary_rollback_policy_v0",
        "fallback_authoritative": bool(source.get("fallback_authoritative", True)),
        "fallback_backend": str(source.get("fallback_backend", "pytorch_adamw") or "pytorch_adamw"),
        "disable_for_run_on_native_error": bool(source.get("disable_for_run_on_native_error", True)),
        "disable_for_run_on_state_sync_failure": bool(source.get("disable_for_run_on_state_sync_failure", True)),
        "disable_for_run_on_checkpoint_resume_mismatch": bool(
            source.get("disable_for_run_on_checkpoint_resume_mismatch", True)
        ),
        "disable_for_run_on_config_mismatch": bool(source.get("disable_for_run_on_config_mismatch", True)),
        "disable_for_run_on_non_finite": bool(source.get("disable_for_run_on_non_finite", True)),
        "rollback_on_resume_mismatch": bool(source.get("rollback_on_resume_mismatch", True)),
        "rollback_on_performance_regression": bool(source.get("rollback_on_performance_regression", True)),
        "default_training_path_enabled": False,
    }


def _rollback_ready(rollback: Mapping[str, Any]) -> bool:
    return bool(
        rollback.get("fallback_authoritative", False)
        and rollback.get("disable_for_run_on_native_error", False)
        and rollback.get("disable_for_run_on_state_sync_failure", False)
        and rollback.get("disable_for_run_on_checkpoint_resume_mismatch", False)
        and rollback.get("rollback_on_resume_mismatch", False)
        and rollback.get("rollback_on_performance_regression", False)
    )


def _review_confirmed_default_off(review: Mapping[str, Any]) -> bool:
    return bool(
        review.get("confirm_default_training_path_enabled") is False
        and review.get("confirm_training_path_enabled") is False
        and review.get("confirm_default_rollout_allowed") is False
        and review.get("confirm_auto_rollout_allowed") is False
    )


def _scope_ok(review: Mapping[str, Any]) -> bool:
    return str(review.get("requested_scope", "") or "") == "manual_wider_canary"


def _blockers(progress_gates: Mapping[str, bool], stability: Mapping[str, Any]) -> list[str]:
    blocked = [f"v5_p4_{name}_missing" for name, ok in progress_gates.items() if not ok]
    if not bool(stability.get("stability_gate_ready", False)):
        blocked.extend(str(item) for item in list(stability.get("blocked_reasons", []) or []))
    return _dedupe(blocked)


def _hold_decision(progress_gates: Mapping[str, bool]) -> str:
    if not bool(progress_gates.get("p3_stability_gate_ready", False)):
        return "hold_for_v5_p3_stability_gate"
    if not bool(progress_gates.get("manual_owner_review_present", False)):
        return "hold_for_manual_owner_review"
    if not bool(progress_gates.get("manual_owner_approved", False)):
        return "hold_for_manual_owner_approval"
    return "hold_for_review_contract_completion"


def _recommended_next_step(ready: bool, progress_gates: Mapping[str, bool]) -> str:
    if ready:
        return "manual wider canary can be requested explicitly; default and auto remain off"
    if not bool(progress_gates.get("p3_stability_gate_ready", False)):
        return "complete V5-P3 replicate stability gate before manual wider canary review"
    if not bool(progress_gates.get("manual_owner_review_present", False)):
        return "record owner approval, rollback acknowledgement, and default-off confirmation"
    return "fix manual review contract blockers before widening canary scope"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_v5_manual_wider_canary_review"]
