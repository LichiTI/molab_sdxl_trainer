"""Promotion review gate for the V3 exact AdamW native canary."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_v3_exact_adamw_config_adapter_scorecard import (
    build_v3_exact_adamw_config_adapter_scorecard,
)


def build_v3_exact_adamw_promotion_review_scorecard(
    *,
    p4_config_adapter: Mapping[str, Any] | None = None,
    run_live_training: bool = True,
) -> dict[str, Any]:
    """Build the final V3 review gate without enabling default dispatch."""

    p4 = dict(
        p4_config_adapter
        or build_v3_exact_adamw_config_adapter_scorecard(run_live_training=run_live_training)
    )
    review = _review_package(p4)
    rollback = _rollback_policy()
    progress_gates = {
        "p4_config_adapter_complete": bool(p4.get("config_adapter_ready", False)),
        "explicit_canary_review_ready": bool(review.get("explicit_canary_review_ready", False)),
        "default_and_auto_blocked": not bool(review.get("default_rollout_allowed", True))
        and not bool(review.get("auto_rollout_allowed", True)),
        "manual_review_required": bool(review.get("manual_review_required", False)),
        "fallback_rollback_ready": bool(rollback.get("fallback_authoritative", False))
        and bool(rollback.get("disable_for_run_on_native_error", False))
        and bool(rollback.get("disable_for_run_on_state_mismatch", False))
        and bool(rollback.get("disable_for_run_on_config_mismatch", False)),
        "performance_boundary_recorded": bool(review.get("representative_performance_required", False))
        and bool(review.get("tiny_matrix_not_representative", False)),
        "default_behavior_unchanged": (
            not bool(p4.get("default_behavior_changed", True))
            and not bool(p4.get("default_training_path_enabled", True))
        ),
    }
    ready = all(progress_gates.values())
    blockers = [f"v3_p5_{name}_missing" for name, ok in progress_gates.items() if not ok]
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v3_exact_adamw_promotion_review_scorecard_v0",
        "gate": "v3_exact_adamw_promotion_review",
        "ok": bool(p4.get("ok", False)),
        "milestone_completed": ready,
        "promotion_review_ready": ready,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "explicit_canary_allowed": bool(review.get("explicit_canary_review_ready", False)),
        "manual_review_required": True,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "p4_summary": {
            "config_adapter_ready": bool(p4.get("config_adapter_ready", False)),
            "milestone_completed": bool(p4.get("milestone_completed", False)),
        },
        "review_package": review,
        "rollback_policy": rollback,
        "progress_gates": progress_gates,
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "V3 exact AdamW canary roadmap complete; keep default off until representative training benchmark review"
            if ready
            else "complete V3-P5 promotion review blockers"
        ),
        "notes": [
            "This gate completes the V3 canary readiness review but does not promote default dispatch.",
            "The current matrix proves correctness and recovery behavior, not representative end-to-end speed.",
            "A larger canary still requires explicit opt-in and human review.",
        ],
    }


def _review_package(p4: Mapping[str, Any]) -> dict[str, Any]:
    adapter_cases = p4.get("adapter_cases") if isinstance(p4.get("adapter_cases"), Mapping) else {}
    exact_case = adapter_cases.get("explicit_exact_adamw") if isinstance(adapter_cases.get("explicit_exact_adamw"), Mapping) else {}
    resolved = exact_case.get("resolved_fields") if isinstance(exact_case.get("resolved_fields"), Mapping) else {}
    request_fields_ready = bool(
        resolved.get("turbocore_native_update_mode") == "native_experimental"
        and resolved.get("turbocore_native_update_dispatch_enabled") is True
        and resolved.get("turbocore_native_update_training_path_enabled") is True
        and resolved.get("turbocore_native_update_require_native_cuda") is True
    )
    return {
        "schema_version": 1,
        "review": "v3_exact_adamw_promotion_review_package_v0",
        "explicit_canary_review_ready": bool(p4.get("config_adapter_ready", False)) and request_fields_ready,
        "request_fields_ready": request_fields_ready,
        "safe_modes": ["off", "observe", "explicit_canary"],
        "blocked_modes_until_review": ["default", "auto"],
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "manual_review_required": True,
        "tiny_matrix_not_representative": True,
        "representative_performance_required": True,
        "larger_canary_requires": [
            "explicit_config_adapter_request",
            "runtime_recovery_latch",
            "checkpoint_resume_review",
            "representative_lora_or_anima_training_benchmark",
            "manual_owner_review",
        ],
    }


def _rollback_policy() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy": "v3_exact_adamw_promotion_review_rollback_policy_v0",
        "fallback_authoritative": True,
        "fallback_backend": "pytorch_adamw",
        "disable_for_run_on_native_error": True,
        "disable_for_run_on_state_mismatch": True,
        "disable_for_run_on_state_sync_failure": True,
        "disable_for_run_on_config_mismatch": True,
        "disable_for_run_on_checkpoint_resume_mismatch": True,
        "default_training_path_enabled": False,
    }


__all__ = ["build_v3_exact_adamw_promotion_review_scorecard"]
