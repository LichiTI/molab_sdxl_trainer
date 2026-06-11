"""V4 promotion review gate for exact AdamW native canary."""

from __future__ import annotations

from typing import Any, Mapping


def build_v4_promotion_review_scorecard(
    *,
    p3_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Review V4 canary promotion readiness without enabling defaults."""

    sections = p3_audit.get("sections") if isinstance(p3_audit.get("sections"), Mapping) else {}
    single = _as_dict(sections.get("single_run_explicit_canary_policy"))
    wider = _as_dict(sections.get("wider_canary_current_block"))
    auto = _as_dict(sections.get("auto_rollout_block"))
    benchmark_state = _benchmark_state(p3_audit)
    real_ready = bool(benchmark_state["ready"])
    review = _review_package(p3_audit, single, wider, auto, benchmark_state)
    rollback = _rollback_policy(single)
    progress_gates = {
        "p3_policy_contract_complete": bool(p3_audit.get("milestone_completed", False)),
        "single_run_explicit_canary_ready": bool(single.get("ok", False))
        and bool(single.get("explicit_canary_allowed", False)),
        "wider_canary_real_benchmark_gate_recorded": _wider_gate_recorded(wider, real_ready),
        "auto_rollout_blocked": not bool(auto.get("ok", True))
        and "v4_p3_auto_rollout_blocked" in list(auto.get("blocked_reasons", []) or []),
        "manual_review_required": bool(review.get("manual_review_required", False)),
        "fallback_rollback_ready": _rollback_ready(rollback),
        "default_and_auto_blocked": _default_off(single) and _default_off(wider) and _default_off(auto),
        "default_behavior_unchanged": True,
    }
    review_complete = all(progress_gates.values())
    blockers = [f"v4_p4_{name}_missing" for name, ok in progress_gates.items() if not ok]
    manual_wider_allowed = bool(review.get("manual_wider_canary_allowed", False))
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v4_promotion_review_scorecard_v0",
        "gate": "v4_promotion_review",
        "ok": review_complete,
        "milestone_completed": review_complete,
        "promotion_review_ready": review_complete,
        "promotion_decision": str(review.get("promotion_decision") or ""),
        "real_benchmark_result_ready": real_ready,
        "real_benchmark_input_present": bool(benchmark_state["input_present"]),
        "real_benchmark_executed": bool(benchmark_state["executed"]),
        "real_benchmark_contract_ready": bool(benchmark_state["contract_ready"]),
        "real_benchmark_performance_gate_ready": bool(benchmark_state["performance_gate_ready"]),
        "real_benchmark_status": str(benchmark_state["status"]),
        "real_benchmark_performance_blockers": list(benchmark_state["performance_blockers"]),
        "manual_review_required": True,
        "manual_wider_canary_allowed": manual_wider_allowed,
        "explicit_single_run_canary_allowed": bool(single.get("explicit_canary_allowed", False)),
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "review_package": review,
        "rollback_policy": rollback,
        "progress_gates": progress_gates,
        "promotion_hold_reasons": list(review.get("promotion_hold_reasons", []) or []),
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": _recommended_next_step(review),
        "notes": [
            "This gate records the review decision; it does not promote default dispatch.",
            "Missing real benchmark evidence is a hold decision, not permission to widen rollout.",
            "Auto/default rollout remains blocked even when manual wider canary becomes eligible.",
        ],
    }


def _review_package(
    p3: Mapping[str, Any],
    single: Mapping[str, Any],
    wider: Mapping[str, Any],
    auto: Mapping[str, Any],
    benchmark_state: Mapping[str, Any],
) -> dict[str, Any]:
    real_ready = bool(benchmark_state.get("ready", False))
    wider_allowed = real_ready and bool(wider.get("ok", False)) and bool(wider.get("larger_manual_canary_allowed", False))
    hold_reasons: list[str] = []
    if not real_ready:
        hold_reasons.append(_benchmark_hold_reason(benchmark_state))
    if not bool(single.get("ok", False)):
        hold_reasons.append("single_run_explicit_canary_not_ready")
    if bool(auto.get("ok", False)):
        hold_reasons.append("auto_rollout_unexpectedly_allowed")
    decision = "manual_wider_canary_review_ready" if wider_allowed else _hold_decision(benchmark_state)
    return {
        "schema_version": 1,
        "review": "v4_promotion_review_package_v0",
        "p3_policy_contract_complete": bool(p3.get("milestone_completed", False)),
        "real_benchmark_result_ready": real_ready,
        "real_benchmark_input_present": bool(benchmark_state.get("input_present", False)),
        "real_benchmark_executed": bool(benchmark_state.get("executed", False)),
        "real_benchmark_contract_ready": bool(benchmark_state.get("contract_ready", False)),
        "real_benchmark_performance_gate_ready": bool(benchmark_state.get("performance_gate_ready", False)),
        "real_benchmark_status": str(benchmark_state.get("status") or "missing"),
        "real_benchmark_performance_blockers": list(benchmark_state.get("performance_blockers", []) or []),
        "promotion_decision": decision,
        "manual_wider_canary_allowed": wider_allowed,
        "explicit_single_run_canary_allowed": bool(single.get("explicit_canary_allowed", False)),
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "manual_review_required": True,
        "safe_modes": ["off", "observe", "explicit_single_run_canary"],
        "blocked_modes_until_review": ["default", "auto"],
        "promotion_hold_reasons": hold_reasons,
        "larger_canary_requires": [
            "representative_benchmark_matrix_result",
            "checkpoint_resume_boundary",
            "explicit_canary_policy",
            "manual_owner_review",
        ],
    }


def _rollback_policy(single: Mapping[str, Any]) -> dict[str, Any]:
    source = single.get("rollback_policy") if isinstance(single.get("rollback_policy"), Mapping) else {}
    return {
        "schema_version": 1,
        "policy": "v4_promotion_review_rollback_policy_v0",
        "fallback_authoritative": bool(source.get("fallback_authoritative", True)),
        "fallback_backend": str(source.get("fallback_backend") or "pytorch_adamw"),
        "disable_for_run_on_native_error": bool(source.get("disable_for_run_on_native_error", True)),
        "disable_for_run_on_state_sync_failure": bool(source.get("disable_for_run_on_state_sync_failure", True)),
        "disable_for_run_on_checkpoint_resume_mismatch": bool(
            source.get("disable_for_run_on_checkpoint_resume_mismatch", True)
        ),
        "disable_for_run_on_config_mismatch": bool(source.get("disable_for_run_on_config_mismatch", True)),
        "disable_for_run_on_non_finite": bool(source.get("disable_for_run_on_non_finite", True)),
        "rollback_on_resume_mismatch": bool(source.get("rollback_on_resume_mismatch", True)),
        "default_training_path_enabled": False,
    }


def _benchmark_state(source: Mapping[str, Any]) -> dict[str, Any]:
    ready = bool(source.get("real_benchmark_result_ready", False))
    performance_gate_ready = bool(source.get("real_benchmark_performance_gate_ready", ready))
    return {
        "ready": ready,
        "input_present": bool(source.get("real_benchmark_input_present", ready)),
        "executed": bool(source.get("real_benchmark_executed", ready)),
        "contract_ready": bool(source.get("real_benchmark_contract_ready", ready)),
        "performance_gate_ready": performance_gate_ready,
        "status": str(source.get("real_benchmark_status") or ("promotion_ready" if ready else "missing")),
        "performance_blockers": [
            str(item) for item in list(source.get("real_benchmark_performance_blockers", []) or [])
        ],
    }


def _benchmark_hold_reason(state: Mapping[str, Any]) -> str:
    if bool(state.get("contract_ready", False)) and not bool(state.get("ready", False)):
        return "real_benchmark_performance_gate_blocked"
    if bool(state.get("executed", False)):
        return "real_benchmark_contract_blocked"
    if bool(state.get("input_present", False)):
        return "real_benchmark_input_not_executed"
    return "real_benchmark_result_missing"


def _hold_decision(state: Mapping[str, Any]) -> str:
    if _benchmark_hold_reason(state) == "real_benchmark_performance_gate_blocked":
        return "hold_for_representative_performance_gate"
    return "hold_for_representative_benchmark"


def _wider_gate_recorded(wider: Mapping[str, Any], real_ready: bool) -> bool:
    if real_ready:
        return bool(wider.get("ok", False)) and bool(wider.get("larger_manual_canary_allowed", False))
    blockers = list(wider.get("blocked_reasons", []) or [])
    return not bool(wider.get("ok", True)) and (
        "v4_p3_real_benchmark_result_missing" in blockers
        or "v4_p3_real_benchmark_performance_gate_blocked" in blockers
    )


def _default_off(section: Mapping[str, Any]) -> bool:
    return (
        section.get("training_path_enabled") is False
        and section.get("default_training_path_enabled") is False
        and section.get("default_rollout_allowed") is False
        and section.get("auto_rollout_allowed") is False
    )


def _rollback_ready(rollback: Mapping[str, Any]) -> bool:
    return bool(
        rollback.get("fallback_authoritative", False)
        and rollback.get("disable_for_run_on_native_error", False)
        and rollback.get("disable_for_run_on_checkpoint_resume_mismatch", False)
        and rollback.get("rollback_on_resume_mismatch", False)
    )


def _recommended_next_step(review: Mapping[str, Any]) -> str:
    if review.get("manual_wider_canary_allowed") is True:
        return "manual wider canary is review-ready; default and auto remain off"
    if review.get("promotion_decision") == "hold_for_representative_performance_gate":
        return "optimize native dispatch overhead before manual wider canary"
    return "collect real V4 representative benchmark before manual wider canary"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = ["build_v4_promotion_review_scorecard"]
