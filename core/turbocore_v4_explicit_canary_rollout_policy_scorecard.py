"""V4 explicit canary rollout policy for exact AdamW native update."""

from __future__ import annotations

from typing import Any, Mapping


OPTIMIZER_KIND = "exact_adamw"
OPTIMIZER_FAMILY = "adamw_exact"
NATIVE_BACKEND = "rust_cuda_adamw_v0"
VALID_MODES = {"off", "observe", "canary", "auto"}
VALID_SCOPES = {"single_run", "wider_manual_canary"}


def build_v4_explicit_canary_rollout_policy_scorecard(
    *,
    p2_audit: Mapping[str, Any] | None = None,
    native_training_mode: str = "canary",
    requested_scope: str = "single_run",
) -> dict[str, Any]:
    """Build a default-off canary policy without dispatching training."""

    p2_ready = bool(p2_audit.get("milestone_completed", False)) if isinstance(p2_audit, Mapping) else True
    benchmark_state = _benchmark_state(p2_audit)
    real_benchmark_ready = bool(benchmark_state["ready"])
    mode = _normalize(native_training_mode, VALID_MODES, "canary")
    scope = _normalize(requested_scope, VALID_SCOPES, "single_run")
    route = _route_decision(mode, scope, p2_ready, benchmark_state)
    rollback = _rollback_policy(p2_ready)
    progress_gates = {
        "p2_checkpoint_resume_complete": p2_ready,
        "route_decision_present": bool(route.get("decision")),
        "explicit_opt_in_required": bool(route.get("requires_explicit_opt_in", False)),
        "request_fields_present": _request_fields_present(route),
        "default_and_auto_blocked": _default_blocked(route),
        "rollback_policy_ready": _rollback_ready(rollback),
        "wider_canary_requires_real_benchmark": _wider_canary_requirement_ok(scope, real_benchmark_ready, route),
        "default_behavior_unchanged": True,
    }
    policy_contract_ready = all(progress_gates.values())
    route_allowed = bool(route.get("route_allowed", False))
    blockers = _blockers(progress_gates, route)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v4_explicit_canary_rollout_policy_scorecard_v0",
        "gate": "v4_explicit_canary_rollout_policy",
        "ok": route_allowed and policy_contract_ready,
        "milestone_completed": route_allowed and policy_contract_ready,
        "policy_contract_ready": policy_contract_ready,
        "route_allowed": route_allowed,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "native_backend": NATIVE_BACKEND,
        "native_training_mode": mode,
        "requested_scope": scope,
        "real_benchmark_result_ready": real_benchmark_ready,
        "real_benchmark_input_present": bool(benchmark_state["input_present"]),
        "real_benchmark_executed": bool(benchmark_state["executed"]),
        "real_benchmark_contract_ready": bool(benchmark_state["contract_ready"]),
        "real_benchmark_performance_gate_ready": bool(benchmark_state["performance_gate_ready"]),
        "real_benchmark_status": str(benchmark_state["status"]),
        "real_benchmark_performance_blockers": list(benchmark_state["performance_blockers"]),
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_dispatch_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "requires_explicit_opt_in": True,
        "explicit_canary_allowed": bool(route.get("explicit_canary_allowed", False)),
        "larger_manual_canary_allowed": bool(route.get("larger_manual_canary_allowed", False)),
        "route_decision": route,
        "rollback_policy": rollback,
        "progress_gates": progress_gates,
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": _recommended_next_step(route, real_benchmark_ready),
        "notes": [
            "This is a policy contract, not a dispatcher.",
            "Single-run canary remains explicit opt-in only.",
            "Wider manual canary is blocked until representative benchmark evidence is present.",
            "Default and auto rollout stay disabled regardless of policy readiness.",
        ],
    }


def _route_decision(
    mode: str,
    scope: str,
    p2_ready: bool,
    benchmark_state: Mapping[str, Any],
) -> dict[str, Any]:
    real_benchmark_ready = bool(benchmark_state.get("ready", False))
    benchmark_perf_blocked = (
        bool(benchmark_state.get("input_present", False))
        and bool(benchmark_state.get("executed", False))
        and bool(benchmark_state.get("contract_ready", False))
        and not real_benchmark_ready
    )
    if mode == "off":
        decision = "off"
        reason = "native_training_mode_off"
        route_allowed = False
    elif not p2_ready:
        decision = "blocked"
        reason = "v4_p2_checkpoint_resume_missing"
        route_allowed = False
    elif mode == "observe":
        decision = "observe_ready_but_default_off"
        reason = "observe_requires_explicit_dev_opt_in"
        route_allowed = True
    elif mode == "auto":
        decision = "auto_blocked"
        reason = "auto_rollout_blocked_for_v4"
        route_allowed = False
    elif scope == "wider_manual_canary" and benchmark_perf_blocked:
        decision = "wider_canary_blocked_until_performance_gate"
        reason = "representative_benchmark_performance_gate_blocked"
        route_allowed = False
    elif scope == "wider_manual_canary" and not real_benchmark_ready:
        decision = "wider_canary_blocked_until_real_benchmark"
        reason = "representative_benchmark_result_missing"
        route_allowed = False
    elif scope == "wider_manual_canary":
        decision = "wider_manual_canary_ready"
        reason = "explicit_manual_canary_after_representative_benchmark"
        route_allowed = True
    else:
        decision = "explicit_single_run_canary_ready"
        reason = "explicit_single_run_canary_requires_dev_opt_in"
        route_allowed = True
    return {
        "schema_version": 1,
        "feature": "v4_exact_adamw_native_update",
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "native_backend": NATIVE_BACKEND,
        "native_training_mode": mode,
        "requested_scope": scope,
        "decision": decision,
        "reason": reason,
        "route_allowed": route_allowed,
        "requires_explicit_opt_in": True,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_dispatch_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "explicit_canary_allowed": decision in {"explicit_single_run_canary_ready", "wider_manual_canary_ready"},
        "larger_manual_canary_allowed": decision == "wider_manual_canary_ready",
        "request_fields": {
            "turbocoreExactAdamwCanary": True,
            "turbocoreNativeUpdateCanary": True,
            "turbocoreNativeUpdateCanaryOptimizer": OPTIMIZER_KIND,
            "turbocoreNativeUpdateCanaryScope": scope,
            "turbocore_native_update_mode": "native_experimental",
            "turbocore_native_update_dispatch_enabled": True,
            "turbocore_native_update_training_path_enabled": True,
            "turbocore_native_update_require_native_cuda": True,
        },
        "missing_before_wider_canary": [] if real_benchmark_ready or benchmark_perf_blocked else [
            "representative_benchmark_matrix_result"
        ],
        "blocked_before_wider_canary": list(benchmark_state.get("performance_blockers", []) or [])
        if benchmark_perf_blocked
        else [],
        "missing_before_auto": [
            "representative_benchmark_matrix_result",
            "manual_owner_review",
            "v4_p4_promotion_review",
        ],
    }


def _rollback_policy(p2_ready: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy": "v4_explicit_canary_rollout_rollback_policy_v0",
        "fallback_authoritative": True,
        "fallback_backend": "pytorch_adamw",
        "p2_checkpoint_resume_ready": bool(p2_ready),
        "disable_for_run_on_native_error": True,
        "disable_for_run_on_state_sync_failure": True,
        "disable_for_run_on_checkpoint_resume_mismatch": True,
        "disable_for_run_on_config_mismatch": True,
        "disable_for_run_on_non_finite": True,
        "rollback_on_resume_mismatch": True,
        "rollback_on_optimizer_kind_mismatch": True,
        "rollback_on_native_backend_mismatch": True,
        "default_training_path_enabled": False,
    }


def _request_fields_present(route: Mapping[str, Any]) -> bool:
    fields = route.get("request_fields") if isinstance(route.get("request_fields"), Mapping) else {}
    return bool(
        fields.get("turbocoreExactAdamwCanary") is True
        and fields.get("turbocoreNativeUpdateCanary") is True
        and fields.get("turbocoreNativeUpdateCanaryOptimizer") == OPTIMIZER_KIND
        and fields.get("turbocore_native_update_mode") == "native_experimental"
        and fields.get("turbocore_native_update_dispatch_enabled") is True
        and fields.get("turbocore_native_update_training_path_enabled") is True
        and fields.get("turbocore_native_update_require_native_cuda") is True
    )


def _default_blocked(route: Mapping[str, Any]) -> bool:
    return (
        route.get("default_training_path_enabled") is False
        and route.get("training_path_enabled") is False
        and route.get("default_dispatch_allowed") is False
        and route.get("default_rollout_allowed") is False
        and route.get("auto_rollout_allowed") is False
    )


def _rollback_ready(rollback: Mapping[str, Any]) -> bool:
    return bool(
        rollback.get("fallback_authoritative", False)
        and rollback.get("disable_for_run_on_native_error", False)
        and rollback.get("disable_for_run_on_state_sync_failure", False)
        and rollback.get("disable_for_run_on_checkpoint_resume_mismatch", False)
        and rollback.get("rollback_on_resume_mismatch", False)
    )


def _wider_canary_requirement_ok(scope: str, real_benchmark_ready: bool, route: Mapping[str, Any]) -> bool:
    if scope != "wider_manual_canary":
        return True
    if real_benchmark_ready:
        return route.get("decision") == "wider_manual_canary_ready"
    return route.get("decision") == "wider_canary_blocked_until_real_benchmark"


def _blockers(progress_gates: Mapping[str, bool], route: Mapping[str, Any]) -> list[str]:
    blockers = [f"v4_p3_{name}_missing" for name, ok in progress_gates.items() if not ok]
    decision = str(route.get("decision") or "")
    if decision == "off":
        blockers.append("v4_p3_native_training_mode_off")
    elif decision == "blocked":
        blockers.append("v4_p3_checkpoint_resume_missing")
    elif decision == "auto_blocked":
        blockers.append("v4_p3_auto_rollout_blocked")
    elif decision == "wider_canary_blocked_until_performance_gate":
        blockers.append("v4_p3_real_benchmark_performance_gate_blocked")
        blockers.extend(str(item) for item in list(route.get("blocked_before_wider_canary", []) or []))
    elif decision == "wider_canary_blocked_until_real_benchmark":
        blockers.append("v4_p3_real_benchmark_result_missing")
    return _dedupe(blockers)


def _recommended_next_step(route: Mapping[str, Any], real_benchmark_ready: bool) -> str:
    decision = str(route.get("decision") or "")
    if decision == "wider_manual_canary_ready":
        return "proceed to V4-P4 promotion review with manual wider canary evidence"
    if decision == "wider_canary_blocked_until_performance_gate":
        return "optimize native dispatch overhead; representative benchmark ran but performance gate is blocked"
    if decision == "explicit_single_run_canary_ready":
        return "single-run explicit canary is ready; collect real V4 representative benchmark before wider canary"
    if not real_benchmark_ready:
        return "run V4 representative benchmark matrix before wider canary"
    return "keep default off and complete V4-P4 promotion review"


def _normalize(value: str, allowed: set[str], default: str) -> str:
    normalized = str(value or default).strip().lower()
    return normalized if normalized in allowed else default


def _benchmark_state(source: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        return {
            "ready": False,
            "input_present": False,
            "executed": False,
            "contract_ready": False,
            "performance_gate_ready": False,
            "status": "missing",
            "performance_blockers": [],
        }
    ready = bool(source.get("real_benchmark_result_ready", False))
    performance_gate_ready = bool(source.get("real_benchmark_performance_gate_ready", ready))
    blockers = [str(item) for item in list(source.get("real_benchmark_performance_blockers", []) or [])]
    return {
        "ready": ready,
        "input_present": bool(source.get("real_benchmark_input_present", ready)),
        "executed": bool(source.get("real_benchmark_executed", ready)),
        "contract_ready": bool(source.get("real_benchmark_contract_ready", ready)),
        "performance_gate_ready": performance_gate_ready,
        "status": str(source.get("real_benchmark_status") or ("promotion_ready" if ready else "missing")),
        "performance_blockers": blockers,
    }


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_v4_explicit_canary_rollout_policy_scorecard"]
