"""Runtime recovery hardening gate for V3 exact AdamW native canary."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_native_update_dispatch_runtime import TurboCoreNativeUpdateDispatchRuntime
from core.turbocore_native_update_recovery import build_native_update_runtime_recovery_policy
from core.turbocore_v3_exact_adamw_short_matrix_scorecard import (
    build_v3_exact_adamw_short_matrix_scorecard,
)


NATIVE_BACKEND = "rust_cuda_adamw_v0"


def build_v3_exact_adamw_runtime_recovery_scorecard(
    *,
    p2_short_matrix: Mapping[str, Any] | None = None,
    run_live_training: bool = True,
) -> dict[str, Any]:
    """Build a recovery hardening scorecard around the existing runtime facade."""

    p2 = dict(
        p2_short_matrix
        or build_v3_exact_adamw_short_matrix_scorecard(run_live_training=run_live_training)
    )
    cases = {
        "runtime_error": _latch_case("runtime_error", "native_runtime_error_observed"),
        "state_mismatch": _state_mismatch_case(),
        "resume_mismatch": _latch_case("resume_mismatch", "native_resume_mismatch_observed"),
        "optimizer_state_sync_failure": _latch_case(
            "optimizer_state_sync_failure",
            "native_optimizer_state_sync_failure",
        ),
        "shadow_autostop_skip": _shadow_autostop_skip_case(),
        "clean_policy": _clean_policy_case(),
    }
    progress_gates = {
        "p2_short_matrix_complete": bool(p2.get("short_matrix_ready", False)),
        "runtime_error_latches": _case_latched(cases["runtime_error"]),
        "state_mismatch_latches": _case_latched(cases["state_mismatch"]),
        "resume_mismatch_latches": _case_latched(cases["resume_mismatch"]),
        "optimizer_state_sync_failure_latches": _case_latched(cases["optimizer_state_sync_failure"]),
        "shadow_autostop_skip_not_latched": not bool(
            cases["shadow_autostop_skip"].get("disabled_for_run", True)
        ),
        "clean_policy_not_latched": not bool(cases["clean_policy"].get("disabled_for_run", True))
        and bool(cases["clean_policy"].get("training_dispatch_recovery_ready", False)),
        "default_behavior_unchanged": (
            not bool(p2.get("default_behavior_changed", True))
            and not bool(p2.get("default_training_path_enabled", True))
        ),
    }
    ready = all(progress_gates.values())
    blockers = [f"v3_p3_{name}_missing" for name, ok in progress_gates.items() if not ok]
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v3_exact_adamw_runtime_recovery_scorecard_v0",
        "gate": "v3_exact_adamw_runtime_recovery_hardening",
        "ok": bool(p2.get("ok", False)) and all(bool(case.get("ok", False)) for case in cases.values()),
        "milestone_completed": ready,
        "runtime_recovery_hardened": ready,
        "native_backend": NATIVE_BACKEND,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "p2_summary": {
            "short_matrix_ready": bool(p2.get("short_matrix_ready", False)),
            "milestone_completed": bool(p2.get("milestone_completed", False)),
        },
        "recovery_cases": cases,
        "progress_gates": progress_gates,
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add V3 exact AdamW canary config adapter on the request boundary"
            if ready
            else "complete V3-P3 runtime recovery hardening blockers"
        ),
        "notes": [
            "Recovery is evaluated through the existing runtime facade, not a new training entry.",
            "Error and mismatch cases must latch native dispatch off for the current run.",
            "Clean policy and shadow autostop skip must not poison future explicit canary steps.",
        ],
    }


def _latch_case(name: str, reason: str) -> dict[str, Any]:
    runtime = TurboCoreNativeUpdateDispatchRuntime()
    policy = {
        "schema_version": 1,
        "policy": "v3_exact_adamw_direct_recovery_policy_v0",
        "mode": "native_experimental",
        "policy_defined": True,
        "disable_native_update_for_run": True,
        "runtime": {"runtime_error_observed": reason == "native_runtime_error_observed"},
        "state_safety": {"state_mismatch_observed": reason != "native_runtime_error_observed"},
        "blocked_reasons": [reason],
    }
    observation = runtime.observe_recovery_policy(policy)
    next_report = _prepare_after_observation(runtime)
    return _case_report(name, reason, policy, observation, next_report)


def _state_mismatch_case() -> dict[str, Any]:
    runtime = TurboCoreNativeUpdateDispatchRuntime()
    policy = build_native_update_runtime_recovery_policy(
        mode="native_experimental",
        shadow_report={
            "owner_native_launch_probe": {
                "attempted": True,
                "ok": True,
                "kernel_executed": True,
                "native_launch_attempted": True,
                "native_launch_ok": True,
                "parity_ok": False,
                "persistent_owner_mutated": False,
            }
        },
        runtime_context=_explicit_runtime_context(),
    )
    observation = runtime.observe_recovery_policy(policy)
    next_report = _prepare_after_observation(runtime)
    return _case_report("state_mismatch", "native_state_mismatch_observed", policy, observation, next_report)


def _shadow_autostop_skip_case() -> dict[str, Any]:
    runtime = TurboCoreNativeUpdateDispatchRuntime()
    policy = build_native_update_runtime_recovery_policy(
        mode="native_experimental",
        shadow_report={
            "reason": "auto_stopped_after_consecutive_passes",
            "after_optimizer": {
                "skipped": True,
                "reason": "auto_stopped_after_consecutive_passes",
            },
        },
        runtime_context=_explicit_runtime_context(),
    )
    observation = runtime.observe_recovery_policy(policy)
    return {
        "schema_version": 1,
        "case": "shadow_autostop_skip",
        "ok": bool(policy.get("disable_native_update_for_run", True)) is False
        and bool(observation.get("disabled_for_run", True)) is False,
        "disabled_for_run": bool(observation.get("disabled_for_run", False)),
        "policy_disable_requested": bool(policy.get("disable_native_update_for_run", False)),
        "training_dispatch_recovery_ready": bool(policy.get("training_dispatch_recovery_ready", False)),
        "blocked_reasons": list(policy.get("blocked_reasons", []) or []),
        "observation": observation,
    }


def _clean_policy_case() -> dict[str, Any]:
    runtime = TurboCoreNativeUpdateDispatchRuntime()
    policy = build_native_update_runtime_recovery_policy(
        mode="native_experimental",
        shadow_report={"owner_native_launch_probe": {"attempted": False, "ok": True}},
        runtime_context=_explicit_runtime_context(),
    )
    observation = runtime.observe_recovery_policy(policy)
    return {
        "schema_version": 1,
        "case": "clean_policy",
        "ok": bool(policy.get("disable_native_update_for_run", True)) is False
        and bool(observation.get("disabled_for_run", True)) is False
        and bool(policy.get("training_dispatch_recovery_ready", False)),
        "disabled_for_run": bool(observation.get("disabled_for_run", False)),
        "policy_disable_requested": bool(policy.get("disable_native_update_for_run", False)),
        "training_dispatch_recovery_ready": bool(policy.get("training_dispatch_recovery_ready", False)),
        "blocked_reasons": list(policy.get("blocked_reasons", []) or []),
        "observation": observation,
    }


def _case_report(
    name: str,
    reason: str,
    policy: Mapping[str, Any],
    observation: Mapping[str, Any],
    next_report: Mapping[str, Any],
) -> dict[str, Any]:
    blocked = set(next_report.get("blocked_reasons", []) or [])
    latched = bool(observation.get("disabled_for_run", False)) and "native_dispatch_disabled_for_run" in blocked
    return {
        "schema_version": 1,
        "case": name,
        "ok": latched
        and not bool(next_report.get("native_step_executed", True))
        and bool(next_report.get("should_call_pytorch_optimizer_step", False)),
        "reason": reason,
        "disabled_for_run": bool(observation.get("disabled_for_run", False)),
        "disable_reason": str(observation.get("disable_reason", "") or ""),
        "next_step_native_executed": bool(next_report.get("native_step_executed", False)),
        "next_step_calls_pytorch": bool(next_report.get("should_call_pytorch_optimizer_step", False)),
        "blocked_reasons": list(next_report.get("blocked_reasons", []) or []),
        "policy": dict(policy),
        "observation": dict(observation),
        "next_step_report": dict(next_report),
    }


def _prepare_after_observation(runtime: TurboCoreNativeUpdateDispatchRuntime) -> dict[str, Any]:
    return runtime.prepare_step(
        step=99,
        arming_report={
            "previous_request_requested": True,
            "armed_for_native_dispatch": True,
            "execute_native_step": True,
        },
        kernel_launch_plan={"launch_allowed": True, "launch_attempted": False},
        runtime_context=_explicit_runtime_context(),
        native_executor=lambda _request: {
            "ok": True,
            "native_step_executed": True,
            "native_kernel_launched": True,
            "training_parameters_mutated": True,
            "should_call_pytorch_optimizer_step": False,
        },
    )


def _explicit_runtime_context() -> dict[str, Any]:
    return {
        "native_update_executor_present": True,
        "native_update_runtime_execution_guard_enabled": True,
        "native_update_training_mutation_guard_enabled": True,
        "native_update_training_dispatch_enabled": True,
        "native_update_runtime_dispatch_available": True,
        "training_path_enabled": True,
    }


def _case_latched(case: Mapping[str, Any]) -> bool:
    return bool(case.get("ok", False)) and bool(case.get("disabled_for_run", False))


__all__ = ["build_v3_exact_adamw_runtime_recovery_scorecard"]
