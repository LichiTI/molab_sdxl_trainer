"""Report-only recovery integration status for TurboCore native updates."""

from __future__ import annotations

from typing import Any, Mapping


TRAINING_RECOVERY_BLOCKER = "native_runtime_recovery_training_dispatch_disabled"
LEGACY_RECOVERY_BLOCKER = "native_runtime_recovery_dispatch_not_integrated"


def build_native_update_recovery_integration_report(
    *,
    mode: str,
    policy_defined: bool,
    disable_native_update_for_run: bool = False,
    runtime_error_observed: bool = False,
    state_mismatch_observed: bool = False,
    runtime_state: Mapping[str, Any] | None = None,
    runtime_observation: Mapping[str, Any] | None = None,
    runtime_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the default-off bridge between recovery policy and runtime latch."""

    state = _as_dict(runtime_state)
    observation = _as_dict(runtime_observation)
    context = _as_dict(runtime_context)
    disabled_for_run = bool(state.get("disabled_for_run", False) or observation.get("disabled_for_run", False))
    disable_reason = str(state.get("disable_reason", "") or observation.get("disable_reason", "") or "")
    recovery_requested = bool(disable_native_update_for_run or runtime_error_observed or state_mismatch_observed)
    observed = bool(observation.get("recovery_policy_present", False))
    default_off_bridge_ready = bool(policy_defined)
    explicit_training_context = bool(
        _normalize_mode(mode) == "native_experimental"
        and context.get("native_update_training_dispatch_enabled", False)
        and context.get("training_path_enabled", False)
        and context.get("native_update_runtime_dispatch_available", False)
        and context.get("native_update_runtime_execution_guard_enabled", False)
        and context.get("native_update_training_mutation_guard_enabled", False)
    )
    training_dispatch_recovery_ready = bool(
        default_off_bridge_ready
        and explicit_training_context
        and not recovery_requested
        and not disabled_for_run
    )
    blocked = [] if training_dispatch_recovery_ready else [TRAINING_RECOVERY_BLOCKER]
    if recovery_requested and observed and not disabled_for_run:
        blocked.append("native_runtime_recovery_latch_not_active")
    return {
        "schema_version": 1,
        "integration": "turbocore_native_update_recovery_integration_v0",
        "mode": _normalize_mode(mode),
        "training_dispatch": training_dispatch_recovery_ready,
        "training_path_enabled": training_dispatch_recovery_ready,
        "pytorch_optimizer_authoritative": not training_dispatch_recovery_ready,
        "policy_defined": bool(policy_defined),
        "recovery_policy_observation_integrated": True,
        "run_disable_latch_integrated": True,
        "pre_step_arming_observes_latch": True,
        "default_off_recovery_bridge_ready": default_off_bridge_ready,
        "recovery_observation_bridge_ready": default_off_bridge_ready,
        "explicit_training_context_requested": explicit_training_context,
        "training_dispatch_recovery_ready": training_dispatch_recovery_ready,
        "training_dispatch_recovery_blocked": not training_dispatch_recovery_ready,
        "training_dispatch_recovery_blocker": "" if training_dispatch_recovery_ready else TRAINING_RECOVERY_BLOCKER,
        "dispatch_integration_ready": True,
        "disable_native_update_for_run_requested": bool(disable_native_update_for_run),
        "runtime_error_observed": bool(runtime_error_observed),
        "state_mismatch_observed": bool(state_mismatch_observed),
        "recovery_requested": recovery_requested,
        "runtime_observation_present": observed,
        "runtime_disabled_for_run": disabled_for_run,
        "runtime_disable_reason": disable_reason,
        "actions": _actions(
            recovery_requested=recovery_requested,
            training_dispatch_recovery_ready=training_dispatch_recovery_ready,
        ),
        "blocked_reasons": _dedupe(blocked),
    }


def _actions(*, recovery_requested: bool, training_dispatch_recovery_ready: bool) -> list[str]:
    actions = [
        "observe_recovery_policy_before_dispatch_arming",
        "latch_native_update_disabled_for_run_when_recovery_requests_it",
    ]
    if training_dispatch_recovery_ready:
        actions.append("native_training_dispatch_recovery_bridge_ready")
    else:
        actions.extend(
            [
                "keep_pytorch_optimizer_authoritative",
                "keep_training_dispatch_disabled_until_native_recovery_can_restore_state",
            ]
        )
    if recovery_requested:
        actions.append("require_shadow_parity_revalidation_after_latched_recovery")
    return actions


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalize_mode(value: str) -> str:
    normalized = str(value or "off").strip().lower().replace("-", "_")
    return normalized if normalized in {"off", "profile", "native_experimental"} else "off"


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = [
    "LEGACY_RECOVERY_BLOCKER",
    "TRAINING_RECOVERY_BLOCKER",
    "build_native_update_recovery_integration_report",
]
