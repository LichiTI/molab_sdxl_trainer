"""Blocker layering for TurboCore native update promotion reports."""

from __future__ import annotations


PRIMARY_PROMOTION_BLOCKERS = frozenset(
    {
        "native_dispatch_runtime_not_implemented",
        "native_dispatch_training_path_disabled",
        "native_dispatch_native_kernel_not_promoted",
        "native_runtime_recovery_training_dispatch_disabled",
        "training_dispatch_recovery_default_off",
        "direct_gradient_write_default_off",
        "direct_gradient_write_not_native_supported",
        "owner_gradient_sync_default_off",
        "owner_gradient_sync_not_supported",
        "owner_gradient_sync_not_training_integrated",
        "owner_gradient_sync_guard_disabled",
        "owner_gradient_sync_not_promoted",
        "native_training_flat_owner_unavailable",
        "native_training_flat_owner_default_off",
        "native_training_flat_owner_not_promoted",
        "native_training_dispatch_kernel_missing",
        "native_training_dispatch_kernel_default_off",
        "native_training_dispatch_kernel_not_promoted",
        "stream_lifetime_unbound",
        "stream_lifetime_ownership_default_off",
        "stream_lifetime_ownership_not_promoted",
        "native_dispatch_training_runtime_executor_default_off",
        "native_dispatch_training_path_default_off",
        "representative_performance_gate_missing",
        "native_update_product_exposure_decision_missing",
        "native_update_product_exposure_decision_not_ok",
        "native_update_product_exposure_evidence_not_ready",
        "native_update_product_exposure_not_ready_for_review",
        "native_update_product_exposure_request_fields_present",
        "native_update_release_review_package_missing",
        "native_update_release_review_package_not_ok",
        "native_update_release_review_evidence_not_ready",
        "native_update_release_review_not_ready_for_owner_review",
        "native_update_release_review_not_recorded",
        "native_update_release_review_expected_gates_not_present",
        "native_update_release_review_gates_not_default_off",
        "native_update_release_review_request_fields_present",
    }
)


DERIVED_PROMOTION_BLOCKERS = frozenset(
    {
        "native_recovery_keeps_dispatch_disabled",
        "native_update_gate_not_enabled",
        "native_dispatch_rehearsal_not_ready",
        "native_dispatch_contract_not_allowing_dispatch",
        "dispatch_request_not_allowed",
        "dispatch_contract_not_allowing_launch",
        "dispatch_not_armed",
        "kernel_launch_not_allowed",
        "native_step_execution_disabled",
        "native_dispatch_runtime_default_off",
        "native_dispatch_diagnostic_executor_call_disabled",
        "native_dispatch_diagnostic_executor_replay_disabled",
    }
)


EXPLICIT_TRAINING_PROMOTION_BLOCKERS = frozenset(
    {
        "native_dispatch_runtime_executor_missing",
        "native_dispatch_runtime_execution_guard_disabled",
        "native_dispatch_training_mutation_guard_disabled",
        "native_dispatch_training_path_not_requested",
    }
)


def split_promotion_blockers(
    blockers: list[str],
    *,
    promotion_dispatch: bool,
    native_step_executed: bool = False,
) -> dict[str, list[str]]:
    """Split promotion blockers into actionable primary and derived layers.

    ``promotion_blockers`` is intentionally kept as the compatibility field in
    the scorecard. This helper gives UI and probes a cleaner default view:
    primary blockers are the remaining engineering/product gates, while derived
    blockers are lower-level consequences of those gates staying closed.
    """

    primary: list[str] = []
    derived: list[str] = []
    for blocker in blockers:
        if blocker in {"native_dispatch_runtime_not_implemented", "native_dispatch_training_path_disabled"}:
            if not promotion_dispatch or native_step_executed:
                derived.append(blocker)
                continue
        if blocker in PRIMARY_PROMOTION_BLOCKERS:
            primary.append(blocker)
            continue
        if blocker in DERIVED_PROMOTION_BLOCKERS:
            derived.append(blocker)
            continue
        if blocker in EXPLICIT_TRAINING_PROMOTION_BLOCKERS and not promotion_dispatch:
            derived.append(blocker)
            continue
        primary.append(blocker)
    return {
        "primary_promotion_blockers": _dedupe(primary),
        "derived_promotion_blockers": _dedupe(derived),
    }


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = [
    "DERIVED_PROMOTION_BLOCKERS",
    "EXPLICIT_TRAINING_PROMOTION_BLOCKERS",
    "PRIMARY_PROMOTION_BLOCKERS",
    "split_promotion_blockers",
]
