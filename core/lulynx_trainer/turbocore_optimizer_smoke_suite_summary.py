"""Summary helpers for the profiled TurboCore optimizer smoke suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


STREAM_LIFETIME_SUMMARY_KEYS = tuple(
    "stream_lifetime_lease_ready_count "
    "stream_lifetime_lease_blocked_case_count "
    "stream_lifetime_default_behavior_changed_count "
    "stream_lifetime_requires_explicit_opt_in_count "
    "stream_lifetime_training_path_enabled_count "
    "stream_lifetime_request_training_path_enabled_count".split()
)

EXACT_ADAMW_TRAINING_LOOP_SUMMARY_KEYS = (
    "exact_adamw_training_loop_canary_case_count",
    "exact_adamw_training_loop_canary_skipped_count",
    "exact_adamw_training_loop_native_dispatch_ready_count",
    "exact_adamw_training_loop_native_kernel_launched_count",
    "exact_adamw_training_loop_pytorch_optimizer_step_skipped_count",
    "exact_adamw_training_loop_optimizer_state_synced_count",
    "exact_adamw_training_loop_direct_grad_ready_count",
    "exact_adamw_training_loop_direct_grad_hook_write_count",
    "exact_adamw_training_loop_direct_grad_written_numel",
    "exact_adamw_training_loop_checkpoint_owner_state_included_count",
    "exact_adamw_training_loop_checkpoint_roundtrip_ok_count",
    "exact_adamw_training_loop_checkpoint_resume_compatible_count",
    "exact_adamw_training_loop_checkpoint_resume_mismatch_rejected_count",
    "exact_adamw_training_loop_checkpoint_training_path_enabled_count",
    "exact_adamw_training_loop_canary_training_path_enabled_count",
    "exact_adamw_training_loop_product_default_changed_count",
)

EXACT_ADAMW_RECOVERY_SUMMARY_KEYS = tuple(
    "exact_adamw_recovery_canary_case_count "
    "exact_adamw_recovery_canary_skipped_count "
    "exact_adamw_recovery_failure_latched_count "
    "exact_adamw_recovery_failure_fallback_pytorch_step_count "
    "exact_adamw_recovery_post_latch_pytorch_fallback_count "
    "exact_adamw_recovery_fallback_state_sync_call_count "
    "exact_adamw_recovery_optimizer_step_after_recovery_count "
    "exact_adamw_recovery_checkpoint_owner_state_included_count "
    "exact_adamw_recovery_checkpoint_epoch_owner_state_included_count "
    "exact_adamw_recovery_checkpoint_resume_mismatch_rejected_count "
    "exact_adamw_recovery_deferred_native_success_count "
    "exact_adamw_recovery_deferred_optimizer_state_sync_deferred_count "
    "exact_adamw_recovery_deferred_fallback_state_sync_flush_count "
    "exact_adamw_recovery_deferred_failure_latched_count "
    "exact_adamw_recovery_deferred_post_latch_pytorch_fallback_count "
    "exact_adamw_recovery_deferred_epoch_state_sync_flush_count "
    "exact_adamw_recovery_deferred_epoch_state_sync_step_count "
    "exact_adamw_recovery_deferred_epoch_close_already_clean_count "
    "exact_adamw_recovery_deferred_epoch_optimizer_step_synced_count "
    "exact_adamw_recovery_deferred_epoch_sync_failure_latched_count "
    "exact_adamw_recovery_deferred_epoch_sync_failure_dirty_state_retained_count "
    "exact_adamw_recovery_deferred_epoch_sync_failure_optimizer_step_stale_count "
    "exact_adamw_recovery_deferred_state_save_flush_count "
    "exact_adamw_recovery_deferred_state_save_optimizer_step_synced_count "
    "exact_adamw_recovery_deferred_state_save_owner_state_included_count "
    "exact_adamw_recovery_deferred_state_save_checkpoint_contract_integrated_count "
    "exact_adamw_recovery_training_path_enabled_count".split()
)

EXACT_ADAMW_RESUME_STATE_GUARD_SUMMARY_KEYS = tuple(
    "exact_adamw_resume_state_guard_canary_case_count "
    "exact_adamw_resume_state_guard_canary_skipped_count "
    "exact_adamw_resume_state_guard_compatible_loaded_count "
    "exact_adamw_resume_state_guard_owner_state_pending_count "
    "exact_adamw_resume_state_guard_pending_consumed_count "
    "exact_adamw_resume_state_guard_native_step_after_resume_count "
    "exact_adamw_resume_state_guard_owner_step_restored_count "
    "exact_adamw_resume_state_guard_checkpoint_contract_integrated_count "
    "exact_adamw_resume_state_guard_mismatch_rejected_count "
    "exact_adamw_resume_state_guard_mismatch_pending_blocked_count "
    "exact_adamw_resume_state_guard_file_roundtrip_loaded_count "
    "exact_adamw_resume_state_guard_file_roundtrip_pending_consumed_count "
    "exact_adamw_resume_state_guard_file_roundtrip_native_step_count "
    "exact_adamw_resume_state_guard_training_path_enabled_count".split()
)
TRAINING_LOOP_SUMMARY_KEYS = (
    EXACT_ADAMW_TRAINING_LOOP_SUMMARY_KEYS
    + EXACT_ADAMW_RECOVERY_SUMMARY_KEYS
    + EXACT_ADAMW_RESUME_STATE_GUARD_SUMMARY_KEYS
)

NATIVE_UPDATE_EXECUTION_LADDER_SMOKE_IDS = (
    "native_update_training_dispatch_integration_contract",
    "native_update_activation_contract",
    "native_update_runtime_execution_contract",
    "native_update_runtime_dispatch_contract",
    "native_update_native_dispatch_execution_contract",
    "native_update_kernel_launch_execution_contract",
    "native_update_parity_execution_contract",
    "native_update_training_step_execution_contract",
    "native_update_training_launch_contract",
)


def payload_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"payload_type": type(payload).__name__}
    keys = (
        "probe",
        "ok",
        "skipped",
        "roadmap",
        "artifact_mode",
        "case_count",
        "optimizer_count",
        "selected_plugin_optimizer_case_count",
        "family_specific_runtime_launch_adapter_ready_count",
        "runtime_dispatch_ready_count",
        "runtime_precondition_ready_count",
        "milestone_completed",
        "product_exposure_decision_recorded_count",
        "ready_for_owner_signature",
        "release_review_recorded",
        "ready_for_product_exposure_review",
        "product_exposure_review_action_required",
        "product_launch_staging_wired_count",
        "stable_first_release_turbocore_optimizer_blocker_count",
        "turbocore_optimizer_default_off_release_scope_ready_count",
        "recommended_next_step",
        "pending_decision",
        "signed_decision",
        "real_artifact_checked",
        "p6_ready_gate_count",
        "p6_required_gate_count",
        "p6_remaining_blocker_count",
        "p6_milestone_completed_count",
        "p6_section_count",
        "p6_required_promotion_count",
        "p6_recommended_next_step",
        "synthetic_owner_direction_end_to_end_validated",
        "real_artifact_approval_recorded_count",
        "actual_training_complete",
        "roadmap_v2_open_work",
        "default_off_product_summary",
        "signed_post_approval_preview_summary",
    )
    summary = {key: payload[key] for key in keys if key in payload}
    nested = payload.get("summary")
    if isinstance(nested, dict):
        for key in _NESTED_SUMMARY_KEYS:
            if key in nested:
                summary[key] = nested[key]
    post_approval = payload.get("post_approval_summary")
    if isinstance(post_approval, dict):
        for key in (
            "post_approval_ready",
            "route_binding_preflight_ready",
            "candidate_ready",
            "runtime_dispatch_ready_family_count",
            "runtime_dispatch_ready_optimizer_count",
            "native_dispatch_allowed_family_count",
            "native_dispatch_allowed_optimizer_count",
            "training_path_enabled_family_count",
            "training_path_enabled_optimizer_count",
            "product_native_ready_family_count",
            "product_native_ready_optimizer_count",
        ):
            if key in post_approval:
                summary[f"post_approval_{key}" if key != "post_approval_ready" else "post_approval_ready"] = post_approval[key]
    return summary


def suite_status_summary(
    results: list[dict[str, Any]],
    *,
    roadmap: str,
    script_root: Path,
    registered_smoke_count: int,
    turbocore_related_smoke_file_count: int,
    specialized_individual_turbocore_smoke_file_count: int,
    include_group_policies: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    payloads = [
        item.get("payload_summary")
        for item in results
        if item.get("ok") and isinstance(item.get("payload_summary"), dict)
    ]
    group_policies = include_group_policies or {}
    real_reviewer_groups = _policy_groups(group_policies, "real_reviewer_input_required", True)
    real_record_gate_groups = _policy_groups(
        group_policies,
        "release_evidence_role",
        "real_record_gate_validation_only",
    )
    real_record_gate_confirmation_groups = _policy_groups(
        group_policies,
        "execution_confirmation_flag",
        "--allow-real-record-gate",
    )
    real_record_gate_confirmation_flags = _policy_values(
        group_policies,
        "execution_confirmation_flag",
    )
    real_record_input_manifest_groups = _policy_groups(
        group_policies,
        "real_record_input_manifest_required",
        True,
    )
    real_record_input_manifest_flags = _policy_values(
        group_policies,
        "real_record_input_manifest_flag",
    )
    real_record_input_manifest_schemas = _policy_values(
        group_policies,
        "real_record_input_manifest_schema",
    )
    approval_artifact_groups = _policy_groups(group_policies, "approval_artifact_allowed", True)
    real_record_gate_default_blocked_groups = real_record_gate_confirmation_groups
    real_record_gate_default_allowed_count = max(
        0,
        len(real_record_gate_groups) - len(real_record_gate_default_blocked_groups),
    )
    ladder_payloads = _payloads_for_smoke_ids(results, NATIVE_UPDATE_EXECUTION_LADDER_SMOKE_IDS)
    ladder_ready_count = _passed_smoke_count(results, NATIVE_UPDATE_EXECUTION_LADDER_SMOKE_IDS)
    return {
        "roadmap": roadmap,
        "profile_entrypoint": "turbocore_optimizer_smoke_suite.py",
        "selected_include_group_policy_count": len(group_policies),
        "selected_real_reviewer_input_required_count": len(real_reviewer_groups),
        "selected_real_reviewer_input_required_groups": real_reviewer_groups,
        "selected_real_record_gate_validation_only_count": len(real_record_gate_groups),
        "selected_real_record_gate_validation_only_groups": real_record_gate_groups,
        "selected_real_record_gate_confirmation_required_count": len(real_record_gate_confirmation_groups),
        "selected_real_record_gate_confirmation_required_groups": real_record_gate_confirmation_groups,
        "selected_real_record_gate_confirmation_flag_count": len(real_record_gate_confirmation_flags),
        "selected_real_record_gate_confirmation_flags": real_record_gate_confirmation_flags,
        "selected_real_record_input_manifest_required_count": len(real_record_input_manifest_groups),
        "selected_real_record_input_manifest_required_groups": real_record_input_manifest_groups,
        "selected_real_record_input_manifest_flag_count": len(real_record_input_manifest_flags),
        "selected_real_record_input_manifest_flags": real_record_input_manifest_flags,
        "selected_real_record_input_manifest_schema_count": len(real_record_input_manifest_schemas),
        "selected_real_record_input_manifest_schemas": real_record_input_manifest_schemas,
        "selected_real_record_gate_execution_allowed_by_default_count": real_record_gate_default_allowed_count,
        "selected_real_record_gate_execution_blocked_by_default_count": len(real_record_gate_default_blocked_groups),
        "selected_real_record_gate_execution_blocked_by_default_groups": real_record_gate_default_blocked_groups,
        "selected_approval_artifact_allowed_group_count": len(approval_artifact_groups),
        "selected_approval_artifact_allowed_groups": approval_artifact_groups,
        "turbocore_related_smoke_file_count": turbocore_related_smoke_file_count,
        "optimizer_suite_registered_smoke_count": registered_smoke_count,
        "factored_custom_optimizer_count": _max_int(
            [payload for payload in payloads if _probe(payload) == "turbocore_factored_custom_optimizer_family_batch_scorecard_smoke"],
            "optimizer_count",
        ),
        "factored_custom_training_loop_canary_ready_count": _max_int(payloads, "training_loop_canary_ready_count"),
        "factored_custom_dispatch_review_ready_count": _max_int(payloads, "dispatch_integration_review_ready_count"),
        "specialized_individual_turbocore_smoke_file_count": specialized_individual_turbocore_smoke_file_count,
        "selected_plugin_optimizer_count": _max_int(
            payloads,
            "selected_plugin_optimizer_count",
            "plugin_optimizer_count",
            "optimizer_count",
        ),
        "selected_plugin_family_count": _max_int(
            payloads,
            "selected_plugin_family_count",
            "family_count",
            "case_count",
            "required_family_present_count",
            "optimizer_family_contract_count",
        ),
        "optimizer_family_kernel_contract_ready_count": _max_int(
            payloads,
            "optimizer_family_contract_count",
            "optimizer_family_kernel_contract_ready_count",
        ),
        "native_readiness_family_evidence_ready_count": _max_int(payloads, "family_evidence_ready_count"),
        "native_readiness_runtime_rehearsal_ready_family_count": _max_int(payloads, "runtime_rehearsal_ready_family_count"),
        "native_readiness_runtime_precondition_ready_family_count": _max_int(payloads, "runtime_precondition_ready_family_count"),
        "native_readiness_runtime_launch_adapter_ready_family_count": _max_int(payloads, "family_specific_runtime_launch_adapter_ready_family_count"),
        "native_readiness_runtime_launch_adapter_ready_optimizer_count": _max_int(payloads, "family_specific_runtime_launch_adapter_ready_optimizer_count", "family_specific_runtime_launch_adapter_ready_count"),
        "roadmap_v2_open_work_category_count": _max_int(payloads, "roadmap_v2_open_work_category_count"),
        "roadmap_v2_open_work_item_count": _max_int(payloads, "roadmap_v2_open_work_item_count"),
        "roadmap_v2_open_work_open_category_count": _max_int(payloads, "roadmap_v2_open_work_open_category_count"),
        "roadmap_v2_open_work_open_item_count": _max_int(payloads, "roadmap_v2_open_work_open_item_count"),
        "roadmap_v2_open_work": _first_value(payloads, "roadmap_v2_open_work"),
        "family_follow_up_family_count": _max_int(payloads, "family_follow_up_family_count"),
        "family_follow_up_base_canary_ready_count": _max_int(payloads, "family_follow_up_base_canary_ready_count"),
        "family_follow_up_remaining_branch_count": _max_int(payloads, "family_follow_up_remaining_branch_count"),
        "family_follow_up_native_step_count": _max_int(payloads, "family_follow_up_native_step_count"),
        "family_follow_up_native_kernel_launch_count": _max_int(payloads, "family_follow_up_native_kernel_launch_count"),
        "family_follow_up_branch_contract_tracked_count": _max_int(payloads, "family_follow_up_branch_contract_tracked_count"),
        "family_follow_up_branch_reference_ready_count": _max_int(payloads, "family_follow_up_branch_reference_ready_count"),
        "family_follow_up_branch_implementation_ready_count": _max_int(payloads, "family_follow_up_branch_implementation_ready_count"),
        "family_follow_up_branch_native_gap_count": _max_int(payloads, "family_follow_up_branch_native_gap_count"),
        "rmsprop_centered_reference_ready_count": _max_int(payloads, "rmsprop_centered_reference_ready_count"),
        "rmsprop_momentum_reference_ready_count": _max_int(payloads, "rmsprop_momentum_reference_ready_count"),
        "sgdp_projection_reference_ready_count": _max_int(payloads, "sgdp_projection_reference_ready_count"),
        "sgdp_decoupled_decay_reference_ready_count": _max_int(payloads, "sgdp_decoupled_decay_reference_ready_count"),
        "fromage_per_tensor_norm_reference_ready_count": _max_int(
            payloads,
            "fromage_per_tensor_norm_reference_ready_count",
        ),
        "fromage_p_bound_reference_ready_count": _max_int(payloads, "fromage_p_bound_reference_ready_count"),
        "rmsprop_centered_launch_config_ready_count": _max_int(payloads, "rmsprop_centered_launch_config_ready_count"),
        "rmsprop_momentum_launch_config_ready_count": _max_int(payloads, "rmsprop_momentum_launch_config_ready_count"),
        "rmsprop_branch_state_layout_ready_count": _max_int(payloads, "rmsprop_branch_state_layout_ready_count"),
        "rmsprop_first_class_contract_ready_count": _max_int(payloads, "rmsprop_first_class_contract_ready_count"),
        "rmsprop_first_class_contract_training_path_enabled_count": _max_int(
            payloads,
            "rmsprop_first_class_contract_training_path_enabled_count",
        ),
        "rmsprop_first_class_contract_native_kernel_present_count": _max_int(
            payloads,
            "rmsprop_first_class_contract_native_kernel_present_count",
        ),
        "rmsprop_branch_first_class_contract_count": _max_int(
            payloads,
            "rmsprop_branch_first_class_contract_count",
        ),
        "rmsprop_centered_branch_contract_pending_count": _max_int(
            payloads,
            "rmsprop_centered_branch_contract_pending_count",
        ),
        "rmsprop_momentum_branch_contract_pending_count": _max_int(
            payloads,
            "rmsprop_momentum_branch_contract_pending_count",
        ),
        "rmsprop_centered_branch_kernel_supported_count": _max_int(
            payloads,
            "rmsprop_centered_branch_kernel_supported_count",
        ),
        "rmsprop_momentum_branch_kernel_supported_count": _max_int(
            payloads,
            "rmsprop_momentum_branch_kernel_supported_count",
        ),
        "rmsprop_branch_native_abi_guard_ready_count": _max_int(
            payloads,
            "rmsprop_branch_native_abi_guard_ready_count",
        ),
        "rmsprop_branch_native_abi_fail_closed_count": _max_int(
            payloads,
            "rmsprop_branch_native_abi_fail_closed_count",
        ),
        "rmsprop_branch_native_kernel_launch_count": _max_int(
            payloads,
            "rmsprop_branch_native_kernel_launch_count",
        ),
        "rmsprop_branch_native_kernel_parity_ready_count": _max_int(
            payloads,
            "rmsprop_branch_native_kernel_parity_ready_count",
        ),
        "pid_momentum_three_buffer_launch_config_ready_count": _max_int(
            payloads,
            "pid_momentum_three_buffer_launch_config_ready_count",
        ),
        "pid_momentum_three_buffer_state_layout_ready_count": _max_int(
            payloads,
            "pid_momentum_three_buffer_state_layout_ready_count",
        ),
        "pid_momentum_three_buffer_native_abi_guard_ready_count": _max_int(
            payloads,
            "pid_momentum_three_buffer_native_abi_guard_ready_count",
        ),
        "pid_momentum_three_buffer_native_kernel_launch_count": _max_int(
            payloads,
            "pid_momentum_three_buffer_native_kernel_launch_count",
        ),
        "pid_momentum_three_buffer_native_kernel_parity_ready_count": _max_int(
            payloads,
            "pid_momentum_three_buffer_native_kernel_parity_ready_count",
        ),
        "sgdp_projection_launch_config_ready_count": _max_int(
            payloads,
            "sgdp_projection_launch_config_ready_count",
        ),
        "sgdp_decoupled_decay_launch_config_ready_count": _max_int(
            payloads,
            "sgdp_decoupled_decay_launch_config_ready_count",
        ),
        "sgdp_branch_state_layout_ready_count": _max_int(payloads, "sgdp_branch_state_layout_ready_count"),
        "sgdp_projection_state_layout_ready_count": _max_int(
            payloads,
            "sgdp_projection_state_layout_ready_count",
        ),
        "sgdp_projection_native_abi_guard_ready_count": _max_int(
            payloads,
            "sgdp_projection_native_abi_guard_ready_count",
        ),
        "sgdp_decoupled_decay_native_abi_guard_ready_count": _max_int(
            payloads,
            "sgdp_decoupled_decay_native_abi_guard_ready_count",
        ),
        "sgdp_branch_native_kernel_launch_count": _max_int(payloads, "sgdp_branch_native_kernel_launch_count"),
        "sgdp_branch_native_kernel_parity_ready_count": _max_int(
            payloads,
            "sgdp_branch_native_kernel_parity_ready_count",
        ),
        "fromage_per_tensor_norm_launch_config_ready_count": _max_int(
            payloads,
            "fromage_per_tensor_norm_launch_config_ready_count",
        ),
        "fromage_p_bound_launch_config_ready_count": _max_int(
            payloads,
            "fromage_p_bound_launch_config_ready_count",
        ),
        "fromage_branch_state_layout_ready_count": _max_int(payloads, "fromage_branch_state_layout_ready_count"),
        "fromage_per_tensor_norm_state_layout_ready_count": _max_int(
            payloads,
            "fromage_per_tensor_norm_state_layout_ready_count",
        ),
        "fromage_p_bound_state_layout_ready_count": _max_int(
            payloads,
            "fromage_p_bound_state_layout_ready_count",
        ),
        "fromage_per_tensor_norm_native_abi_guard_ready_count": _max_int(
            payloads,
            "fromage_per_tensor_norm_native_abi_guard_ready_count",
        ),
        "fromage_p_bound_native_abi_guard_ready_count": _max_int(
            payloads,
            "fromage_p_bound_native_abi_guard_ready_count",
        ),
        "fromage_branch_native_kernel_launch_count": _max_int(payloads, "fromage_branch_native_kernel_launch_count"),
        "fromage_branch_native_kernel_parity_ready_count": _max_int(
            payloads,
            "fromage_branch_native_kernel_parity_ready_count",
        ),
        "native_readiness_runtime_launch_coverage_ready_family_count": _max_int(payloads, "runtime_launch_coverage_ready_family_count"),
        "native_readiness_runtime_launch_coverage_ready_optimizer_count": _max_int(payloads, "runtime_launch_coverage_ready_optimizer_count"),
        "native_readiness_runtime_launch_coverage_mode_counts": _first_mapping(payloads, "runtime_launch_coverage_mode_counts"),
        "native_readiness_owner_release_hold_ready_family_count": _max_int(payloads, "owner_release_hold_ready_family_count"),
        "default_off_product_runtime_dispatch_ready_family_count": _max_int(
            payloads,
            "default_off_product_runtime_dispatch_ready_family_count",
        ),
        "default_off_product_runtime_dispatch_ready_optimizer_count": _max_int(
            payloads,
            "default_off_product_runtime_dispatch_ready_optimizer_count",
        ),
        "default_off_product_native_dispatch_allowed_family_count": _max_int(
            payloads,
            "default_off_product_native_dispatch_allowed_family_count",
        ),
        "default_off_product_native_dispatch_allowed_optimizer_count": _max_int(
            payloads,
            "default_off_product_native_dispatch_allowed_optimizer_count",
        ),
        "default_off_product_training_path_enabled_family_count": _max_int(
            payloads,
            "default_off_product_training_path_enabled_family_count",
        ),
        "default_off_product_training_path_enabled_optimizer_count": _max_int(
            payloads,
            "default_off_product_training_path_enabled_optimizer_count",
        ),
        "default_off_product_product_native_ready_family_count": _max_int(
            payloads,
            "default_off_product_product_native_ready_family_count",
        ),
        "default_off_product_product_native_ready_optimizer_count": _max_int(
            payloads,
            "default_off_product_product_native_ready_optimizer_count",
        ),
        "signed_post_approval_preview_ready_count": _max_int(payloads, "signed_post_approval_preview_ready_count"),
        "signed_post_approval_preview_runtime_dispatch_ready_family_count": _max_int(
            payloads,
            "signed_post_approval_preview_runtime_dispatch_ready_family_count",
        ),
        "signed_post_approval_preview_runtime_dispatch_ready_optimizer_count": _max_int(
            payloads,
            "signed_post_approval_preview_runtime_dispatch_ready_optimizer_count",
        ),
        "signed_post_approval_preview_native_dispatch_allowed_family_count": _max_int(
            payloads,
            "signed_post_approval_preview_native_dispatch_allowed_family_count",
        ),
        "signed_post_approval_preview_native_dispatch_allowed_optimizer_count": _max_int(
            payloads,
            "signed_post_approval_preview_native_dispatch_allowed_optimizer_count",
        ),
        "signed_post_approval_preview_training_path_enabled_family_count": _max_int(
            payloads,
            "signed_post_approval_preview_training_path_enabled_family_count",
        ),
        "signed_post_approval_preview_training_path_enabled_optimizer_count": _max_int(
            payloads,
            "signed_post_approval_preview_training_path_enabled_optimizer_count",
        ),
        "signed_post_approval_preview_product_native_ready_family_count": _max_int(
            payloads,
            "signed_post_approval_preview_product_native_ready_family_count",
        ),
        "signed_post_approval_preview_product_native_ready_optimizer_count": _max_int(
            payloads,
            "signed_post_approval_preview_product_native_ready_optimizer_count",
        ),
        "owner_release_hold_package_family_count": _max_int(payloads, "owner_release_hold_package_family_count"),
        "owner_release_hold_package_ready_family_count": _max_int(payloads, "owner_release_hold_package_ready_family_count"),
        "owner_release_hold_package_manual_review_required_count": _max_int(
            payloads,
            "owner_release_hold_package_manual_review_required_count",
        ),
        "owner_release_hold_package_owner_approval_missing_count": _max_int(
            payloads,
            "owner_release_hold_package_owner_approval_missing_count",
        ),
        "owner_release_hold_package_release_approval_missing_count": _max_int(
            payloads,
            "owner_release_hold_package_release_approval_missing_count",
        ),
        "owner_release_hold_package_runtime_dispatch_ready_count": _max_int(
            payloads,
            "owner_release_hold_package_runtime_dispatch_ready_count",
        ),
        "owner_release_hold_package_native_dispatch_allowed_count": _max_int(
            payloads,
            "owner_release_hold_package_native_dispatch_allowed_count",
        ),
        "owner_release_hold_package_training_path_enabled_count": _max_int(
            payloads,
            "owner_release_hold_package_training_path_enabled_count",
        ),
        "owner_release_hold_package_default_behavior_changed_count": _max_int(
            payloads,
            "owner_release_hold_package_default_behavior_changed_count",
        ),
        "owner_release_hold_package_product_native_ready_count": _max_int(
            payloads,
            "owner_release_hold_package_product_native_ready_count",
        ),
        "adaptive_lr_chain_stage_count": _max_int(payloads, "adaptive_lr_chain_stage_count"),
        "adaptive_lr_chain_ready_stage_count": _max_int(payloads, "adaptive_lr_chain_ready_stage_count"),
        "adaptive_lr_chain_open_stage_count": _max_int(payloads, "adaptive_lr_chain_open_stage_count"),
        "adaptive_lr_chain_target_optimizer_count": _max_int(payloads, "adaptive_lr_chain_target_optimizer_count"),
        "adaptive_lr_chain_product_exposure_gate_ready_count": _max_int(
            payloads,
            "adaptive_lr_chain_product_exposure_gate_ready_count",
        ),
        "adaptive_lr_chain_runtime_dispatch_ready_count": _max_int(
            payloads,
            "adaptive_lr_chain_runtime_dispatch_ready_count",
        ),
        "adaptive_lr_chain_native_dispatch_allowed_count": _max_int(
            payloads,
            "adaptive_lr_chain_native_dispatch_allowed_count",
        ),
        "adaptive_lr_chain_training_path_enabled_count": _max_int(
            payloads,
            "adaptive_lr_chain_training_path_enabled_count",
        ),
        "adaptive_lr_chain_default_behavior_changed_count": _max_int(
            payloads,
            "adaptive_lr_chain_default_behavior_changed_count",
        ),
        "adaptive_lr_chain_product_native_ready_count": _max_int(
            payloads,
            "adaptive_lr_chain_product_native_ready_count",
        ),
        "native_readiness_request_schema_ui_non_exposure_ready_family_count": _max_int(payloads, "request_schema_ui_non_exposure_ready_family_count"),
        "native_readiness_representative_runtime_rehearsal_ready_count": _max_int(payloads, "representative_runtime_rehearsal_ready_count"),
        "native_readiness_representative_runtime_ready_family_count": _max_int(payloads, "representative_runtime_ready_family_count"),
        "native_readiness_runtime_launch_absent_family_count": _max_int(payloads, "runtime_launch_absent_family_count"),
        "native_readiness_runtime_launch_missing_count": _max_int(payloads, "family_specific_runtime_launch_missing_count"),
        "native_readiness_product_training_route_missing_count": _max_int(payloads, "product_training_route_missing_count"),
        "native_readiness_owner_release_approval_missing_count": _max_int(payloads, "owner_release_approval_missing_count"),
        "native_readiness_post_approval_ready_count": _max_int(payloads, "post_approval_ready"),
        "native_readiness_post_approval_runtime_dispatch_ready_family_count": _max_int(
            payloads,
            "post_approval_runtime_dispatch_ready_family_count",
        ),
        "native_readiness_post_approval_runtime_dispatch_ready_optimizer_count": _max_int(
            payloads,
            "post_approval_runtime_dispatch_ready_optimizer_count",
        ),
        "native_readiness_post_approval_native_dispatch_allowed_family_count": _max_int(
            payloads,
            "post_approval_native_dispatch_allowed_family_count",
        ),
        "native_readiness_post_approval_native_dispatch_allowed_optimizer_count": _max_int(
            payloads,
            "post_approval_native_dispatch_allowed_optimizer_count",
        ),
        "native_readiness_post_approval_training_path_enabled_family_count": _max_int(
            payloads,
            "post_approval_training_path_enabled_family_count",
        ),
        "native_readiness_post_approval_training_path_enabled_optimizer_count": _max_int(
            payloads,
            "post_approval_training_path_enabled_optimizer_count",
        ),
        "native_readiness_post_approval_product_native_ready_family_count": _max_int(
            payloads,
            "post_approval_product_native_ready_family_count",
        ),
        "native_readiness_post_approval_product_native_ready_optimizer_count": _max_int(
            payloads,
            "post_approval_product_native_ready_optimizer_count",
        ),
        "owner_release_approval_recorded_count": _max_int(payloads, "owner_release_approval_recorded_count"),
        "owner_release_review_packet_ready_for_signature_count": _max_int(
            payloads,
            "ready_for_owner_signature",
        ),
        "owner_release_review_recorded_count": _max_int(
            payloads,
            "release_review_recorded_count",
            "release_review_recorded",
        ),
        "owner_release_direction_ready_for_signature_count": _max_int(payloads, "owner_release_direction_ready_for_signature_count"),
        "owner_release_direction_recorded_count": _max_int(payloads, "owner_release_direction_recorded_count"),
        "owner_release_direction_approval_recorded_count": _max_int(payloads, "owner_release_direction_approval_recorded_count"),
        "release_review_archive_ready_count": _max_int(payloads, "release_review_archive_ready_count"),
        "release_artifact_first_required_artifact_count": _max_int(
            payloads,
            "release_artifact_first_required_artifact_count",
        ),
        "release_artifact_first_ready_artifact_count": _max_int(
            payloads,
            "release_artifact_first_ready_artifact_count",
        ),
        "release_artifact_first_missing_artifact_count": _max_int(
            payloads,
            "release_artifact_first_missing_artifact_count",
        ),
        "release_artifact_first_parse_error_count": _max_int(
            payloads,
            "release_artifact_first_parse_error_count",
        ),
        "release_artifact_first_cross_check_count": _max_int(
            payloads,
            "release_artifact_first_cross_check_count",
        ),
        "release_artifact_first_cross_check_ready_count": _max_int(
            payloads,
            "release_artifact_first_cross_check_ready_count",
        ),
        "release_artifact_first_v2_record_input_item_count": _max_int(
            payloads,
            "release_artifact_first_v2_record_input_item_count",
        ),
        "release_artifact_first_v2_record_input_phase1_item_count": _max_int(
            payloads,
            "release_artifact_first_v2_record_input_phase1_item_count",
        ),
        "release_artifact_first_v2_record_input_phase2_item_count": _max_int(
            payloads,
            "release_artifact_first_v2_record_input_phase2_item_count",
        ),
        "release_artifact_first_v2_phase2_refresh_inputs_tracked_count": _max_int(
            payloads,
            "release_artifact_first_v2_phase2_refresh_inputs_tracked_count",
        ),
        "release_artifact_first_v2_record_input_checklist_shape_ready_count": _max_int(
            payloads,
            "release_artifact_first_v2_record_input_checklist_shape_ready_count",
        ),
        "release_artifact_first_v2_record_input_support_check_count": _max_int(
            payloads,
            "release_artifact_first_v2_record_input_support_check_count",
        ),
        "release_artifact_first_v2_record_input_support_ready_count": _max_int(
            payloads,
            "release_artifact_first_v2_record_input_support_ready_count",
        ),
        "release_artifact_first_v2_record_input_support_blocked_item_count": _max_int(
            payloads,
            "release_artifact_first_v2_record_input_support_blocked_item_count",
        ),
        "release_artifact_first_v2_record_input_support_blocker_count": _max_int(
            payloads,
            "release_artifact_first_v2_record_input_support_blocker_count",
        ),
        "release_artifact_first_v2_record_input_support_source_binding_ready_count": _max_int(
            payloads,
            "release_artifact_first_v2_record_input_support_source_binding_ready_count",
        ),
        "release_artifact_first_v2_record_input_support_source_binding_blocker_count": _max_int(
            payloads,
            "release_artifact_first_v2_record_input_support_source_binding_blocker_count",
        ),
        "release_artifact_first_v2_record_input_support_shape_ready_count": _max_int(
            payloads,
            "release_artifact_first_v2_record_input_support_shape_ready_count",
        ),
        "release_artifact_first_v2_reviewer_handoff_phase_metadata_ready_count": _max_int(
            payloads,
            "release_artifact_first_v2_reviewer_handoff_phase_metadata_ready_count",
        ),
        "release_artifact_first_v2_reviewer_handoff_phase1_template_entry_count": _max_int(
            payloads,
            "release_artifact_first_v2_reviewer_handoff_phase1_template_entry_count",
        ),
        "release_artifact_first_v2_reviewer_handoff_phase2_deferred_signature_count": _max_int(
            payloads,
            "release_artifact_first_v2_reviewer_handoff_phase2_deferred_signature_count",
        ),
        "release_artifact_first_v2_reviewer_handoff_phase_shape_ready_count": _max_int(
            payloads,
            "release_artifact_first_v2_reviewer_handoff_phase_shape_ready_count",
        ),
        "release_artifact_first_v2_command_audit_expected_path_binding_ready_count": _max_int(
            payloads,
            "release_artifact_first_v2_command_audit_expected_path_binding_ready_count",
        ),
        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_count": _max_int(
            payloads,
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_count",
        ),
        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_pre_record_command_count": _max_int(
            payloads,
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_pre_record_command_count",
        ),
        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_approval_record_command_count": _max_int(
            payloads,
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_approval_record_command_count",
        ),
        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_match_count": _max_int(
            payloads,
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_match_count",
        ),
        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_mismatch_count": _max_int(
            payloads,
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_mismatch_count",
        ),
        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_ready_count": _max_int(
            payloads,
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_ready_count",
        ),
        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_shape_ready_count": _max_int(
            payloads,
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_shape_ready_count",
        ),
        "release_artifact_first_validation_ready_count": _max_int(
            payloads,
            "release_artifact_first_validation_ready_count",
        ),
        "release_artifact_first_runtime_dispatch_ready_count": _max_int(
            payloads,
            "release_artifact_first_runtime_dispatch_ready_count",
        ),
        "release_artifact_first_native_dispatch_allowed_count": _max_int(
            payloads,
            "release_artifact_first_native_dispatch_allowed_count",
        ),
        "release_artifact_first_training_path_enabled_count": _max_int(
            payloads,
            "release_artifact_first_training_path_enabled_count",
        ),
        "release_artifact_first_default_behavior_changed_count": _max_int(
            payloads,
            "release_artifact_first_default_behavior_changed_count",
        ),
        "release_artifact_first_product_native_ready_count": _max_int(
            payloads,
            "release_artifact_first_product_native_ready_count",
        ),
        "product_exposure_decision_recorded_count": _max_int(payloads, "product_exposure_decision_recorded_count"),
        "product_exposure_ready_for_review_count": _max_int(payloads, "ready_for_product_exposure_review"),
        "product_exposure_review_action_required_count": _max_int(
            payloads,
            "product_exposure_review_action_required",
        ),
        "v2_remaining_gate_total_count": _max_int(payloads, "v2_remaining_gate_total_count"),
        "v2_remaining_gate_open_count": _max_int(payloads, "v2_remaining_gate_open_count"),
        "v2_remaining_gate_closed_count": _max_int(payloads, "v2_remaining_gate_closed_count"),
        "v2_remaining_gate_owner_release_open_count": _max_int(
            payloads,
            "v2_remaining_gate_owner_release_open_count",
        ),
        "v2_remaining_gate_product_exposure_open_count": _max_int(
            payloads,
            "v2_remaining_gate_product_exposure_open_count",
        ),
        "v2_remaining_gate_handoff_ready_count": _max_int(payloads, "v2_remaining_gate_handoff_ready_count"),
        "v2_remaining_gate_release_artifact_ready_count": _max_int(
            payloads,
            "v2_remaining_gate_release_artifact_ready_count",
        ),
        "v2_remaining_gate_default_off_guard_count": _max_int(
            payloads,
            "v2_remaining_gate_default_off_guard_count",
        ),
        "v2_remaining_gate_runtime_dispatch_ready_count": _max_int(
            payloads,
            "v2_remaining_gate_runtime_dispatch_ready_count",
        ),
        "v2_remaining_gate_native_dispatch_allowed_count": _max_int(
            payloads,
            "v2_remaining_gate_native_dispatch_allowed_count",
        ),
        "v2_remaining_gate_training_path_enabled_count": _max_int(
            payloads,
            "v2_remaining_gate_training_path_enabled_count",
        ),
        "v2_remaining_gate_product_native_ready_count": _max_int(
            payloads,
            "v2_remaining_gate_product_native_ready_count",
        ),
        "v2_remaining_gate_default_behavior_changed_count": _max_int(
            payloads,
            "v2_remaining_gate_default_behavior_changed_count",
        ),
        "v2_remaining_gate_unsafe_claim_count": _max_int(payloads, "v2_remaining_gate_unsafe_claim_count"),
        "v2_signature_bundle_entry_count": _max_int(payloads, "v2_signature_bundle_entry_count"),
        "v2_signature_bundle_ready_for_signature_count": _max_int(
            payloads,
            "v2_signature_bundle_ready_for_signature_count",
        ),
        "v2_signature_bundle_blocked_entry_count": _max_int(
            payloads,
            "v2_signature_bundle_blocked_entry_count",
        ),
        "v2_signature_bundle_owner_review_ready_count": _max_int(
            payloads,
            "v2_signature_bundle_owner_review_ready_count",
        ),
        "v2_signature_bundle_product_exposure_ready_count": _max_int(
            payloads,
            "v2_signature_bundle_product_exposure_ready_count",
        ),
        "v2_signature_bundle_owner_direction_ready_count": _max_int(
            payloads,
            "v2_signature_bundle_owner_direction_ready_count",
        ),
        "v2_signature_bundle_approval_recorded_count": _max_int(
            payloads,
            "v2_signature_bundle_approval_recorded_count",
        ),
        "v2_signature_bundle_runtime_dispatch_ready_count": _max_int(
            payloads,
            "v2_signature_bundle_runtime_dispatch_ready_count",
        ),
        "v2_signature_bundle_native_dispatch_allowed_count": _max_int(
            payloads,
            "v2_signature_bundle_native_dispatch_allowed_count",
        ),
        "v2_signature_bundle_training_path_enabled_count": _max_int(
            payloads,
            "v2_signature_bundle_training_path_enabled_count",
        ),
        "v2_signature_bundle_product_native_ready_count": _max_int(
            payloads,
            "v2_signature_bundle_product_native_ready_count",
        ),
        "v2_signature_bundle_default_behavior_changed_count": _max_int(
            payloads,
            "v2_signature_bundle_default_behavior_changed_count",
        ),
        "v2_signature_bundle_unsafe_claim_count": _max_int(payloads, "v2_signature_bundle_unsafe_claim_count"),
        "v2_reviewer_handoff_entry_count": _max_int(payloads, "v2_reviewer_handoff_entry_count"),
        "v2_reviewer_handoff_ready_entry_count": _max_int(payloads, "v2_reviewer_handoff_ready_entry_count"),
        "v2_reviewer_handoff_blocked_entry_count": _max_int(payloads, "v2_reviewer_handoff_blocked_entry_count"),
        "v2_reviewer_handoff_signed_template_entry_count": _max_int(
            payloads,
            "v2_reviewer_handoff_signed_template_entry_count",
        ),
        "v2_reviewer_handoff_phase_metadata_ready_count": _max_int(
            payloads,
            "v2_reviewer_handoff_phase_metadata_ready_count",
        ),
        "v2_reviewer_handoff_phase1_template_entry_count": _max_int(
            payloads,
            "v2_reviewer_handoff_phase1_template_entry_count",
        ),
        "v2_reviewer_handoff_phase2_deferred_signature_count": _max_int(
            payloads,
            "v2_reviewer_handoff_phase2_deferred_signature_count",
        ),
        "v2_reviewer_handoff_required_manual_field_signature_count": _max_int(
            payloads,
            "v2_reviewer_handoff_required_manual_field_signature_count",
        ),
        "v2_reviewer_handoff_required_manual_field_count": _max_int(
            payloads,
            "v2_reviewer_handoff_required_manual_field_count",
        ),
        "v2_reviewer_handoff_phase1_signature_count": _max_int(
            payloads,
            "v2_reviewer_handoff_phase1_signature_count",
        ),
        "v2_reviewer_handoff_phase2_blocked_signature_count": _max_int(
            payloads,
            "v2_reviewer_handoff_phase2_blocked_signature_count",
        ),
        "v2_reviewer_handoff_phase1_required_manual_field_count": _max_int(
            payloads,
            "v2_reviewer_handoff_phase1_required_manual_field_count",
        ),
        "v2_reviewer_handoff_template_dry_run_command_count": _max_int(
            payloads,
            "v2_reviewer_handoff_template_dry_run_command_count",
        ),
        "v2_reviewer_handoff_real_return_validation_command_count": _max_int(
            payloads,
            "v2_reviewer_handoff_real_return_validation_command_count",
        ),
        "v2_reviewer_handoff_phase1_post_return_command_count": _max_int(
            payloads,
            "v2_reviewer_handoff_phase1_post_return_command_count",
        ),
        "v2_reviewer_handoff_phase1_post_return_pre_record_command_count": _max_int(
            payloads,
            "v2_reviewer_handoff_phase1_post_return_pre_record_command_count",
        ),
        "v2_reviewer_handoff_phase1_post_return_approval_record_command_count": _max_int(
            payloads,
            "v2_reviewer_handoff_phase1_post_return_approval_record_command_count",
        ),
        "v2_reviewer_handoff_packet_ready_count": _max_int(payloads, "v2_reviewer_handoff_packet_ready_count"),
        "v2_reviewer_handoff_approval_recorded_count": _max_int(
            payloads,
            "v2_reviewer_handoff_approval_recorded_count",
        ),
        "v2_reviewer_handoff_runtime_dispatch_ready_count": _max_int(
            payloads,
            "v2_reviewer_handoff_runtime_dispatch_ready_count",
        ),
        "v2_reviewer_handoff_native_dispatch_allowed_count": _max_int(
            payloads,
            "v2_reviewer_handoff_native_dispatch_allowed_count",
        ),
        "v2_reviewer_handoff_training_path_enabled_count": _max_int(
            payloads,
            "v2_reviewer_handoff_training_path_enabled_count",
        ),
        "v2_reviewer_handoff_product_native_ready_count": _max_int(
            payloads,
            "v2_reviewer_handoff_product_native_ready_count",
        ),
        "v2_reviewer_handoff_default_behavior_changed_count": _max_int(
            payloads,
            "v2_reviewer_handoff_default_behavior_changed_count",
        ),
        "v2_reviewer_handoff_unsafe_claim_count": _max_int(payloads, "v2_reviewer_handoff_unsafe_claim_count"),
        "v2_approval_execution_step_count": _max_int(payloads, "v2_approval_execution_step_count"),
        "v2_approval_execution_phase1_step_count": _max_int(payloads, "v2_approval_execution_phase1_step_count"),
        "v2_approval_execution_phase2_step_count": _max_int(payloads, "v2_approval_execution_phase2_step_count"),
        "v2_approval_execution_shared_step_count": _max_int(payloads, "v2_approval_execution_shared_step_count"),
        "v2_approval_execution_ready_step_count": _max_int(payloads, "v2_approval_execution_ready_step_count"),
        "v2_approval_execution_manual_signature_step_count": _max_int(
            payloads,
            "v2_approval_execution_manual_signature_step_count",
        ),
        "v2_approval_execution_record_step_count": _max_int(payloads, "v2_approval_execution_record_step_count"),
        "v2_approval_execution_extraction_step_count": _max_int(
            payloads,
            "v2_approval_execution_extraction_step_count",
        ),
        "v2_approval_execution_preflight_step_count": _max_int(
            payloads,
            "v2_approval_execution_preflight_step_count",
        ),
        "v2_approval_execution_phase1_record_chain_step_count": _max_int(
            payloads,
            "v2_approval_execution_phase1_record_chain_step_count",
        ),
        "v2_approval_execution_phase2_record_chain_step_count": _max_int(
            payloads,
            "v2_approval_execution_phase2_record_chain_step_count",
        ),
        "v2_approval_execution_reviewer_template_entry_count": _max_int(
            payloads,
            "v2_approval_execution_reviewer_template_entry_count",
        ),
        "v2_approval_execution_blocked_signature_entry_count": _max_int(
            payloads,
            "v2_approval_execution_blocked_signature_entry_count",
        ),
        "v2_approval_execution_phase1_handoff_post_return_command_count": _max_int(
            payloads,
            "v2_approval_execution_phase1_handoff_post_return_command_count",
        ),
        "v2_approval_execution_phase1_handoff_post_return_pre_record_command_count": _max_int(
            payloads,
            "v2_approval_execution_phase1_handoff_post_return_pre_record_command_count",
        ),
        "v2_approval_execution_phase1_handoff_post_return_approval_record_command_count": _max_int(
            payloads,
            "v2_approval_execution_phase1_handoff_post_return_approval_record_command_count",
        ),
        "v2_approval_execution_phase1_handoff_post_return_command_match_count": _max_int(
            payloads,
            "v2_approval_execution_phase1_handoff_post_return_command_match_count",
        ),
        "v2_approval_execution_phase1_handoff_post_return_command_mismatch_count": _max_int(
            payloads,
            "v2_approval_execution_phase1_handoff_post_return_command_mismatch_count",
        ),
        "v2_approval_execution_phase1_handoff_post_return_ready_count": _max_int(
            payloads,
            "v2_approval_execution_phase1_handoff_post_return_ready_count",
        ),
        "v2_approval_execution_plan_ready_count": _max_int(payloads, "v2_approval_execution_plan_ready_count"),
        "v2_approval_execution_approval_recorded_count": _max_int(
            payloads,
            "v2_approval_execution_approval_recorded_count",
        ),
        "v2_approval_execution_runtime_dispatch_ready_count": _max_int(
            payloads,
            "v2_approval_execution_runtime_dispatch_ready_count",
        ),
        "v2_approval_execution_native_dispatch_allowed_count": _max_int(
            payloads,
            "v2_approval_execution_native_dispatch_allowed_count",
        ),
        "v2_approval_execution_training_path_enabled_count": _max_int(
            payloads,
            "v2_approval_execution_training_path_enabled_count",
        ),
        "v2_approval_execution_product_native_ready_count": _max_int(
            payloads,
            "v2_approval_execution_product_native_ready_count",
        ),
        "v2_approval_execution_default_behavior_changed_count": _max_int(
            payloads,
            "v2_approval_execution_default_behavior_changed_count",
        ),
        "v2_approval_execution_unsafe_claim_count": _max_int(payloads, "v2_approval_execution_unsafe_claim_count"),
        "v2_approval_command_audit_step_count": _max_int(payloads, "v2_approval_command_audit_step_count"),
        "v2_approval_command_audit_command_step_count": _max_int(
            payloads,
            "v2_approval_command_audit_command_step_count",
        ),
        "v2_approval_command_audit_entrypoint_exists_count": _max_int(
            payloads,
            "v2_approval_command_audit_entrypoint_exists_count",
        ),
        "v2_approval_command_audit_missing_entrypoint_count": _max_int(
            payloads,
            "v2_approval_command_audit_missing_entrypoint_count",
        ),
        "v2_approval_command_audit_order_valid_count": _max_int(
            payloads,
            "v2_approval_command_audit_order_valid_count",
        ),
        "v2_approval_command_audit_record_after_preflight_count": _max_int(
            payloads,
            "v2_approval_command_audit_record_after_preflight_count",
        ),
        "v2_approval_command_audit_record_before_preflight_count": _max_int(
            payloads,
            "v2_approval_command_audit_record_before_preflight_count",
        ),
        "v2_approval_command_audit_record_after_real_signature_count": _max_int(
            payloads,
            "v2_approval_command_audit_record_after_real_signature_count",
        ),
        "v2_approval_command_audit_record_before_real_signature_count": _max_int(
            payloads,
            "v2_approval_command_audit_record_before_real_signature_count",
        ),
        "v2_approval_command_audit_phase2_order_valid_count": _max_int(
            payloads,
            "v2_approval_command_audit_phase2_order_valid_count",
        ),
        "v2_approval_command_audit_phase2_blocker_count": _max_int(
            payloads,
            "v2_approval_command_audit_phase2_blocker_count",
        ),
        "v2_approval_command_audit_signature_order_valid_count": _max_int(
            payloads,
            "v2_approval_command_audit_signature_order_valid_count",
        ),
        "v2_approval_command_audit_signature_blocker_count": _max_int(
            payloads,
            "v2_approval_command_audit_signature_blocker_count",
        ),
        "v2_approval_command_audit_phase_marker_valid_count": _max_int(
            payloads,
            "v2_approval_command_audit_phase_marker_valid_count",
        ),
        "v2_approval_command_audit_phase_marker_blocker_count": _max_int(
            payloads,
            "v2_approval_command_audit_phase_marker_blocker_count",
        ),
        "v2_approval_command_audit_required_marker_missing_count": _max_int(
            payloads,
            "v2_approval_command_audit_required_marker_missing_count",
        ),
        "v2_approval_command_audit_preflight_artifact_write_count": _max_int(
            payloads,
            "v2_approval_command_audit_preflight_artifact_write_count",
        ),
        "v2_approval_command_audit_preflight_no_artifact_blocker_count": _max_int(
            payloads,
            "v2_approval_command_audit_preflight_no_artifact_blocker_count",
        ),
        "v2_approval_command_audit_record_command_preflight_arg_count": _max_int(
            payloads,
            "v2_approval_command_audit_record_command_preflight_arg_count",
        ),
        "v2_approval_command_audit_unsigned_template_allowed_count": _max_int(
            payloads,
            "v2_approval_command_audit_unsigned_template_allowed_count",
        ),
        "v2_approval_command_audit_expected_arg_binding_count": _max_int(
            payloads,
            "v2_approval_command_audit_expected_arg_binding_count",
        ),
        "v2_approval_command_audit_expected_arg_mismatch_count": _max_int(
            payloads,
            "v2_approval_command_audit_expected_arg_mismatch_count",
        ),
        "v2_approval_command_audit_expected_path_binding_ready_count": _max_int(
            payloads,
            "v2_approval_command_audit_expected_path_binding_ready_count",
        ),
        "v2_approval_command_audit_phase1_handoff_post_return_command_count": _max_int(
            payloads,
            "v2_approval_command_audit_phase1_handoff_post_return_command_count",
        ),
        "v2_approval_command_audit_phase1_handoff_post_return_pre_record_command_count": _max_int(
            payloads,
            "v2_approval_command_audit_phase1_handoff_post_return_pre_record_command_count",
        ),
        "v2_approval_command_audit_phase1_handoff_post_return_approval_record_command_count": _max_int(
            payloads,
            "v2_approval_command_audit_phase1_handoff_post_return_approval_record_command_count",
        ),
        "v2_approval_command_audit_phase1_handoff_post_return_command_match_count": _max_int(
            payloads,
            "v2_approval_command_audit_phase1_handoff_post_return_command_match_count",
        ),
        "v2_approval_command_audit_phase1_handoff_post_return_command_mismatch_count": _max_int(
            payloads,
            "v2_approval_command_audit_phase1_handoff_post_return_command_mismatch_count",
        ),
        "v2_approval_command_audit_phase1_handoff_post_return_ready_count": _max_int(
            payloads,
            "v2_approval_command_audit_phase1_handoff_post_return_ready_count",
        ),
        "v2_approval_command_audit_ready_count": _max_int(payloads, "v2_approval_command_audit_ready_count"),
        "v2_approval_command_audit_approval_recorded_count": _max_int(
            payloads,
            "v2_approval_command_audit_approval_recorded_count",
        ),
        "v2_approval_command_audit_runtime_dispatch_ready_count": _max_int(
            payloads,
            "v2_approval_command_audit_runtime_dispatch_ready_count",
        ),
        "v2_approval_command_audit_native_dispatch_allowed_count": _max_int(
            payloads,
            "v2_approval_command_audit_native_dispatch_allowed_count",
        ),
        "v2_approval_command_audit_training_path_enabled_count": _max_int(
            payloads,
            "v2_approval_command_audit_training_path_enabled_count",
        ),
        "v2_approval_command_audit_product_native_ready_count": _max_int(
            payloads,
            "v2_approval_command_audit_product_native_ready_count",
        ),
        "v2_approval_command_audit_default_behavior_changed_count": _max_int(
            payloads,
            "v2_approval_command_audit_default_behavior_changed_count",
        ),
        "v2_approval_command_audit_unsafe_claim_count": _max_int(
            payloads,
            "v2_approval_command_audit_unsafe_claim_count",
        ),
        "v2_record_input_checklist_item_count": _max_int(payloads, "v2_record_input_checklist_item_count"),
        "v2_record_input_checklist_present_item_count": _max_int(
            payloads,
            "v2_record_input_checklist_present_item_count",
        ),
        "v2_record_input_checklist_valid_json_item_count": _max_int(
            payloads,
            "v2_record_input_checklist_valid_json_item_count",
        ),
        "v2_record_input_checklist_missing_item_count": _max_int(
            payloads,
            "v2_record_input_checklist_missing_item_count",
        ),
        "v2_record_input_checklist_ready_item_count": _max_int(
            payloads,
            "v2_record_input_checklist_ready_item_count",
        ),
        "v2_record_input_checklist_phase1_item_count": _max_int(
            payloads,
            "v2_record_input_checklist_phase1_item_count",
        ),
        "v2_record_input_checklist_phase1_ready_item_count": _max_int(
            payloads,
            "v2_record_input_checklist_phase1_ready_item_count",
        ),
        "v2_record_input_checklist_phase1_missing_item_count": _max_int(
            payloads,
            "v2_record_input_checklist_phase1_missing_item_count",
        ),
        "v2_record_input_checklist_phase1_ready_count": _max_int(
            payloads,
            "v2_record_input_checklist_phase1_ready_count",
        ),
        "v2_record_input_checklist_phase2_item_count": _max_int(
            payloads,
            "v2_record_input_checklist_phase2_item_count",
        ),
        "v2_record_input_checklist_phase2_ready_item_count": _max_int(
            payloads,
            "v2_record_input_checklist_phase2_ready_item_count",
        ),
        "v2_record_input_checklist_phase2_missing_item_count": _max_int(
            payloads,
            "v2_record_input_checklist_phase2_missing_item_count",
        ),
        "v2_record_input_checklist_phase2_ready_count": _max_int(
            payloads,
            "v2_record_input_checklist_phase2_ready_count",
        ),
        "v2_record_input_checklist_full_ready_count": _max_int(
            payloads,
            "v2_record_input_checklist_full_ready_count",
        ),
        "v2_record_input_checklist_support_check_count": _max_int(
            payloads,
            "v2_record_input_checklist_support_check_count",
        ),
        "v2_record_input_checklist_support_ready_count": _max_int(
            payloads,
            "v2_record_input_checklist_support_ready_count",
        ),
        "v2_record_input_checklist_support_blocked_item_count": _max_int(
            payloads,
            "v2_record_input_checklist_support_blocked_item_count",
        ),
        "v2_record_input_checklist_support_blocker_count": _max_int(
            payloads,
            "v2_record_input_checklist_support_blocker_count",
        ),
        "v2_record_input_checklist_support_source_binding_ready_count": _max_int(
            payloads,
            "v2_record_input_checklist_support_source_binding_ready_count",
        ),
        "v2_record_input_checklist_support_source_binding_blocker_count": _max_int(
            payloads,
            "v2_record_input_checklist_support_source_binding_blocker_count",
        ),
        "v2_record_input_checklist_artifact_ready_count": _max_int(
            payloads,
            "v2_record_input_checklist_artifact_ready_count",
        ),
        "v2_record_input_checklist_approval_recorded_count": _max_int(
            payloads,
            "v2_record_input_checklist_approval_recorded_count",
        ),
        "v2_record_input_checklist_runtime_dispatch_ready_count": _max_int(
            payloads,
            "v2_record_input_checklist_runtime_dispatch_ready_count",
        ),
        "v2_record_input_checklist_native_dispatch_allowed_count": _max_int(
            payloads,
            "v2_record_input_checklist_native_dispatch_allowed_count",
        ),
        "v2_record_input_checklist_training_path_enabled_count": _max_int(
            payloads,
            "v2_record_input_checklist_training_path_enabled_count",
        ),
        "v2_record_input_checklist_product_native_ready_count": _max_int(
            payloads,
            "v2_record_input_checklist_product_native_ready_count",
        ),
        "v2_record_input_checklist_default_behavior_changed_count": _max_int(
            payloads,
            "v2_record_input_checklist_default_behavior_changed_count",
        ),
        "v2_record_input_checklist_unsafe_claim_count": _max_int(
            payloads,
            "v2_record_input_checklist_unsafe_claim_count",
        ),
        "v2_signed_bundle_extraction_entry_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_entry_count",
        ),
        "v2_signed_bundle_extraction_present_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_present_count",
        ),
        "v2_signed_bundle_extraction_source_digest_match_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_source_digest_match_count",
        ),
        "v2_signed_bundle_extraction_source_digest_stale_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_source_digest_stale_count",
        ),
        "v2_signed_bundle_extraction_unsigned_template_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_unsigned_template_count",
        ),
        "v2_signed_bundle_extraction_template_digest_mismatch_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_template_digest_mismatch_count",
        ),
        "v2_signed_bundle_extraction_unknown_entry_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_unknown_entry_count",
        ),
        "v2_signed_bundle_extraction_missing_signature_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_missing_signature_count",
        ),
        "v2_signed_bundle_extraction_phase1_missing_signature_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_phase1_missing_signature_count",
        ),
        "v2_signed_bundle_extraction_phase2_missing_signature_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_phase2_missing_signature_count",
        ),
        "v2_signed_bundle_extraction_not_ready_entry_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_not_ready_entry_count",
        ),
        "v2_signed_bundle_extraction_signed_entry_digest_present_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_signed_entry_digest_present_count",
        ),
        "v2_signed_bundle_extraction_extractable_signed_entry_digest_present_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_extractable_signed_entry_digest_present_count",
        ),
        "v2_signed_bundle_extraction_extractable_entry_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_extractable_entry_count",
        ),
        "v2_signed_bundle_extraction_phase1_extractable_entry_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_phase1_extractable_entry_count",
        ),
        "v2_signed_bundle_extraction_phase1_ready_for_record_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_phase1_ready_for_record_count",
        ),
        "v2_signed_bundle_extraction_phase2_extractable_entry_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_phase2_extractable_entry_count",
        ),
        "v2_signed_bundle_extraction_phase2_ready_for_record_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_phase2_ready_for_record_count",
        ),
        "v2_signed_bundle_extraction_full_ready_for_record_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_full_ready_for_record_count",
        ),
        "v2_signed_bundle_extraction_missing_entry_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_missing_entry_count",
        ),
        "v2_signed_bundle_extraction_owner_review_extracted_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_owner_review_extracted_count",
        ),
        "v2_signed_bundle_extraction_product_exposure_extracted_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_product_exposure_extracted_count",
        ),
        "v2_signed_bundle_extraction_owner_direction_extracted_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_owner_direction_extracted_count",
        ),
        "v2_signed_bundle_extraction_artifact_written_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_artifact_written_count",
        ),
        "v2_signed_bundle_extraction_ready_for_record_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_ready_for_record_count",
        ),
        "v2_signed_bundle_extraction_approval_recorded_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_approval_recorded_count",
        ),
        "v2_signed_bundle_extraction_runtime_dispatch_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_runtime_dispatch_ready_count",
        ),
        "v2_signed_bundle_extraction_native_dispatch_allowed_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_native_dispatch_allowed_count",
        ),
        "v2_signed_bundle_extraction_training_path_enabled_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_training_path_enabled_count",
        ),
        "v2_signed_bundle_extraction_product_native_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_product_native_ready_count",
        ),
        "v2_signed_bundle_extraction_default_behavior_changed_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_default_behavior_changed_count",
        ),
        "v2_signed_bundle_extraction_unsafe_claim_count": _max_int(
            payloads,
            "v2_signed_bundle_extraction_unsafe_claim_count",
        ),
        "v2_approval_preflight_file_count": _max_int(payloads, "v2_approval_preflight_file_count"),
        "v2_approval_preflight_present_file_count": _max_int(
            payloads,
            "v2_approval_preflight_present_file_count",
        ),
        "v2_approval_preflight_valid_json_file_count": _max_int(
            payloads,
            "v2_approval_preflight_valid_json_file_count",
        ),
        "v2_approval_preflight_missing_file_count": _max_int(
            payloads,
            "v2_approval_preflight_missing_file_count",
        ),
        "v2_approval_preflight_missing_shared_input_count": _max_int(
            payloads,
            "v2_approval_preflight_missing_shared_input_count",
        ),
        "v2_approval_preflight_missing_phase1_input_count": _max_int(
            payloads,
            "v2_approval_preflight_missing_phase1_input_count",
        ),
        "v2_approval_preflight_missing_phase2_input_count": _max_int(
            payloads,
            "v2_approval_preflight_missing_phase2_input_count",
        ),
        "v2_approval_preflight_parse_error_count": _max_int(payloads, "v2_approval_preflight_parse_error_count"),
        "v2_approval_preflight_signature_bundle_valid_count": _max_int(
            payloads,
            "v2_approval_preflight_signature_bundle_valid_count",
        ),
        "v2_approval_preflight_signed_bundle_valid_count": _max_int(
            payloads,
            "v2_approval_preflight_signed_bundle_valid_count",
        ),
        "v2_approval_preflight_source_digest_match_count": _max_int(
            payloads,
            "v2_approval_preflight_source_digest_match_count",
        ),
        "v2_approval_preflight_source_digest_stale_count": _max_int(
            payloads,
            "v2_approval_preflight_source_digest_stale_count",
        ),
        "v2_approval_preflight_template_digest_mismatch_count": _max_int(
            payloads,
            "v2_approval_preflight_template_digest_mismatch_count",
        ),
        "v2_approval_preflight_unknown_entry_count": _max_int(
            payloads,
            "v2_approval_preflight_unknown_entry_count",
        ),
        "v2_approval_preflight_not_ready_entry_count": _max_int(
            payloads,
            "v2_approval_preflight_not_ready_entry_count",
        ),
        "v2_approval_preflight_unsigned_template_count": _max_int(
            payloads,
            "v2_approval_preflight_unsigned_template_count",
        ),
        "v2_approval_preflight_signed_payload_digest_present_count": _max_int(
            payloads,
            "v2_approval_preflight_signed_payload_digest_present_count",
        ),
        "v2_approval_preflight_signed_payload_digest_missing_count": _max_int(
            payloads,
            "v2_approval_preflight_signed_payload_digest_missing_count",
        ),
        "v2_approval_preflight_signed_bundle_entry_digest_present_count": _max_int(
            payloads,
            "v2_approval_preflight_signed_bundle_entry_digest_present_count",
        ),
        "v2_approval_preflight_signed_payload_bundle_digest_match_count": _max_int(
            payloads,
            "v2_approval_preflight_signed_payload_bundle_digest_match_count",
        ),
        "v2_approval_preflight_extracted_entry_digest_match_count": _max_int(
            payloads,
            "v2_approval_preflight_extracted_entry_digest_match_count",
        ),
        "v2_approval_preflight_extracted_entry_digest_mismatch_count": _max_int(
            payloads,
            "v2_approval_preflight_extracted_entry_digest_mismatch_count",
        ),
        "v2_approval_preflight_extracted_entry_source_missing_count": _max_int(
            payloads,
            "v2_approval_preflight_extracted_entry_source_missing_count",
        ),
        "v2_approval_preflight_support_ready_count": _max_int(
            payloads,
            "v2_approval_preflight_support_ready_count",
        ),
        "v2_approval_preflight_support_invalid_count": _max_int(
            payloads,
            "v2_approval_preflight_support_invalid_count",
        ),
        "v2_approval_preflight_support_source_binding_ready_count": _max_int(
            payloads,
            "v2_approval_preflight_support_source_binding_ready_count",
        ),
        "v2_approval_preflight_support_source_binding_blocker_count": _max_int(
            payloads,
            "v2_approval_preflight_support_source_binding_blocker_count",
        ),
        "v2_approval_preflight_owner_review_ready_count": _max_int(
            payloads,
            "v2_approval_preflight_owner_review_ready_count",
        ),
        "v2_approval_preflight_product_exposure_ready_count": _max_int(
            payloads,
            "v2_approval_preflight_product_exposure_ready_count",
        ),
        "v2_approval_preflight_owner_direction_ready_count": _max_int(
            payloads,
            "v2_approval_preflight_owner_direction_ready_count",
        ),
        "v2_approval_preflight_phase1_ready_count": _max_int(
            payloads,
            "v2_approval_preflight_phase1_ready_count",
        ),
        "v2_approval_preflight_phase2_ready_count": _max_int(
            payloads,
            "v2_approval_preflight_phase2_ready_count",
        ),
        "v2_approval_preflight_full_ready_count": _max_int(payloads, "v2_approval_preflight_full_ready_count"),
        "v2_approval_preflight_hard_fail_count": _max_int(payloads, "v2_approval_preflight_hard_fail_count"),
        "v2_approval_preflight_approval_recorded_count": _max_int(
            payloads,
            "v2_approval_preflight_approval_recorded_count",
        ),
        "v2_approval_preflight_post_record_request_fields_ready_count": _max_int(
            payloads,
            "v2_approval_preflight_post_record_request_fields_ready_count",
        ),
        "v2_approval_preflight_post_record_request_field_emission_count": _max_int(
            payloads,
            "v2_approval_preflight_post_record_request_field_emission_count",
        ),
        "v2_approval_preflight_runtime_dispatch_ready_count": _max_int(
            payloads,
            "v2_approval_preflight_runtime_dispatch_ready_count",
        ),
        "v2_approval_preflight_native_dispatch_allowed_count": _max_int(
            payloads,
            "v2_approval_preflight_native_dispatch_allowed_count",
        ),
        "v2_approval_preflight_training_path_enabled_count": _max_int(
            payloads,
            "v2_approval_preflight_training_path_enabled_count",
        ),
        "v2_approval_preflight_product_native_ready_count": _max_int(
            payloads,
            "v2_approval_preflight_product_native_ready_count",
        ),
        "v2_approval_preflight_default_behavior_changed_count": _max_int(
            payloads,
            "v2_approval_preflight_default_behavior_changed_count",
        ),
        "v2_approval_preflight_unsafe_claim_count": _max_int(payloads, "v2_approval_preflight_unsafe_claim_count"),
        "v2_signed_bundle_entry_count": _max_int(payloads, "v2_signed_bundle_entry_count"),
        "v2_signed_bundle_present_count": _max_int(payloads, "v2_signed_bundle_present_count"),
        "v2_signed_bundle_valid_record_count": _max_int(payloads, "v2_signed_bundle_valid_record_count"),
        "v2_signed_bundle_phase1_valid_record_count": _max_int(
            payloads,
            "v2_signed_bundle_phase1_valid_record_count",
        ),
        "v2_signed_bundle_phase1_ready_count": _max_int(payloads, "v2_signed_bundle_phase1_ready_count"),
        "v2_signed_bundle_phase2_valid_record_count": _max_int(
            payloads,
            "v2_signed_bundle_phase2_valid_record_count",
        ),
        "v2_signed_bundle_phase2_ready_count": _max_int(payloads, "v2_signed_bundle_phase2_ready_count"),
        "v2_signed_bundle_full_ready_count": _max_int(payloads, "v2_signed_bundle_full_ready_count"),
        "v2_signed_bundle_missing_signature_count": _max_int(payloads, "v2_signed_bundle_missing_signature_count"),
        "v2_signed_bundle_phase1_missing_signature_count": _max_int(
            payloads,
            "v2_signed_bundle_phase1_missing_signature_count",
        ),
        "v2_signed_bundle_phase2_missing_signature_count": _max_int(
            payloads,
            "v2_signed_bundle_phase2_missing_signature_count",
        ),
        "v2_signed_bundle_unsigned_template_count": _max_int(payloads, "v2_signed_bundle_unsigned_template_count"),
        "v2_signed_bundle_template_digest_mismatch_count": _max_int(
            payloads,
            "v2_signed_bundle_template_digest_mismatch_count",
        ),
        "v2_signed_bundle_unknown_entry_count": _max_int(payloads, "v2_signed_bundle_unknown_entry_count"),
        "v2_signed_bundle_not_ready_entry_count": _max_int(payloads, "v2_signed_bundle_not_ready_entry_count"),
        "v2_signed_bundle_manual_field_check_count": _max_int(
            payloads,
            "v2_signed_bundle_manual_field_check_count",
        ),
        "v2_signed_bundle_manual_field_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_manual_field_ready_count",
        ),
        "v2_signed_bundle_manual_field_missing_count": _max_int(
            payloads,
            "v2_signed_bundle_manual_field_missing_count",
        ),
        "v2_signed_bundle_manual_field_missing_signature_count": _max_int(
            payloads,
            "v2_signed_bundle_manual_field_missing_signature_count",
        ),
        "v2_signed_bundle_manual_field_shape_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_manual_field_shape_ready_count",
        ),
        "v2_signed_bundle_phase1_manual_field_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_phase1_manual_field_ready_count",
        ),
        "v2_signed_bundle_phase2_manual_field_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_phase2_manual_field_ready_count",
        ),
        "v2_signed_bundle_source_digest_match_count": _max_int(
            payloads,
            "v2_signed_bundle_source_digest_match_count",
        ),
        "v2_signed_bundle_source_digest_stale_count": _max_int(
            payloads,
            "v2_signed_bundle_source_digest_stale_count",
        ),
        "v2_signed_bundle_owner_review_recorded_count": _max_int(
            payloads,
            "v2_signed_bundle_owner_review_recorded_count",
        ),
        "v2_signed_bundle_product_exposure_recorded_count": _max_int(
            payloads,
            "v2_signed_bundle_product_exposure_recorded_count",
        ),
        "v2_signed_bundle_owner_direction_recorded_count": _max_int(
            payloads,
            "v2_signed_bundle_owner_direction_recorded_count",
        ),
        "v2_signed_bundle_approval_artifact_written_count": _max_int(
            payloads,
            "v2_signed_bundle_approval_artifact_written_count",
        ),
        "v2_signed_bundle_runtime_dispatch_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_runtime_dispatch_ready_count",
        ),
        "v2_signed_bundle_native_dispatch_allowed_count": _max_int(
            payloads,
            "v2_signed_bundle_native_dispatch_allowed_count",
        ),
        "v2_signed_bundle_training_path_enabled_count": _max_int(
            payloads,
            "v2_signed_bundle_training_path_enabled_count",
        ),
        "v2_signed_bundle_product_native_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_product_native_ready_count",
        ),
        "v2_signed_bundle_default_behavior_changed_count": _max_int(
            payloads,
            "v2_signed_bundle_default_behavior_changed_count",
        ),
        "v2_signed_bundle_unsafe_claim_count": _max_int(payloads, "v2_signed_bundle_unsafe_claim_count"),
        "v2_signed_bundle_freshness_guard_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_freshness_guard_ready_count",
        ),
        "v2_signed_bundle_freshness_current_digest_present_count": _max_int(
            payloads,
            "v2_signed_bundle_freshness_current_digest_present_count",
        ),
        "v2_signed_bundle_freshness_template_digest_match_count": _max_int(
            payloads,
            "v2_signed_bundle_freshness_template_digest_match_count",
        ),
        "v2_signed_bundle_freshness_signed_bundle_present_count": _max_int(
            payloads,
            "v2_signed_bundle_freshness_signed_bundle_present_count",
        ),
        "v2_signed_bundle_freshness_signed_bundle_digest_match_count": _max_int(
            payloads,
            "v2_signed_bundle_freshness_signed_bundle_digest_match_count",
        ),
        "v2_signed_bundle_freshness_stale_signed_bundle_count": _max_int(
            payloads,
            "v2_signed_bundle_freshness_stale_signed_bundle_count",
        ),
        "v2_signed_bundle_freshness_unknown_signed_entry_count": _max_int(
            payloads,
            "v2_signed_bundle_freshness_unknown_signed_entry_count",
        ),
        "v2_signed_bundle_freshness_approval_recorded_count": _max_int(
            payloads,
            "v2_signed_bundle_freshness_approval_recorded_count",
        ),
        "v2_signed_bundle_freshness_approval_artifact_written_count": _max_int(
            payloads,
            "v2_signed_bundle_freshness_approval_artifact_written_count",
        ),
        "v2_signed_bundle_freshness_runtime_dispatch_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_freshness_runtime_dispatch_ready_count",
        ),
        "v2_signed_bundle_freshness_native_dispatch_allowed_count": _max_int(
            payloads,
            "v2_signed_bundle_freshness_native_dispatch_allowed_count",
        ),
        "v2_signed_bundle_freshness_training_path_enabled_count": _max_int(
            payloads,
            "v2_signed_bundle_freshness_training_path_enabled_count",
        ),
        "v2_signed_bundle_freshness_product_native_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_freshness_product_native_ready_count",
        ),
        "v2_signed_bundle_freshness_default_behavior_changed_count": _max_int(
            payloads,
            "v2_signed_bundle_freshness_default_behavior_changed_count",
        ),
        "v2_signed_bundle_freshness_unsafe_claim_count": _max_int(
            payloads,
            "v2_signed_bundle_freshness_unsafe_claim_count",
        ),
        "v2_signed_bundle_roundtrip_ready_template_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_ready_template_count",
        ),
        "v2_signed_bundle_roundtrip_signed_bundle_present_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_signed_bundle_present_count",
        ),
        "v2_signed_bundle_roundtrip_intake_valid_record_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_intake_valid_record_count",
        ),
        "v2_signed_bundle_roundtrip_extracted_entry_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_extracted_entry_count",
        ),
        "v2_signed_bundle_roundtrip_extraction_signed_entry_digest_present_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_extraction_signed_entry_digest_present_count",
        ),
        "v2_signed_bundle_roundtrip_extraction_extractable_signed_entry_digest_present_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_extraction_extractable_signed_entry_digest_present_count",
        ),
        "v2_signed_bundle_roundtrip_extraction_digest_shape_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_extraction_digest_shape_ready_count",
        ),
        "v2_signed_bundle_roundtrip_extraction_artifact_written_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_extraction_artifact_written_count",
        ),
        "v2_signed_bundle_roundtrip_preflight_signed_payload_digest_present_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_preflight_signed_payload_digest_present_count",
        ),
        "v2_signed_bundle_roundtrip_preflight_signed_payload_digest_missing_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_preflight_signed_payload_digest_missing_count",
        ),
        "v2_signed_bundle_roundtrip_preflight_signed_bundle_entry_digest_present_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_preflight_signed_bundle_entry_digest_present_count",
        ),
        "v2_signed_bundle_roundtrip_preflight_signed_payload_bundle_digest_match_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_preflight_signed_payload_bundle_digest_match_count",
        ),
        "v2_signed_bundle_roundtrip_preflight_digest_shape_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_preflight_digest_shape_ready_count",
        ),
        "v2_signed_bundle_roundtrip_phase1_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_phase1_ready_count",
        ),
        "v2_signed_bundle_roundtrip_phase2_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_phase2_ready_count",
        ),
        "v2_signed_bundle_roundtrip_full_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_full_ready_count",
        ),
        "v2_signed_bundle_roundtrip_owner_direction_blocked_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_owner_direction_blocked_count",
        ),
        "v2_signed_bundle_roundtrip_approval_recorded_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_approval_recorded_count",
        ),
        "v2_signed_bundle_roundtrip_approval_artifact_written_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_approval_artifact_written_count",
        ),
        "v2_signed_bundle_roundtrip_runtime_dispatch_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_runtime_dispatch_ready_count",
        ),
        "v2_signed_bundle_roundtrip_native_dispatch_allowed_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_native_dispatch_allowed_count",
        ),
        "v2_signed_bundle_roundtrip_training_path_enabled_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_training_path_enabled_count",
        ),
        "v2_signed_bundle_roundtrip_product_native_ready_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_product_native_ready_count",
        ),
        "v2_signed_bundle_roundtrip_default_behavior_changed_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_default_behavior_changed_count",
        ),
        "v2_signed_bundle_roundtrip_unsafe_claim_count": _max_int(
            payloads,
            "v2_signed_bundle_roundtrip_unsafe_claim_count",
        ),
        "v2_approval_state_stage_count": _max_int(payloads, "v2_approval_state_stage_count"),
        "v2_approval_state_ready_stage_count": _max_int(payloads, "v2_approval_state_ready_stage_count"),
        "v2_approval_state_waiting_signature_stage_count": _max_int(
            payloads,
            "v2_approval_state_waiting_signature_stage_count",
        ),
        "v2_approval_state_ready_for_record_stage_count": _max_int(
            payloads,
            "v2_approval_state_ready_for_record_stage_count",
        ),
        "v2_approval_state_recorded_stage_count": _max_int(payloads, "v2_approval_state_recorded_stage_count"),
        "v2_approval_state_missing_stage_count": _max_int(payloads, "v2_approval_state_missing_stage_count"),
        "v2_approval_state_blocked_stage_count": _max_int(payloads, "v2_approval_state_blocked_stage_count"),
        "v2_approval_state_remaining_gate_open_count": _max_int(
            payloads,
            "v2_approval_state_remaining_gate_open_count",
        ),
        "v2_approval_state_signature_ready_count": _max_int(payloads, "v2_approval_state_signature_ready_count"),
        "v2_approval_state_signed_bundle_present_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_present_count",
        ),
        "v2_approval_state_signed_bundle_freshness_guard_ready_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_freshness_guard_ready_count",
        ),
        "v2_approval_state_signed_bundle_freshness_template_digest_match_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_freshness_template_digest_match_count",
        ),
        "v2_approval_state_signed_bundle_freshness_stale_signed_bundle_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_freshness_stale_signed_bundle_count",
        ),
        "v2_approval_state_signed_bundle_freshness_unknown_signed_entry_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_freshness_unknown_signed_entry_count",
        ),
        "v2_approval_state_signed_bundle_freshness_unsafe_claim_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_freshness_unsafe_claim_count",
        ),
        "v2_approval_state_phase1_signed_bundle_ready_count": _max_int(
            payloads,
            "v2_approval_state_phase1_signed_bundle_ready_count",
        ),
        "v2_approval_state_phase2_signed_bundle_ready_count": _max_int(
            payloads,
            "v2_approval_state_phase2_signed_bundle_ready_count",
        ),
        "v2_approval_state_full_signed_bundle_ready_count": _max_int(
            payloads,
            "v2_approval_state_full_signed_bundle_ready_count",
        ),
        "v2_approval_state_signed_bundle_missing_signature_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_missing_signature_count",
        ),
        "v2_approval_state_phase1_signed_bundle_missing_signature_count": _max_int(
            payloads,
            "v2_approval_state_phase1_signed_bundle_missing_signature_count",
        ),
        "v2_approval_state_phase2_signed_bundle_missing_signature_count": _max_int(
            payloads,
            "v2_approval_state_phase2_signed_bundle_missing_signature_count",
        ),
        "v2_approval_state_signed_bundle_not_ready_entry_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_not_ready_entry_count",
        ),
        "v2_approval_state_signed_bundle_source_digest_match_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_source_digest_match_count",
        ),
        "v2_approval_state_signed_bundle_source_digest_stale_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_source_digest_stale_count",
        ),
        "v2_approval_state_signed_bundle_template_digest_mismatch_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_template_digest_mismatch_count",
        ),
        "v2_approval_state_signed_bundle_unknown_entry_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_unknown_entry_count",
        ),
        "v2_approval_state_signed_bundle_unsigned_template_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_unsigned_template_count",
        ),
        "v2_approval_state_signed_bundle_intake_integrity_ready_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_intake_integrity_ready_count",
        ),
        "v2_approval_state_signed_bundle_manual_field_check_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_manual_field_check_count",
        ),
        "v2_approval_state_signed_bundle_manual_field_ready_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_manual_field_ready_count",
        ),
        "v2_approval_state_signed_bundle_manual_field_missing_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_manual_field_missing_count",
        ),
        "v2_approval_state_signed_bundle_manual_field_shape_ready_count": _max_int(
            payloads,
            "v2_approval_state_signed_bundle_manual_field_shape_ready_count",
        ),
        "v2_approval_state_phase1_signed_bundle_manual_field_ready_count": _max_int(
            payloads,
            "v2_approval_state_phase1_signed_bundle_manual_field_ready_count",
        ),
        "v2_approval_state_phase2_signed_bundle_manual_field_ready_count": _max_int(
            payloads,
            "v2_approval_state_phase2_signed_bundle_manual_field_ready_count",
        ),
        "v2_approval_state_phase1_signed_bundle_manual_shape_ready_count": _max_int(
            payloads,
            "v2_approval_state_phase1_signed_bundle_manual_shape_ready_count",
        ),
        "v2_approval_state_phase2_signed_bundle_manual_shape_ready_count": _max_int(
            payloads,
            "v2_approval_state_phase2_signed_bundle_manual_shape_ready_count",
        ),
        "v2_approval_state_full_signed_bundle_manual_shape_ready_count": _max_int(
            payloads,
            "v2_approval_state_full_signed_bundle_manual_shape_ready_count",
        ),
        "v2_approval_state_phase1_extraction_ready_count": _max_int(
            payloads,
            "v2_approval_state_phase1_extraction_ready_count",
        ),
        "v2_approval_state_phase2_extraction_ready_count": _max_int(
            payloads,
            "v2_approval_state_phase2_extraction_ready_count",
        ),
        "v2_approval_state_full_extraction_ready_count": _max_int(
            payloads,
            "v2_approval_state_full_extraction_ready_count",
        ),
        "v2_approval_state_extraction_missing_signature_count": _max_int(
            payloads,
            "v2_approval_state_extraction_missing_signature_count",
        ),
        "v2_approval_state_phase1_extraction_missing_signature_count": _max_int(
            payloads,
            "v2_approval_state_phase1_extraction_missing_signature_count",
        ),
        "v2_approval_state_phase2_extraction_missing_signature_count": _max_int(
            payloads,
            "v2_approval_state_phase2_extraction_missing_signature_count",
        ),
        "v2_approval_state_extraction_not_ready_entry_count": _max_int(
            payloads,
            "v2_approval_state_extraction_not_ready_entry_count",
        ),
        "v2_approval_state_extraction_source_digest_match_count": _max_int(
            payloads,
            "v2_approval_state_extraction_source_digest_match_count",
        ),
        "v2_approval_state_extraction_source_digest_stale_count": _max_int(
            payloads,
            "v2_approval_state_extraction_source_digest_stale_count",
        ),
        "v2_approval_state_extraction_template_digest_mismatch_count": _max_int(
            payloads,
            "v2_approval_state_extraction_template_digest_mismatch_count",
        ),
        "v2_approval_state_extraction_unknown_entry_count": _max_int(
            payloads,
            "v2_approval_state_extraction_unknown_entry_count",
        ),
        "v2_approval_state_extraction_unsigned_template_count": _max_int(
            payloads,
            "v2_approval_state_extraction_unsigned_template_count",
        ),
        "v2_approval_state_extraction_integrity_ready_count": _max_int(
            payloads,
            "v2_approval_state_extraction_integrity_ready_count",
        ),
        "v2_approval_state_extraction_signed_entry_digest_present_count": _max_int(
            payloads,
            "v2_approval_state_extraction_signed_entry_digest_present_count",
        ),
        "v2_approval_state_extraction_extractable_signed_entry_digest_present_count": _max_int(
            payloads,
            "v2_approval_state_extraction_extractable_signed_entry_digest_present_count",
        ),
        "v2_approval_state_extraction_digest_shape_ready_count": _max_int(
            payloads,
            "v2_approval_state_extraction_digest_shape_ready_count",
        ),
        "v2_approval_state_command_audit_ready_count": _max_int(
            payloads,
            "v2_approval_state_command_audit_ready_count",
        ),
        "v2_approval_state_command_audit_expected_arg_binding_count": _max_int(
            payloads,
            "v2_approval_state_command_audit_expected_arg_binding_count",
        ),
        "v2_approval_state_command_audit_expected_arg_mismatch_count": _max_int(
            payloads,
            "v2_approval_state_command_audit_expected_arg_mismatch_count",
        ),
        "v2_approval_state_command_audit_expected_path_binding_ready_count": _max_int(
            payloads,
            "v2_approval_state_command_audit_expected_path_binding_ready_count",
        ),
        "v2_approval_state_command_audit_preflight_artifact_write_count": _max_int(
            payloads,
            "v2_approval_state_command_audit_preflight_artifact_write_count",
        ),
        "v2_approval_state_command_audit_preflight_no_artifact_blocker_count": _max_int(
            payloads,
            "v2_approval_state_command_audit_preflight_no_artifact_blocker_count",
        ),
        "v2_approval_state_command_audit_record_command_preflight_arg_count": _max_int(
            payloads,
            "v2_approval_state_command_audit_record_command_preflight_arg_count",
        ),
        "v2_approval_state_command_audit_phase1_handoff_post_return_command_count": _max_int(
            payloads,
            "v2_approval_state_command_audit_phase1_handoff_post_return_command_count",
        ),
        "v2_approval_state_command_audit_phase1_handoff_post_return_pre_record_command_count": _max_int(
            payloads,
            "v2_approval_state_command_audit_phase1_handoff_post_return_pre_record_command_count",
        ),
        "v2_approval_state_command_audit_phase1_handoff_post_return_approval_record_command_count": _max_int(
            payloads,
            "v2_approval_state_command_audit_phase1_handoff_post_return_approval_record_command_count",
        ),
        "v2_approval_state_command_audit_phase1_handoff_post_return_command_match_count": _max_int(
            payloads,
            "v2_approval_state_command_audit_phase1_handoff_post_return_command_match_count",
        ),
        "v2_approval_state_command_audit_phase1_handoff_post_return_command_mismatch_count": _max_int(
            payloads,
            "v2_approval_state_command_audit_phase1_handoff_post_return_command_mismatch_count",
        ),
        "v2_approval_state_command_audit_phase1_handoff_post_return_ready_count": _max_int(
            payloads,
            "v2_approval_state_command_audit_phase1_handoff_post_return_ready_count",
        ),
        "v2_approval_state_record_input_checklist_ready_count": _max_int(
            payloads,
            "v2_approval_state_record_input_checklist_ready_count",
        ),
        "v2_approval_state_record_input_support_check_count": _max_int(
            payloads,
            "v2_approval_state_record_input_support_check_count",
        ),
        "v2_approval_state_record_input_support_ready_count": _max_int(
            payloads,
            "v2_approval_state_record_input_support_ready_count",
        ),
        "v2_approval_state_record_input_support_blocked_item_count": _max_int(
            payloads,
            "v2_approval_state_record_input_support_blocked_item_count",
        ),
        "v2_approval_state_record_input_support_blocker_count": _max_int(
            payloads,
            "v2_approval_state_record_input_support_blocker_count",
        ),
        "v2_approval_state_record_input_support_source_binding_ready_count": _max_int(
            payloads,
            "v2_approval_state_record_input_support_source_binding_ready_count",
        ),
        "v2_approval_state_record_input_support_source_binding_blocker_count": _max_int(
            payloads,
            "v2_approval_state_record_input_support_source_binding_blocker_count",
        ),
        "v2_approval_state_record_input_support_shape_ready_count": _max_int(
            payloads,
            "v2_approval_state_record_input_support_shape_ready_count",
        ),
        "v2_approval_state_phase1_preflight_ready_count": _max_int(
            payloads,
            "v2_approval_state_phase1_preflight_ready_count",
        ),
        "v2_approval_state_phase2_preflight_ready_count": _max_int(
            payloads,
            "v2_approval_state_phase2_preflight_ready_count",
        ),
        "v2_approval_state_full_preflight_ready_count": _max_int(
            payloads,
            "v2_approval_state_full_preflight_ready_count",
        ),
        "v2_approval_state_preflight_not_ready_entry_count": _max_int(
            payloads,
            "v2_approval_state_preflight_not_ready_entry_count",
        ),
        "v2_approval_state_preflight_source_digest_match_count": _max_int(
            payloads,
            "v2_approval_state_preflight_source_digest_match_count",
        ),
        "v2_approval_state_preflight_source_digest_stale_count": _max_int(
            payloads,
            "v2_approval_state_preflight_source_digest_stale_count",
        ),
        "v2_approval_state_preflight_template_digest_mismatch_count": _max_int(
            payloads,
            "v2_approval_state_preflight_template_digest_mismatch_count",
        ),
        "v2_approval_state_preflight_unknown_entry_count": _max_int(
            payloads,
            "v2_approval_state_preflight_unknown_entry_count",
        ),
        "v2_approval_state_preflight_unsigned_template_count": _max_int(
            payloads,
            "v2_approval_state_preflight_unsigned_template_count",
        ),
        "v2_approval_state_preflight_source_integrity_ready_count": _max_int(
            payloads,
            "v2_approval_state_preflight_source_integrity_ready_count",
        ),
        "v2_approval_state_preflight_support_ready_count": _max_int(
            payloads,
            "v2_approval_state_preflight_support_ready_count",
        ),
        "v2_approval_state_preflight_support_invalid_count": _max_int(
            payloads,
            "v2_approval_state_preflight_support_invalid_count",
        ),
        "v2_approval_state_preflight_support_source_binding_ready_count": _max_int(
            payloads,
            "v2_approval_state_preflight_support_source_binding_ready_count",
        ),
        "v2_approval_state_preflight_support_source_binding_blocker_count": _max_int(
            payloads,
            "v2_approval_state_preflight_support_source_binding_blocker_count",
        ),
        "v2_approval_state_preflight_post_record_request_field_emission_count": _max_int(
            payloads,
            "v2_approval_state_preflight_post_record_request_field_emission_count",
        ),
        "v2_approval_state_preflight_hard_fail_count": _max_int(
            payloads,
            "v2_approval_state_preflight_hard_fail_count",
        ),
        "v2_approval_state_preflight_support_integrity_ready_count": _max_int(
            payloads,
            "v2_approval_state_preflight_support_integrity_ready_count",
        ),
        "v2_approval_state_phase1_preflight_support_integrity_ready_count": _max_int(
            payloads,
            "v2_approval_state_phase1_preflight_support_integrity_ready_count",
        ),
        "v2_approval_state_phase2_preflight_support_integrity_ready_count": _max_int(
            payloads,
            "v2_approval_state_phase2_preflight_support_integrity_ready_count",
        ),
        "v2_approval_state_preflight_signed_payload_digest_present_count": _max_int(
            payloads,
            "v2_approval_state_preflight_signed_payload_digest_present_count",
        ),
        "v2_approval_state_preflight_signed_payload_digest_missing_count": _max_int(
            payloads,
            "v2_approval_state_preflight_signed_payload_digest_missing_count",
        ),
        "v2_approval_state_preflight_signed_bundle_entry_digest_present_count": _max_int(
            payloads,
            "v2_approval_state_preflight_signed_bundle_entry_digest_present_count",
        ),
        "v2_approval_state_preflight_signed_payload_bundle_digest_match_count": _max_int(
            payloads,
            "v2_approval_state_preflight_signed_payload_bundle_digest_match_count",
        ),
        "v2_approval_state_preflight_digest_shape_ready_count": _max_int(
            payloads,
            "v2_approval_state_preflight_digest_shape_ready_count",
        ),
        "v2_approval_state_phase1_record_chain_ready_count": _max_int(
            payloads,
            "v2_approval_state_phase1_record_chain_ready_count",
        ),
        "v2_approval_state_phase2_record_chain_ready_count": _max_int(
            payloads,
            "v2_approval_state_phase2_record_chain_ready_count",
        ),
        "v2_approval_state_full_record_chain_ready_count": _max_int(
            payloads,
            "v2_approval_state_full_record_chain_ready_count",
        ),
        "v2_approval_state_owner_review_recorded_count": _max_int(
            payloads,
            "v2_approval_state_owner_review_recorded_count",
        ),
        "v2_approval_state_product_exposure_recorded_count": _max_int(
            payloads,
            "v2_approval_state_product_exposure_recorded_count",
        ),
        "v2_approval_state_owner_direction_recorded_count": _max_int(
            payloads,
            "v2_approval_state_owner_direction_recorded_count",
        ),
        "v2_approval_state_phase1_record_preflight_binding_ready_count": _max_int(
            payloads,
            "v2_approval_state_phase1_record_preflight_binding_ready_count",
        ),
        "v2_approval_state_phase2_record_preflight_binding_ready_count": _max_int(
            payloads,
            "v2_approval_state_phase2_record_preflight_binding_ready_count",
        ),
        "v2_approval_state_full_record_preflight_binding_ready_count": _max_int(
            payloads,
            "v2_approval_state_full_record_preflight_binding_ready_count",
        ),
        "v2_approval_state_record_preflight_binding_ready_count": _max_int(
            payloads,
            "v2_approval_state_record_preflight_binding_ready_count",
        ),
        "v2_approval_state_approval_recorded_count": _max_int(
            payloads,
            "v2_approval_state_approval_recorded_count",
        ),
        "v2_approval_state_runtime_dispatch_ready_count": _max_int(
            payloads,
            "v2_approval_state_runtime_dispatch_ready_count",
        ),
        "v2_approval_state_native_dispatch_allowed_count": _max_int(
            payloads,
            "v2_approval_state_native_dispatch_allowed_count",
        ),
        "v2_approval_state_training_path_enabled_count": _max_int(
            payloads,
            "v2_approval_state_training_path_enabled_count",
        ),
        "v2_approval_state_product_native_ready_count": _max_int(
            payloads,
            "v2_approval_state_product_native_ready_count",
        ),
        "v2_approval_state_default_behavior_changed_count": _max_int(
            payloads,
            "v2_approval_state_default_behavior_changed_count",
        ),
        "v2_approval_state_unsafe_claim_count": _max_int(payloads, "v2_approval_state_unsafe_claim_count"),
        "v2_completion_audit_requirement_count": _max_int(payloads, "v2_completion_audit_requirement_count"),
        "v2_completion_audit_passed_requirement_count": _max_int(
            payloads,
            "v2_completion_audit_passed_requirement_count",
        ),
        "v2_completion_audit_failed_requirement_count": _max_int(
            payloads,
            "v2_completion_audit_failed_requirement_count",
        ),
        "v2_completion_audit_remaining_gate_open_count": _max_int(
            payloads,
            "v2_completion_audit_remaining_gate_open_count",
        ),
        "v2_completion_audit_command_audit_ready_count": _max_int(
            payloads,
            "v2_completion_audit_command_audit_ready_count",
        ),
        "v2_completion_audit_command_audit_expected_arg_binding_count": _max_int(
            payloads,
            "v2_completion_audit_command_audit_expected_arg_binding_count",
        ),
        "v2_completion_audit_command_audit_expected_arg_mismatch_count": _max_int(
            payloads,
            "v2_completion_audit_command_audit_expected_arg_mismatch_count",
        ),
        "v2_completion_audit_command_audit_expected_path_binding_ready_count": _max_int(
            payloads,
            "v2_completion_audit_command_audit_expected_path_binding_ready_count",
        ),
        "v2_completion_audit_command_audit_phase1_handoff_post_return_command_count": _max_int(
            payloads,
            "v2_completion_audit_command_audit_phase1_handoff_post_return_command_count",
        ),
        "v2_completion_audit_command_audit_phase1_handoff_post_return_pre_record_command_count": _max_int(
            payloads,
            "v2_completion_audit_command_audit_phase1_handoff_post_return_pre_record_command_count",
        ),
        "v2_completion_audit_command_audit_phase1_handoff_post_return_approval_record_command_count": _max_int(
            payloads,
            "v2_completion_audit_command_audit_phase1_handoff_post_return_approval_record_command_count",
        ),
        "v2_completion_audit_command_audit_phase1_handoff_post_return_command_match_count": _max_int(
            payloads,
            "v2_completion_audit_command_audit_phase1_handoff_post_return_command_match_count",
        ),
        "v2_completion_audit_command_audit_phase1_handoff_post_return_command_mismatch_count": _max_int(
            payloads,
            "v2_completion_audit_command_audit_phase1_handoff_post_return_command_mismatch_count",
        ),
        "v2_completion_audit_command_audit_phase1_handoff_post_return_ready_count": _max_int(
            payloads,
            "v2_completion_audit_command_audit_phase1_handoff_post_return_ready_count",
        ),
        "v2_completion_audit_phase1_ready_count": _max_int(payloads, "v2_completion_audit_phase1_ready_count"),
        "v2_completion_audit_phase2_ready_count": _max_int(payloads, "v2_completion_audit_phase2_ready_count"),
        "v2_completion_audit_record_input_item_count": _max_int(
            payloads,
            "v2_completion_audit_record_input_item_count",
        ),
        "v2_completion_audit_record_input_phase1_item_count": _max_int(
            payloads,
            "v2_completion_audit_record_input_phase1_item_count",
        ),
        "v2_completion_audit_record_input_phase2_item_count": _max_int(
            payloads,
            "v2_completion_audit_record_input_phase2_item_count",
        ),
        "v2_completion_audit_phase2_refresh_inputs_tracked_count": _max_int(
            payloads,
            "v2_completion_audit_phase2_refresh_inputs_tracked_count",
        ),
        "v2_completion_audit_record_input_checklist_shape_ready_count": _max_int(
            payloads,
            "v2_completion_audit_record_input_checklist_shape_ready_count",
        ),
        "v2_completion_audit_record_input_support_check_count": _max_int(
            payloads,
            "v2_completion_audit_record_input_support_check_count",
        ),
        "v2_completion_audit_record_input_support_ready_count": _max_int(
            payloads,
            "v2_completion_audit_record_input_support_ready_count",
        ),
        "v2_completion_audit_record_input_support_blocked_item_count": _max_int(
            payloads,
            "v2_completion_audit_record_input_support_blocked_item_count",
        ),
        "v2_completion_audit_record_input_support_blocker_count": _max_int(
            payloads,
            "v2_completion_audit_record_input_support_blocker_count",
        ),
        "v2_completion_audit_record_input_support_source_binding_ready_count": _max_int(
            payloads,
            "v2_completion_audit_record_input_support_source_binding_ready_count",
        ),
        "v2_completion_audit_record_input_support_source_binding_blocker_count": _max_int(
            payloads,
            "v2_completion_audit_record_input_support_source_binding_blocker_count",
        ),
        "v2_completion_audit_record_input_support_shape_ready_count": _max_int(
            payloads,
            "v2_completion_audit_record_input_support_shape_ready_count",
        ),
        "v2_completion_audit_phase1_signed_bundle_ready_count": _max_int(
            payloads,
            "v2_completion_audit_phase1_signed_bundle_ready_count",
        ),
        "v2_completion_audit_phase2_signed_bundle_ready_count": _max_int(
            payloads,
            "v2_completion_audit_phase2_signed_bundle_ready_count",
        ),
        "v2_completion_audit_signed_bundle_missing_signature_count": _max_int(
            payloads,
            "v2_completion_audit_signed_bundle_missing_signature_count",
        ),
        "v2_completion_audit_phase1_signed_bundle_missing_signature_count": _max_int(
            payloads,
            "v2_completion_audit_phase1_signed_bundle_missing_signature_count",
        ),
        "v2_completion_audit_phase2_signed_bundle_missing_signature_count": _max_int(
            payloads,
            "v2_completion_audit_phase2_signed_bundle_missing_signature_count",
        ),
        "v2_completion_audit_signed_bundle_freshness_guard_ready_count": _max_int(
            payloads,
            "v2_completion_audit_signed_bundle_freshness_guard_ready_count",
        ),
        "v2_completion_audit_signed_bundle_freshness_template_digest_match_count": _max_int(
            payloads,
            "v2_completion_audit_signed_bundle_freshness_template_digest_match_count",
        ),
        "v2_completion_audit_signed_bundle_freshness_stale_signed_bundle_count": _max_int(
            payloads,
            "v2_completion_audit_signed_bundle_freshness_stale_signed_bundle_count",
        ),
        "v2_completion_audit_signed_bundle_freshness_unknown_signed_entry_count": _max_int(
            payloads,
            "v2_completion_audit_signed_bundle_freshness_unknown_signed_entry_count",
        ),
        "v2_completion_audit_signed_bundle_freshness_unsafe_claim_count": _max_int(
            payloads,
            "v2_completion_audit_signed_bundle_freshness_unsafe_claim_count",
        ),
        "v2_completion_audit_signed_bundle_not_ready_entry_count": _max_int(
            payloads,
            "v2_completion_audit_signed_bundle_not_ready_entry_count",
        ),
        "v2_completion_audit_signed_bundle_source_digest_match_count": _max_int(
            payloads,
            "v2_completion_audit_signed_bundle_source_digest_match_count",
        ),
        "v2_completion_audit_signed_bundle_source_digest_stale_count": _max_int(
            payloads,
            "v2_completion_audit_signed_bundle_source_digest_stale_count",
        ),
        "v2_completion_audit_signed_bundle_template_digest_mismatch_count": _max_int(
            payloads,
            "v2_completion_audit_signed_bundle_template_digest_mismatch_count",
        ),
        "v2_completion_audit_signed_bundle_unknown_entry_count": _max_int(
            payloads,
            "v2_completion_audit_signed_bundle_unknown_entry_count",
        ),
        "v2_completion_audit_signed_bundle_unsigned_template_count": _max_int(
            payloads,
            "v2_completion_audit_signed_bundle_unsigned_template_count",
        ),
        "v2_completion_audit_signed_bundle_intake_integrity_ready_count": _max_int(
            payloads,
            "v2_completion_audit_signed_bundle_intake_integrity_ready_count",
        ),
        "v2_completion_audit_signed_bundle_manual_field_check_count": _max_int(
            payloads,
            "v2_completion_audit_signed_bundle_manual_field_check_count",
        ),
        "v2_completion_audit_signed_bundle_manual_field_ready_count": _max_int(
            payloads,
            "v2_completion_audit_signed_bundle_manual_field_ready_count",
        ),
        "v2_completion_audit_signed_bundle_manual_field_missing_count": _max_int(
            payloads,
            "v2_completion_audit_signed_bundle_manual_field_missing_count",
        ),
        "v2_completion_audit_signed_bundle_manual_field_shape_ready_count": _max_int(
            payloads,
            "v2_completion_audit_signed_bundle_manual_field_shape_ready_count",
        ),
        "v2_completion_audit_phase1_signed_bundle_manual_field_ready_count": _max_int(
            payloads,
            "v2_completion_audit_phase1_signed_bundle_manual_field_ready_count",
        ),
        "v2_completion_audit_phase2_signed_bundle_manual_field_ready_count": _max_int(
            payloads,
            "v2_completion_audit_phase2_signed_bundle_manual_field_ready_count",
        ),
        "v2_completion_audit_phase1_signed_bundle_manual_shape_ready_count": _max_int(
            payloads,
            "v2_completion_audit_phase1_signed_bundle_manual_shape_ready_count",
        ),
        "v2_completion_audit_phase2_signed_bundle_manual_shape_ready_count": _max_int(
            payloads,
            "v2_completion_audit_phase2_signed_bundle_manual_shape_ready_count",
        ),
        "v2_completion_audit_full_signed_bundle_manual_shape_ready_count": _max_int(
            payloads,
            "v2_completion_audit_full_signed_bundle_manual_shape_ready_count",
        ),
        "v2_completion_audit_phase1_extraction_ready_count": _max_int(
            payloads,
            "v2_completion_audit_phase1_extraction_ready_count",
        ),
        "v2_completion_audit_phase2_extraction_ready_count": _max_int(
            payloads,
            "v2_completion_audit_phase2_extraction_ready_count",
        ),
        "v2_completion_audit_extraction_missing_signature_count": _max_int(
            payloads,
            "v2_completion_audit_extraction_missing_signature_count",
        ),
        "v2_completion_audit_phase1_extraction_missing_signature_count": _max_int(
            payloads,
            "v2_completion_audit_phase1_extraction_missing_signature_count",
        ),
        "v2_completion_audit_phase2_extraction_missing_signature_count": _max_int(
            payloads,
            "v2_completion_audit_phase2_extraction_missing_signature_count",
        ),
        "v2_completion_audit_extraction_not_ready_entry_count": _max_int(
            payloads,
            "v2_completion_audit_extraction_not_ready_entry_count",
        ),
        "v2_completion_audit_extraction_source_digest_match_count": _max_int(
            payloads,
            "v2_completion_audit_extraction_source_digest_match_count",
        ),
        "v2_completion_audit_extraction_source_digest_stale_count": _max_int(
            payloads,
            "v2_completion_audit_extraction_source_digest_stale_count",
        ),
        "v2_completion_audit_extraction_template_digest_mismatch_count": _max_int(
            payloads,
            "v2_completion_audit_extraction_template_digest_mismatch_count",
        ),
        "v2_completion_audit_extraction_unknown_entry_count": _max_int(
            payloads,
            "v2_completion_audit_extraction_unknown_entry_count",
        ),
        "v2_completion_audit_extraction_unsigned_template_count": _max_int(
            payloads,
            "v2_completion_audit_extraction_unsigned_template_count",
        ),
        "v2_completion_audit_extraction_integrity_ready_count": _max_int(
            payloads,
            "v2_completion_audit_extraction_integrity_ready_count",
        ),
        "v2_completion_audit_extraction_signed_entry_digest_present_count": _max_int(
            payloads,
            "v2_completion_audit_extraction_signed_entry_digest_present_count",
        ),
        "v2_completion_audit_extraction_extractable_signed_entry_digest_present_count": _max_int(
            payloads,
            "v2_completion_audit_extraction_extractable_signed_entry_digest_present_count",
        ),
        "v2_completion_audit_extraction_digest_shape_ready_count": _max_int(
            payloads,
            "v2_completion_audit_extraction_digest_shape_ready_count",
        ),
        "v2_completion_audit_phase1_preflight_ready_count": _max_int(
            payloads,
            "v2_completion_audit_phase1_preflight_ready_count",
        ),
        "v2_completion_audit_phase2_preflight_ready_count": _max_int(
            payloads,
            "v2_completion_audit_phase2_preflight_ready_count",
        ),
        "v2_completion_audit_phase1_record_chain_ready_count": _max_int(
            payloads,
            "v2_completion_audit_phase1_record_chain_ready_count",
        ),
        "v2_completion_audit_phase2_record_chain_ready_count": _max_int(
            payloads,
            "v2_completion_audit_phase2_record_chain_ready_count",
        ),
        "v2_completion_audit_full_record_chain_ready_count": _max_int(
            payloads,
            "v2_completion_audit_full_record_chain_ready_count",
        ),
        "v2_completion_audit_preflight_full_ready_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_full_ready_count",
        ),
        "v2_completion_audit_preflight_integrity_ready_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_integrity_ready_count",
        ),
        "v2_completion_audit_preflight_signed_bundle_valid_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_signed_bundle_valid_count",
        ),
        "v2_completion_audit_preflight_source_digest_match_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_source_digest_match_count",
        ),
        "v2_completion_audit_preflight_source_digest_stale_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_source_digest_stale_count",
        ),
        "v2_completion_audit_preflight_template_digest_mismatch_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_template_digest_mismatch_count",
        ),
        "v2_completion_audit_preflight_unknown_entry_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_unknown_entry_count",
        ),
        "v2_completion_audit_preflight_not_ready_entry_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_not_ready_entry_count",
        ),
        "v2_completion_audit_preflight_source_integrity_ready_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_source_integrity_ready_count",
        ),
        "v2_completion_audit_preflight_support_integrity_ready_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_support_integrity_ready_count",
        ),
        "v2_completion_audit_preflight_unsigned_template_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_unsigned_template_count",
        ),
        "v2_completion_audit_preflight_signed_payload_digest_present_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_signed_payload_digest_present_count",
        ),
        "v2_completion_audit_preflight_signed_payload_digest_missing_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_signed_payload_digest_missing_count",
        ),
        "v2_completion_audit_preflight_signed_bundle_entry_digest_present_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_signed_bundle_entry_digest_present_count",
        ),
        "v2_completion_audit_preflight_signed_payload_bundle_digest_match_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_signed_payload_bundle_digest_match_count",
        ),
        "v2_completion_audit_preflight_extracted_entry_digest_match_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_extracted_entry_digest_match_count",
        ),
        "v2_completion_audit_preflight_extracted_entry_digest_mismatch_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_extracted_entry_digest_mismatch_count",
        ),
        "v2_completion_audit_preflight_extracted_entry_source_missing_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_extracted_entry_source_missing_count",
        ),
        "v2_completion_audit_preflight_support_ready_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_support_ready_count",
        ),
        "v2_completion_audit_preflight_support_invalid_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_support_invalid_count",
        ),
        "v2_completion_audit_preflight_support_source_binding_ready_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_support_source_binding_ready_count",
        ),
        "v2_completion_audit_preflight_support_source_binding_blocker_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_support_source_binding_blocker_count",
        ),
        "v2_completion_audit_preflight_post_record_request_field_emission_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_post_record_request_field_emission_count",
        ),
        "v2_completion_audit_preflight_hard_fail_count": _max_int(
            payloads,
            "v2_completion_audit_preflight_hard_fail_count",
        ),
        "v2_completion_audit_owner_review_recorded_count": _max_int(
            payloads,
            "v2_completion_audit_owner_review_recorded_count",
        ),
        "v2_completion_audit_product_exposure_recorded_count": _max_int(
            payloads,
            "v2_completion_audit_product_exposure_recorded_count",
        ),
        "v2_completion_audit_owner_direction_recorded_count": _max_int(
            payloads,
            "v2_completion_audit_owner_direction_recorded_count",
        ),
        "v2_completion_audit_phase1_record_preflight_binding_ready_count": _max_int(
            payloads,
            "v2_completion_audit_phase1_record_preflight_binding_ready_count",
        ),
        "v2_completion_audit_phase2_record_preflight_binding_ready_count": _max_int(
            payloads,
            "v2_completion_audit_phase2_record_preflight_binding_ready_count",
        ),
        "v2_completion_audit_full_record_preflight_binding_ready_count": _max_int(
            payloads,
            "v2_completion_audit_full_record_preflight_binding_ready_count",
        ),
        "v2_completion_audit_record_preflight_binding_ready_count": _max_int(
            payloads,
            "v2_completion_audit_record_preflight_binding_ready_count",
        ),
        "v2_completion_audit_artifact_first_required_artifact_count": _max_int(
            payloads,
            "v2_completion_audit_artifact_first_required_artifact_count",
        ),
        "v2_completion_audit_artifact_first_ready_artifact_count": _max_int(
            payloads,
            "v2_completion_audit_artifact_first_ready_artifact_count",
        ),
        "v2_completion_audit_artifact_first_ready_count": _max_int(
            payloads,
            "v2_completion_audit_artifact_first_ready_count",
        ),
        "v2_completion_audit_artifact_first_record_input_checklist_shape_ready_count": _max_int(
            payloads,
            "v2_completion_audit_artifact_first_record_input_checklist_shape_ready_count",
        ),
        "v2_completion_audit_artifact_first_record_input_support_shape_ready_count": _max_int(
            payloads,
            "v2_completion_audit_artifact_first_record_input_support_shape_ready_count",
        ),
        "v2_completion_audit_artifact_first_reviewer_handoff_phase_metadata_ready_count": _max_int(
            payloads,
            "v2_completion_audit_artifact_first_reviewer_handoff_phase_metadata_ready_count",
        ),
        "v2_completion_audit_artifact_first_reviewer_handoff_phase1_template_entry_count": _max_int(
            payloads,
            "v2_completion_audit_artifact_first_reviewer_handoff_phase1_template_entry_count",
        ),
        "v2_completion_audit_artifact_first_reviewer_handoff_phase2_deferred_signature_count": _max_int(
            payloads,
            "v2_completion_audit_artifact_first_reviewer_handoff_phase2_deferred_signature_count",
        ),
        "v2_completion_audit_artifact_first_reviewer_handoff_phase_shape_ready_count": _max_int(
            payloads,
            "v2_completion_audit_artifact_first_reviewer_handoff_phase_shape_ready_count",
        ),
        "v2_completion_audit_artifact_first_command_audit_expected_path_binding_ready_count": _max_int(
            payloads,
            "v2_completion_audit_artifact_first_command_audit_expected_path_binding_ready_count",
        ),
        "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_count": _max_int(
            payloads,
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_count",
        ),
        "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_pre_record_command_count": (
            _max_int(
                payloads,
                "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_pre_record_command_count",
            )
        ),
        "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_approval_record_command_count": (
            _max_int(
                payloads,
                (
                    "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_"
                    "approval_record_command_count"
                ),
            )
        ),
        "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_match_count": (
            _max_int(
                payloads,
                "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_match_count",
            )
        ),
        "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_mismatch_count": (
            _max_int(
                payloads,
                "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_mismatch_count",
            )
        ),
        "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_ready_count": _max_int(
            payloads,
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_ready_count",
        ),
        "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_shape_ready_count": _max_int(
            payloads,
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_shape_ready_count",
        ),
        "v2_completion_audit_default_off_guard_count": _max_int(
            payloads,
            "v2_completion_audit_default_off_guard_count",
        ),
        "v2_completion_audit_completion_ready_count": _max_int(
            payloads,
            "v2_completion_audit_completion_ready_count",
        ),
        "v2_completion_audit_approval_recorded_count": _max_int(
            payloads,
            "v2_completion_audit_approval_recorded_count",
        ),
        "v2_completion_audit_runtime_dispatch_ready_count": _max_int(
            payloads,
            "v2_completion_audit_runtime_dispatch_ready_count",
        ),
        "v2_completion_audit_native_dispatch_allowed_count": _max_int(
            payloads,
            "v2_completion_audit_native_dispatch_allowed_count",
        ),
        "v2_completion_audit_training_path_enabled_count": _max_int(
            payloads,
            "v2_completion_audit_training_path_enabled_count",
        ),
        "v2_completion_audit_product_native_ready_count": _max_int(
            payloads,
            "v2_completion_audit_product_native_ready_count",
        ),
        "v2_completion_audit_default_behavior_changed_count": _max_int(
            payloads,
            "v2_completion_audit_default_behavior_changed_count",
        ),
        "v2_completion_audit_unsafe_claim_count": _max_int(payloads, "v2_completion_audit_unsafe_claim_count"),
        "product_training_route_binding_ready_count": _max_int(payloads, "product_training_route_binding_ready_count"),
        "product_route_binding_chain_stage_count": _max_int(payloads, "product_route_binding_chain_stage_count"),
        "product_route_binding_chain_ready_stage_count": _max_int(
            payloads,
            "product_route_binding_chain_ready_stage_count",
        ),
        "product_route_binding_chain_open_stage_count": _max_int(
            payloads,
            "product_route_binding_chain_open_stage_count",
        ),
        "product_route_binding_chain_contract_ready_count": _max_int(
            payloads,
            "product_route_binding_chain_contract_ready_count",
        ),
        "product_route_binding_chain_approval_missing_count": _max_int(
            payloads,
            "product_route_binding_chain_approval_missing_count",
        ),
        "product_route_binding_chain_product_training_route_bound_count": _max_int(
            payloads,
            "product_route_binding_chain_product_training_route_bound_count",
        ),
        "product_route_binding_chain_runtime_dispatch_ready_count": _max_int(
            payloads,
            "product_route_binding_chain_runtime_dispatch_ready_count",
        ),
        "product_route_binding_chain_native_dispatch_allowed_count": _max_int(
            payloads,
            "product_route_binding_chain_native_dispatch_allowed_count",
        ),
        "product_route_binding_chain_training_path_enabled_count": _max_int(
            payloads,
            "product_route_binding_chain_training_path_enabled_count",
        ),
        "product_route_binding_chain_default_behavior_changed_count": _max_int(
            payloads,
            "product_route_binding_chain_default_behavior_changed_count",
        ),
        "product_route_binding_chain_product_native_ready_count": _max_int(
            payloads,
            "product_route_binding_chain_product_native_ready_count",
        ),
        "training_loop_contract_open_training_path_enabled_count": _max_int(payloads, "open_training_path_enabled"),
        "training_loop_contract_candidate_switch_count": _max_int(payloads, "candidate_switch_count"),
        "route_binding_config_patch_ready_count": _max_int(payloads, "route_binding_config_patch_ready_count", "product_training_route_binding_config_patch_ready_count"),
        "product_training_route_binding_kwargs_wired_count": _max_int(payloads, "product_training_route_binding_kwargs_wired_count"),
        "runtime_config_patch_applied_count": _max_int(payloads, "runtime_config_patch_applied_count"),
        "run_local_adapter_staged_count": _max_int(payloads, "run_local_adapter_staged_count"),
        "product_launch_staging_wired_count": _max_int(payloads, "product_launch_staging_wired_count"),
        "stable_first_release_turbocore_optimizer_blocker_count": _max_int(payloads, "stable_first_release_turbocore_optimizer_blocker_count"),
        "turbocore_optimizer_default_off_release_scope_ready_count": _max_int(payloads, "turbocore_optimizer_default_off_release_scope_ready_count"),
        "native_update_rollout_review_package_ready_count": _passed_smoke_count(results, ("native_update_rollout_review_package",)),
        "native_update_execution_ladder_ready_count": ladder_ready_count,
        "native_update_execution_ladder_default_off_count": ladder_ready_count,
        "native_update_execution_ladder_training_path_enabled_count": _max_int(
            ladder_payloads,
            "training_path_enabled_count",
            "training_path_enabled",
        ),
        "phase1_success_review_ready_count": _passed_smoke_count(results, ("phase1_success_review",)),
        "phase1_success_review_default_off_count": _passed_smoke_count(results, ("phase1_success_review",)),
        "p6_audit_ready_gate_count": _max_int(payloads, "p6_ready_gate_count"),
        "p6_audit_required_gate_count": _max_int(payloads, "p6_required_gate_count"),
        "p6_audit_remaining_blocker_count": _max_int(payloads, "p6_remaining_blocker_count"),
        "p6_audit_milestone_completed_count": _max_int(payloads, "p6_milestone_completed_count"),
        "p6_audit_section_count": _max_int(payloads, "p6_section_count"),
        "p6_audit_required_promotion_count": _max_int(payloads, "p6_required_promotion_count"),
        "native_kernel_inventory_source_ready_count": _max_int(payloads, "optimizer_native_kernel_inventory_source_ready_count", "kernel_source_present_count"),
        "native_kernel_inventory_probe_ready_count": _max_int(payloads, "optimizer_native_kernel_inventory_probe_ready_count", "rust_probe_present_count"),
        "product_native_ready_count": _max_int(payloads, "product_native_ready_count"),
        "runtime_dispatch_ready_count": _max_int(payloads, "runtime_dispatch_ready_count"),
        "plugin_actual_training_count": _max_int(payloads, "per_optimizer_native_training_count"),
        "plugin_actual_training_gap_count": _max_int(payloads, "actual_training_gap_count"),
        "plugin_trainer_resume_parity_proven_count": _max_int(payloads, "trainer_resume_parity_proven_count"),
        "plugin_representative_native_training_mapped_count": _max_int(
            payloads,
            "representative_native_training_mapped_count",
        ),
        "plugin_actual_training_complete_count": _max_int(payloads, "actual_training_complete_count"),
        "plugin_actual_training_status_counts": _first_mapping(payloads, "actual_training_status_counts"),
        "plugin_adaptivelr_training_loop_case_count": _max_int(
            payloads,
            "plugin_adaptivelr_training_loop_case_count",
        ),
        "plugin_adaptivelr_training_loop_native_step_count": _max_int(
            payloads,
            "plugin_adaptivelr_training_loop_native_step_count",
        ),
        "plugin_adaptivelr_training_loop_native_kernel_launch_count": _max_int(
            payloads,
            "plugin_adaptivelr_training_loop_native_kernel_launch_count",
        ),
        "plugin_adaptivelr_training_loop_training_executor_called_count": _max_int(
            payloads,
            "plugin_adaptivelr_training_loop_training_executor_called_count",
        ),
        "plugin_adaptivelr_training_loop_skip_pytorch_count": _max_int(
            payloads,
            "plugin_adaptivelr_training_loop_skip_pytorch_count",
        ),
        "plugin_adaptivelr_training_loop_native_family_alias_count": _max_int(
            payloads,
            "plugin_adaptivelr_training_loop_native_family_alias_count",
        ),
        "plugin_fused_backward_hook_canary_case_count": _max_int(
            payloads,
            "plugin_fused_backward_hook_canary_case_count",
        ),
        "plugin_fused_backward_hook_canary_native_step_count": _max_int(
            payloads,
            "plugin_fused_backward_hook_canary_native_step_count",
        ),
        "plugin_fused_backward_hook_canary_native_kernel_launch_count": _max_int(
            payloads,
            "plugin_fused_backward_hook_canary_native_kernel_launch_count",
        ),
        "plugin_fused_backward_hook_canary_public_step_called_count": _max_int(
            payloads,
            "plugin_fused_backward_hook_canary_public_step_called_count",
        ),
        "plugin_bridge_training_loop_canary_case_count": _max_int(
            payloads,
            "plugin_bridge_training_loop_canary_case_count",
        ),
        "plugin_bridge_training_loop_canary_native_step_count": _max_int(
            payloads,
            "plugin_bridge_training_loop_canary_native_step_count",
        ),
        "plugin_bridge_training_loop_canary_native_kernel_launch_count": _max_int(
            payloads,
            "plugin_bridge_training_loop_canary_native_kernel_launch_count",
        ),
        "plugin_bridge_training_loop_canary_training_executor_called_count": _max_int(
            payloads,
            "plugin_bridge_training_loop_canary_training_executor_called_count",
        ),
        "plugin_bridge_training_loop_canary_skip_pytorch_count": _max_int(
            payloads,
            "plugin_bridge_training_loop_canary_skip_pytorch_count",
        ),
        "plugin_training_loop_passed_count": _max_int(payloads, "plugin_training_loop_passed_count"),
        "plugin_training_loop_failed_count": _max_int(payloads, "plugin_training_loop_failed_count"),
        "plugin_training_loop_step_route_executed_count": _max_int(
            payloads,
            "plugin_training_loop_step_route_executed_count",
        ),
        "plugin_training_loop_param_updated_count": _max_int(payloads, "plugin_training_loop_param_updated_count"),
        "plugin_training_loop_route_counts": _first_mapping(payloads, "plugin_training_loop_route_counts"),
        **_max_int_summary(payloads, STREAM_LIFETIME_SUMMARY_KEYS),
        **_max_int_summary(payloads, TRAINING_LOOP_SUMMARY_KEYS),
        "native_dispatch_allowed_count": _max_int(payloads, "native_dispatch_allowed_count"),
        "training_path_enabled_count": _max_int(payloads, "training_path_enabled_count"),
        "route_family_counts": _first_mapping(payloads, "route_family_counts"),
        "selected_family_counts": _first_mapping(payloads, "selected_family_counts"),
        "workflow": (
            "Use this profiled suite first. Run individual smoke files only after "
            "the suite identifies the failing area; use explicit rebuild entrypoints "
            "only when artifacts must be refreshed."
        ),
    }


_NESTED_SUMMARY_KEYS = (
    "case_count",
    "optimizer_count",
    "plugin_optimizer_count",
    "product_native_ready_count",
    "top_level_native_dispatch_allowed_count",
    "native_kernel_launch_count",
    "optimizer_family_contract_count",
    "required_family_present_count",
    "entrypoint_present_count",
    "kernel_source_present_count",
    "rust_probe_present_count",
    "family_count",
    "route_family_counts",
    "selected_family_counts",
    "selected_plugin_family_count",
    "runtime_dispatch_ready_count",
    "selected_plugin_optimizer_count",
    "trainer_resume_parity_proven_count",
    "per_optimizer_native_training_count",
    "actual_training_gap_count",
    "representative_native_training_mapped_count",
    "runtime_precondition_rehearsal_ready_count",
    "actual_training_complete_count",
    "actual_training_status_counts",
    "plugin_training_loop_total_count",
    "plugin_training_loop_passed_count",
    "plugin_training_loop_failed_count",
    "plugin_training_loop_skipped_count",
    "plugin_training_loop_step_route_executed_count",
    "plugin_training_loop_standard_step_called_count",
    "plugin_training_loop_closure_step_called_count",
    "plugin_training_loop_fused_backward_route_count",
    "plugin_training_loop_fused_backward_call_count",
    "plugin_training_loop_create_graph_backward_count",
    "plugin_training_loop_zero_grad_called_count",
    "plugin_training_loop_param_updated_count",
    "plugin_training_loop_grad_cleared_count",
    "plugin_training_loop_global_step_advanced_count",
    "plugin_training_loop_finite_loss_count",
    "plugin_training_loop_route_counts",
    "plugin_adaptivelr_training_loop_case_count",
    "plugin_adaptivelr_training_loop_native_step_count",
    "plugin_adaptivelr_training_loop_native_kernel_launch_count",
    "plugin_adaptivelr_training_loop_training_executor_called_count",
    "plugin_adaptivelr_training_loop_skip_pytorch_count",
    "plugin_adaptivelr_training_loop_native_family_alias_count",
    "plugin_fused_backward_hook_canary_case_count",
    "plugin_fused_backward_hook_canary_native_step_count",
    "plugin_fused_backward_hook_canary_native_kernel_launch_count",
    "plugin_fused_backward_hook_canary_public_step_called_count",
    "plugin_bridge_training_loop_canary_case_count",
    "plugin_bridge_training_loop_canary_native_step_count",
    "plugin_bridge_training_loop_canary_native_kernel_launch_count",
    "plugin_bridge_training_loop_canary_training_executor_called_count",
    "plugin_bridge_training_loop_canary_skip_pytorch_count",
    *STREAM_LIFETIME_SUMMARY_KEYS,
    *TRAINING_LOOP_SUMMARY_KEYS,
    "native_dispatch_allowed_count",
    "training_path_enabled_count",
    "route_family_count",
    "family_evidence_ready_count",
    "runtime_rehearsal_ready_family_count",
    "runtime_precondition_ready_family_count",
    "family_specific_runtime_launch_adapter_ready_count",
    "family_specific_runtime_launch_adapter_ready_family_count",
    "family_specific_runtime_launch_adapter_ready_optimizer_count",
    "roadmap_v2_open_work_category_count",
    "roadmap_v2_open_work_item_count",
    "roadmap_v2_open_work_open_category_count",
    "roadmap_v2_open_work_open_item_count",
    "roadmap_v2_open_work",
    "family_follow_up_family_count",
    "family_follow_up_base_canary_ready_count",
    "family_follow_up_remaining_branch_count",
    "family_follow_up_native_step_count",
    "family_follow_up_native_kernel_launch_count",
    "family_follow_up_branch_contract_tracked_count",
    "family_follow_up_branch_reference_ready_count",
    "family_follow_up_branch_implementation_ready_count",
    "family_follow_up_branch_native_gap_count",
    "rmsprop_centered_reference_ready_count",
    "rmsprop_momentum_reference_ready_count",
    "sgdp_projection_reference_ready_count",
    "sgdp_decoupled_decay_reference_ready_count",
    "fromage_per_tensor_norm_reference_ready_count",
    "fromage_p_bound_reference_ready_count",
    "rmsprop_centered_launch_config_ready_count",
    "rmsprop_momentum_launch_config_ready_count",
    "rmsprop_branch_state_layout_ready_count",
    "rmsprop_first_class_contract_ready_count",
    "rmsprop_first_class_contract_training_path_enabled_count",
    "rmsprop_first_class_contract_native_kernel_present_count",
    "rmsprop_branch_first_class_contract_count",
    "rmsprop_centered_branch_contract_pending_count",
    "rmsprop_momentum_branch_contract_pending_count",
    "rmsprop_centered_branch_kernel_supported_count",
    "rmsprop_momentum_branch_kernel_supported_count",
    "rmsprop_branch_native_abi_guard_ready_count",
    "rmsprop_branch_native_abi_fail_closed_count",
    "rmsprop_branch_native_kernel_launch_count",
    "rmsprop_branch_native_kernel_parity_ready_count",
    "pid_momentum_three_buffer_launch_config_ready_count",
    "pid_momentum_three_buffer_state_layout_ready_count",
    "pid_momentum_three_buffer_native_abi_guard_ready_count",
    "pid_momentum_three_buffer_native_kernel_launch_count",
    "pid_momentum_three_buffer_native_kernel_parity_ready_count",
    "sgdp_projection_launch_config_ready_count",
    "sgdp_decoupled_decay_launch_config_ready_count",
    "sgdp_branch_state_layout_ready_count",
    "sgdp_projection_state_layout_ready_count",
    "sgdp_projection_native_abi_guard_ready_count",
    "sgdp_decoupled_decay_native_abi_guard_ready_count",
    "sgdp_branch_native_kernel_launch_count",
    "sgdp_branch_native_kernel_parity_ready_count",
    "fromage_per_tensor_norm_launch_config_ready_count",
    "fromage_p_bound_launch_config_ready_count",
    "fromage_branch_state_layout_ready_count",
    "fromage_per_tensor_norm_state_layout_ready_count",
    "fromage_p_bound_state_layout_ready_count",
    "fromage_per_tensor_norm_native_abi_guard_ready_count",
    "fromage_p_bound_native_abi_guard_ready_count",
    "fromage_branch_native_kernel_launch_count",
    "fromage_branch_native_kernel_parity_ready_count",
    "runtime_launch_coverage_ready_family_count",
    "runtime_launch_coverage_ready_optimizer_count",
    "runtime_launch_coverage_mode_counts",
    "owner_release_hold_ready_family_count",
    "default_off_product_runtime_dispatch_ready_family_count",
    "default_off_product_runtime_dispatch_ready_optimizer_count",
    "default_off_product_native_dispatch_allowed_family_count",
    "default_off_product_native_dispatch_allowed_optimizer_count",
    "default_off_product_training_path_enabled_family_count",
    "default_off_product_training_path_enabled_optimizer_count",
    "default_off_product_product_native_ready_family_count",
    "default_off_product_product_native_ready_optimizer_count",
    "signed_post_approval_preview_ready_count",
    "signed_post_approval_preview_runtime_dispatch_ready_family_count",
    "signed_post_approval_preview_runtime_dispatch_ready_optimizer_count",
    "signed_post_approval_preview_native_dispatch_allowed_family_count",
    "signed_post_approval_preview_native_dispatch_allowed_optimizer_count",
    "signed_post_approval_preview_training_path_enabled_family_count",
    "signed_post_approval_preview_training_path_enabled_optimizer_count",
    "signed_post_approval_preview_product_native_ready_family_count",
    "signed_post_approval_preview_product_native_ready_optimizer_count",
    "owner_release_hold_package_family_count",
    "owner_release_hold_package_ready_family_count",
    "owner_release_hold_package_manual_review_required_count",
    "owner_release_hold_package_owner_approval_missing_count",
    "owner_release_hold_package_release_approval_missing_count",
    "owner_release_hold_package_runtime_dispatch_ready_count",
    "owner_release_hold_package_native_dispatch_allowed_count",
    "owner_release_hold_package_training_path_enabled_count",
    "owner_release_hold_package_default_behavior_changed_count",
    "owner_release_hold_package_product_native_ready_count",
    "adaptive_lr_chain_stage_count",
    "adaptive_lr_chain_ready_stage_count",
    "adaptive_lr_chain_open_stage_count",
    "adaptive_lr_chain_target_optimizer_count",
    "adaptive_lr_chain_product_exposure_gate_ready_count",
    "adaptive_lr_chain_runtime_dispatch_ready_count",
    "adaptive_lr_chain_native_dispatch_allowed_count",
    "adaptive_lr_chain_training_path_enabled_count",
    "adaptive_lr_chain_default_behavior_changed_count",
    "adaptive_lr_chain_product_native_ready_count",
    "request_schema_ui_non_exposure_ready_family_count",
    "representative_runtime_rehearsal_ready_count",
    "representative_runtime_ready_family_count",
    "runtime_launch_absent_family_count",
    "family_specific_runtime_launch_missing_count",
    "product_training_route_missing_count",
    "owner_release_approval_missing_count",
    "owner_release_approval_recorded_count",
    "owner_release_review_packet_ready_for_signature_count",
    "owner_release_review_recorded_count",
    "owner_release_direction_ready_for_signature_count",
    "owner_release_direction_recorded_count",
    "owner_release_direction_approval_recorded_count",
    "synthetic_owner_release_direction_recorded_count",
    "synthetic_owner_release_direction_approval_recorded_count",
    "release_review_archive_ready_count",
    "release_artifact_first_required_artifact_count",
    "release_artifact_first_ready_artifact_count",
    "release_artifact_first_missing_artifact_count",
    "release_artifact_first_parse_error_count",
    "release_artifact_first_cross_check_count",
    "release_artifact_first_cross_check_ready_count",
    "release_artifact_first_v2_record_input_item_count",
    "release_artifact_first_v2_record_input_phase1_item_count",
    "release_artifact_first_v2_record_input_phase2_item_count",
    "release_artifact_first_v2_phase2_refresh_inputs_tracked_count",
    "release_artifact_first_v2_record_input_checklist_shape_ready_count",
    "release_artifact_first_v2_record_input_support_check_count",
    "release_artifact_first_v2_record_input_support_ready_count",
    "release_artifact_first_v2_record_input_support_blocked_item_count",
    "release_artifact_first_v2_record_input_support_blocker_count",
    "release_artifact_first_v2_record_input_support_source_binding_ready_count",
    "release_artifact_first_v2_record_input_support_source_binding_blocker_count",
    "release_artifact_first_v2_record_input_support_shape_ready_count",
    "release_artifact_first_v2_reviewer_handoff_phase_metadata_ready_count",
    "release_artifact_first_v2_reviewer_handoff_phase1_template_entry_count",
    "release_artifact_first_v2_reviewer_handoff_phase2_deferred_signature_count",
    "release_artifact_first_v2_reviewer_handoff_phase_shape_ready_count",
    "release_artifact_first_v2_command_audit_expected_path_binding_ready_count",
    "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_count",
    "release_artifact_first_v2_command_audit_phase1_handoff_post_return_pre_record_command_count",
    "release_artifact_first_v2_command_audit_phase1_handoff_post_return_approval_record_command_count",
    "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_match_count",
    "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_mismatch_count",
    "release_artifact_first_v2_command_audit_phase1_handoff_post_return_ready_count",
    "release_artifact_first_v2_command_audit_phase1_handoff_post_return_shape_ready_count",
    "release_artifact_first_validation_ready_count",
    "release_artifact_first_runtime_dispatch_ready_count",
    "release_artifact_first_native_dispatch_allowed_count",
    "release_artifact_first_training_path_enabled_count",
    "release_artifact_first_default_behavior_changed_count",
    "release_artifact_first_product_native_ready_count",
    "product_exposure_decision_recorded_count",
    "product_exposure_ready_for_review_count",
    "product_exposure_review_action_required_count",
    "v2_remaining_gate_total_count",
    "v2_remaining_gate_open_count",
    "v2_remaining_gate_closed_count",
    "v2_remaining_gate_owner_release_open_count",
    "v2_remaining_gate_product_exposure_open_count",
    "v2_remaining_gate_handoff_ready_count",
    "v2_remaining_gate_release_artifact_ready_count",
    "v2_remaining_gate_default_off_guard_count",
    "v2_remaining_gate_runtime_dispatch_ready_count",
    "v2_remaining_gate_native_dispatch_allowed_count",
    "v2_remaining_gate_training_path_enabled_count",
    "v2_remaining_gate_product_native_ready_count",
    "v2_remaining_gate_default_behavior_changed_count",
    "v2_remaining_gate_unsafe_claim_count",
    "v2_signature_bundle_entry_count",
    "v2_signature_bundle_ready_for_signature_count",
    "v2_signature_bundle_blocked_entry_count",
    "v2_signature_bundle_owner_review_ready_count",
    "v2_signature_bundle_product_exposure_ready_count",
    "v2_signature_bundle_owner_direction_ready_count",
    "v2_signature_bundle_approval_recorded_count",
    "v2_signature_bundle_runtime_dispatch_ready_count",
    "v2_signature_bundle_native_dispatch_allowed_count",
    "v2_signature_bundle_training_path_enabled_count",
    "v2_signature_bundle_product_native_ready_count",
    "v2_signature_bundle_default_behavior_changed_count",
    "v2_signature_bundle_unsafe_claim_count",
    "v2_reviewer_handoff_entry_count",
    "v2_reviewer_handoff_ready_entry_count",
    "v2_reviewer_handoff_blocked_entry_count",
    "v2_reviewer_handoff_signed_template_entry_count",
    "v2_reviewer_handoff_phase_metadata_ready_count",
    "v2_reviewer_handoff_phase1_template_entry_count",
    "v2_reviewer_handoff_phase2_deferred_signature_count",
    "v2_reviewer_handoff_required_manual_field_signature_count",
    "v2_reviewer_handoff_required_manual_field_count",
    "v2_reviewer_handoff_phase1_signature_count",
    "v2_reviewer_handoff_phase2_blocked_signature_count",
    "v2_reviewer_handoff_phase1_required_manual_field_count",
    "v2_reviewer_handoff_template_dry_run_command_count",
    "v2_reviewer_handoff_real_return_validation_command_count",
    "v2_reviewer_handoff_phase1_post_return_command_count",
    "v2_reviewer_handoff_phase1_post_return_pre_record_command_count",
    "v2_reviewer_handoff_phase1_post_return_approval_record_command_count",
    "v2_reviewer_handoff_packet_ready_count",
    "v2_reviewer_handoff_approval_recorded_count",
    "v2_reviewer_handoff_runtime_dispatch_ready_count",
    "v2_reviewer_handoff_native_dispatch_allowed_count",
    "v2_reviewer_handoff_training_path_enabled_count",
    "v2_reviewer_handoff_product_native_ready_count",
    "v2_reviewer_handoff_default_behavior_changed_count",
    "v2_reviewer_handoff_unsafe_claim_count",
    "v2_approval_execution_step_count",
    "v2_approval_execution_phase1_step_count",
    "v2_approval_execution_phase2_step_count",
    "v2_approval_execution_shared_step_count",
    "v2_approval_execution_ready_step_count",
    "v2_approval_execution_manual_signature_step_count",
    "v2_approval_execution_record_step_count",
    "v2_approval_execution_extraction_step_count",
    "v2_approval_execution_preflight_step_count",
    "v2_approval_execution_phase1_record_chain_step_count",
    "v2_approval_execution_phase2_record_chain_step_count",
    "v2_approval_execution_reviewer_template_entry_count",
    "v2_approval_execution_blocked_signature_entry_count",
    "v2_approval_execution_phase1_handoff_post_return_command_count",
    "v2_approval_execution_phase1_handoff_post_return_pre_record_command_count",
    "v2_approval_execution_phase1_handoff_post_return_approval_record_command_count",
    "v2_approval_execution_phase1_handoff_post_return_command_match_count",
    "v2_approval_execution_phase1_handoff_post_return_command_mismatch_count",
    "v2_approval_execution_phase1_handoff_post_return_ready_count",
    "v2_approval_execution_plan_ready_count",
    "v2_approval_execution_approval_recorded_count",
    "v2_approval_execution_runtime_dispatch_ready_count",
    "v2_approval_execution_native_dispatch_allowed_count",
    "v2_approval_execution_training_path_enabled_count",
    "v2_approval_execution_product_native_ready_count",
    "v2_approval_execution_default_behavior_changed_count",
    "v2_approval_execution_unsafe_claim_count",
    "v2_approval_command_audit_step_count",
    "v2_approval_command_audit_command_step_count",
    "v2_approval_command_audit_entrypoint_exists_count",
    "v2_approval_command_audit_missing_entrypoint_count",
    "v2_approval_command_audit_order_valid_count",
    "v2_approval_command_audit_record_after_preflight_count",
    "v2_approval_command_audit_record_before_preflight_count",
    "v2_approval_command_audit_record_after_real_signature_count",
    "v2_approval_command_audit_record_before_real_signature_count",
    "v2_approval_command_audit_phase2_order_valid_count",
    "v2_approval_command_audit_phase2_blocker_count",
    "v2_approval_command_audit_signature_order_valid_count",
    "v2_approval_command_audit_signature_blocker_count",
    "v2_approval_command_audit_phase_marker_valid_count",
    "v2_approval_command_audit_phase_marker_blocker_count",
    "v2_approval_command_audit_required_marker_missing_count",
    "v2_approval_command_audit_preflight_artifact_write_count",
    "v2_approval_command_audit_preflight_no_artifact_blocker_count",
    "v2_approval_command_audit_record_command_preflight_arg_count",
    "v2_approval_command_audit_unsigned_template_allowed_count",
    "v2_approval_command_audit_expected_arg_binding_count",
    "v2_approval_command_audit_expected_arg_mismatch_count",
    "v2_approval_command_audit_expected_path_binding_ready_count",
    "v2_approval_command_audit_phase1_handoff_post_return_command_count",
    "v2_approval_command_audit_phase1_handoff_post_return_pre_record_command_count",
    "v2_approval_command_audit_phase1_handoff_post_return_approval_record_command_count",
    "v2_approval_command_audit_phase1_handoff_post_return_command_match_count",
    "v2_approval_command_audit_phase1_handoff_post_return_command_mismatch_count",
    "v2_approval_command_audit_phase1_handoff_post_return_ready_count",
    "v2_approval_command_audit_ready_count",
    "v2_approval_command_audit_approval_recorded_count",
    "v2_approval_command_audit_runtime_dispatch_ready_count",
    "v2_approval_command_audit_native_dispatch_allowed_count",
    "v2_approval_command_audit_training_path_enabled_count",
    "v2_approval_command_audit_product_native_ready_count",
    "v2_approval_command_audit_default_behavior_changed_count",
    "v2_approval_command_audit_unsafe_claim_count",
    "v2_record_input_checklist_item_count",
    "v2_record_input_checklist_present_item_count",
    "v2_record_input_checklist_valid_json_item_count",
    "v2_record_input_checklist_missing_item_count",
    "v2_record_input_checklist_ready_item_count",
    "v2_record_input_checklist_phase1_item_count",
    "v2_record_input_checklist_phase1_ready_item_count",
    "v2_record_input_checklist_phase1_missing_item_count",
    "v2_record_input_checklist_phase1_ready_count",
    "v2_record_input_checklist_phase2_item_count",
    "v2_record_input_checklist_phase2_ready_item_count",
    "v2_record_input_checklist_phase2_missing_item_count",
    "v2_record_input_checklist_phase2_ready_count",
    "v2_record_input_checklist_full_ready_count",
    "v2_record_input_checklist_support_check_count",
    "v2_record_input_checklist_support_ready_count",
    "v2_record_input_checklist_support_blocked_item_count",
    "v2_record_input_checklist_support_blocker_count",
    "v2_record_input_checklist_support_source_binding_ready_count",
    "v2_record_input_checklist_support_source_binding_blocker_count",
    "v2_record_input_checklist_artifact_ready_count",
    "v2_record_input_checklist_approval_recorded_count",
    "v2_record_input_checklist_runtime_dispatch_ready_count",
    "v2_record_input_checklist_native_dispatch_allowed_count",
    "v2_record_input_checklist_training_path_enabled_count",
    "v2_record_input_checklist_product_native_ready_count",
    "v2_record_input_checklist_default_behavior_changed_count",
    "v2_record_input_checklist_unsafe_claim_count",
    "v2_signed_bundle_extraction_entry_count",
    "v2_signed_bundle_extraction_present_count",
    "v2_signed_bundle_extraction_source_digest_match_count",
    "v2_signed_bundle_extraction_source_digest_stale_count",
    "v2_signed_bundle_extraction_unsigned_template_count",
    "v2_signed_bundle_extraction_template_digest_mismatch_count",
    "v2_signed_bundle_extraction_unknown_entry_count",
    "v2_signed_bundle_extraction_missing_signature_count",
    "v2_signed_bundle_extraction_phase1_missing_signature_count",
    "v2_signed_bundle_extraction_phase2_missing_signature_count",
    "v2_signed_bundle_extraction_not_ready_entry_count",
    "v2_signed_bundle_extraction_signed_entry_digest_present_count",
    "v2_signed_bundle_extraction_extractable_signed_entry_digest_present_count",
    "v2_signed_bundle_extraction_extractable_entry_count",
    "v2_signed_bundle_extraction_phase1_extractable_entry_count",
    "v2_signed_bundle_extraction_phase1_ready_for_record_count",
    "v2_signed_bundle_extraction_phase2_extractable_entry_count",
    "v2_signed_bundle_extraction_phase2_ready_for_record_count",
    "v2_signed_bundle_extraction_full_ready_for_record_count",
    "v2_signed_bundle_extraction_missing_entry_count",
    "v2_signed_bundle_extraction_owner_review_extracted_count",
    "v2_signed_bundle_extraction_product_exposure_extracted_count",
    "v2_signed_bundle_extraction_owner_direction_extracted_count",
    "v2_signed_bundle_extraction_artifact_written_count",
    "v2_signed_bundle_extraction_ready_for_record_count",
    "v2_signed_bundle_extraction_approval_recorded_count",
    "v2_signed_bundle_extraction_runtime_dispatch_ready_count",
    "v2_signed_bundle_extraction_native_dispatch_allowed_count",
    "v2_signed_bundle_extraction_training_path_enabled_count",
    "v2_signed_bundle_extraction_product_native_ready_count",
    "v2_signed_bundle_extraction_default_behavior_changed_count",
    "v2_signed_bundle_extraction_unsafe_claim_count",
    "v2_approval_preflight_file_count",
    "v2_approval_preflight_present_file_count",
    "v2_approval_preflight_valid_json_file_count",
    "v2_approval_preflight_missing_file_count",
    "v2_approval_preflight_missing_shared_input_count",
    "v2_approval_preflight_missing_phase1_input_count",
    "v2_approval_preflight_missing_phase2_input_count",
    "v2_approval_preflight_parse_error_count",
    "v2_approval_preflight_signature_bundle_valid_count",
    "v2_approval_preflight_signed_bundle_valid_count",
    "v2_approval_preflight_source_digest_match_count",
    "v2_approval_preflight_source_digest_stale_count",
    "v2_approval_preflight_template_digest_mismatch_count",
    "v2_approval_preflight_unknown_entry_count",
    "v2_approval_preflight_not_ready_entry_count",
    "v2_approval_preflight_unsigned_template_count",
    "v2_approval_preflight_signed_payload_digest_present_count",
    "v2_approval_preflight_signed_payload_digest_missing_count",
    "v2_approval_preflight_signed_bundle_entry_digest_present_count",
    "v2_approval_preflight_signed_payload_bundle_digest_match_count",
    "v2_approval_preflight_extracted_entry_digest_match_count",
    "v2_approval_preflight_extracted_entry_digest_mismatch_count",
    "v2_approval_preflight_extracted_entry_source_missing_count",
    "v2_approval_preflight_support_ready_count",
    "v2_approval_preflight_support_invalid_count",
    "v2_approval_preflight_owner_review_ready_count",
    "v2_approval_preflight_product_exposure_ready_count",
    "v2_approval_preflight_owner_direction_ready_count",
    "v2_approval_preflight_phase1_ready_count",
    "v2_approval_preflight_phase2_ready_count",
    "v2_approval_preflight_full_ready_count",
    "v2_approval_preflight_hard_fail_count",
    "v2_approval_preflight_approval_recorded_count",
    "v2_approval_preflight_post_record_request_fields_ready_count",
    "v2_approval_preflight_post_record_request_field_emission_count",
    "v2_approval_preflight_runtime_dispatch_ready_count",
    "v2_approval_preflight_native_dispatch_allowed_count",
    "v2_approval_preflight_training_path_enabled_count",
    "v2_approval_preflight_product_native_ready_count",
    "v2_approval_preflight_default_behavior_changed_count",
    "v2_approval_preflight_unsafe_claim_count",
    "v2_signed_bundle_entry_count",
    "v2_signed_bundle_present_count",
    "v2_signed_bundle_valid_record_count",
    "v2_signed_bundle_phase1_valid_record_count",
    "v2_signed_bundle_phase1_ready_count",
    "v2_signed_bundle_phase2_valid_record_count",
    "v2_signed_bundle_phase2_ready_count",
    "v2_signed_bundle_full_ready_count",
    "v2_signed_bundle_missing_signature_count",
    "v2_signed_bundle_phase1_missing_signature_count",
    "v2_signed_bundle_phase2_missing_signature_count",
    "v2_signed_bundle_unsigned_template_count",
    "v2_signed_bundle_template_digest_mismatch_count",
    "v2_signed_bundle_unknown_entry_count",
    "v2_signed_bundle_not_ready_entry_count",
    "v2_signed_bundle_manual_field_check_count",
    "v2_signed_bundle_manual_field_ready_count",
    "v2_signed_bundle_manual_field_missing_count",
    "v2_signed_bundle_manual_field_missing_signature_count",
    "v2_signed_bundle_manual_field_shape_ready_count",
    "v2_signed_bundle_phase1_manual_field_ready_count",
    "v2_signed_bundle_phase2_manual_field_ready_count",
    "v2_signed_bundle_source_digest_match_count",
    "v2_signed_bundle_source_digest_stale_count",
    "v2_signed_bundle_owner_review_recorded_count",
    "v2_signed_bundle_product_exposure_recorded_count",
    "v2_signed_bundle_owner_direction_recorded_count",
    "v2_signed_bundle_approval_artifact_written_count",
    "v2_signed_bundle_runtime_dispatch_ready_count",
    "v2_signed_bundle_native_dispatch_allowed_count",
    "v2_signed_bundle_training_path_enabled_count",
    "v2_signed_bundle_product_native_ready_count",
    "v2_signed_bundle_default_behavior_changed_count",
    "v2_signed_bundle_unsafe_claim_count",
    "v2_signed_bundle_freshness_guard_ready_count",
    "v2_signed_bundle_freshness_current_digest_present_count",
    "v2_signed_bundle_freshness_template_digest_match_count",
    "v2_signed_bundle_freshness_signed_bundle_present_count",
    "v2_signed_bundle_freshness_signed_bundle_digest_match_count",
    "v2_signed_bundle_freshness_stale_signed_bundle_count",
    "v2_signed_bundle_freshness_unknown_signed_entry_count",
    "v2_signed_bundle_freshness_approval_recorded_count",
    "v2_signed_bundle_freshness_approval_artifact_written_count",
    "v2_signed_bundle_freshness_runtime_dispatch_ready_count",
    "v2_signed_bundle_freshness_native_dispatch_allowed_count",
    "v2_signed_bundle_freshness_training_path_enabled_count",
    "v2_signed_bundle_freshness_product_native_ready_count",
    "v2_signed_bundle_freshness_default_behavior_changed_count",
    "v2_signed_bundle_freshness_unsafe_claim_count",
    "v2_signed_bundle_roundtrip_ready_template_count",
    "v2_signed_bundle_roundtrip_signed_bundle_present_count",
    "v2_signed_bundle_roundtrip_intake_valid_record_count",
    "v2_signed_bundle_roundtrip_extracted_entry_count",
    "v2_signed_bundle_roundtrip_extraction_signed_entry_digest_present_count",
    "v2_signed_bundle_roundtrip_extraction_extractable_signed_entry_digest_present_count",
    "v2_signed_bundle_roundtrip_extraction_digest_shape_ready_count",
    "v2_signed_bundle_roundtrip_extraction_artifact_written_count",
    "v2_signed_bundle_roundtrip_preflight_signed_payload_digest_present_count",
    "v2_signed_bundle_roundtrip_preflight_signed_payload_digest_missing_count",
    "v2_signed_bundle_roundtrip_preflight_signed_bundle_entry_digest_present_count",
    "v2_signed_bundle_roundtrip_preflight_signed_payload_bundle_digest_match_count",
    "v2_signed_bundle_roundtrip_preflight_digest_shape_ready_count",
    "v2_signed_bundle_roundtrip_phase1_ready_count",
    "v2_signed_bundle_roundtrip_phase2_ready_count",
    "v2_signed_bundle_roundtrip_full_ready_count",
    "v2_signed_bundle_roundtrip_owner_direction_blocked_count",
    "v2_signed_bundle_roundtrip_approval_recorded_count",
    "v2_signed_bundle_roundtrip_approval_artifact_written_count",
    "v2_signed_bundle_roundtrip_runtime_dispatch_ready_count",
    "v2_signed_bundle_roundtrip_native_dispatch_allowed_count",
    "v2_signed_bundle_roundtrip_training_path_enabled_count",
    "v2_signed_bundle_roundtrip_product_native_ready_count",
    "v2_signed_bundle_roundtrip_default_behavior_changed_count",
    "v2_signed_bundle_roundtrip_unsafe_claim_count",
    "v2_approval_state_stage_count",
    "v2_approval_state_ready_stage_count",
    "v2_approval_state_waiting_signature_stage_count",
    "v2_approval_state_ready_for_record_stage_count",
    "v2_approval_state_recorded_stage_count",
    "v2_approval_state_missing_stage_count",
    "v2_approval_state_blocked_stage_count",
    "v2_approval_state_remaining_gate_open_count",
    "v2_approval_state_signature_ready_count",
    "v2_approval_state_signed_bundle_present_count",
    "v2_approval_state_signed_bundle_freshness_guard_ready_count",
    "v2_approval_state_signed_bundle_freshness_template_digest_match_count",
    "v2_approval_state_signed_bundle_freshness_stale_signed_bundle_count",
    "v2_approval_state_signed_bundle_freshness_unknown_signed_entry_count",
    "v2_approval_state_signed_bundle_freshness_unsafe_claim_count",
    "v2_approval_state_phase1_signed_bundle_ready_count",
    "v2_approval_state_phase2_signed_bundle_ready_count",
    "v2_approval_state_full_signed_bundle_ready_count",
    "v2_approval_state_signed_bundle_missing_signature_count",
    "v2_approval_state_phase1_signed_bundle_missing_signature_count",
    "v2_approval_state_phase2_signed_bundle_missing_signature_count",
    "v2_approval_state_signed_bundle_not_ready_entry_count",
    "v2_approval_state_signed_bundle_source_digest_match_count",
    "v2_approval_state_signed_bundle_source_digest_stale_count",
    "v2_approval_state_signed_bundle_template_digest_mismatch_count",
    "v2_approval_state_signed_bundle_unknown_entry_count",
    "v2_approval_state_signed_bundle_unsigned_template_count",
    "v2_approval_state_signed_bundle_intake_integrity_ready_count",
    "v2_approval_state_signed_bundle_manual_field_check_count",
    "v2_approval_state_signed_bundle_manual_field_ready_count",
    "v2_approval_state_signed_bundle_manual_field_missing_count",
    "v2_approval_state_signed_bundle_manual_field_shape_ready_count",
    "v2_approval_state_phase1_signed_bundle_manual_field_ready_count",
    "v2_approval_state_phase2_signed_bundle_manual_field_ready_count",
    "v2_approval_state_phase1_signed_bundle_manual_shape_ready_count",
    "v2_approval_state_phase2_signed_bundle_manual_shape_ready_count",
    "v2_approval_state_full_signed_bundle_manual_shape_ready_count",
    "v2_approval_state_phase1_extraction_ready_count",
    "v2_approval_state_phase2_extraction_ready_count",
    "v2_approval_state_full_extraction_ready_count",
    "v2_approval_state_extraction_missing_signature_count",
    "v2_approval_state_phase1_extraction_missing_signature_count",
    "v2_approval_state_phase2_extraction_missing_signature_count",
    "v2_approval_state_extraction_not_ready_entry_count",
    "v2_approval_state_extraction_source_digest_match_count",
    "v2_approval_state_extraction_source_digest_stale_count",
    "v2_approval_state_extraction_template_digest_mismatch_count",
    "v2_approval_state_extraction_unknown_entry_count",
    "v2_approval_state_extraction_unsigned_template_count",
    "v2_approval_state_extraction_integrity_ready_count",
    "v2_approval_state_extraction_signed_entry_digest_present_count",
    "v2_approval_state_extraction_extractable_signed_entry_digest_present_count",
    "v2_approval_state_extraction_digest_shape_ready_count",
    "v2_approval_state_command_audit_ready_count",
    "v2_approval_state_command_audit_expected_arg_binding_count",
    "v2_approval_state_command_audit_expected_arg_mismatch_count",
    "v2_approval_state_command_audit_expected_path_binding_ready_count",
    "v2_approval_state_command_audit_preflight_artifact_write_count",
    "v2_approval_state_command_audit_preflight_no_artifact_blocker_count",
    "v2_approval_state_command_audit_record_command_preflight_arg_count",
    "v2_approval_state_command_audit_phase1_handoff_post_return_command_count",
    "v2_approval_state_command_audit_phase1_handoff_post_return_pre_record_command_count",
    "v2_approval_state_command_audit_phase1_handoff_post_return_approval_record_command_count",
    "v2_approval_state_command_audit_phase1_handoff_post_return_command_match_count",
    "v2_approval_state_command_audit_phase1_handoff_post_return_command_mismatch_count",
    "v2_approval_state_command_audit_phase1_handoff_post_return_ready_count",
    "v2_approval_state_record_input_checklist_ready_count",
    "v2_approval_state_record_input_support_check_count",
    "v2_approval_state_record_input_support_ready_count",
    "v2_approval_state_record_input_support_blocked_item_count",
    "v2_approval_state_record_input_support_blocker_count",
    "v2_approval_state_record_input_support_source_binding_ready_count",
    "v2_approval_state_record_input_support_source_binding_blocker_count",
    "v2_approval_state_record_input_support_shape_ready_count",
    "v2_approval_state_phase1_preflight_ready_count",
    "v2_approval_state_phase2_preflight_ready_count",
    "v2_approval_state_full_preflight_ready_count",
    "v2_approval_state_preflight_not_ready_entry_count",
    "v2_approval_state_preflight_source_digest_match_count",
    "v2_approval_state_preflight_source_digest_stale_count",
    "v2_approval_state_preflight_template_digest_mismatch_count",
    "v2_approval_state_preflight_unknown_entry_count",
    "v2_approval_state_preflight_unsigned_template_count",
    "v2_approval_state_preflight_source_integrity_ready_count",
    "v2_approval_state_preflight_support_ready_count",
    "v2_approval_state_preflight_support_invalid_count",
    "v2_approval_state_preflight_support_source_binding_ready_count",
    "v2_approval_state_preflight_support_source_binding_blocker_count",
    "v2_approval_state_preflight_post_record_request_field_emission_count",
    "v2_approval_state_preflight_hard_fail_count",
    "v2_approval_state_preflight_support_integrity_ready_count",
    "v2_approval_state_phase1_preflight_support_integrity_ready_count",
    "v2_approval_state_phase2_preflight_support_integrity_ready_count",
    "v2_approval_state_preflight_signed_payload_digest_present_count",
    "v2_approval_state_preflight_signed_payload_digest_missing_count",
    "v2_approval_state_preflight_signed_bundle_entry_digest_present_count",
    "v2_approval_state_preflight_signed_payload_bundle_digest_match_count",
    "v2_approval_state_preflight_digest_shape_ready_count",
    "v2_approval_state_phase1_record_chain_ready_count",
    "v2_approval_state_phase2_record_chain_ready_count",
    "v2_approval_state_full_record_chain_ready_count",
    "v2_approval_state_owner_review_recorded_count",
    "v2_approval_state_product_exposure_recorded_count",
    "v2_approval_state_owner_direction_recorded_count",
    "v2_approval_state_phase1_record_preflight_binding_ready_count",
    "v2_approval_state_phase2_record_preflight_binding_ready_count",
    "v2_approval_state_full_record_preflight_binding_ready_count",
    "v2_approval_state_record_preflight_binding_ready_count",
    "v2_approval_state_approval_recorded_count",
    "v2_approval_state_runtime_dispatch_ready_count",
    "v2_approval_state_native_dispatch_allowed_count",
    "v2_approval_state_training_path_enabled_count",
    "v2_approval_state_product_native_ready_count",
    "v2_approval_state_default_behavior_changed_count",
    "v2_approval_state_unsafe_claim_count",
    "v2_completion_audit_requirement_count",
    "v2_completion_audit_passed_requirement_count",
    "v2_completion_audit_failed_requirement_count",
    "v2_completion_audit_remaining_gate_open_count",
    "v2_completion_audit_command_audit_ready_count",
    "v2_completion_audit_command_audit_expected_arg_binding_count",
    "v2_completion_audit_command_audit_expected_arg_mismatch_count",
    "v2_completion_audit_command_audit_expected_path_binding_ready_count",
    "v2_completion_audit_command_audit_phase1_handoff_post_return_command_count",
    "v2_completion_audit_command_audit_phase1_handoff_post_return_pre_record_command_count",
    "v2_completion_audit_command_audit_phase1_handoff_post_return_approval_record_command_count",
    "v2_completion_audit_command_audit_phase1_handoff_post_return_command_match_count",
    "v2_completion_audit_command_audit_phase1_handoff_post_return_command_mismatch_count",
    "v2_completion_audit_command_audit_phase1_handoff_post_return_ready_count",
    "v2_completion_audit_phase1_ready_count",
    "v2_completion_audit_phase2_ready_count",
    "v2_completion_audit_record_input_item_count",
    "v2_completion_audit_record_input_phase1_item_count",
    "v2_completion_audit_record_input_phase2_item_count",
    "v2_completion_audit_phase2_refresh_inputs_tracked_count",
    "v2_completion_audit_record_input_checklist_shape_ready_count",
    "v2_completion_audit_record_input_support_check_count",
    "v2_completion_audit_record_input_support_ready_count",
    "v2_completion_audit_record_input_support_blocked_item_count",
    "v2_completion_audit_record_input_support_blocker_count",
    "v2_completion_audit_record_input_support_source_binding_ready_count",
    "v2_completion_audit_record_input_support_source_binding_blocker_count",
    "v2_completion_audit_record_input_support_shape_ready_count",
    "v2_completion_audit_phase1_signed_bundle_ready_count",
    "v2_completion_audit_phase2_signed_bundle_ready_count",
    "v2_completion_audit_signed_bundle_missing_signature_count",
    "v2_completion_audit_phase1_signed_bundle_missing_signature_count",
    "v2_completion_audit_phase2_signed_bundle_missing_signature_count",
    "v2_completion_audit_signed_bundle_freshness_guard_ready_count",
    "v2_completion_audit_signed_bundle_freshness_template_digest_match_count",
    "v2_completion_audit_signed_bundle_freshness_stale_signed_bundle_count",
    "v2_completion_audit_signed_bundle_freshness_unknown_signed_entry_count",
    "v2_completion_audit_signed_bundle_freshness_unsafe_claim_count",
    "v2_completion_audit_signed_bundle_not_ready_entry_count",
    "v2_completion_audit_signed_bundle_source_digest_match_count",
    "v2_completion_audit_signed_bundle_source_digest_stale_count",
    "v2_completion_audit_signed_bundle_template_digest_mismatch_count",
    "v2_completion_audit_signed_bundle_unknown_entry_count",
    "v2_completion_audit_signed_bundle_unsigned_template_count",
    "v2_completion_audit_signed_bundle_intake_integrity_ready_count",
    "v2_completion_audit_signed_bundle_manual_field_check_count",
    "v2_completion_audit_signed_bundle_manual_field_ready_count",
    "v2_completion_audit_signed_bundle_manual_field_missing_count",
    "v2_completion_audit_signed_bundle_manual_field_shape_ready_count",
    "v2_completion_audit_phase1_signed_bundle_manual_field_ready_count",
    "v2_completion_audit_phase2_signed_bundle_manual_field_ready_count",
    "v2_completion_audit_phase1_signed_bundle_manual_shape_ready_count",
    "v2_completion_audit_phase2_signed_bundle_manual_shape_ready_count",
    "v2_completion_audit_full_signed_bundle_manual_shape_ready_count",
    "v2_completion_audit_phase1_extraction_ready_count",
    "v2_completion_audit_phase2_extraction_ready_count",
    "v2_completion_audit_extraction_missing_signature_count",
    "v2_completion_audit_phase1_extraction_missing_signature_count",
    "v2_completion_audit_phase2_extraction_missing_signature_count",
    "v2_completion_audit_extraction_not_ready_entry_count",
    "v2_completion_audit_extraction_source_digest_match_count",
    "v2_completion_audit_extraction_source_digest_stale_count",
    "v2_completion_audit_extraction_template_digest_mismatch_count",
    "v2_completion_audit_extraction_unknown_entry_count",
    "v2_completion_audit_extraction_unsigned_template_count",
    "v2_completion_audit_extraction_integrity_ready_count",
    "v2_completion_audit_extraction_signed_entry_digest_present_count",
    "v2_completion_audit_extraction_extractable_signed_entry_digest_present_count",
    "v2_completion_audit_extraction_digest_shape_ready_count",
    "v2_completion_audit_phase1_preflight_ready_count",
    "v2_completion_audit_phase2_preflight_ready_count",
    "v2_completion_audit_phase1_record_chain_ready_count",
    "v2_completion_audit_phase2_record_chain_ready_count",
    "v2_completion_audit_full_record_chain_ready_count",
    "v2_completion_audit_preflight_full_ready_count",
    "v2_completion_audit_preflight_integrity_ready_count",
    "v2_completion_audit_preflight_signed_bundle_valid_count",
    "v2_completion_audit_preflight_source_digest_match_count",
    "v2_completion_audit_preflight_source_digest_stale_count",
    "v2_completion_audit_preflight_template_digest_mismatch_count",
    "v2_completion_audit_preflight_unknown_entry_count",
    "v2_completion_audit_preflight_not_ready_entry_count",
    "v2_completion_audit_preflight_source_integrity_ready_count",
    "v2_completion_audit_preflight_support_integrity_ready_count",
    "v2_completion_audit_preflight_unsigned_template_count",
    "v2_completion_audit_preflight_signed_payload_digest_present_count",
    "v2_completion_audit_preflight_signed_payload_digest_missing_count",
    "v2_completion_audit_preflight_signed_bundle_entry_digest_present_count",
    "v2_completion_audit_preflight_signed_payload_bundle_digest_match_count",
    "v2_completion_audit_preflight_extracted_entry_digest_match_count",
    "v2_completion_audit_preflight_extracted_entry_digest_mismatch_count",
    "v2_completion_audit_preflight_extracted_entry_source_missing_count",
    "v2_completion_audit_preflight_support_ready_count",
    "v2_completion_audit_preflight_support_invalid_count",
    "v2_completion_audit_preflight_support_source_binding_ready_count",
    "v2_completion_audit_preflight_support_source_binding_blocker_count",
    "v2_completion_audit_preflight_post_record_request_field_emission_count",
    "v2_completion_audit_preflight_hard_fail_count",
    "v2_completion_audit_owner_review_recorded_count",
    "v2_completion_audit_product_exposure_recorded_count",
    "v2_completion_audit_owner_direction_recorded_count",
    "v2_completion_audit_phase1_record_preflight_binding_ready_count",
    "v2_completion_audit_phase2_record_preflight_binding_ready_count",
    "v2_completion_audit_full_record_preflight_binding_ready_count",
    "v2_completion_audit_record_preflight_binding_ready_count",
    "v2_completion_audit_artifact_first_required_artifact_count",
    "v2_completion_audit_artifact_first_ready_artifact_count",
    "v2_completion_audit_artifact_first_ready_count",
    "v2_completion_audit_artifact_first_record_input_checklist_shape_ready_count",
    "v2_completion_audit_artifact_first_record_input_support_shape_ready_count",
    "v2_completion_audit_artifact_first_reviewer_handoff_phase_metadata_ready_count",
    "v2_completion_audit_artifact_first_reviewer_handoff_phase1_template_entry_count",
    "v2_completion_audit_artifact_first_reviewer_handoff_phase2_deferred_signature_count",
    "v2_completion_audit_artifact_first_reviewer_handoff_phase_shape_ready_count",
    "v2_completion_audit_artifact_first_command_audit_expected_path_binding_ready_count",
    "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_count",
    "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_pre_record_command_count",
    "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_approval_record_command_count",
    "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_match_count",
    "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_mismatch_count",
    "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_ready_count",
    "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_shape_ready_count",
    "v2_completion_audit_default_off_guard_count",
    "v2_completion_audit_completion_ready_count",
    "v2_completion_audit_approval_recorded_count",
    "v2_completion_audit_runtime_dispatch_ready_count",
    "v2_completion_audit_native_dispatch_allowed_count",
    "v2_completion_audit_training_path_enabled_count",
    "v2_completion_audit_product_native_ready_count",
    "v2_completion_audit_default_behavior_changed_count",
    "v2_completion_audit_unsafe_claim_count",
    "product_training_route_binding_ready_count",
    "product_route_binding_chain_stage_count",
    "product_route_binding_chain_ready_stage_count",
    "product_route_binding_chain_open_stage_count",
    "product_route_binding_chain_contract_ready_count",
    "product_route_binding_chain_approval_missing_count",
    "product_route_binding_chain_product_training_route_bound_count",
    "product_route_binding_chain_runtime_dispatch_ready_count",
    "product_route_binding_chain_native_dispatch_allowed_count",
    "product_route_binding_chain_training_path_enabled_count",
    "product_route_binding_chain_default_behavior_changed_count",
    "product_route_binding_chain_product_native_ready_count",
    "open_training_path_enabled",
    "candidate_switch_count",
    "product_route_binding_preflight_ready_count",
    "training_loop_route_candidate_switch_count",
    "route_binding_config_patch_ready_count",
    "product_training_route_binding_config_patch_ready_count",
    "product_training_route_binding_kwargs_wired_count",
    "runtime_config_patch_applied_count",
    "run_local_adapter_staged_count",
    "product_launch_staging_wired_count",
    "stable_first_release_turbocore_optimizer_blocker_count",
    "turbocore_optimizer_default_off_release_scope_ready_count",
    "product_native_ready_family_count",
    "runtime_dispatch_ready_family_count",
    "native_dispatch_allowed_family_count",
    "training_path_enabled_family_count",
    "native_scratch_kernel_ready_count",
    "training_tensor_binding_canary_ready_count",
    "runtime_dispatch_adapter_shadow_ready_count",
    "training_loop_canary_ready_count",
    "e2e_shadow_matrix_ready_count",
    "canary_rollout_policy_ready_count",
    "dispatch_integration_review_ready_count",
)


def _max_int(payloads: list[Any], *keys: str) -> int:
    values: list[int] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key in keys:
            try:
                values.append(int(payload.get(key, 0) or 0))
            except (TypeError, ValueError):
                continue
    return max(values, default=0)


def _max_int_summary(payloads: list[Any], keys: tuple[str, ...]) -> dict[str, int]:
    return {key: _max_int(payloads, key) for key in keys}


def _passed_smoke_count(results: list[dict[str, Any]], smoke_ids: tuple[str, ...]) -> int:
    expected = set(smoke_ids)
    return sum(1 for item in results if item.get("ok") and item.get("smoke_id") in expected)


def _payloads_for_smoke_ids(results: list[dict[str, Any]], smoke_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    expected = set(smoke_ids)
    return [
        item["payload_summary"]
        for item in results
        if item.get("smoke_id") in expected and isinstance(item.get("payload_summary"), dict)
    ]


def _policy_groups(
    group_policies: Mapping[str, Mapping[str, Any]],
    key: str,
    expected: Any,
) -> list[str]:
    return [
        group_name
        for group_name, policy in sorted(group_policies.items())
        if policy.get(key) == expected
    ]


def _policy_values(
    group_policies: Mapping[str, Mapping[str, Any]],
    key: str,
) -> list[str]:
    return sorted(
        {
            value
            for policy in group_policies.values()
            for value in (policy.get(key),)
            if isinstance(value, str)
        }
    )


def _first_mapping(payloads: list[Any], key: str) -> dict[str, int]:
    for payload in payloads:
        value = payload.get(key) if isinstance(payload, dict) else None
        if not isinstance(value, dict):
            continue
        out: dict[str, int] = {}
        for item_key, item_value in value.items():
            try:
                out[str(item_key)] = int(item_value or 0)
            except (TypeError, ValueError):
                continue
        if out:
            return dict(sorted(out.items()))
    return {}


def _first_value(payloads: list[Any], key: str) -> Any:
    for payload in payloads:
        if isinstance(payload, dict) and key in payload:
            return payload[key]
    return None


def _probe(payload: Any) -> str:
    return str(payload.get("probe", "") or "") if isinstance(payload, dict) else ""
