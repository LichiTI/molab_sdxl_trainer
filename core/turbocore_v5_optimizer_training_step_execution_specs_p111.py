"""Stage spec for optimizer training-step execution contract P111."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_tensor_transfer_execution_specs_p110 import P110_SPEC


COMMON_SECTIONS = (
    "training_step_execution_plan_inventory",
    "training_step_execution_precondition_inventory",
    "training_step_authorization_boundary",
    "gradient_boundary",
    "optimizer_step_boundary",
    "parameter_update_boundary",
    "optimizer_state_update_boundary",
    "loss_scale_boundary",
    "training_launch_boundary",
    "operator_training_step_boundary",
    "observability_boundary",
    "rollback_policy_summary",
    "no_training_step_execution_boundary",
    "no_training_launch_execution_boundary",
    "no_tensor_transfer_execution_boundary",
    "no_parity_execution_boundary",
    "no_kernel_launch_execution_boundary",
    "no_native_dispatch_execution_boundary",
    "no_runtime_dispatch_execution_boundary",
    "no_runtime_execution_boundary",
    "no_runtime_state_refresh_boundary",
    "no_runtime_adapter_enabled_boundary",
    "no_request_execution_boundary",
    "no_job_creation_boundary",
    "no_queue_enqueue_boundary",
    "no_request_submission_boundary",
    "no_schema_config_router_ui_patch_boundary",
    "no_default_rollout_boundary",
)


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


def _unsafe(*fields: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(fields))


P111_SPEC = OptimizerLateStageSpec(
    stage_id=111,
    token="optimizer_training_step_execution",
    scope="optimizer_training_step_execution_contract",
    title="optimizer training-step execution",
    previous_token="optimizer_tensor_transfer_execution",
    previous_label="P110 optimizer tensor-transfer execution contract",
    previous_ready_decision=P110_SPEC.ready_decision,
    previous_ready_field="optimizer_tensor_transfer_execution_contract_ready",
    previous_evidence_field="optimizer_tensor_transfer_execution_evidence_recorded",
    previous_signed_field="optimizer_tensor_transfer_execution_signed",
    previous_post_fields="post_p110_request_fields",
    previous_ack="acknowledge_p110_optimizer_tensor_transfer_execution_contract_recorded",
    package_ready_field="optimizer_training_step_execution_package_ready",
    policy_ready_field="training_step_execution_policy_ready",
    row_keys=("optimizer_training_step_execution_rows", "optimizer_step_execution_rows"),
    row_ready_field="training_step_execution_review_ready",
    later_field="later_optimizer_training_launch_execution_contract_required",
    later_ack="acknowledge_later_optimizer_training_launch_execution_contract_required",
    next_contract="optimizer_training_launch_execution",
    allowed_intents=frozenset(
        {
            "training_step_execution_candidate",
            "hold_for_more_evidence",
            "reject_training_step_execution",
        }
    ),
    required_sections=_sections(
        "p110_optimizer_tensor_transfer_execution_contract_reference",
        "optimizer_training_step_execution_package",
        "per_optimizer_training_step_execution_rows",
    ),
    unsafe_true_fields=_unsafe(
        "optimizer_training_step_execution_applied",
        "optimizer_training_step_execution_enabled",
        "optimizer_training_step_execution_executed",
        "training_step_approved",
        "training_step_allowed",
        "training_step_enabled",
        "training_step_requested",
        "training_step_executed",
        "training_step_started",
        "gradients_materialized",
        "optimizer_step_executed",
        "parameters_updated",
        "optimizer_state_updated",
        "loss_scale_updated",
        "training_launch_allowed",
        "training_launch_executed",
    ),
    unsafe_non_empty_fields=_unsafe(
        "optimizer_training_step_execution_payload",
        "training_step_execution_payload",
        "training_step_authorization_payload",
        "gradient_payload",
        "optimizer_step_payload",
        "parameter_update_payload",
        "optimizer_state_update_payload",
        "loss_scale_payload",
        "training_launch_payload",
    ),
    inherited_unsafe_true_fields=P110_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P110_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P111_SPEC"]
