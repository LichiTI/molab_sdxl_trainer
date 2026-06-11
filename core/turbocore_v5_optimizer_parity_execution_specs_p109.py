"""Stage spec for optimizer parity execution contract P109."""

from __future__ import annotations

from core.turbocore_v5_optimizer_kernel_launch_execution_specs_p108 import P108_SPEC
from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec


COMMON_SECTIONS = (
    "parity_execution_plan_inventory",
    "parity_execution_precondition_inventory",
    "parity_authorization_boundary",
    "parity_input_boundary",
    "parity_output_boundary",
    "parity_tolerance_boundary",
    "parity_comparison_boundary",
    "tensor_transfer_boundary",
    "training_step_boundary",
    "operator_parity_boundary",
    "observability_boundary",
    "rollback_policy_summary",
    "no_parity_execution_boundary",
    "no_tensor_transfer_execution_boundary",
    "no_training_step_execution_boundary",
    "no_kernel_launch_execution_boundary",
    "no_native_dispatch_execution_boundary",
    "no_runtime_dispatch_execution_boundary",
    "no_runtime_execution_boundary",
    "no_runtime_state_refresh_boundary",
    "no_runtime_adapter_enabled_boundary",
    "no_training_launch_execution_boundary",
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


P109_SPEC = OptimizerLateStageSpec(
    stage_id=109,
    token="optimizer_parity_execution",
    scope="optimizer_parity_execution_contract",
    title="optimizer parity execution",
    previous_token="optimizer_kernel_launch_execution",
    previous_label="P108 optimizer kernel-launch execution contract",
    previous_ready_decision=P108_SPEC.ready_decision,
    previous_ready_field="optimizer_kernel_launch_execution_contract_ready",
    previous_evidence_field="optimizer_kernel_launch_execution_evidence_recorded",
    previous_signed_field="optimizer_kernel_launch_execution_signed",
    previous_post_fields="post_p108_request_fields",
    previous_ack="acknowledge_p108_optimizer_kernel_launch_execution_contract_recorded",
    package_ready_field="optimizer_parity_execution_package_ready",
    policy_ready_field="parity_execution_policy_ready",
    row_keys=("optimizer_parity_execution_rows", "optimizer_execution_parity_rows"),
    row_ready_field="parity_execution_review_ready",
    later_field="later_optimizer_tensor_transfer_execution_contract_required",
    later_ack="acknowledge_later_optimizer_tensor_transfer_execution_contract_required",
    next_contract="optimizer_tensor_transfer_execution",
    allowed_intents=frozenset(
        {
            "parity_execution_candidate",
            "hold_for_more_evidence",
            "reject_parity_execution",
        }
    ),
    required_sections=_sections(
        "p108_optimizer_kernel_launch_execution_contract_reference",
        "optimizer_parity_execution_package",
        "per_optimizer_parity_execution_rows",
    ),
    unsafe_true_fields=_unsafe(
        "optimizer_parity_execution_applied",
        "optimizer_parity_execution_enabled",
        "optimizer_parity_execution_executed",
        "parity_approved",
        "parity_allowed",
        "parity_enabled",
        "parity_executed",
        "parity_started",
        "parity_request_executed",
        "parity_inputs_materialized",
        "parity_outputs_materialized",
        "parity_comparison_executed",
        "parity_result_recorded",
        "tensor_transfer_allowed",
        "tensor_transfer_executed",
        "training_step_executed",
    ),
    unsafe_non_empty_fields=_unsafe(
        "optimizer_parity_execution_payload",
        "parity_execution_payload",
        "parity_authorization_payload",
        "parity_input_payload",
        "parity_output_payload",
        "parity_tolerance_payload",
        "parity_comparison_payload",
        "tensor_transfer_payload",
        "training_step_payload",
    ),
    inherited_unsafe_true_fields=P108_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P108_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P109_SPEC"]
