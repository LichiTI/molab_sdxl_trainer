"""Stage spec for optimizer tensor-transfer execution contract P110."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_parity_execution_specs_p109 import P109_SPEC


COMMON_SECTIONS = (
    "tensor_transfer_execution_plan_inventory",
    "tensor_transfer_execution_precondition_inventory",
    "tensor_transfer_authorization_boundary",
    "tensor_transfer_source_boundary",
    "tensor_transfer_destination_boundary",
    "tensor_transfer_stream_event_boundary",
    "tensor_transfer_copy_boundary",
    "tensor_transfer_result_boundary",
    "training_step_boundary",
    "operator_tensor_transfer_boundary",
    "observability_boundary",
    "rollback_policy_summary",
    "no_tensor_transfer_execution_boundary",
    "no_training_step_execution_boundary",
    "no_parity_execution_boundary",
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


P110_SPEC = OptimizerLateStageSpec(
    stage_id=110,
    token="optimizer_tensor_transfer_execution",
    scope="optimizer_tensor_transfer_execution_contract",
    title="optimizer tensor-transfer execution",
    previous_token="optimizer_parity_execution",
    previous_label="P109 optimizer parity execution contract",
    previous_ready_decision=P109_SPEC.ready_decision,
    previous_ready_field="optimizer_parity_execution_contract_ready",
    previous_evidence_field="optimizer_parity_execution_evidence_recorded",
    previous_signed_field="optimizer_parity_execution_signed",
    previous_post_fields="post_p109_request_fields",
    previous_ack="acknowledge_p109_optimizer_parity_execution_contract_recorded",
    package_ready_field="optimizer_tensor_transfer_execution_package_ready",
    policy_ready_field="tensor_transfer_execution_policy_ready",
    row_keys=("optimizer_tensor_transfer_execution_rows", "optimizer_transfer_execution_rows"),
    row_ready_field="tensor_transfer_execution_review_ready",
    later_field="later_optimizer_training_step_execution_contract_required",
    later_ack="acknowledge_later_optimizer_training_step_execution_contract_required",
    next_contract="optimizer_training_step_execution",
    allowed_intents=frozenset(
        {
            "tensor_transfer_execution_candidate",
            "hold_for_more_evidence",
            "reject_tensor_transfer_execution",
        }
    ),
    required_sections=_sections(
        "p109_optimizer_parity_execution_contract_reference",
        "optimizer_tensor_transfer_execution_package",
        "per_optimizer_tensor_transfer_execution_rows",
    ),
    unsafe_true_fields=_unsafe(
        "optimizer_tensor_transfer_execution_applied",
        "optimizer_tensor_transfer_execution_enabled",
        "optimizer_tensor_transfer_execution_executed",
        "tensor_transfer_approved",
        "tensor_transfer_allowed",
        "tensor_transfer_enabled",
        "tensor_transfer_requested",
        "tensor_transfer_executed",
        "tensor_transfer_started",
        "source_tensors_materialized",
        "destination_tensors_materialized",
        "transfer_copy_executed",
        "transfer_result_recorded",
        "training_step_allowed",
        "training_step_executed",
    ),
    unsafe_non_empty_fields=_unsafe(
        "optimizer_tensor_transfer_execution_payload",
        "tensor_transfer_execution_payload",
        "tensor_transfer_authorization_payload",
        "tensor_transfer_source_payload",
        "tensor_transfer_destination_payload",
        "tensor_transfer_stream_event_payload",
        "tensor_transfer_copy_payload",
        "tensor_transfer_result_payload",
        "training_step_payload",
    ),
    inherited_unsafe_true_fields=P109_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P109_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P110_SPEC"]
