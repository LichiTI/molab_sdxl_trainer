"""Stage spec for optimizer training-launch execution contract P112."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_training_step_execution_specs_p111 import P111_SPEC


COMMON_SECTIONS = (
    "training_launch_execution_plan_inventory",
    "training_launch_execution_precondition_inventory",
    "training_launch_authorization_boundary",
    "training_runtime_boundary",
    "training_process_boundary",
    "run_dispatch_boundary",
    "scheduler_dispatch_boundary",
    "request_execution_boundary",
    "job_creation_boundary",
    "operator_training_launch_boundary",
    "observability_boundary",
    "rollback_policy_summary",
    "no_training_launch_execution_boundary",
    "no_training_runtime_start_boundary",
    "no_training_process_start_boundary",
    "no_run_dispatch_boundary",
    "no_run_record_write_boundary",
    "no_scheduler_dispatch_boundary",
    "no_request_execution_boundary",
    "no_job_creation_boundary",
    "no_job_record_write_boundary",
    "no_queue_enqueue_boundary",
    "no_training_step_execution_boundary",
    "no_tensor_transfer_execution_boundary",
    "no_parity_execution_boundary",
    "no_kernel_launch_execution_boundary",
    "no_native_dispatch_execution_boundary",
    "no_runtime_dispatch_execution_boundary",
    "no_runtime_execution_boundary",
    "no_runtime_state_refresh_boundary",
    "no_runtime_adapter_enabled_boundary",
    "no_request_submission_boundary",
    "no_schema_config_router_ui_patch_boundary",
    "no_default_rollout_boundary",
)


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


def _unsafe(*fields: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(fields))


P112_SPEC = OptimizerLateStageSpec(
    stage_id=112,
    token="optimizer_training_launch_execution",
    scope="optimizer_training_launch_execution_contract",
    title="optimizer training-launch execution",
    previous_token="optimizer_training_step_execution",
    previous_label="P111 optimizer training-step execution contract",
    previous_ready_decision=P111_SPEC.ready_decision,
    previous_ready_field="optimizer_training_step_execution_contract_ready",
    previous_evidence_field="optimizer_training_step_execution_evidence_recorded",
    previous_signed_field="optimizer_training_step_execution_signed",
    previous_post_fields="post_p111_request_fields",
    previous_ack="acknowledge_p111_optimizer_training_step_execution_contract_recorded",
    package_ready_field="optimizer_training_launch_execution_package_ready",
    policy_ready_field="training_launch_execution_policy_ready",
    row_keys=("optimizer_training_launch_execution_rows", "optimizer_launch_execution_rows"),
    row_ready_field="training_launch_execution_review_ready",
    later_field="later_optimizer_runtime_execution_preflight_contract_required",
    later_ack="acknowledge_later_optimizer_runtime_execution_preflight_contract_required",
    next_contract="optimizer_runtime_execution_preflight",
    allowed_intents=frozenset(
        {
            "training_launch_execution_candidate",
            "hold_for_more_evidence",
            "reject_training_launch_execution",
        }
    ),
    required_sections=_sections(
        "p111_optimizer_training_step_execution_contract_reference",
        "optimizer_training_launch_execution_package",
        "per_optimizer_training_launch_execution_rows",
    ),
    unsafe_true_fields=_unsafe(
        "optimizer_training_launch_execution_applied",
        "optimizer_training_launch_execution_enabled",
        "optimizer_training_launch_execution_executed",
        "training_launch_approved",
        "training_launch_allowed",
        "training_launch_enabled",
        "training_launch_requested",
        "training_launch_started",
        "training_launch_executed",
        "training_runtime_started",
        "training_process_started",
        "run_dispatch_allowed",
        "run_dispatch_executed",
        "scheduler_dispatch_executed",
        "request_execution_executed",
        "job_created",
        "job_record_written",
        "queue_enqueue_executed",
    ),
    unsafe_non_empty_fields=_unsafe(
        "optimizer_training_launch_execution_payload",
        "training_launch_execution_payload",
        "training_launch_authorization_payload",
        "training_runtime_payload",
        "training_process_payload",
        "run_dispatch_payload",
        "scheduler_dispatch_payload",
        "request_execution_payload",
        "job_creation_payload",
    ),
    inherited_unsafe_true_fields=P111_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P111_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P112_SPEC"]
