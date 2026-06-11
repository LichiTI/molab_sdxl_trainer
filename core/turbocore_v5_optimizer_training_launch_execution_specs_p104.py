"""Stage spec for optimizer training-launch execution contract P104."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_run_dispatch_specs_p103 import P103_SPEC


COMMON_SECTIONS = (
    "training_launch_execution_plan_inventory",
    "training_launch_execution_precondition_inventory",
    "training_launch_authorization_boundary",
    "launch_request_boundary",
    "launch_config_boundary",
    "launch_job_boundary",
    "launch_run_boundary",
    "runtime_dispatch_boundary",
    "training_runtime_boundary",
    "operator_launch_boundary",
    "observability_boundary",
    "rollback_policy_summary",
    "no_training_launch_execution_boundary",
    "no_training_runtime_start_boundary",
    "no_runtime_dispatch_execution_boundary",
    "no_native_dispatch_execution_boundary",
    "no_kernel_launch_execution_boundary",
    "no_parity_execution_boundary",
    "no_tensor_transfer_execution_boundary",
    "no_training_step_execution_boundary",
    "no_run_dispatch_boundary",
    "no_run_record_write_boundary",
    "no_scheduler_dispatch_boundary",
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


P104_SPEC = OptimizerLateStageSpec(
    stage_id=104,
    token="optimizer_training_launch_execution",
    scope="optimizer_training_launch_execution_contract",
    title="optimizer training-launch execution",
    previous_token="optimizer_run_dispatch",
    previous_label="P103 optimizer run-dispatch contract",
    previous_ready_decision=P103_SPEC.ready_decision,
    previous_ready_field="optimizer_run_dispatch_contract_ready",
    previous_evidence_field="optimizer_run_dispatch_evidence_recorded",
    previous_signed_field="optimizer_run_dispatch_signed",
    previous_post_fields="post_p103_request_fields",
    previous_ack="acknowledge_p103_optimizer_run_dispatch_contract_recorded",
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
        "p103_optimizer_run_dispatch_contract_reference",
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
        "training_launch_executed",
        "training_launch_applied",
        "training_runtime_started",
        "training_process_started",
        "launch_request_submitted",
        "launch_request_executed",
        "launch_job_started",
        "launch_run_started",
        "runtime_dispatch_allowed",
        "runtime_dispatch_enabled",
        "runtime_dispatch_executed",
        "runtime_execution_executed",
        "native_dispatch_executed",
        "kernel_launch_executed",
        "parity_executed",
        "tensor_transfer_executed",
        "training_step_executed",
    ),
    unsafe_non_empty_fields=_unsafe(
        "optimizer_training_launch_execution_payload",
        "training_launch_execution_payload",
        "launch_request_payload",
        "launch_config_payload",
        "launch_job_payload",
        "launch_run_payload",
        "runtime_dispatch_payload",
        "training_runtime_payload",
    ),
    inherited_unsafe_true_fields=P103_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P103_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P104_SPEC"]
