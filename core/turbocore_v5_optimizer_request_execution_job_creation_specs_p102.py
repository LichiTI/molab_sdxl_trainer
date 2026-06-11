"""Stage spec for optimizer request-execution/job-creation contract P102."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_request_submission_specs_p101 import P101_SPEC


COMMON_SECTIONS = (
    "request_execution_plan_inventory",
    "request_execution_precondition_inventory",
    "job_creation_plan_inventory",
    "job_creation_precondition_inventory",
    "request_store_boundary",
    "job_record_boundary",
    "queue_enqueue_boundary",
    "run_dispatch_boundary",
    "training_launch_boundary",
    "execution_resolution_boundary",
    "operator_execution_boundary",
    "observability_boundary",
    "rollback_policy_summary",
    "no_request_execution_boundary",
    "no_job_creation_boundary",
    "no_job_record_write_boundary",
    "no_queue_enqueue_boundary",
    "no_run_dispatch_boundary",
    "no_training_launch_boundary",
    "no_request_submission_boundary",
    "no_request_payload_materialization_boundary",
    "no_schema_config_router_ui_patch_boundary",
    "no_default_rollout_boundary",
)


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


def _unsafe(*fields: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(fields))


P102_SPEC = OptimizerLateStageSpec(
    stage_id=102,
    token="optimizer_request_execution_job_creation",
    scope="optimizer_request_execution_job_creation_contract",
    title="optimizer request-execution/job-creation",
    previous_token="optimizer_request_submission",
    previous_label="P101 optimizer request-submission contract",
    previous_ready_decision=P101_SPEC.ready_decision,
    previous_ready_field="optimizer_request_submission_contract_ready",
    previous_evidence_field="optimizer_request_submission_evidence_recorded",
    previous_signed_field="optimizer_request_submission_signed",
    previous_post_fields="post_p101_request_fields",
    previous_ack="acknowledge_p101_optimizer_request_submission_contract_recorded",
    package_ready_field="optimizer_request_execution_job_creation_package_ready",
    policy_ready_field="request_execution_job_creation_policy_ready",
    row_keys=("optimizer_request_execution_job_creation_rows", "optimizer_job_creation_rows"),
    row_ready_field="request_execution_job_creation_review_ready",
    later_field="later_optimizer_run_dispatch_contract_required",
    later_ack="acknowledge_later_optimizer_run_dispatch_contract_required",
    next_contract="optimizer_run_dispatch",
    allowed_intents=frozenset(
        {
            "request_execution_job_creation_candidate",
            "hold_for_more_evidence",
            "reject_request_execution_job_creation",
        }
    ),
    required_sections=_sections(
        "p101_optimizer_request_submission_contract_reference",
        "optimizer_request_execution_job_creation_package",
        "per_optimizer_request_execution_job_creation_rows",
    ),
    unsafe_true_fields=_unsafe(
        "optimizer_request_execution_job_creation_applied",
        "optimizer_request_execution_job_creation_enabled",
        "optimizer_request_execution_job_creation_executed",
        "request_execution_allowed",
        "request_execution_enabled",
        "request_execution_executed",
        "request_execution_applied",
        "job_creation_allowed",
        "job_creation_enabled",
        "job_creation_executed",
        "job_record_written",
        "job_store_written",
        "task_record_written",
        "queue_enqueue_allowed",
        "queue_enqueued",
        "scheduler_enqueued",
        "run_dispatch_allowed",
        "run_dispatch_executed",
        "run_record_written",
        "run_dispatched",
        "execution_resolution_invoked",
        "training_job_created",
        "training_launch_allowed",
        "training_launch_executed",
    ),
    unsafe_non_empty_fields=_unsafe(
        "optimizer_request_execution_job_creation_payload",
        "request_execution_payload",
        "job_creation_payload",
        "job_record_payload",
        "job_store_payload",
        "task_record_payload",
        "queue_enqueue_payload",
        "scheduler_payload",
        "execution_resolution_payload",
        "run_dispatch_payload",
        "run_record_payload",
        "training_launch_payload",
        "training_job_payload",
        "training_run_payload",
    ),
    inherited_unsafe_true_fields=P101_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P101_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P102_SPEC"]
