"""Stage spec for optimizer run-dispatch contract P103."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_request_execution_job_creation_specs_p102 import P102_SPEC


COMMON_SECTIONS = (
    "run_dispatch_plan_inventory",
    "run_dispatch_precondition_inventory",
    "run_record_boundary",
    "scheduler_dispatch_boundary",
    "training_launch_boundary",
    "runtime_dispatch_boundary",
    "execution_resolution_boundary",
    "operator_dispatch_boundary",
    "observability_boundary",
    "rollback_policy_summary",
    "no_run_dispatch_boundary",
    "no_run_record_write_boundary",
    "no_scheduler_dispatch_boundary",
    "no_training_launch_boundary",
    "no_runtime_dispatch_boundary",
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


P103_SPEC = OptimizerLateStageSpec(
    stage_id=103,
    token="optimizer_run_dispatch",
    scope="optimizer_run_dispatch_contract",
    title="optimizer run-dispatch",
    previous_token="optimizer_request_execution_job_creation",
    previous_label="P102 optimizer request-execution/job-creation contract",
    previous_ready_decision=P102_SPEC.ready_decision,
    previous_ready_field="optimizer_request_execution_job_creation_contract_ready",
    previous_evidence_field="optimizer_request_execution_job_creation_evidence_recorded",
    previous_signed_field="optimizer_request_execution_job_creation_signed",
    previous_post_fields="post_p102_request_fields",
    previous_ack="acknowledge_p102_optimizer_request_execution_job_creation_contract_recorded",
    package_ready_field="optimizer_run_dispatch_package_ready",
    policy_ready_field="run_dispatch_policy_ready",
    row_keys=("optimizer_run_dispatch_rows", "optimizer_dispatch_run_rows"),
    row_ready_field="run_dispatch_review_ready",
    later_field="later_optimizer_training_launch_execution_contract_required",
    later_ack="acknowledge_later_optimizer_training_launch_execution_contract_required",
    next_contract="optimizer_training_launch_execution",
    allowed_intents=frozenset(
        {
            "run_dispatch_candidate",
            "hold_for_more_evidence",
            "reject_run_dispatch",
        }
    ),
    required_sections=_sections(
        "p102_optimizer_request_execution_job_creation_contract_reference",
        "optimizer_run_dispatch_package",
        "per_optimizer_run_dispatch_rows",
    ),
    unsafe_true_fields=_unsafe(
        "optimizer_run_dispatch_applied",
        "optimizer_run_dispatch_enabled",
        "optimizer_run_dispatch_executed",
        "run_dispatch_allowed",
        "run_dispatch_enabled",
        "run_dispatch_executed",
        "runs_dispatched",
        "run_record_written",
        "scheduler_dispatch_allowed",
        "scheduler_dispatch_enabled",
        "scheduler_dispatch_executed",
        "training_launch_allowed",
        "training_launch_enabled",
        "training_launch_executed",
        "runtime_dispatch_allowed",
        "runtime_dispatch_enabled",
        "runtime_dispatch_executed",
        "runtime_execution_executed",
        "native_dispatch_executed",
        "kernel_launch_executed",
        "operator_dispatch_executed",
        "execution_resolution_invoked",
    ),
    unsafe_non_empty_fields=_unsafe(
        "optimizer_run_dispatch_payload",
        "run_dispatch_payload",
        "run_record_payload",
        "scheduler_dispatch_payload",
        "training_launch_payload",
        "runtime_dispatch_payload",
        "training_run_payload",
    ),
    inherited_unsafe_true_fields=P102_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P102_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P103_SPEC"]
