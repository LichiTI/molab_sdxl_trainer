"""Stage spec for optimizer request-submission contract P101."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_request_field_emission_specs_p100 import P100_SPEC


COMMON_SECTIONS = (
    "request_submission_plan_inventory",
    "request_submission_precondition_inventory",
    "request_payload_boundary",
    "request_validation_boundary",
    "backend_router_boundary",
    "router_submission_boundary",
    "request_store_boundary",
    "job_creation_boundary",
    "run_dispatch_boundary",
    "training_launch_boundary",
    "operator_submission_boundary",
    "observability_boundary",
    "rollback_policy_summary",
    "no_request_submission_boundary",
    "no_request_payload_materialization_boundary",
    "no_request_field_emission_boundary",
    "no_backend_router_registration_boundary",
    "no_router_submission_boundary",
    "no_request_store_write_boundary",
    "no_training_request_submit_boundary",
    "no_training_job_creation_boundary",
    "no_run_dispatch_boundary",
    "no_training_step_or_launch_boundary",
    "no_default_rollout_boundary",
)


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


def _unsafe(*fields: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(fields))


P101_SPEC = OptimizerLateStageSpec(
    stage_id=101,
    token="optimizer_request_submission",
    scope="optimizer_request_submission_contract",
    title="optimizer request-submission",
    previous_token="optimizer_request_field_emission",
    previous_label="P100 optimizer request-field emission contract",
    previous_ready_decision=P100_SPEC.ready_decision,
    previous_ready_field="optimizer_request_field_emission_contract_ready",
    previous_evidence_field="optimizer_request_field_emission_evidence_recorded",
    previous_signed_field="optimizer_request_field_emission_signed",
    previous_post_fields="post_p100_request_fields",
    previous_ack="acknowledge_p100_optimizer_request_field_emission_contract_recorded",
    package_ready_field="optimizer_request_submission_package_ready",
    policy_ready_field="request_submission_policy_ready",
    row_keys=("optimizer_request_submission_rows", "optimizer_submission_rows"),
    row_ready_field="request_submission_review_ready",
    later_field="later_optimizer_request_execution_job_creation_contract_required",
    later_ack="acknowledge_later_optimizer_request_execution_job_creation_contract_required",
    next_contract="optimizer_request_execution_job_creation",
    allowed_intents=frozenset(
        {
            "request_submission_candidate",
            "hold_for_more_evidence",
            "reject_request_submission",
        }
    ),
    required_sections=_sections(
        "p100_optimizer_request_field_emission_contract_reference",
        "optimizer_request_submission_package",
        "per_optimizer_request_submission_rows",
    ),
    unsafe_true_fields=_unsafe(
        "optimizer_request_submission_applied",
        "optimizer_request_submission_enabled",
        "optimizer_request_submission_executed",
        "request_submission_allowed",
        "request_submission_enabled",
        "request_submission_executed",
        "request_submission_applied",
        "request_submitted",
        "request_payload_materialized",
        "request_payload_built",
        "request_payload_submitted",
        "backend_router_request_sent",
        "router_submission_allowed",
        "router_submission_executed",
        "request_store_written",
        "training_request_submitted",
        "job_creation_allowed",
        "training_job_created",
        "run_dispatch_allowed",
        "run_dispatched",
        "training_launch_allowed",
        "training_launch_executed",
    ),
    unsafe_non_empty_fields=_unsafe(
        "optimizer_request_submission_payload",
        "request_submission_payload",
        "submitted_request_payload",
        "request_payload",
        "request_validation_payload",
        "backend_router_payload",
        "router_submission_payload",
        "request_store_payload",
        "training_request_payload",
        "job_creation_payload",
        "run_dispatch_payload",
        "training_launch_payload",
        "training_job_payload",
    ),
    inherited_unsafe_true_fields=P100_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P100_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P101_SPEC"]
